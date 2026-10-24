use super::*;

/// Vulkan graphical tests share SDL video ownership and a singleton GPU;
/// synthetic failures can poison them across Rust's serial test threads.
/// Give each scenario its own SDL/Vulkan process while leaving pure tests in
/// this process. Exact selection prevents recursive execution of the suite.
pub(crate) fn isolate_vulkan_graphical_test(name: &str) -> bool {
    if std::env::var_os("ROSEMARY_TEST_VULKAN").is_none()
        || std::env::var_os("ROSEMARY_VULKAN_GRAPHICS_CHILD").is_some()
    {
        return false;
    }
    let output = std::process::Command::new(std::env::current_exe().expect("test executable"))
        .arg("--exact")
        .arg(if name.contains("::") {
            name.to_owned()
        } else {
            format!("swapchain::tests::{name}")
        })
        .arg("--nocapture")
        .env("ROSEMARY_VULKAN_GRAPHICS_CHILD", "1")
        .output()
        .expect("start isolated Vulkan graphical test");
    assert!(
        output.status.success(),
        "isolated {name} failed ({:?}):\n{}\n{}",
        output.status.code(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        String::from_utf8_lossy(&output.stdout).contains("1 passed"),
        "isolated {name} did not execute: {}",
        String::from_utf8_lossy(&output.stdout)
    );
    true
}

/// Tests run without the background compositor worker, so they can prove several
/// independent layer copies are coalesced into one actual WSI presentation.
fn flush_if_composed(window: sdl3::Window) {
    let entry = SWAPCHAINS
        .lock()
        .expect("window swapchains")
        .as_ref()
        .and_then(|all| all.get(&(window as usize)).cloned())
        .expect("swapchain for test window");
    let _presenting = entry.presenting.lock().expect("window presentation");
    let mut swapchain = entry.state.lock().expect("window swapchain");
    if !uses_layer_compositor(swapchain.present_mode) {
        return;
    }
    assert!(
        swapchain.composite_dirty,
        "independent layer updates were not staged"
    );
    let before = swapchain.frame;
    swapchain
        .flush_composite(window)
        .expect("one coalesced window presentation");
    assert!(
        !swapchain.composite_dirty,
        "window presentation did not drain the accumulator"
    );
    assert_eq!(
        swapchain.frame,
        before + 1,
        "the flush submitted multiple WSI frames"
    );
}

#[test]
#[cfg_attr(target_os = "macos", ignore = "SDL main thread required")]
fn vulkan_graphical_test_prerequisites_are_available() {
    if isolate_vulkan_graphical_test("vulkan_graphical_test_prerequisites_are_available") {
        return;
    }
    if std::env::var_os("ROSEMARY_TEST_VULKAN").is_none() {
        return;
    }
    assert!(sdl3::video::init_video(), "SDL video subsystem unavailable");
    assert!(
        sdl3::may_pump(),
        "Vulkan tests must run on SDL's video owner thread"
    );
    let adapters = crate::vulkan::adapters().expect("Vulkan adapters unavailable");
    let gpu = adapters
        .first()
        .expect("no Vulkan adapter")
        .gpu()
        .expect("Vulkan logical device unavailable");
    let window = sdl3::video::create_window(c"Vulkan test prerequisites", 80, 64, 0)
        .expect("SDL video window unavailable");
    sdl3::video::presentation_window(window)
        .expect("presentation metadata unavailable on SDL owner thread");
    let instance = crate::vulkan::instance_handle().expect("Vulkan instance unavailable");
    let surface = sdl3::video::vulkan::create_surface(window, instance, crate::vulkan::library())
        .expect("SDL Vulkan surface creation failed");
    assert!(
        gpu.can_present(vk::SurfaceKHR::from_raw(surface)),
        "the Vulkan device queue cannot present to this window"
    );
    assert!(
        sdl3::video::vulkan::owns_window(window),
        "SDL conversion not registered"
    );
    sdl3::video::vulkan::destroy_surface(instance, surface);
    sdl3::video::destroy_window(window);
}

#[test]
fn present_fences_are_not_awaited_after_unqueued_errors() {
    use vk::Result as R;
    for gamescope in [false, true] {
        for error in [
            R::ERROR_DEVICE_LOST,
            R::ERROR_OUT_OF_HOST_MEMORY,
            R::ERROR_OUT_OF_DEVICE_MEMORY,
        ] {
            assert!(!present_fence_will_signal(&Err(error), gamescope));
        }
        assert!(present_fence_will_signal(&Ok(false), gamescope));
        assert!(present_fence_will_signal(&Ok(true), gamescope));
    }
    // The specification queues semaphore/fence work for these results, but
    // Gamescope's alternate WSI path has a known missing-signal behavior.
    for error in [R::ERROR_OUT_OF_DATE_KHR, R::ERROR_SURFACE_LOST_KHR] {
        assert!(present_fence_will_signal(&Err(error), false));
        assert!(!present_fence_will_signal(&Err(error), true));
    }
}

#[test]
fn fifo_limits_wsi_images_without_changing_mailbox_capacity() {
    assert_eq!(desired_image_count(vk::PresentModeKHR::FIFO, 3, 8), 4);
    assert_eq!(desired_image_count(vk::PresentModeKHR::FIFO, 3, 0), 4);
    assert_eq!(desired_image_count(vk::PresentModeKHR::FIFO, 3, 3), 3);
    assert_eq!(desired_image_count(vk::PresentModeKHR::MAILBOX, 3, 8), 4);
    assert_eq!(desired_image_count(vk::PresentModeKHR::IMMEDIATE, 2, 0), 2);
    assert_eq!(desired_image_count(vk::PresentModeKHR::IMMEDIATE, 3, 3), 3);
    assert_eq!(desired_image_count(vk::PresentModeKHR::IMMEDIATE, 2, 2), 2);
    assert_eq!(
        desired_image_count(vk::PresentModeKHR::FIFO_RELAXED, 3, 8),
        4
    );
}

#[test]
fn all_present_modes_compose_layers_before_wsi() {
    for mode in [
        vk::PresentModeKHR::FIFO,
        vk::PresentModeKHR::FIFO_RELAXED,
        vk::PresentModeKHR::MAILBOX,
        vk::PresentModeKHR::IMMEDIATE,
    ] {
        assert!(uses_layer_compositor(mode));
        assert_eq!(frame_slots(mode), FRAMES_IN_FLIGHT);
    }
}

#[test]
fn layer_compositing_batch_delay_distinguishes_wsi_modes() {
    for rate in [30.0, 60.0, 165.0, 240.0] {
        let immediate = compositor_batch_delay(vk::PresentModeKHR::IMMEDIATE, rate);
        let mailbox = compositor_batch_delay(vk::PresentModeKHR::MAILBOX, rate);
        let fifo = compositor_batch_delay(vk::PresentModeKHR::FIFO, rate);
        assert_eq!(mailbox, immediate);
        assert_eq!(
            compositor_batch_delay(vk::PresentModeKHR::FIFO_RELAXED, rate),
            fifo
        );
        assert!(immediate < fifo);
        assert!(immediate <= Duration::from_millis(2));
        assert!(fifo <= Duration::from_millis(4));
    }
}

#[test]
fn single_layer_fast_path_requires_an_owned_opaque_full_window() {
    let frame = appkit::view::Frame {
        window: std::ptr::null_mut(),
        x: 0,
        y: 0,
        width: 320,
        height: 240,
        window_width: 320,
        window_height: 240,
    };
    for mode in [vk::PresentModeKHR::IMMEDIATE, vk::PresentModeKHR::MAILBOX] {
        assert!(fast_direct_layer(mode, 1, &frame, None, false, true));
        assert!(!fast_direct_layer(mode, 1, &frame, None, false, false));
        assert!(!fast_direct_layer(mode, 2, &frame, None, false, true));
        assert!(!fast_direct_layer(mode, 1, &frame, None, true, true));
        assert!(!fast_direct_layer(
            mode,
            1,
            &frame,
            Some(&[Region {
                x: 0,
                y: 0,
                width: 32,
                height: 32
            }]),
            false,
            true,
        ));
        let shifted = appkit::view::Frame { x: 1, ..frame };
        assert!(!fast_direct_layer(mode, 1, &shifted, None, false, true));
    }
    for mode in [vk::PresentModeKHR::FIFO, vk::PresentModeKHR::FIFO_RELAXED] {
        assert!(!fast_direct_layer(mode, 1, &frame, None, false, true));
    }
}

#[test]
fn mailbox_presentation_is_preferred_when_available() {
    if isolate_vulkan_graphical_test("mailbox_presentation_is_preferred_when_available") {
        return;
    }
    assert_eq!(
        choose_present_mode(&[vk::PresentModeKHR::FIFO, vk::PresentModeKHR::MAILBOX]),
        vk::PresentModeKHR::MAILBOX
    );
    assert_eq!(
        choose_present_mode(&[vk::PresentModeKHR::IMMEDIATE, vk::PresentModeKHR::FIFO]),
        vk::PresentModeKHR::FIFO
    );
}

/// A window, a swapchain and a frame that reaches it.
///
/// Everything this module does is a call into a driver and a window
/// system, so there is nothing here a unit test can stand in for: the
/// test opens a real window on the host it runs on, presents into it, and
/// is skipped where there is no such host. `ROSEMARY_TEST_VULKAN=1` asks
/// for it, the same switch the adapter test uses.
#[test]
#[cfg_attr(
    target_os = "macos",
    ignore = "Cocoa needs the main thread; use rosemary's Metal guest tests"
)]
fn a_window_is_presented_to_through_its_swapchain() {
    if isolate_vulkan_graphical_test("a_window_is_presented_to_through_its_swapchain") {
        return;
    }
    if std::env::var_os("ROSEMARY_TEST_VULKAN").is_none() {
        return;
    }
    let Ok(adapters) = crate::vulkan::adapters() else {
        return;
    };
    let Some(gpu) = adapters.first().and_then(|adapter| adapter.gpu().ok()) else {
        return;
    };
    assert!(
        crate::vulkan::surface_api().is_some(),
        "the instance was created without VK_KHR_surface"
    );
    assert!(
        gpu.swapchain_api().is_some(),
        "the device was created without VK_KHR_swapchain"
    );

    // The flags a window of this runtime is actually made with: no
    // graphics API at all, so that this is the call that decides. A test
    // which asked for `VULKAN` here would not be testing the conversion
    // every real window goes through.
    let flags = sdl3::video::flags::HIGH_PIXEL_DENSITY;
    let Some(window) = sdl3::video::create_window(c"rosemary swapchain", 320, 240, flags) else {
        return;
    };
    sdl3::video::show_window(window);
    let (width, height) =
        sdl3::video::window_size_in_pixels(window).expect("a window SDL just made");
    let frame = appkit::view::Frame {
        window,
        x: 0,
        y: 0,
        width: width as u32,
        height: height as u32,
        window_width: width as u32,
        window_height: height as u32,
    };
    assert!(frame.covers_window());
    // The window is converted by the first presentation and not before.
    assert_eq!(
        sdl3::video::window_flags(window) & sdl3::video::flags::VULKAN,
        0,
        "the window was a Vulkan window before anything presented to it"
    );

    let source = gpu
        .create_image(&ImageDescription {
            image_type: vk::ImageType::TYPE_2D,
            view_type: vk::ImageViewType::TYPE_2D,
            format: vk::Format::B8G8R8A8_UNORM,
            width: frame.width,
            height: frame.height,
            depth: 1,
            levels: 1,
            layers: 1,
            samples: 1,
            usage: gpu::USAGE_RENDER_TARGET,
            cube: false,
            components: vk::ComponentMapping::default(),
        })
        .expect("a colour image");

    // More frames than the swapchain has images times the frames in
    // flight, so that every fence, every acquisition semaphore and every
    // per-image presentation semaphore is reused several times over.
    // Reuse is where a synchronization mistake shows up; the first frame
    // never does.
    let handle = || {
        SWAPCHAINS
            .lock()
            .expect("window swapchains")
            .as_ref()
            .and_then(|all| all.get(&(window as usize)))
            .map(|window| window.state.lock().expect("window swapchain").handle)
    };
    for _ in 0..12 {
        assert_eq!(
            present(gpu, 1, &frame, &source, &[]),
            Ok(true),
            "the window was not presented to through a swapchain"
        );
    }
    flush_if_composed(window);
    // ...and is one afterwards, having been converted in place from a
    // window with no graphics API - the case a real guest's window is
    // in, whatever order it created its `MTLDevice` in.
    assert_ne!(
        sdl3::video::window_flags(window) & sdl3::video::flags::VULKAN,
        0,
        "presenting did not make the window a Vulkan window"
    );
    let steady = handle().expect("a swapchain for the window");
    for _ in 0..12 {
        assert_eq!(present(gpu, 1, &frame, &source, &[]), Ok(true));
    }
    flush_if_composed(window);
    // A window that has not resized keeps the swapchain it has. A
    // surface with a `currentExtent` of its own answers a size that is
    // not the one that was asked for, and comparing against *that* would
    // rebuild the swapchain on every single frame.
    assert_eq!(
        handle(),
        Some(steady),
        "the swapchain was rebuilt without the window resizing"
    );

    // Presentation survives the window changing size, which is a
    // different swapchain and a different set of images.
    sdl3::video::set_window_size(window, 400, 300);
    let (width, height) = sdl3::video::window_size_in_pixels(window).expect("still a window");
    let resized = appkit::view::Frame {
        width: width as u32,
        height: height as u32,
        window_width: width as u32,
        window_height: height as u32,
        ..frame
    };
    assert_eq!(present(gpu, 1, &resized, &source, &[]), Ok(true));
    flush_if_composed(window);

    // Take the swapchain out of the map so that its `Drop` runs the
    // teardown while the window it presented to is still alive.
    SWAPCHAINS
        .lock()
        .expect("window swapchains")
        .as_mut()
        .and_then(|all| all.remove(&(window as usize)));
    gpu.destroy_image(source);
    sdl3::video::destroy_window(window);
}

/// Exercise the real asynchronous scheduler, which the deterministic
/// coalescing tests intentionally keep disabled. Run separately with
/// ROSEMARY_TEST_VULKAN=1 and a selected WSI mode (IMMEDIATE or FIFO).
#[test]
#[cfg_attr(target_os = "macos", ignore = "Cocoa needs the main thread")]
fn two_composited_windows_present_on_independent_workers() {
    if isolate_vulkan_graphical_test("two_composited_windows_present_on_independent_workers") {
        return;
    }
    let _retire_guard = AwaitTestWindowRetirements;
    if std::env::var_os("ROSEMARY_TEST_VULKAN").is_none() {
        return;
    }
    let Ok(adapters) = crate::vulkan::adapters() else {
        return;
    };
    let Some(gpu) = adapters.first().and_then(|adapter| adapter.gpu().ok()) else {
        return;
    };
    let mut windows = Vec::new();
    for _ in 0..4 {
        let window = sdl3::video::create_window(
            c"rosemary fifo independent window",
            320,
            240,
            sdl3::video::flags::HIGH_PIXEL_DENSITY,
        )
        .expect("X11 test window");
        sdl3::video::show_window(window);
        let (w, h) = sdl3::video::window_size_in_pixels(window).expect("window size");
        let source = gpu
            .create_image(&ImageDescription {
                image_type: vk::ImageType::TYPE_2D,
                view_type: vk::ImageViewType::TYPE_2D,
                format: vk::Format::B8G8R8A8_UNORM,
                width: w as u32,
                height: h as u32,
                depth: 1,
                levels: 1,
                layers: 1,
                samples: 1,
                usage: gpu::USAGE_RENDER_TARGET,
                cube: false,
                components: vk::ComponentMapping::default(),
            })
            .expect("source image");
        let frame = appkit::view::Frame {
            window,
            x: 0,
            y: 0,
            width: w as u32,
            height: h as u32,
            window_width: w as u32,
            window_height: h as u32,
        };
        assert_eq!(
            present(gpu, window as usize, &frame, &source, &[]),
            Ok(true)
        );
        let entry = SWAPCHAINS
            .lock()
            .expect("swapchains")
            .as_ref()
            .and_then(|all| all.get(&(window as usize)).cloned())
            .expect("test window swapchain");
        assert!(uses_layer_compositor(
            entry.state.lock().expect("state").present_mode
        ));
        ensure_layer_compositor_worker(window as usize, &entry);
        windows.push((window, source, entry));
    }
    let deadline = Instant::now() + Duration::from_secs(3);
    for (_, _, entry) in &windows {
        loop {
            let state = entry.state.lock().expect("window state");
            if !state.composite_dirty && state.frame == 2 {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "FIFO worker did not drain its window"
            );
            drop(state);
            std::thread::sleep(Duration::from_millis(2));
        }
    }
    // Deliberately hold window A's acquisition/presentation lock. The
    // global drawable dispatcher must defer A immediately and still stage B.
    let first = &windows[0];
    let second = &windows[1];
    let make_frame = |window: sdl3::Window| {
        let (w, h) = sdl3::video::window_size_in_pixels(window).expect("window size");
        appkit::view::Frame {
            window,
            x: 0,
            y: 0,
            width: w as u32,
            height: h as u32,
            window_width: w as u32,
            window_height: h as u32,
        }
    };
    let window_key = first.0 as usize;
    let snapshot = sdl3::video::presentation_window(first.0).expect("main-thread video snapshot");
    let from_worker =
        std::thread::spawn(move || sdl3::video::presentation_window(window_key as sdl3::Window))
            .join()
            .expect("worker snapshot");
    assert_eq!(
        from_worker,
        Some(snapshot),
        "worker must use main-thread snapshot"
    );
    let window_id = sdl3::video::window_id(first.0);
    sdl3::video::invalidate_presentation_window(window_id, &[first.0]);
    let uninitialized =
        std::thread::spawn(move || sdl3::video::presentation_window(window_key as sdl3::Window))
            .join()
            .expect("worker query");
    assert_eq!(
        uninitialized, None,
        "worker may not call SDL for uncached geometry"
    );
    assert!(
        sdl3::video::presentation_window(first.0).is_some(),
        "main thread repairs cache"
    );
    let first_guard = first
        .2
        .presenting
        .lock()
        .expect("deliberate WSI contention");
    assert!(
        window_presenting_busy(first.0),
        "AppKit must defer busy window A"
    );
    assert!(
        !window_presenting_busy(second.0),
        "window B remains available"
    );
    assert_eq!(
        present_nonblocking(gpu, first.0 as usize, &make_frame(first.0), &first.1, &[]),
        Ok(None),
        "backing-store staging must not block AppKit on A"
    );
    assert_eq!(
        present_nonblocking(
            gpu,
            second.0 as usize,
            &make_frame(second.0),
            &second.1,
            &[]
        ),
        Ok(Some(true)),
        "backing-store staging must still progress on B"
    );
    let before = Instant::now();
    assert_eq!(
        present_owned(
            gpu,
            first.0 as usize,
            &make_frame(first.0),
            &first.1,
            &[],
            0
        ),
        Ok(None),
        "contended window must defer rather than block the dispatcher"
    );
    assert_eq!(
        present_owned(
            gpu,
            second.0 as usize,
            &make_frame(second.0),
            &second.1,
            &[],
            0
        ),
        Ok(Some(true)),
        "uncontended window must still stage"
    );
    assert!(
        before.elapsed() < Duration::from_millis(200),
        "window B stalled on A"
    );
    drop(first_guard);

    // Surface loss is window-local. Another window must continue.
    *first.2.injected_acquire_error.lock().expect("injection") =
        Some(vk::Result::ERROR_SURFACE_LOST_KHR);
    assert_eq!(
        present(gpu, first.0 as usize, &make_frame(first.0), &first.1, &[]),
        Ok(true)
    );
    assert_eq!(
        present(
            gpu,
            second.0 as usize,
            &make_frame(second.0),
            &second.1,
            &[]
        ),
        Ok(true)
    );
    let until = Instant::now() + Duration::from_secs(3);
    while first.2.present_failure.load(Ordering::Acquire) != 1 {
        assert!(
            Instant::now() < until,
            "terminal WSI error must stop the worker"
        );
        std::thread::sleep(Duration::from_millis(2));
    }
    assert!(
        first
            .2
            .injected_acquire_error
            .lock()
            .expect("injection")
            .is_none()
    );
    assert_eq!(
        present_owned(
            gpu,
            first.0 as usize,
            &make_frame(first.0),
            &first.1,
            &[],
            0
        ),
        Ok(Some(true)),
        "surface-lost window must not transition to incompatible SDL window surfaces"
    );
    let until = Instant::now() + Duration::from_secs(3);
    while second.2.state.lock().expect("state").composite_dirty {
        assert!(
            Instant::now() < until,
            "healthy window B cannot be stalled by A"
        );
        std::thread::sleep(Duration::from_millis(2));
    }
    // SURFACE_LOST terminates its worker too, but chooses the SDL readback
    // fallback instead of submitting more work to the invalid WSI surface.
    *second.2.injected_acquire_error.lock().expect("injection") =
        Some(vk::Result::ERROR_SURFACE_LOST_KHR);
    assert_eq!(
        present(
            gpu,
            second.0 as usize,
            &make_frame(second.0),
            &second.1,
            &[]
        ),
        Ok(true)
    );
    let until = Instant::now() + Duration::from_secs(3);
    while second.2.present_failure.load(Ordering::Acquire) != 1 {
        assert!(Instant::now() < until, "surface-lost worker must exit");
        std::thread::sleep(Duration::from_millis(2));
    }
    assert_eq!(
        present_owned(
            gpu,
            second.0 as usize,
            &make_frame(second.0),
            &second.1,
            &[],
            0
        ),
        Ok(Some(true)),
        "surface-lost window must be disabled until re-creation"
    );
    // Repeat terminal failure injection *at vkQueuePresentKHR*, after a
    // successful WSI image acquisition, not at vkAcquireNextImageKHR.
    for (at, error, expected, should_fallback) in [
        (2usize, vk::Result::ERROR_SURFACE_LOST_KHR, 1, true),
        (3usize, vk::Result::ERROR_DEVICE_LOST, 2, true),
    ] {
        let (window, source, entry) = &windows[at];
        entry
            .state
            .lock()
            .expect("swapchain state")
            .injected_queue_present_error = Some(error);
        assert_eq!(
            present(gpu, *window as usize, &make_frame(*window), source, &[]),
            Ok(true)
        );
        let until = Instant::now() + Duration::from_secs(3);
        while entry.present_failure.load(Ordering::Acquire) != expected {
            assert!(
                Instant::now() < until,
                "queue-present terminal error must stop worker: {error:?}"
            );
            std::thread::sleep(Duration::from_millis(2));
        }
        let state = entry.state.lock().expect("state");
        assert!(
            state.composite_dirty,
            "failed presentation cannot clear dirty accumulator"
        );
        assert!(
            state.injected_queue_present_error.is_none(),
            "injected fault not consumed"
        );
        drop(state);
        assert_eq!(
            present_nonblocking(gpu, *window as usize, &make_frame(*window), source, &[]),
            Ok(Some(should_fallback)),
            "wrong policy after queue-present error: {error:?}"
        );
    }
    assert!(
        gpu.lost.load(Ordering::Acquire),
        "device loss must mark the logical device"
    );
    for (window, _, entry) in &windows {
        assert_eq!(
            entry.present_failure.load(Ordering::Acquire),
            2,
            "device loss must stop every window of this GPU: {window:p}"
        );
    }
    for (window, source, entry) in windows {
        forget_window(window);
        drop(entry);
        gpu.destroy_image(source);
        sdl3::video::destroy_window(window);
    }
    // The fault was injected after successful real Vulkan presentation.
    // The host device remains operational for independent unit tests.
    gpu.lost.store(false, Ordering::Release);
}

/// SDL creates the surface on its video thread; the Vulkan initializer must
/// not serialize another window behind its (deliberately slow) driver call.
#[test]
#[cfg_attr(target_os = "macos", ignore = "SDL/Cocoa main thread integration")]
fn slow_window_initialization_does_not_block_other_windows_or_teardown() {
    if isolate_vulkan_graphical_test(
        "slow_window_initialization_does_not_block_other_windows_or_teardown",
    ) {
        return;
    }
    let _retire_guard = AwaitTestWindowRetirements;
    if std::env::var_os("ROSEMARY_TEST_VULKAN").is_none() {
        return;
    }
    let Ok(adapters) = crate::vulkan::adapters() else {
        return;
    };
    let Some(gpu) = adapters.first().and_then(|adapter| adapter.gpu().ok()) else {
        return;
    };
    let mut created = Vec::new();
    for _ in 0..2 {
        let window = sdl3::video::create_window(c"independent initialization", 240, 180, 0)
            .expect("host SDL window");
        sdl3::video::show_window(window);
        let _ = sdl3::video::presentation_window(window).expect("video snapshot");
        let source = gpu
            .create_image(&ImageDescription {
                image_type: vk::ImageType::TYPE_2D,
                view_type: vk::ImageViewType::TYPE_2D,
                format: vk::Format::B8G8R8A8_UNORM,
                width: 240,
                height: 180,
                depth: 1,
                levels: 1,
                layers: 1,
                samples: 1,
                usage: gpu::USAGE_RENDER_TARGET,
                cube: false,
                components: vk::ComponentMapping::default(),
            })
            .expect("source");
        let frame = appkit::view::Frame {
            window,
            x: 0,
            y: 0,
            width: 240,
            height: 180,
            window_width: 240,
            window_height: 180,
        };
        created.push((window, source, frame));
    }
    let a = &created[0];
    let b = &created[1];
    TEST_SLOW_INITIALIZER_STARTED.store(false, Ordering::Release);
    TEST_SLOW_INITIALIZER.store(a.0 as usize, Ordering::Release);
    assert_eq!(
        present_impl(gpu, a.0 as usize, &a.2, &a.1, &[], None, None),
        Ok(None)
    );
    let deadline = Instant::now() + Duration::from_secs(2);
    while !TEST_SLOW_INITIALIZER_STARTED.load(Ordering::Acquire) {
        assert!(Instant::now() < deadline, "slow initializer did not start");
        std::thread::sleep(Duration::from_millis(1));
    }
    let start = Instant::now();
    assert_eq!(present(gpu, b.0 as usize, &b.2, &b.1, &[]), Ok(true));
    assert!(
        start.elapsed() < Duration::from_millis(400),
        "window B initialization stalled behind window A's 600ms Vulkan initializer"
    );
    forget_window(a.0);
    sdl3::video::destroy_window(a.0);
    // SDL_Window destruction may be deferred until A's cancellation guard
    // retires; the pump completes all outstanding surface cleanup.
    std::thread::sleep(Duration::from_millis(650));
    sdl3::video::vulkan::service_surface_requests();
    TEST_SLOW_INITIALIZER.store(0, Ordering::Release);
    forget_window(b.0);
    sdl3::video::destroy_window(b.0);
    for (_, source, _) in created {
        gpu.destroy_image(source);
    }
}

/// A clean layer compositor worker must terminate when its Vulkan device is lost,
/// without needing another layer frame or SDL window destruction to wake it.
#[test]
#[cfg_attr(target_os = "macos", ignore = "Cocoa needs the main thread")]
fn idle_compositor_worker_exits_on_device_loss_without_new_frames() {
    if isolate_vulkan_graphical_test(
        "idle_compositor_worker_exits_on_device_loss_without_new_frames",
    ) {
        return;
    }
    let _retire_guard = AwaitTestWindowRetirements;
    if std::env::var_os("ROSEMARY_TEST_VULKAN").is_none() {
        return;
    }
    let adapters = crate::vulkan::adapters().expect("Vulkan adapter discovery");
    let gpu = adapters
        .first()
        .expect("Vulkan adapter")
        .gpu()
        .expect("Vulkan GPU");
    let window =
        sdl3::video::create_window(c"idle FIFO worker loss", 160, 120, 0).expect("SDL window");
    sdl3::video::show_window(window);
    let _ = sdl3::video::presentation_window(window).expect("SDL owner snapshot");
    let source = gpu
        .create_image(&ImageDescription {
            image_type: vk::ImageType::TYPE_2D,
            view_type: vk::ImageViewType::TYPE_2D,
            format: vk::Format::B8G8R8A8_UNORM,
            width: 160,
            height: 120,
            depth: 1,
            levels: 1,
            layers: 1,
            samples: 1,
            usage: gpu::USAGE_RENDER_TARGET,
            cube: false,
            components: vk::ComponentMapping::default(),
        })
        .expect("layer source");
    let frame = appkit::view::Frame {
        window,
        x: 0,
        y: 0,
        width: 160,
        height: 120,
        window_width: 160,
        window_height: 120,
    };
    assert_eq!(
        present(gpu, window as usize, &frame, &source, &[]),
        Ok(true)
    );
    let entry = SWAPCHAINS
        .lock()
        .expect("swapchains")
        .as_ref()
        .unwrap()
        .get(&(window as usize))
        .expect("FIFO swapchain")
        .clone();
    assert!(uses_layer_compositor(
        entry.state.lock().unwrap().present_mode
    ));
    TEST_COMPOSITOR_IDLE_PARKED.store(0, Ordering::Release);
    ensure_layer_compositor_worker(window as usize, &entry);
    let deadline = Instant::now() + Duration::from_secs(3);
    loop {
        let state = entry.state.lock().unwrap();
        if !state.composite_dirty
            && TEST_COMPOSITOR_IDLE_PARKED.load(Ordering::Acquire) == window as usize
        {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "FIFO worker did not reach idle sleep"
        );
        drop(state);
        std::thread::sleep(Duration::from_millis(2));
    }
    // Notifying without a new frame must wake the sleeping worker.
    record_device_loss(gpu);
    let deadline = Instant::now() + Duration::from_secs(3);
    loop {
        let finished = entry
            .worker
            .lock()
            .unwrap()
            .as_ref()
            .is_some_and(|worker| worker.is_finished());
        if finished {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "idle FIFO worker retained after device loss"
        );
        std::thread::sleep(Duration::from_millis(2));
    }
    assert_eq!(entry.present_failure.load(Ordering::Acquire), 2);
    assert!(!entry.state.lock().unwrap().composite_dirty);
    forget_window(window);
    drop(entry);
    sdl3::video::destroy_window(window);
    gpu.lost.store(false, Ordering::Release); // synthetic loss only
    gpu.destroy_image(source);
}

#[test]
fn initializer_scope_guard_cannot_clear_replacement_token() {
    let key = 0x7f34_9125usize;
    let first = Arc::new(AtomicBool::new(false));
    let replacement = Arc::new(AtomicBool::new(false));
    let first_guard = InitializerRegistration {
        window: key,
        token: Arc::clone(&first),
    };
    INITIALIZING
        .lock()
        .expect("initializers")
        .get_or_insert_with(HashMap::new)
        .insert(key, Arc::clone(&first));
    let replacement_guard = InitializerRegistration {
        window: key,
        token: Arc::clone(&replacement),
    };
    INITIALIZING
        .lock()
        .expect("initializers")
        .as_mut()
        .unwrap()
        .insert(key, Arc::clone(&replacement));
    drop(first_guard);
    assert!(
        INITIALIZING
            .lock()
            .unwrap()
            .as_ref()
            .unwrap()
            .get(&key)
            .is_some_and(|current| Arc::ptr_eq(current, &replacement)),
        "old initializer cleared the next generation"
    );
    drop(replacement_guard);
    assert!(
        !INITIALIZING
            .lock()
            .unwrap()
            .as_ref()
            .unwrap()
            .contains_key(&key),
        "initializer scope guard did not clear its own generation"
    );
}

#[test]
#[cfg_attr(target_os = "macos", ignore = "SDL/Cocoa main-thread integration")]
fn device_loss_before_swapchain_create_cleans_pending_initializer() {
    if isolate_vulkan_graphical_test(
        "device_loss_before_swapchain_create_cleans_pending_initializer",
    ) {
        return;
    }
    if std::env::var_os("ROSEMARY_TEST_VULKAN").is_none() {
        return;
    }
    let adapters = crate::vulkan::adapters().expect("real Vulkan adapters required");
    let gpu = adapters
        .first()
        .expect("Vulkan device")
        .gpu()
        .expect("usable GPU");
    let window =
        sdl3::video::create_window(c"early device loss", 170, 120, 0).expect("SDL video window");
    sdl3::video::show_window(window);
    sdl3::video::presentation_window(window).expect("SDL video-thread metadata");
    let key = window as usize;
    let frame = appkit::view::Frame {
        window,
        x: 0,
        y: 0,
        width: 170,
        height: 120,
        window_width: 170,
        window_height: 120,
    };
    let source = gpu
        .create_image(&ImageDescription {
            image_type: vk::ImageType::TYPE_2D,
            view_type: vk::ImageViewType::TYPE_2D,
            format: vk::Format::B8G8R8A8_UNORM,
            width: 170,
            height: 120,
            depth: 1,
            levels: 1,
            layers: 1,
            samples: 1,
            usage: gpu::USAGE_RENDER_TARGET,
            cube: false,
            components: vk::ComponentMapping::default(),
        })
        .expect("test GPU image");
    TEST_SLOW_INITIALIZER_STARTED.store(false, Ordering::Release);
    TEST_SLOW_INITIALIZER.store(key, Ordering::Release);
    let deadline = Instant::now() + Duration::from_secs(3);
    while !TEST_SLOW_INITIALIZER_STARTED.load(Ordering::Acquire) {
        assert!(
            Instant::now() < deadline,
            "initializer did not reach pre-create checkpoint"
        );
        let _ = present_impl(gpu, key, &frame, &source, &[], None, None);
        sdl3::video::vulkan::service_surface_requests();
        std::thread::sleep(Duration::from_millis(2));
    }
    assert!(
        sdl3::video::vulkan::owns_window(window),
        "SDL window must be converted"
    );
    assert!(
        INITIALIZING
            .lock()
            .unwrap()
            .as_ref()
            .unwrap()
            .contains_key(&key)
    );
    gpu.lost.store(true, Ordering::Release);
    let deadline = Instant::now() + Duration::from_secs(3);
    while INITIALIZING
        .lock()
        .unwrap()
        .as_ref()
        .is_some_and(|all| all.contains_key(&key))
    {
        assert!(
            Instant::now() < deadline,
            "orphaned initializer entry after device loss"
        );
        std::thread::sleep(Duration::from_millis(2));
    }
    assert!(
        !sdl3::video::vulkan::surface_request_pending(window),
        "aborted converted surface remained taken"
    );
    assert!(
        !window_initialization_pending(window),
        "backing store permanently deferred"
    );
    assert_eq!(
        crate::present::present_store(window, &appkit::backing::Store::new(170, 120), &[], &[]),
        appkit::backing::PresentOutcome::Displayed,
        "Vulkan-converted window cannot use SDL CPU fallback after loss"
    );
    TEST_SLOW_INITIALIZER.store(0, Ordering::Release);
    gpu.lost.store(false, Ordering::Release); // synthetic flag; real device works
    forget_window(window);
    sdl3::video::vulkan::service_surface_requests();
    sdl3::video::destroy_window(window);
    gpu.destroy_image(source);
}

#[test]
fn a_transferred_surface_is_retired_once_when_metadata_disappears() {
    if isolate_vulkan_graphical_test(
        "a_transferred_surface_is_retired_once_when_metadata_disappears",
    ) {
        return;
    }
    if std::env::var_os("ROSEMARY_TEST_VULKAN").is_none() {
        return;
    }
    let Ok(adapters) = crate::vulkan::adapters() else {
        return;
    };
    let Some(gpu) = adapters.first().and_then(|a| a.gpu().ok()) else {
        return;
    };
    let window = sdl3::video::create_window(c"surface ownership", 160, 100, 0).unwrap();
    let _ = sdl3::video::presentation_window(window).unwrap();
    let id = sdl3::video::window_id(window);
    sdl3::video::invalidate_presentation_window(id, &[window]);
    TEST_SURFACE_GUARD_RELEASES.store(0, Ordering::Release);
    let key = window as usize;
    let instance = crate::vulkan::instance_handle().unwrap();
    let result = std::thread::spawn(move || {
        Swapchain::create(
            gpu,
            key as sdl3::Window,
            vk::Extent2D {
                width: 160,
                height: 100,
            },
            u64::MAX,
            instance,
        )
    })
    .join()
    .expect("initializer");
    assert!(
        result.is_err(),
        "missing SDL snapshot must fail before accessing fake VkSurface"
    );
    assert_eq!(
        TEST_SURFACE_GUARD_RELEASES.load(Ordering::Acquire),
        1,
        "VkSurface ownership must be retired exactly once on every early failure"
    );
    sdl3::video::destroy_window(window);
}

#[test]
fn minimized_window_retries_transient_initialization_after_restore() {
    if isolate_vulkan_graphical_test(
        "minimized_window_retries_transient_initialization_after_restore",
    ) {
        return;
    }
    let _retire_guard = AwaitTestWindowRetirements;
    if std::env::var_os("ROSEMARY_TEST_VULKAN").is_none() {
        return;
    }
    let Ok(adapters) = crate::vulkan::adapters() else {
        return;
    };
    let Some(gpu) = adapters.first().and_then(|a| a.gpu().ok()) else {
        return;
    };
    let window = sdl3::video::create_window(c"minimized Vulkan recovery", 240, 180, 0).unwrap();
    sdl3::video::show_window(window);
    let _ = sdl3::video::presentation_window(window).unwrap();
    let frame = appkit::view::Frame {
        window,
        x: 0,
        y: 0,
        width: 240,
        height: 180,
        window_width: 240,
        window_height: 180,
    };
    let source = gpu
        .create_image(&ImageDescription {
            image_type: vk::ImageType::TYPE_2D,
            view_type: vk::ImageViewType::TYPE_2D,
            format: vk::Format::B8G8R8A8_UNORM,
            width: 240,
            height: 180,
            depth: 1,
            levels: 1,
            layers: 1,
            samples: 1,
            usage: gpu::USAGE_RENDER_TARGET,
            cube: false,
            components: vk::ComponentMapping::default(),
        })
        .unwrap();
    sdl3::video::minimize_window(window);
    TEST_FAIL_FIRST_CREATE_WINDOW.store(window as usize, Ordering::Release);
    let deadline = Instant::now() + Duration::from_secs(3);
    while !INIT_RETRIES
        .lock()
        .unwrap()
        .as_ref()
        .is_some_and(|all| all.contains_key(&(window as usize)))
    {
        assert!(
            Instant::now() < deadline,
            "transient failure did not schedule retry"
        );
        let _ = present_impl(gpu, 42, &frame, &source, &[], None, None);
        sdl3::video::vulkan::service_surface_requests();
        std::thread::sleep(Duration::from_millis(2));
    }
    // Simulate >8 transient retries while minimized without waiting for
    // twelve real backoff intervals.
    for _ in 0..12 {
        assert!(schedule_retry(window as usize, false));
    }
    assert_eq!(
        INIT_RETRIES
            .lock()
            .unwrap()
            .as_ref()
            .unwrap()
            .get(&(window as usize))
            .unwrap()
            .delay,
        Duration::from_secs(2),
    );
    assert!(
        !REFUSED
            .lock()
            .unwrap()
            .as_ref()
            .is_some_and(|r| r.contains_key(&(window as usize))),
        "temporary initialization failures may not permanently refuse a window"
    );
    INIT_RETRIES
        .lock()
        .unwrap()
        .as_mut()
        .unwrap()
        .get_mut(&(window as usize))
        .unwrap()
        .after = Instant::now();
    sdl3::video::restore_window(window);
    let until = Instant::now() + Duration::from_secs(5);
    loop {
        if SWAPCHAINS
            .lock()
            .unwrap()
            .as_ref()
            .is_some_and(|all| all.contains_key(&(window as usize)))
        {
            break;
        }
        assert!(
            Instant::now() < until,
            "restored window never recovered Vulkan presentation"
        );
        let _ = present_impl(gpu, 42, &frame, &source, &[], None, None);
        sdl3::video::vulkan::service_surface_requests();
        std::thread::sleep(Duration::from_millis(2));
    }
    assert_eq!(present(gpu, 42, &frame, &source, &[]), Ok(true));
    flush_if_composed(window);
    assert!(sdl3::video::vulkan::owns_window(window));
    forget_window(window);
    sdl3::video::destroy_window(window);
    gpu.destroy_image(source);
}

#[test]
fn backing_store_preflight_defers_until_retry_deadline() {
    let key = 0x794514usize;
    reset_retry(key);
    assert!(!window_initialization_pending(key as sdl3::Window));
    assert!(schedule_retry(key, false));
    assert!(window_initialization_pending(key as sdl3::Window));
    reset_retry(key);
    assert!(!window_initialization_pending(key as sdl3::Window));
}

/// Emulate the failed vkQueueSubmit path: its fence was reset, but no command
/// was queued to signal it. The next FIFO stage must not wait on that fence.
#[test]
#[cfg_attr(target_os = "macos", ignore = "SDL/Cocoa main thread integration")]
fn unsubmitted_unsignaled_fifo_fence_does_not_stall_next_frame() {
    if isolate_vulkan_graphical_test("unsubmitted_unsignaled_fifo_fence_does_not_stall_next_frame")
    {
        return;
    }
    if std::env::var_os("ROSEMARY_TEST_VULKAN").is_none() {
        return;
    }
    let Ok(adapters) = crate::vulkan::adapters() else {
        return;
    };
    let Some(gpu) = adapters.first().and_then(|a| a.gpu().ok()) else {
        return;
    };
    let Some(window) = sdl3::video::create_window(c"failed FIFO fence", 160, 120, 0) else {
        return;
    };
    sdl3::video::show_window(window);
    sdl3::video::presentation_window(window).expect("SDL video snapshot");
    let instance = crate::vulkan::instance_handle().expect("Vulkan instance");
    let surface = sdl3::video::vulkan::create_surface(window, instance, crate::vulkan::library())
        .expect("SDL Vulkan surface");
    let mut swapchain = Swapchain::create(
        gpu,
        window,
        vk::Extent2D {
            width: 160,
            height: 120,
        },
        surface,
        instance,
    )
    .expect("Vulkan swapchain");
    let source = gpu
        .create_image(&ImageDescription {
            image_type: vk::ImageType::TYPE_2D,
            view_type: vk::ImageViewType::TYPE_2D,
            format: vk::Format::B8G8R8A8_UNORM,
            width: 160,
            height: 120,
            depth: 1,
            levels: 1,
            layers: 1,
            samples: 1,
            usage: gpu::USAGE_RENDER_TARGET,
            cube: false,
            components: vk::ComponentMapping::default(),
        })
        .expect("source image");
    let frame = appkit::view::Frame {
        window,
        x: 0,
        y: 0,
        width: 160,
        height: 120,
        window_width: 160,
        window_height: 120,
    };
    let slot = 0;
    // SAFETY: the new frame's fence is unused and belongs to this GPU.
    unsafe { gpu.device.reset_fences(&[swapchain.frames[slot].done]) }
        .expect("simulate fence reset before failed submission");
    assert_eq!(
        unsafe { gpu.device.get_fence_status(swapchain.frames[slot].done) },
        Ok(false)
    );
    // An actually submitted but unsignaled slot must defer without waiting.
    swapchain.frames[slot].submitted = true;
    assert_eq!(
        swapchain.stage_composite(&frame, &source, None, &mut SourceOwner(None), true),
        Ok(None)
    );
    // A failed queue submit never marked the frame as submitted. Despite the
    // unsignaled fence, it is immediately safe to reuse the command buffer.
    swapchain.frames[slot].submitted = false;
    assert_eq!(
        swapchain.stage_composite(&frame, &source, None, &mut SourceOwner(None), true),
        Ok(Some(true))
    );
    drop(swapchain);
    gpu.destroy_image(source);
    sdl3::video::destroy_window(window);
}

#[test]
fn injected_fence_status_and_wait_device_loss_propagate() {
    if isolate_vulkan_graphical_test("injected_fence_status_and_wait_device_loss_propagate") {
        return;
    }
    if std::env::var_os("ROSEMARY_TEST_VULKAN").is_none() {
        return;
    }
    let Ok(adapters) = crate::vulkan::adapters() else {
        return;
    };
    let Some(gpu) = adapters.first().and_then(|a| a.gpu().ok()) else {
        return;
    };
    for operation in ["vkGetFenceStatus", "vkWaitForFences"] {
        gpu.lost.store(false, Ordering::Release);
        let result: Result<bool, String> =
            gpu.fence_result(Err(vk::Result::ERROR_DEVICE_LOST), operation);
        assert!(result.unwrap_err().contains("ERROR_DEVICE_LOST"));
        assert!(
            gpu.lost.load(Ordering::Acquire),
            "{operation} did not mark device lost"
        );
    }
    gpu.lost.store(false, Ordering::Release);
}

#[test]
#[cfg_attr(target_os = "macos", ignore = "SDL main thread")]
fn fifo_acquire_submit_failure_consumes_semaphore_and_retires_image() {
    if isolate_vulkan_graphical_test(
        "fifo_acquire_submit_failure_consumes_semaphore_and_retires_image",
    ) {
        return;
    }
    if std::env::var_os("ROSEMARY_TEST_VULKAN").is_none() {
        return;
    }
    let Ok(adapters) = crate::vulkan::adapters() else {
        return;
    };
    let Some(gpu) = adapters.first().and_then(|adapter| adapter.gpu().ok()) else {
        return;
    };
    let Some(window) = sdl3::video::create_window(c"FIFO acquire failure", 160, 120, 0) else {
        return;
    };
    sdl3::video::show_window(window);
    sdl3::video::presentation_window(window).expect("SDL window snapshot");
    let instance = crate::vulkan::instance_handle().expect("Vulkan instance");
    let surface = sdl3::video::vulkan::create_surface(window, instance, crate::vulkan::library())
        .expect("Vulkan WSI surface");
    let mut state = Swapchain::create(
        gpu,
        window,
        vk::Extent2D {
            width: 160,
            height: 120,
        },
        surface,
        instance,
    )
    .expect("swapchain");
    if !uses_layer_compositor(state.present_mode) {
        drop(state);
        sdl3::video::destroy_window(window);
        return;
    }
    let source = gpu
        .create_image(&ImageDescription {
            image_type: vk::ImageType::TYPE_2D,
            view_type: vk::ImageViewType::TYPE_2D,
            format: vk::Format::B8G8R8A8_UNORM,
            width: 160,
            height: 120,
            depth: 1,
            levels: 1,
            layers: 1,
            samples: 1,
            usage: gpu::USAGE_RENDER_TARGET,
            cube: false,
            components: vk::ComponentMapping::default(),
        })
        .expect("GPU image");
    let frame = appkit::view::Frame {
        window,
        x: 0,
        y: 0,
        width: 160,
        height: 120,
        window_width: 160,
        window_height: 120,
    };
    assert_eq!(
        state.stage_composite(&frame, &source, None, &mut SourceOwner(None), false),
        Ok(Some(true))
    );
    let original = state.handle;
    state.injected_queue_submit_error = Some(vk::Result::ERROR_OUT_OF_HOST_MEMORY);
    assert!(
        state.flush_composite(window).is_ok(),
        "failed FIFO submission must recover"
    );
    assert!(
        state.acquired_unsubmitted.is_none(),
        "FIFO failed submit left the acquire semaphore unconsumed"
    );
    assert_ne!(state.handle, original, "acquired image was not retired");
    assert!(
        state.composite_dirty,
        "cached Metal layers must be replayed"
    );
    state
        .flush_composite(window)
        .expect("replay preserved FIFO snapshot");
    assert!(!state.composite_dirty);
    assert_eq!(
        state.stage_composite(&frame, &source, None, &mut SourceOwner(None), false),
        Ok(Some(true))
    );
    state
        .flush_composite(window)
        .expect("next FIFO frame must present normally");
    drop(state);
    gpu.destroy_image(source);
    sdl3::video::destroy_window(window);
}

#[test]
#[cfg_attr(target_os = "macos", ignore = "SDL main thread")]
fn direct_vulkan_surface_creation_registers_cpu_surface_exclusion() {
    if isolate_vulkan_graphical_test(
        "direct_vulkan_surface_creation_registers_cpu_surface_exclusion",
    ) {
        return;
    }
    if std::env::var_os("ROSEMARY_TEST_VULKAN").is_none() {
        return;
    }
    let Ok(adapters) = crate::vulkan::adapters() else {
        return;
    };
    let Some(_gpu) = adapters.first().and_then(|adapter| adapter.gpu().ok()) else {
        return;
    };
    let Some(window) = sdl3::video::create_window(c"Direct Vulkan surface", 80, 64, 0) else {
        return;
    };
    assert!(!sdl3::video::vulkan::owns_window(window));
    let instance = crate::vulkan::instance_handle().expect("Vulkan instance");
    let surface = sdl3::video::vulkan::create_surface(window, instance, crate::vulkan::library())
        .expect("direct Vulkan surface");
    assert!(sdl3::video::vulkan::owns_window(window));
    assert!(
        !sdl3::video::present_window(window, |_| true),
        "Vulkan forbids SDL surface"
    );
    sdl3::video::vulkan::destroy_surface(instance, surface);
    assert!(
        sdl3::video::vulkan::owns_window(window),
        "destruction must not undo conversion"
    );
    sdl3::video::destroy_window(window);
}

/// Independent FIFO WSI scenarios share a real SDL surface and VkSwapchainKHR.
#[cfg(not(target_os = "macos"))]
fn v10_fifo_fixture() -> Option<(
    sdl3::Window,
    &'static Gpu,
    Swapchain,
    GpuImage,
    appkit::view::Frame,
)> {
    if std::env::var_os("ROSEMARY_TEST_VULKAN").is_none() {
        return None;
    }
    let gpu = crate::vulkan::adapters().ok()?.first()?.gpu().ok()?;
    let window = sdl3::video::create_window(c"FIFO persistent layers", 160, 120, 0)?;
    sdl3::video::show_window(window);
    sdl3::video::presentation_window(window)?;
    let instance = crate::vulkan::instance_handle()?;
    let surface = sdl3::video::vulkan::create_surface(window, instance, crate::vulkan::library())?;
    let state = Swapchain::create(
        gpu,
        window,
        vk::Extent2D {
            width: 160,
            height: 120,
        },
        surface,
        instance,
    )
    .expect("WSI swapchain");
    if !uses_layer_compositor(state.present_mode) {
        drop(state);
        sdl3::video::destroy_window(window);
        return None;
    }
    let image = gpu
        .create_image(&ImageDescription {
            image_type: vk::ImageType::TYPE_2D,
            view_type: vk::ImageViewType::TYPE_2D,
            format: vk::Format::B8G8R8A8_UNORM,
            width: 160,
            height: 120,
            depth: 1,
            levels: 1,
            layers: 1,
            samples: 1,
            usage: gpu::USAGE_RENDER_TARGET,
            cube: false,
            components: vk::ComponentMapping::default(),
        })
        .expect("Metal content source");
    let frame = appkit::view::Frame {
        window,
        x: 0,
        y: 0,
        width: 160,
        height: 120,
        window_width: 160,
        window_height: 120,
    };
    Some((window, gpu, state, image, frame))
}

/// Exercises actual Vulkan acquisition and WSI presentation on the direct
/// worker path, including the lazy transition to multi-layer composition.
#[test]
#[cfg(not(target_os = "macos"))]
fn single_opaque_layer_uses_one_wsi_copy_then_snapshots_on_second_layer() {
    if isolate_vulkan_graphical_test(
        "single_opaque_layer_uses_one_wsi_copy_then_snapshots_on_second_layer",
    ) {
        return;
    }
    let expected_mode = match std::env::var("ROSEMARY_METAL_PRESENTATION_MODE").as_deref() {
        Ok("immediate") => vk::PresentModeKHR::IMMEDIATE,
        Ok("mailbox") => vk::PresentModeKHR::MAILBOX,
        _ => return,
    };
    let Some((window, gpu, mut state, image, frame)) = v10_fifo_fixture() else {
        return;
    };
    // A requested unavailable WSI mode is not real validation of that mode.
    assert_eq!(
        state.present_mode, expected_mode,
        "the host did not select the requested WSI mode"
    );
    state.layers.push(8);
    // Reuse every WSI image, acquisition semaphore and presentation fence
    // repeatedly; the first few images alone do not exercise retirement.
    for _ in 0..(FRAMES_IN_FLIGHT * 3 + 2) {
        state.queue_direct(
            &frame,
            PresentSource::from(&image),
            &mut SourceOwner(None),
            None,
        );
        assert_eq!(pending_batch_delay(&state), Duration::ZERO);
        assert!(
            state.composite.is_none(),
            "direct path allocated an intermediate image"
        );
        state
            .flush_composite(window)
            .expect("single-layer direct WSI presentation");
        assert!(
            state.composite.is_none(),
            "direct path copied into an accumulator"
        );
        assert!(!state.composite_dirty);
    }
    let prev_slot = state.direct.as_ref().unwrap().submitted_slot;
    assert!(prev_slot.is_some());
    state.layers.push(9);
    assert_eq!(state.stage_direct_snapshot(false), Ok(Some(true)));
    assert!(state.direct.is_none());
    assert!(
        state.composite.is_some(),
        "second layer must retain prior full-window content"
    );
    assert_eq!(
        state.stage_composite(
            &appkit::view::Frame {
                x: 12,
                y: 15,
                width: 80,
                height: 60,
                ..frame
            },
            &image,
            None,
            &mut SourceOwner(None),
            false,
        ),
        Ok(Some(true)),
    );
    state.flush_composite(window).expect("two-layer WSI frame");
    assert!(!state.composite_dirty);
    state
        .rebuild(state.requested)
        .expect("recreate multi-layer WSI");
    assert!(state.composite_dirty);
    state
        .flush_composite(window)
        .expect("static multi-layer replay");
    drop(state);
    gpu.destroy_image(image);
    sdl3::video::destroy_window(window);
}

/// An acquired WSI image must never force the worker to wait forever on
/// the previous maintenance1 presentation fence.
#[test]
#[cfg(not(target_os = "macos"))]
fn unsignaled_prior_present_fence_does_not_block_acquired_image() {
    if isolate_vulkan_graphical_test("unsignaled_prior_present_fence_does_not_block_acquired_image")
    {
        return;
    }
    let Some((window, gpu, mut state, image, frame)) = v10_fifo_fixture() else {
        return;
    };
    if state.presented.is_empty() {
        drop(state);
        gpu.destroy_image(image);
        sdl3::video::destroy_window(window);
        return;
    }
    assert_eq!(
        state.stage_composite(&frame, &image, None, &mut SourceOwner(None), false),
        Ok(Some(true))
    );
    state.flush_composite(window).expect("initial presentation");
    // Ensure the fault applies to whichever image this MAILBOX or IMMEDIATE
    // driver acquires next, including MAILBOX implementations that skip images.
    for fence in &mut state.presented {
        fence.pending = true;
    }
    state.injected_unsignaled_present_fence = true;
    let before = state.frame;
    // Acquire the same image on repeated presents to exercise its prior fence.
    for _ in 0..(state.images.len() * 3) {
        assert_eq!(
            state.stage_composite(&frame, &image, None, &mut SourceOwner(None), false),
            Ok(Some(true))
        );
        let start = Instant::now();
        state
            .flush_composite(window)
            .expect("present without waiting for stalled fence");
        assert!(
            start.elapsed() < Duration::from_secs(2),
            "previous presentation fence blocked worker"
        );
        if state.skipped_stalled_present_fence {
            break;
        }
    }
    assert!(
        state.skipped_stalled_present_fence,
        "failed to reacquire a previously presented image"
    );
    assert!(state.frame > before);
    assert!(
        state.acquired_unsubmitted.is_none(),
        "acquired semaphore was not consumed"
    );
    assert!(!state.composite_dirty);
    // After a present omits a maintenance fence, even a later signal of the
    // original fence cannot cover the new present. Retire via queue idle.
    assert!(state.stalled_present_fence);
    state.idle();
    assert!(!state.stalled_present_fence);
    assert!(state.presented.iter().all(|fence| !fence.pending));
    drop(state);
    gpu.destroy_image(image);
    sdl3::video::destroy_window(window);
}

#[cfg(not(target_os = "macos"))]
fn direct_layer_resize_preserves_unscaled_static_window_image_impl(wsi_only: bool) {
    let expected_mode = match std::env::var("ROSEMARY_METAL_PRESENTATION_MODE").as_deref() {
        Ok("immediate") => vk::PresentModeKHR::IMMEDIATE,
        Ok("mailbox") => vk::PresentModeKHR::MAILBOX,
        _ => return,
    };
    let Some((window, gpu, mut state, image, frame)) = v10_fifo_fixture() else {
        return;
    };
    assert_eq!(state.present_mode, expected_mode);
    state.layers.push(8);
    state.queue_direct(
        &frame,
        PresentSource::from(&image),
        &mut SourceOwner(None),
        None,
    );
    state
        .flush_composite(window)
        .expect("first direct WSI presentation");
    assert!(state.composite.is_none());
    assert!(sdl3::video::set_window_size(window, 208, 144));
    let deadline = Instant::now() + Duration::from_secs(2);
    loop {
        sdl3::pump();
        let caps = unsafe {
            crate::vulkan::surface_api()
                .expect("WSI surface API")
                .get_physical_device_surface_capabilities(gpu.physical, state.surface)
        }
        .expect("WSI capabilities");
        if caps.current_extent.width == u32::MAX && wsi_only {
            // This host uses the caller's requested extent, not WSI's:
            // cannot force a new effective extent without a new request.
            drop(state);
            gpu.destroy_image(image);
            sdl3::video::destroy_window(window);
            return;
        }
        if caps.current_extent.width == u32::MAX
            || (caps.current_extent.width, caps.current_extent.height) == (208, 144)
        {
            break;
        }
        assert!(Instant::now() < deadline, "Vulkan surface did not resize");
        std::thread::sleep(Duration::from_millis(5));
    }
    let requested_before = state.requested;
    let rebuild_request = if wsi_only {
        requested_before
    } else {
        vk::Extent2D {
            width: 208,
            height: 144,
        }
    };
    state.rebuild(rebuild_request).expect("direct WSI resize");
    if wsi_only {
        assert_eq!(state.requested, requested_before, "rebuild request changed");
        assert_ne!(
            (state.extent.width, state.extent.height),
            (frame.width, frame.height),
            "WSI extent did not change"
        );
    }
    assert!(
        state.direct.is_none(),
        "old geometry must not be stretched as a direct frame"
    );
    let saved = state
        .composite
        .as_ref()
        .expect("resized snapshot of static layer");
    assert_eq!(
        (saved.width, saved.height),
        (state.extent.width, state.extent.height)
    );
    assert!(state.composite_dirty);
    state
        .flush_composite(window)
        .expect("static layer replay without guest redraw");
    assert!(!state.composite_dirty);
    drop(state);
    gpu.destroy_image(image);
    sdl3::video::destroy_window(window);
}

#[test]
#[cfg(not(target_os = "macos"))]
fn direct_layer_resize_preserves_unscaled_static_window_image() {
    direct_layer_resize_preserves_unscaled_static_window_image_impl(false);
}

#[test]
#[cfg(not(target_os = "macos"))]
fn wsi_effective_extent_change_preserves_direct_layer_without_new_request() {
    direct_layer_resize_preserves_unscaled_static_window_image_impl(true);
}

/// Exercise the actual AppKit CPU store upload worker's Vulkan path. Its
/// image cache must rotate while an earlier frame is still retained, and the
/// compositor must select direct WSI instead of an intermediate accumulator.
#[test]
#[cfg(not(target_os = "macos"))]
fn backing_store_uploads_select_direct_wsi_without_overwriting_inflight_source() {
    if isolate_vulkan_graphical_test(
        "backing_store_uploads_select_direct_wsi_without_overwriting_inflight_source",
    ) {
        return;
    }
    let expected = match std::env::var("ROSEMARY_METAL_PRESENTATION_MODE").as_deref() {
        Ok("immediate") => vk::PresentModeKHR::IMMEDIATE,
        Ok("mailbox") => vk::PresentModeKHR::MAILBOX,
        _ => return,
    };
    if std::env::var_os("ROSEMARY_TEST_VULKAN").is_none() {
        return;
    }
    let Some(window) = sdl3::video::create_window(c"direct backing store", 160, 120, 0) else {
        return;
    };
    sdl3::video::show_window(window);
    sdl3::video::presentation_window(window).expect("SDL metadata");
    let mut store = appkit::backing::Store::new(160, 120);
    store.pixels.fill(43);
    let deadline = Instant::now() + Duration::from_secs(4);
    loop {
        match crate::present::present_store_synchronous(window, &store, &[], &[]) {
            appkit::backing::PresentOutcome::Displayed => break,
            appkit::backing::PresentOutcome::Deferred => {
                assert!(
                    Instant::now() < deadline,
                    "store upload never reached Vulkan"
                );
                sdl3::video::vulkan::service_surface_requests();
                std::thread::sleep(Duration::from_millis(2));
            }
            outcome => panic!("store upload failed: {outcome:?}"),
        }
    }
    let entry = SWAPCHAINS
        .lock()
        .unwrap()
        .as_ref()
        .unwrap()
        .get(&(window as usize))
        .unwrap()
        .clone();
    {
        let _presenting = entry.presenting.lock().unwrap();
        let mut state = entry.state.lock().unwrap();
        assert_eq!(state.present_mode, expected);
        assert!(
            state.composite.is_none(),
            "backing store took extra GPU accumulator path"
        );
        assert!(
            state.direct.as_ref().unwrap().backing.is_some(),
            "missing retained backing image"
        );
        state.flush_composite(window).expect("direct store WSI");
    }
    let (old, _) = crate::present::backing_store_image_for_test(window).unwrap();
    store.pixels.fill(77);
    assert_eq!(
        crate::present::present_store_synchronous(window, &store, &[], &[]),
        appkit::backing::PresentOutcome::Displayed
    );
    {
        let (current, spare) = crate::present::backing_store_image_for_test(window).unwrap();
        assert!(
            !Arc::ptr_eq(&old, &current),
            "overwrote an image retained by WSI"
        );
        assert_eq!(spare, 1, "source rotation did not retain old image");
    }
    {
        let _presenting = entry.presenting.lock().unwrap();
        let mut state = entry.state.lock().unwrap();
        assert!(state.composite.is_none());
        state
            .flush_composite(window)
            .expect("new backing-store direct WSI");
        state.direct.as_mut().unwrap().queued_at = Instant::now() - DIRECT_IDLE_SNAPSHOT_DELAY;
        assert!(state.snapshot_idle_direct().expect("idle store snapshot"));
        state.layers.push(0x777); // The next backing-store update joins a second Metal layer.
    }
    store.pixels.fill(91);
    assert_eq!(
        crate::present::present_store_synchronous(window, &store, &[], &[]),
        appkit::backing::PresentOutcome::Displayed,
        "multi-layer backing store should enter the compositor"
    );
    {
        let _presenting = entry.presenting.lock().unwrap();
        let mut state = entry.state.lock().unwrap();
        assert!(state.composite.is_some());
        assert!(
            state
                .frames
                .iter()
                .any(|frame| frame.backing.is_some() && frame.submitted),
            "staged backing source lost its lease before the GPU-copy fence"
        );
        state
            .flush_composite(window)
            .expect("multi-layer backing-store present");
        state.idle();
        assert!(state.frames.iter().all(|frame| frame.backing.is_none()));
    }
    drop(old);
    drop(entry); // Teardown waits for every external WindowSwapchain Arc.
    crate::present::forget_backing_store_for_test(window);
    forget_window(window);
    sdl3::video::destroy_window(window);
    wait_for_test_retirements();
}

/// The real AppKit overlay-present route must retain its blended GPU image
/// and perform one WSI copy. The deliberately invalid DMA-BUF exercises the
/// overlay fallback copy and refusal contract without requiring an exporter;
/// importing an actual foreign buffer needs separate supported-host coverage.
#[test]
#[cfg(not(target_os = "macos"))]
fn backing_store_with_gpu_overlay_uses_direct_wsi_without_extra_composite() {
    if isolate_vulkan_graphical_test(
        "backing_store_with_gpu_overlay_uses_direct_wsi_without_extra_composite",
    ) {
        return;
    }
    let expected = match std::env::var("ROSEMARY_METAL_PRESENTATION_MODE").as_deref() {
        Ok("immediate") => vk::PresentModeKHR::IMMEDIATE,
        Ok("mailbox") => vk::PresentModeKHR::MAILBOX,
        _ => return,
    };
    if std::env::var_os("ROSEMARY_TEST_VULKAN").is_none() {
        return;
    }
    let window = sdl3::video::create_window(c"direct GPU overlay store", 160, 120, 0)
        .expect("SDL overlay test window");
    sdl3::video::show_window(window);
    sdl3::video::presentation_window(window).expect("SDL overlay metadata");
    let mut store = appkit::backing::Store::new(160, 120);
    store.pixels.fill(42);
    let overlays = [appkit::backing::GpuOverlay {
        picture: Arc::new(appkit::backing::Dmabuf {
            width: 160,
            height: 120,
            fourcc: u32::from_le_bytes(*b"AR24"),
            modifier: 0,
            planes: Vec::new(),
        }),
        rect: appkit::backing::Rect {
            x: 0,
            y: 0,
            width: 160,
            height: 120,
        },
    }];
    let deadline = Instant::now() + Duration::from_secs(4);
    loop {
        match crate::present::present_store_synchronous(window, &store, &[], &overlays) {
            appkit::backing::PresentOutcome::Displayed => break,
            appkit::backing::PresentOutcome::Deferred => {
                assert!(
                    Instant::now() < deadline,
                    "overlay backing store did not reach Vulkan"
                );
                sdl3::video::vulkan::service_surface_requests();
                std::thread::sleep(Duration::from_millis(2));
            }
            other => panic!("unexpected overlay backing outcome: {other:?}"),
        }
    }
    let entry = SWAPCHAINS
        .lock()
        .unwrap()
        .as_ref()
        .unwrap()
        .get(&(window as usize))
        .unwrap()
        .clone();
    let old = {
        let _presenting = entry.presenting.lock().unwrap();
        let mut state = entry.state.lock().unwrap();
        assert_eq!(state.present_mode, expected);
        assert!(
            state.composite.is_none(),
            "overlay unnecessarily staged into accumulator"
        );
        let direct = state
            .direct
            .as_ref()
            .expect("eligible overlay must queue directly");
        let lease = direct
            .backing
            .as_ref()
            .expect("blended image lacks GPU-copy lease");
        assert_eq!(
            lease.image.format,
            vk::Format::R8G8B8A8_UNORM,
            "overlay route did not select actual blended target"
        );
        let previous = Arc::clone(lease);
        state
            .flush_composite(window)
            .expect("direct overlay WSI frame");
        assert!(
            state.composite.is_none(),
            "direct overlay frame created full window composite"
        );
        previous
    };
    store.pixels.fill(87);
    assert_eq!(
        crate::present::present_store_synchronous(window, &store, &[], &overlays),
        appkit::backing::PresentOutcome::Displayed
    );
    {
        let _presenting = entry.presenting.lock().unwrap();
        let mut state = entry.state.lock().unwrap();
        let direct = state.direct.as_ref().expect("updated overlay direct frame");
        let fresh = direct.backing.as_ref().expect("overlay source lease");
        assert!(
            !Arc::ptr_eq(&old, fresh),
            "blended image overwritten while WSI retained it"
        );
        assert!(state.composite.is_none());
        state
            .flush_composite(window)
            .expect("second direct overlay WSI");
        state.direct.as_mut().unwrap().queued_at = Instant::now() - DIRECT_IDLE_SNAPSHOT_DELAY;
        assert!(state.snapshot_idle_direct().expect("idle overlay snapshot"));
        state.idle();
    }
    // All eight blended allocations are now retained as though the WSI
    // presenter were slow to return them. Retries must defer before issuing
    // another backing upload, not spin on a costly GPU copy every millisecond.
    let retained_blends = crate::present::hold_every_blended_source_for_test(window);
    let submitted = crate::present::backing_upload_submissions_for_test();
    for _ in 0..5 {
        assert_eq!(
            crate::present::present_store_synchronous(window, &store, &[], &overlays),
            appkit::backing::PresentOutcome::Deferred
        );
        assert_eq!(
            crate::present::backing_upload_submissions_for_test(),
            submitted,
            "pool exhaustion submitted a redundant full-frame GPU upload"
        );
    }
    drop(retained_blends);
    assert_eq!(
        crate::present::present_store_synchronous(window, &store, &[], &overlays),
        appkit::backing::PresentOutcome::Displayed,
        "source lease release must permit upload and presentation"
    );
    assert_eq!(
        crate::present::backing_upload_submissions_for_test(),
        submitted + 1
    );
    drop(old);
    drop(entry);
    crate::present::forget_backing_store_for_test(window);
    appkit::backing::forget_window(window); // Clear the deliberate DMA-BUF refusal.
    forget_window(window);
    sdl3::video::destroy_window(window);
    wait_for_test_retirements();
}

/// A full-window store upload supplies a retained GPU allocation, not a
/// CAMetalDrawable. The WSI worker can copy it directly without an offscreen
/// accumulator or reading a source the next upload will overwrite.
#[test]
#[cfg(not(target_os = "macos"))]
fn retained_backing_source_uses_direct_wsi_and_releases_after_gpu_copy() {
    if isolate_vulkan_graphical_test(
        "retained_backing_source_uses_direct_wsi_and_releases_after_gpu_copy",
    ) {
        return;
    }
    let expected = match std::env::var("ROSEMARY_METAL_PRESENTATION_MODE").as_deref() {
        Ok("immediate") => vk::PresentModeKHR::IMMEDIATE,
        Ok("mailbox") => vk::PresentModeKHR::MAILBOX,
        _ => return,
    };
    let Some((window, gpu, mut state, image, frame)) = v10_fifo_fixture() else {
        return;
    };
    assert_eq!(state.present_mode, expected, "real WSI mode required");
    let lease = BackingImageLease::new(gpu, image);
    state.layers.push(window as usize);
    assert!(fast_direct_layer(
        state.present_mode,
        1,
        &frame,
        None,
        false,
        true
    ));
    for _ in 0..FRAMES_IN_FLIGHT * 2 {
        state.queue_direct(
            &frame,
            PresentSource::from(&lease.image),
            &mut SourceOwner(None),
            Some(Arc::clone(&lease)),
        );
        assert_eq!(pending_batch_delay(&state), Duration::ZERO);
        assert!(state.composite.is_none());
        state
            .flush_composite(window)
            .expect("backing image direct WSI copy");
        assert!(
            state.composite.is_none(),
            "must not allocate offscreen full-frame accumulator"
        );
        assert!(
            Arc::strong_count(&lease) > 1,
            "direct WSI image lost retained lifetime"
        );
    }
    // After idle, a single snapshot is enough to preserve static contents
    // and the last backing-store copy is no longer WSI-owned.
    state.direct.as_mut().unwrap().queued_at = Instant::now() - DIRECT_IDLE_SNAPSHOT_DELAY;
    assert!(state.snapshot_idle_direct().expect("idle source snapshot"));
    state.idle();
    assert_eq!(
        Arc::strong_count(&lease),
        1,
        "GPU fence failed to release backing store"
    );
    assert!(!state.composite_dirty);
    drop(state);
    drop(lease);
    sdl3::video::destroy_window(window);
}

/// Idle direct frames are snapshotted only after they are on screen, then
/// their drawable can be returned without queuing another WSI presentation.
#[test]
#[cfg(not(target_os = "macos"))]
fn idle_direct_frame_snapshots_once_and_releases_the_source_fence() {
    if isolate_vulkan_graphical_test(
        "idle_direct_frame_snapshots_once_and_releases_the_source_fence",
    ) {
        return;
    }
    let expected_mode = match std::env::var("ROSEMARY_METAL_PRESENTATION_MODE").as_deref() {
        Ok("immediate") => vk::PresentModeKHR::IMMEDIATE,
        Ok("mailbox") => vk::PresentModeKHR::MAILBOX,
        _ => return,
    };
    let Some((window, gpu, mut state, image, frame)) = v10_fifo_fixture() else {
        return;
    };
    assert_eq!(state.present_mode, expected_mode);
    state.layers.push(8);
    state.queue_direct(
        &frame,
        PresentSource::from(&image),
        &mut SourceOwner(None),
        None,
    );
    assert!(
        !state
            .snapshot_idle_direct()
            .expect("dirty direct cannot snapshot")
    );
    state.flush_composite(window).expect("direct presentation");
    assert!(state.direct.is_some());
    assert!(
        !state
            .snapshot_idle_direct()
            .expect("recent direct cannot snapshot")
    );
    state.direct.as_mut().unwrap().queued_at = Instant::now() - DIRECT_IDLE_SNAPSHOT_DELAY;
    let before_stage = state.frame;
    assert!(state.snapshot_idle_direct().expect("snapshot idle direct"));
    assert_eq!(state.frame, before_stage + 1, "one GPU staging copy");
    assert!(state.direct.is_none());
    assert!(state.composite.is_some());
    assert!(
        !state.composite_dirty,
        "already-presented image must not re-enter WSI"
    );
    let snapshot_slot = before_stage % state.frames.len();
    assert!(!state.frames[snapshot_slot].submitted);
    assert!(state.frames[snapshot_slot].source.is_none());
    assert!(
        !state
            .snapshot_idle_direct()
            .expect("only one idle snapshot")
    );
    state.layers.push(9);
    assert_eq!(
        state.stage_composite(
            &appkit::view::Frame {
                x: 4,
                y: 3,
                width: 20,
                height: 40,
                ..frame
            },
            &image,
            None,
            &mut SourceOwner(None),
            false,
        ),
        Ok(Some(true)),
    );
    state
        .flush_composite(window)
        .expect("present new layer atop preserved idle snapshot");
    drop(state);
    gpu.destroy_image(image);
    sdl3::video::destroy_window(window);
}

/// Acquired-image recovery must preserve the direct source, not silently
/// consume its drawable on an out-of-date or failed submission.
#[test]
#[cfg(not(target_os = "macos"))]
fn direct_single_layer_replays_after_failed_submit() {
    if isolate_vulkan_graphical_test("direct_single_layer_replays_after_failed_submit") {
        return;
    }
    let expected_mode = match std::env::var("ROSEMARY_METAL_PRESENTATION_MODE").as_deref() {
        Ok("immediate") => vk::PresentModeKHR::IMMEDIATE,
        Ok("mailbox") => vk::PresentModeKHR::MAILBOX,
        _ => return,
    };
    let Some((window, gpu, mut state, image, frame)) = v10_fifo_fixture() else {
        return;
    };
    assert_eq!(state.present_mode, expected_mode);
    state.layers.push(8);
    state.queue_direct(
        &frame,
        PresentSource::from(&image),
        &mut SourceOwner(None),
        None,
    );
    state.injected_queue_submit_error = Some(vk::Result::ERROR_OUT_OF_HOST_MEMORY);
    state
        .flush_composite(window)
        .expect("recover direct acquire after failed submit");
    assert!(state.composite_dirty);
    assert!(state.direct.is_some());
    state
        .flush_composite(window)
        .expect("replay direct without new drawable");
    assert!(!state.composite_dirty);
    assert!(state.composite.is_none());
    drop(state);
    gpu.destroy_image(image);
    sdl3::video::destroy_window(window);
}

#[test]
#[cfg(not(target_os = "macos"))]
fn fifo_static_layer_survives_out_of_date_recreation_without_guest_redraw() {
    if isolate_vulkan_graphical_test(
        "fifo_static_layer_survives_out_of_date_recreation_without_guest_redraw",
    ) {
        return;
    }
    let Some((window, gpu, mut state, image, frame)) = v10_fifo_fixture() else {
        return;
    };
    assert_eq!(
        state.stage_composite(&frame, &image, None, &mut SourceOwner(None), false),
        Ok(Some(true))
    );
    state
        .flush_composite(window)
        .expect("first FIFO presentation");
    let previous = state.composite.as_ref().unwrap().image;
    let old_handle = state.handle;
    assert!(!state.composite_dirty);
    state.rebuild(state.requested).expect("WSI rebuild");
    assert_ne!(state.handle, old_handle);
    assert_eq!(
        state.composite.as_ref().unwrap().image,
        previous,
        "static layer GPU image should survive same-format recreation"
    );
    assert!(
        state.composite_dirty,
        "static layer must be queued for WSI replay"
    );
    // This is another queue presentation only: no stage_composite/guest frame.
    state
        .flush_composite(window)
        .expect("replay last layer frame after recreate");
    assert!(!state.composite_dirty);
    drop(state);
    gpu.destroy_image(image);
    sdl3::video::destroy_window(window);
}

#[test]
#[cfg(not(target_os = "macos"))]
fn fifo_out_of_date_acquire_replays_static_layers_without_guest_submit() {
    if isolate_vulkan_graphical_test(
        "fifo_out_of_date_acquire_replays_static_layers_without_guest_submit",
    ) {
        return;
    }
    let Some((window, gpu, mut state, image, frame)) = v10_fifo_fixture() else {
        return;
    };
    assert_eq!(
        state.stage_composite(&frame, &image, None, &mut SourceOwner(None), false),
        Ok(Some(true))
    );
    state.flush_composite(window).expect("initial present");
    let previous = state.composite.as_ref().unwrap().image;
    assert_eq!(
        state.stage_composite(&frame, &image, None, &mut SourceOwner(None), false),
        Ok(Some(true))
    );
    let before = state.handle;
    let ready = state
        .prepare_composite(window)
        .expect("reserve WSI acquire");
    // WSI reports OUT_OF_DATE before assigning a new image, so the layer
    // cache must remain presentable without another guest drawable.
    assert_eq!(state.complete_composite(ready, Ok(None)), Ok(None));
    assert_ne!(state.handle, before);
    assert!(state.composite_dirty);
    assert_eq!(state.composite.as_ref().unwrap().image, previous);
    state
        .flush_composite(window)
        .expect("present cached layer after OUT_OF_DATE");
    assert!(!state.composite_dirty);
    drop(state);
    gpu.destroy_image(image);
    sdl3::video::destroy_window(window);
}

#[test]
#[cfg(not(target_os = "macos"))]
fn fifo_out_of_date_queue_present_replays_static_layers() {
    if isolate_vulkan_graphical_test("fifo_out_of_date_queue_present_replays_static_layers") {
        return;
    }
    let Some((window, gpu, mut state, image, frame)) = v10_fifo_fixture() else {
        return;
    };
    assert_eq!(
        state.stage_composite(&frame, &image, None, &mut SourceOwner(None), false),
        Ok(Some(true))
    );
    state.injected_queue_present_error = Some(vk::Result::ERROR_OUT_OF_DATE_KHR);
    let before = state.handle;
    state
        .flush_composite(window)
        .expect("WSI out-of-date must rebuild");
    assert_ne!(state.handle, before);
    assert!(state.composite_dirty);
    let saved = state.composite.as_ref().unwrap().image;
    state
        .flush_composite(window)
        .expect("present cached layer after WSI error");
    assert!(!state.composite_dirty);
    assert_eq!(state.composite.as_ref().unwrap().image, saved);
    drop(state);
    gpu.destroy_image(image);
    sdl3::video::destroy_window(window);
}

#[test]
#[cfg(not(target_os = "macos"))]
fn fifo_window_resize_keeps_snapshot_without_new_drawable() {
    if isolate_vulkan_graphical_test("fifo_window_resize_keeps_snapshot_without_new_drawable") {
        return;
    }
    let Some((window, gpu, mut state, image, frame)) = v10_fifo_fixture() else {
        return;
    };
    assert_eq!(
        state.stage_composite(&frame, &image, None, &mut SourceOwner(None), false),
        Ok(Some(true))
    );
    state.flush_composite(window).expect("original present");
    let previous = state.composite.as_ref().unwrap().image;
    assert!(sdl3::video::set_window_size(window, 208, 144));
    // X11 applies resize asynchronously. Do not accidentally test only a
    // logical requested extent while the actual Vulkan surface stays old.
    let deadline = Instant::now() + Duration::from_secs(2);
    loop {
        sdl3::pump();
        let caps = unsafe {
            crate::vulkan::surface_api()
                .expect("WSI surface API")
                .get_physical_device_surface_capabilities(gpu.physical, state.surface)
        }
        .expect("WSI capabilities");
        if caps.current_extent.width == u32::MAX
            || (caps.current_extent.width, caps.current_extent.height) == (208, 144)
        {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "native Vulkan surface never resized"
        );
        std::thread::sleep(Duration::from_millis(5));
    }
    state
        .rebuild(vk::Extent2D {
            width: 208,
            height: 144,
        })
        .expect("WSI resize");
    assert!(state.composite_dirty);
    let retained = state.composite.as_ref().unwrap();
    assert_eq!(
        (retained.width, retained.height),
        (state.extent.width, state.extent.height)
    );
    assert_ne!(
        (state.extent.width, state.extent.height),
        (160, 120),
        "fixture did not resize the actual Vulkan surface"
    );
    assert_ne!(retained.image, previous, "resized snapshot must be copied");
    let resized = retained.image;
    state
        .flush_composite(window)
        .expect("replay resized snapshot without guest frame");
    let extent = state.extent;
    state.rebuild(extent).expect("same-size WSI rebuild");
    assert_eq!(
        state.composite.as_ref().unwrap().image,
        resized,
        "a rebuild that changes nothing must keep the snapshot it has"
    );
    drop(state);
    gpu.destroy_image(image);
    sdl3::video::destroy_window(window);
}

#[test]
#[cfg(not(target_os = "macos"))]
fn secondary_acquire_recovery_failure_waits_wsi_fence_on_teardown() {
    if isolate_vulkan_graphical_test(
        "secondary_acquire_recovery_failure_waits_wsi_fence_on_teardown",
    ) {
        return;
    }
    let Some((window, gpu, mut state, image, frame)) = v10_fifo_fixture() else {
        return;
    };
    assert_eq!(
        state.stage_composite(&frame, &image, None, &mut SourceOwner(None), false),
        Ok(Some(true))
    );
    state.injected_queue_submit_error = Some(vk::Result::ERROR_OUT_OF_HOST_MEMORY);
    state.injected_recovery_error = Some(vk::Result::ERROR_OUT_OF_HOST_MEMORY);
    let result = state.flush_composite(window);
    assert!(
        result.is_err(),
        "double failed queue submissions must disable FIFO"
    );
    let (acquired, _) = state
        .acquired_unsubmitted
        .expect("unconsumed WSI semaphore must remain tracked");
    let acquire_fence = state
        .frames
        .iter()
        .find(|f| f.acquired == acquired)
        .unwrap()
        .acquire_done;
    gpu.wait(acquire_fence)
        .expect("acquire fence proves WSI signal finished");
    drop(state); // Destructor independently waits before retiring semaphore.
    gpu.destroy_image(image);
    sdl3::video::destroy_window(window);
}

#[test]
#[cfg(not(target_os = "macos"))]
fn failed_recreation_never_reuses_retired_old_swapchain() {
    if isolate_vulkan_graphical_test("failed_recreation_never_reuses_retired_old_swapchain") {
        return;
    }
    let Some((window, gpu, mut state, image, frame)) = v10_fifo_fixture() else {
        return;
    };
    assert_eq!(
        state.stage_composite(&frame, &image, None, &mut SourceOwner(None), false),
        Ok(Some(true))
    );
    state
        .flush_composite(window)
        .expect("first FIFO presentation");
    let old_handle = state.handle;
    let saved = state.composite.as_ref().unwrap().image;
    state.injected_build_failure = true;
    assert!(
        state.rebuild(state.requested).is_err(),
        "inject post-create failure after Vulkan retires oldSwapchain"
    );
    assert!(state.retired);
    assert_eq!(state.handle, old_handle, "still owned but unusable");
    state
        .rebuild(state.requested)
        .expect("recreate without retired oldSwapchain");
    assert!(!state.retired);
    assert_eq!(
        state.last_create_old_swapchain,
        vk::SwapchainKHR::null(),
        "retired oldSwapchain cannot be supplied to vkCreateSwapchainKHR twice"
    );
    // Native Vulkan handle values may be reused after destruction.
    assert_eq!(state.composite.as_ref().unwrap().image, saved);
    state
        .flush_composite(window)
        .expect("replay preserved layer after failed recreate");
    drop(state);
    gpu.destroy_image(image);
    sdl3::video::destroy_window(window);
}

#[test]
fn initialization_retry_frequency_is_bounded_without_blacklisting() {
    let key = 0x794513usize;
    reset_retry(key);
    for _ in 0..16 {
        assert!(schedule_retry(key, false));
    }
    let retries = INIT_RETRIES.lock().expect("retries");
    let retry = retries.as_ref().unwrap().get(&key).unwrap();
    assert_eq!(retry.delay, Duration::from_secs(2), "retry must be capped");
    drop(retries);
    assert!(!retry_ready(key));
    reset_retry(key);
    assert!(retry_ready(key));
}

#[test]
fn device_loss_during_swapchain_initialization_does_not_relock_registry() {
    if isolate_vulkan_graphical_test(
        "device_loss_during_swapchain_initialization_does_not_relock_registry",
    ) {
        return;
    }
    if isolate_vulkan_graphical_test(
        "device_loss_during_swapchain_initialization_does_not_relock_registry",
    ) {
        return;
    }
    if std::env::var_os("ROSEMARY_TEST_VULKAN").is_none() {
        return;
    }
    let Ok(adapters) = crate::vulkan::adapters() else {
        return;
    };
    let Some(gpu) = adapters.first().and_then(|a| a.gpu().ok()) else {
        return;
    };
    let window = sdl3::video::create_window(c"device lost in initializer", 190, 130, 0).unwrap();
    sdl3::video::show_window(window);
    let _ = sdl3::video::presentation_window(window).unwrap();
    let frame = appkit::view::Frame {
        window,
        x: 0,
        y: 0,
        width: 190,
        height: 130,
        window_width: 190,
        window_height: 130,
    };
    let source = gpu
        .create_image(&ImageDescription {
            image_type: vk::ImageType::TYPE_2D,
            view_type: vk::ImageViewType::TYPE_2D,
            format: vk::Format::B8G8R8A8_UNORM,
            width: 190,
            height: 130,
            depth: 1,
            levels: 1,
            layers: 1,
            samples: 1,
            usage: gpu::USAGE_RENDER_TARGET,
            cube: false,
            components: vk::ComponentMapping::default(),
        })
        .unwrap();
    TEST_FAIL_DEVICE_CREATE_WINDOW.store(window as usize, Ordering::Release);
    let deadline = Instant::now() + Duration::from_secs(3);
    while !gpu.lost.load(Ordering::Acquire) {
        assert!(
            Instant::now() < deadline,
            "initializer device loss deadlocked the registry"
        );
        let _ = present_impl(gpu, window as usize, &frame, &source, &[], None, None);
        sdl3::video::vulkan::service_surface_requests();
        std::thread::sleep(Duration::from_millis(2));
    }
    assert!(
        SWAPCHAINS.try_lock().is_ok(),
        "device-loss handling must release swapchain mutex"
    );
    let deadline = Instant::now() + Duration::from_secs(3);
    while INITIALIZING
        .lock()
        .unwrap()
        .as_ref()
        .is_some_and(|all| all.contains_key(&(window as usize)))
    {
        assert!(
            Instant::now() < deadline,
            "device-loss initializer not cleared"
        );
        std::thread::sleep(Duration::from_millis(2));
    }
    assert!(
        sdl3::video::vulkan::owns_window(window),
        "SDL already converted this window before Swapchain::create"
    );
    assert!(
        !sdl3::video::vulkan::surface_request_pending(window),
        "device-loss during Swapchain::create left SDL State::Taken"
    );
    assert!(
        !window_initialization_pending(window),
        "converted backing store permanently deferred on device loss"
    );
    assert_eq!(
        crate::present::present_store(window, &appkit::backing::Store::new(190, 130), &[], &[],),
        appkit::backing::PresentOutcome::Displayed,
        "converted terminal window must not keep reporting Deferred or use SDL fallback"
    );
    forget_window(window);
    std::thread::sleep(Duration::from_millis(10));
    sdl3::video::destroy_window(window);
    sdl3::video::vulkan::service_surface_requests();
    gpu.lost.store(false, Ordering::Release); // synthetic result, real device healthy
    gpu.destroy_image(source);
}

/// Two layers in one window, the second of them a rectangle inside it,
/// which is the path that needs the composite image.
#[test]
#[cfg_attr(
    target_os = "macos",
    ignore = "Cocoa needs the main thread; use rosemary's Metal guest tests"
)]
fn two_layers_in_a_window_are_composited_into_one_image() {
    if isolate_vulkan_graphical_test("two_layers_in_a_window_are_composited_into_one_image") {
        return;
    }
    if std::env::var_os("ROSEMARY_TEST_VULKAN").is_none() {
        return;
    }
    let Ok(adapters) = crate::vulkan::adapters() else {
        return;
    };
    let Some(gpu) = adapters.first().and_then(|adapter| adapter.gpu().ok()) else {
        return;
    };
    let flags = sdl3::video::flags::HIGH_PIXEL_DENSITY;
    let Some(window) = sdl3::video::create_window(c"rosemary compositing", 320, 240, flags) else {
        return;
    };
    sdl3::video::show_window(window);
    let (width, height) =
        sdl3::video::window_size_in_pixels(window).expect("a window SDL just made");
    let (width, height) = (width as u32, height as u32);

    let image = |w: u32, h: u32| {
        gpu.create_image(&ImageDescription {
            image_type: vk::ImageType::TYPE_2D,
            view_type: vk::ImageViewType::TYPE_2D,
            format: vk::Format::B8G8R8A8_UNORM,
            width: w,
            height: h,
            depth: 1,
            levels: 1,
            layers: 1,
            samples: 1,
            usage: gpu::USAGE_RENDER_TARGET,
            cube: false,
            components: vk::ComponentMapping::default(),
        })
        .expect("a colour image")
    };
    let background = image(width, height);
    let overlay = image(width / 2, height / 2);

    let whole = appkit::view::Frame {
        window,
        x: 0,
        y: 0,
        width,
        height,
        window_width: width,
        window_height: height,
    };
    // Deliberately overhanging the bottom right corner, so the clipping
    // in `clamp` is what keeps the blit inside the image.
    let corner = appkit::view::Frame {
        x: (width / 2) as i32,
        y: (height / 2) as i32,
        width,
        height,
        ..whole
    };
    assert!(!corner.covers_window());

    // The first layer alone takes the direct path; the second brings the
    // composite into existence, and both are presented after that.
    assert_eq!(present(gpu, 1, &whole, &background, &[]), Ok(true));
    assert_eq!(present(gpu, 2, &corner, &overlay, &[]), Ok(true));
    assert_eq!(present(gpu, 1, &whole, &background, &[]), Ok(true));
    assert_eq!(present(gpu, 2, &corner, &overlay, &[]), Ok(true));
    flush_if_composed(window);

    let composited = SWAPCHAINS
        .lock()
        .expect("window swapchains")
        .as_ref()
        .and_then(|all| all.get(&(window as usize)))
        .is_some_and(|window| {
            window
                .state
                .lock()
                .expect("window swapchain")
                .composite
                .is_some()
        });
    assert!(
        composited,
        "a window with two layers presented without a composite image"
    );

    SWAPCHAINS
        .lock()
        .expect("window swapchains")
        .as_mut()
        .and_then(|all| all.remove(&(window as usize)));
    gpu.destroy_image(background);
    gpu.destroy_image(overlay);
    sdl3::video::destroy_window(window);
}

/// A window made with room for rounded corners is presented through the
/// corners pass, on a surface that shows its alpha, whatever size it is.
#[test]
#[cfg_attr(
    target_os = "macos",
    ignore = "Cocoa needs the main thread; use rosemary's Metal guest tests"
)]
fn a_transparent_window_is_presented_through_the_corners_pass() {
    if isolate_vulkan_graphical_test("a_transparent_window_is_presented_through_the_corners_pass") {
        return;
    }
    if std::env::var_os("ROSEMARY_TEST_VULKAN").is_none() {
        return;
    }
    let Ok(adapters) = crate::vulkan::adapters() else {
        return;
    };
    let Some(gpu) = adapters.first().and_then(|adapter| adapter.gpu().ok()) else {
        return;
    };
    let flags = sdl3::video::flags::HIGH_PIXEL_DENSITY | sdl3::video::flags::TRANSPARENT;
    let Some(window) = sdl3::video::create_window(c"rosemary corners", 320, 240, flags) else {
        return;
    };
    sdl3::video::show_window(window);
    let frame_of = |window| {
        let (width, height) =
            sdl3::video::window_size_in_pixels(window).expect("a window SDL made");
        appkit::view::Frame {
            window,
            x: 0,
            y: 0,
            width: width as u32,
            height: height as u32,
            window_width: width as u32,
            window_height: height as u32,
        }
    };
    let frame = frame_of(window);
    let source = gpu
        .create_image(&ImageDescription {
            image_type: vk::ImageType::TYPE_2D,
            view_type: vk::ImageViewType::TYPE_2D,
            format: vk::Format::B8G8R8A8_UNORM,
            width: frame.width,
            height: frame.height,
            depth: 1,
            levels: 1,
            layers: 1,
            samples: 1,
            usage: gpu::USAGE_RENDER_TARGET,
            cube: false,
            components: vk::ComponentMapping::default(),
        })
        .expect("a colour image");
    // Enough frames to reuse every frame slot's descriptor set.
    for _ in 0..8 {
        assert_eq!(present(gpu, 1, &frame, &source, &[]), Ok(true));
    }
    flush_if_composed(window);
    let sealed = |check: fn(&Swapchain) -> bool| {
        SWAPCHAINS
            .lock()
            .expect("window swapchains")
            .as_ref()
            .and_then(|all| all.get(&(window as usize)))
            .is_some_and(|window| check(&window.state.lock().expect("window swapchain")))
    };
    // A surface that cannot show premultiplied alpha is presented opaque
    // instead, which is not this test's subject.
    if sealed(|swapchain| swapchain.corners.is_some()) {
        assert!(
            sealed(|swapchain| swapchain.composite.is_some()),
            "the corners pass ran without a composite to read"
        );
        sdl3::video::set_window_size(window, 400, 300);
        let resized = frame_of(window);
        for _ in 0..4 {
            assert_eq!(present(gpu, 1, &resized, &source, &[]), Ok(true));
        }
        flush_if_composed(window);
        assert!(sealed(|swapchain| swapchain.corners.is_some()));
    }

    SWAPCHAINS
        .lock()
        .expect("window swapchains")
        .as_mut()
        .and_then(|all| all.remove(&(window as usize)));
    gpu.destroy_image(source);
    sdl3::video::destroy_window(window);
}

/// An OpenGL window is never converted into a Vulkan one.
///
/// SDL will do it, and on Wayland the result kills the process: the
/// `EGLSurface` SDL made with the window stays alive on the
/// `wl_surface`, the Vulkan driver claims the same surface, and the
/// compositer answers `wp_linux_drm_syncobj_manager_v1: surface already
/// exists` and closes the connection. Presentation must refuse such a
/// window and fall back to the readback rather than convert it.
#[test]
#[cfg_attr(
    target_os = "macos",
    ignore = "Cocoa needs the main thread; use rosemary's Metal guest tests"
)]
fn an_opengl_window_is_refused_rather_than_converted() {
    if isolate_vulkan_graphical_test("an_opengl_window_is_refused_rather_than_converted") {
        return;
    }
    if std::env::var_os("ROSEMARY_TEST_VULKAN").is_none() {
        return;
    }
    let Ok(adapters) = crate::vulkan::adapters() else {
        return;
    };
    let Some(gpu) = adapters.first().and_then(|adapter| adapter.gpu().ok()) else {
        return;
    };
    let flags = sdl3::video::flags::OPENGL | sdl3::video::flags::HIGH_PIXEL_DENSITY;
    let Some(window) = sdl3::video::create_window(c"rosemary opengl", 320, 240, flags) else {
        return;
    };
    sdl3::video::show_window(window);
    let (w, h) = sdl3::video::window_size_in_pixels(window).expect("a window");
    let frame = appkit::view::Frame {
        window,
        x: 0,
        y: 0,
        width: w as u32,
        height: h as u32,
        window_width: w as u32,
        window_height: h as u32,
    };
    let source = gpu
        .create_image(&ImageDescription {
            image_type: vk::ImageType::TYPE_2D,
            view_type: vk::ImageViewType::TYPE_2D,
            format: vk::Format::B8G8R8A8_UNORM,
            width: frame.width,
            height: frame.height,
            depth: 1,
            levels: 1,
            layers: 1,
            samples: 1,
            usage: gpu::USAGE_RENDER_TARGET,
            cube: false,
            components: vk::ComponentMapping::default(),
        })
        .expect("a colour image");

    assert_eq!(
        present(gpu, 1, &frame, &source, &[]),
        Ok(false),
        "an OpenGL window was given a Vulkan surface"
    );
    // Still an OpenGL window afterwards: nothing converted it.
    assert_eq!(
        sdl3::video::window_flags(window) & sdl3::video::flags::VULKAN,
        0,
        "the window was converted to a Vulkan window anyway"
    );
    assert_ne!(
        sdl3::video::window_flags(window) & sdl3::video::flags::OPENGL,
        0,
        "the window lost its OpenGL flag"
    );
    // And it is remembered, so a run does not retry it every frame.
    assert_eq!(present(gpu, 1, &frame, &source, &[]), Ok(false));

    REFUSED
        .lock()
        .expect("refused windows")
        .as_mut()
        .and_then(|refused| refused.remove(&(window as usize)));
    gpu.destroy_image(source);
    sdl3::video::destroy_window(window);
}

/// The whole path a guest actually takes: an `NSWindow` made through
/// `-initWithContentRect:`, a view in it, a `CAMetalLayer` on the view,
/// and a drawable presented through the layer.
///
/// The tests above build the SDL window themselves, so they cannot see a
/// mistake in the flags `-initWithContentRect:` chooses - and that is
/// exactly what once left every real window unable to have a swapchain.
#[test]
#[cfg_attr(
    target_os = "macos",
    ignore = "Cocoa needs the main thread; use rosemary's Metal guest tests"
)]
fn a_layer_on_a_guests_own_window_is_presented_through_a_swapchain() {
    if isolate_vulkan_graphical_test(
        "a_layer_on_a_guests_own_window_is_presented_through_a_swapchain",
    ) {
        return;
    }
    if std::env::var_os("ROSEMARY_TEST_VULKAN").is_none() {
        return;
    }
    let Ok(adapters) = crate::vulkan::adapters() else {
        return;
    };
    let Some(gpu) = adapters.first().and_then(|adapter| adapter.gpu().ok()) else {
        return;
    };
    crate::bootstrap();

    let rect = foundation::geometry::NSRect {
        origin: foundation::geometry::NSPoint { x: 0.0, y: 0.0 },
        size: foundation::geometry::NSSize {
            width: 320.0,
            height: 240.0,
        },
    };
    let class = objc::class::lookup(c"NSWindow").expect("NSWindow");
    let window = objc::object::create_instance(class, appkit::window::INSTANCE_SIZE);
    // SAFETY: `-initWithContentRect:styleMask:backing:defer:` is
    // `id (id, SEL, NSRect, NSUInteger, NSUInteger, BOOL)` on a fresh
    // window, which is how a guest makes one.
    unsafe {
        let send: unsafe extern "C" fn(
            objc::object::Id,
            objc::sel::Sel,
            foundation::geometry::NSRect,
            usize,
            usize,
            u8,
        ) = std::mem::transmute(objc::msg::entry());
        send(
            window,
            objc::sel::get("initWithContentRect:styleMask:backing:defer:"),
            rect,
            0, // Borderless: a titled X11 window has host titlebar insets.
            0,
            0,
        );
    }
    // SAFETY: a window this test just made.
    let sdl = unsafe { appkit::window::sdl_window(window) };
    if sdl.is_null() {
        return; // no display on this host
    }
    assert_eq!(
        sdl3::video::window_flags(sdl) & sdl3::video::flags::OPENGL,
        0,
        "a guest's window claimed OpenGL, so it can never have a swapchain"
    );

    let view_class = objc::class::lookup(c"NSView").expect("NSView");
    let view = objc::object::create_instance(view_class, 0);
    // SAFETY: `-initWithFrame:` is `id (id, SEL, NSRect)`.
    unsafe {
        let send: unsafe extern "C" fn(
            objc::object::Id,
            objc::sel::Sel,
            foundation::geometry::NSRect,
        ) -> objc::object::Id = std::mem::transmute(objc::msg::entry());
        send(view, objc::sel::get("initWithFrame:"), rect);
    }
    // SAFETY: `-setContentView:` is `void (id, SEL, id)`.
    unsafe {
        let send: unsafe extern "C" fn(objc::object::Id, objc::sel::Sel, objc::object::Id) =
            std::mem::transmute(objc::msg::entry());
        send(window, objc::sel::get("setContentView:"), view);
    }

    let layer_class = objc::class::lookup(c"CAMetalLayer").expect("CAMetalLayer");
    // SAFETY: `+layer` is `id (Class, SEL)`.
    let layer = unsafe {
        let send: unsafe extern "C" fn(objc::object::Id, objc::sel::Sel) -> objc::object::Id =
            std::mem::transmute(objc::msg::entry());
        send(
            layer_class as *const objc::class::Class as objc::object::Id,
            objc::sel::get("layer"),
        )
    };
    assert!(!layer.is_null());
    // SAFETY: `-setLayer:` is `void (id, SEL, id)`, and this is what
    // records the layer's view so presentation can find its window.
    unsafe {
        let send: unsafe extern "C" fn(objc::object::Id, objc::sel::Sel, objc::object::Id) =
            std::mem::transmute(objc::msg::entry());
        send(view, objc::sel::get("setLayer:"), layer);
    }

    let frame = crate::present::layer_frame(layer)
        .expect("the layer's view is in a window this runtime made");
    assert_eq!(frame.window, sdl);
    assert!(
        frame.covers_window(),
        "a content view the size of its window did not cover it"
    );

    let source = gpu
        .create_image(&ImageDescription {
            image_type: vk::ImageType::TYPE_2D,
            view_type: vk::ImageViewType::TYPE_2D,
            format: vk::Format::B8G8R8A8_UNORM,
            width: frame.width.max(1),
            height: frame.height.max(1),
            depth: 1,
            levels: 1,
            layers: 1,
            samples: 1,
            usage: gpu::USAGE_RENDER_TARGET,
            cube: false,
            components: vk::ComponentMapping::default(),
        })
        .expect("a colour image");
    assert_eq!(
        present(gpu, layer as usize, &frame, &source, &[]),
        Ok(true),
        "a guest's own window fell back to the readback"
    );

    // Regression: an SDL resize/event invalidates the worker-readable
    // snapshot after this drawable was submitted but before its dispatch.
    // The drawable must remain in the Submitted ring until a real Vulkan
    // presentation occurs; CPU readback is forbidden on the converted SDL window.
    let drawable = unsafe { crate::present::next_drawable(layer, objc::sel::get("nextDrawable")) };
    assert!(!drawable.is_null(), "an actual Metal drawable");
    assert!(crate::present::submit(drawable));
    let baseline = {
        let all = SWAPCHAINS.lock().unwrap();
        let entry = all.as_ref().unwrap().get(&(sdl as usize)).unwrap();
        entry.state.lock().unwrap().frame
    };
    let identity = sdl3::video::window_id(sdl);
    sdl3::video::invalidate_presentation_window(identity, &[sdl]);
    // Only the non-SDL owner thread sees the transient cache hole. Calling
    // from the owner would immediately resample SDL and mask the race.
    let address = layer as usize;
    assert!(
        std::thread::spawn(move || {
            crate::present::layer_frame(address as objc::object::Id).is_none()
                && crate::present::layer_waits_for_window_metadata(address as objc::object::Id)
        })
        .join()
        .unwrap()
    );
    let submitted = drawable as usize;
    assert_eq!(
        std::thread::spawn(move || {
            crate::present::enqueue_swapchain(submitted as objc::object::Id)
        })
        .join()
        .unwrap(),
        crate::present::PresentAttempt::DeferredBusy,
        "metadata invalidation must not select CPU readback"
    );
    assert!(
        crate::present::drawable_is_submitted_for_test(drawable),
        "deferred frame was silently completed"
    );
    sdl3::video::publish_presentation_windows(&[sdl]);
    assert!(crate::present::layer_frame(layer).is_some());
    let deadline = Instant::now() + Duration::from_secs(4);
    loop {
        let submitted = drawable as usize;
        match std::thread::spawn(move || {
            crate::present::enqueue_swapchain(submitted as objc::object::Id)
        })
        .join()
        .unwrap()
        {
            crate::present::PresentAttempt::Handled => break,
            crate::present::PresentAttempt::Deferred
            | crate::present::PresentAttempt::DeferredBusy => {
                assert!(
                    Instant::now() < deadline,
                    "deferred drawable never presented"
                );
                sdl3::video::vulkan::service_surface_requests();
                std::thread::sleep(Duration::from_millis(2));
            }
            crate::present::PresentAttempt::Readback => {
                panic!("the converted window incorrectly selected CPU fallback")
            }
        }
    }
    flush_if_composed(sdl);
    let after = {
        let all = SWAPCHAINS.lock().unwrap();
        let entry = all.as_ref().unwrap().get(&(sdl as usize)).unwrap();
        entry.state.lock().unwrap().frame
    };
    assert!(
        after > baseline,
        "the deferred drawable never reached Vulkan"
    );
    // This harness omits the asynchronous production compositor worker.
    // Its direct single-layer source remains retained after WSI copy until
    // the worker's idle snapshot releases the last CAMetalDrawable. Exercise
    // that exact completion contract synchronously rather than timing out.
    {
        let entry = SWAPCHAINS
            .lock()
            .unwrap()
            .as_ref()
            .unwrap()
            .get(&(sdl as usize))
            .unwrap()
            .clone();
        let _presenting = entry.presenting.lock().unwrap();
        let mut state = entry.state.lock().unwrap();
        if state.direct.is_some() {
            assert!(
                crate::present::drawable_is_submitted_for_test(drawable),
                "WSI copy must not finish a drawable before idle snapshot"
            );
            state.direct.as_mut().unwrap().queued_at = Instant::now() - DIRECT_IDLE_SNAPSHOT_DELAY;
            assert!(
                state
                    .snapshot_idle_direct()
                    .expect("idle snapshot after direct WSI")
            );
        }
    }
    // Other backing-store recomposites can create more swapchain frames.
    // This drawable itself was handled once; wait for its real copy fence.
    let deadline = Instant::now() + Duration::from_secs(3);
    while crate::present::drawable_is_submitted_for_test(drawable) {
        crate::swapchain::retire_copied();
        assert!(
            Instant::now() < deadline,
            "the submitted drawable never completed"
        );
        std::thread::sleep(Duration::from_millis(2));
    }

    // A1 loses its cached geometry while the presenter is scanning a batch.
    // A2 belongs to the same window but SDL republishes its metadata between
    // those two dispatch decisions. Neither A2 nor a later batch may overtake
    // the older submitted drawable.
    let first = unsafe { crate::present::next_drawable(layer, objc::sel::get("nextDrawable")) };
    let second = unsafe { crate::present::next_drawable(layer, objc::sel::get("nextDrawable")) };
    assert!(!first.is_null() && !second.is_null() && first != second);
    assert!(crate::present::submit(first));
    assert!(crate::present::submit(second));
    let first_addr = first as usize;
    let second_addr = second as usize;
    sdl3::video::invalidate_presentation_window(identity, &[sdl]);
    let (reached, at_first) = std::sync::mpsc::channel();
    let (resume, after_publish) = std::sync::mpsc::channel();
    let batch = std::thread::spawn(move || {
        let mut attempted = Vec::new();
        let deferred = crate::present::dispatch_batch(
            std::collections::VecDeque::from([first_addr, second_addr]),
            |drawable| crate::present::drawable_window(drawable as objc::object::Id),
            |drawable| {
                attempted.push(drawable);
                let outcome = crate::present::enqueue_swapchain(drawable as objc::object::Id);
                if drawable == first_addr {
                    reached.send(outcome).expect("notify video thread");
                    after_publish.recv().expect("metadata publication");
                }
                outcome
            },
            |_drawable, _outcome| {},
        );
        (deferred, attempted)
    });
    assert_eq!(
        at_first
            .recv_timeout(Duration::from_secs(4))
            .expect("first drawable was examined"),
        crate::present::PresentAttempt::DeferredBusy,
    );
    sdl3::video::publish_presentation_windows(&[sdl]);
    assert!(crate::present::layer_frame(layer).is_some());
    resume.send(()).expect("resume the same batch");
    let (mut deferred, attempted) = batch.join().expect("presenter batch");
    assert_eq!(
        attempted,
        [first_addr],
        "newer A2 overtook metadata-starved A1"
    );
    assert_eq!(
        deferred,
        std::collections::VecDeque::from([first_addr, second_addr]),
        "the original order must survive the cache-publication transition"
    );
    assert!(
        crate::present::drawable_is_submitted_for_test(first)
            && crate::present::drawable_is_submitted_for_test(second),
        "the dispatcher consumed a queued drawable during invalidation"
    );
    // Replay through the same real batch dispatcher, and check the order
    // in which the two distinct Metal drawables actually reach the swapchain.
    let mut handled = Vec::new();
    let deadline = Instant::now() + Duration::from_secs(5);
    while !deferred.is_empty() {
        let (retry, newly_handled) = std::thread::spawn(move || {
            let mut completed = Vec::new();
            let retry = crate::present::dispatch_batch(
                deferred,
                |drawable| crate::present::drawable_window(drawable as objc::object::Id),
                |drawable| crate::present::enqueue_swapchain(drawable as objc::object::Id),
                |drawable, result| {
                    assert_eq!(result, crate::present::PresentAttempt::Handled);
                    completed.push(drawable);
                },
            );
            (retry, completed)
        })
        .join()
        .expect("presenter retry");
        deferred = retry;
        handled.extend(newly_handled);
        if !deferred.is_empty() {
            assert!(Instant::now() < deadline, "queued pair never presented");
            sdl3::video::vulkan::service_surface_requests();
            std::thread::sleep(Duration::from_millis(2));
        }
    }
    assert_eq!(
        handled,
        [first_addr, second_addr],
        "the frames must not reorder"
    );
    flush_if_composed(sdl);
    {
        let entry = SWAPCHAINS
            .lock()
            .unwrap()
            .as_ref()
            .unwrap()
            .get(&(sdl as usize))
            .unwrap()
            .clone();
        let _presenting = entry.presenting.lock().unwrap();
        let mut state = entry.state.lock().unwrap();
        if state.direct.is_some() {
            assert!(
                crate::present::drawable_is_submitted_for_test(second),
                "the last direct drawable remains submitted until GPU idle snapshot"
            );
            state.direct.as_mut().unwrap().queued_at = Instant::now() - DIRECT_IDLE_SNAPSHOT_DELAY;
            assert!(
                state
                    .snapshot_idle_direct()
                    .expect("idle snapshot for deferred pair")
            );
        }
    }
    let deadline = Instant::now() + Duration::from_secs(3);
    while crate::present::drawable_is_submitted_for_test(first)
        || crate::present::drawable_is_submitted_for_test(second)
    {
        crate::swapchain::retire_copied();
        assert!(Instant::now() < deadline, "deferred pair never retired");
        std::thread::sleep(Duration::from_millis(2));
    }

    SWAPCHAINS
        .lock()
        .expect("window swapchains")
        .as_mut()
        .and_then(|all| all.remove(&(sdl as usize)));
    gpu.destroy_image(source);
}

#[test]
fn a_frame_off_the_edge_of_its_window_is_clipped_to_it() {
    let extent = vk::Extent2D {
        width: 100,
        height: 80,
    };
    let frame = |x, y, width, height| appkit::view::Frame {
        window: std::ptr::null_mut(),
        x,
        y,
        width,
        height,
        window_width: extent.width,
        window_height: extent.height,
    };
    // Wholly inside, which is left alone.
    let [from, to] = clamp(&frame(10, 20, 30, 40), extent).expect("inside the window");
    assert_eq!((from.x, from.y, to.x, to.y), (10, 20, 40, 60));
    // Hanging off the far corner, clipped to the image.
    let [from, to] = clamp(&frame(80, 60, 100, 100), extent).expect("partly inside");
    assert_eq!((from.x, from.y, to.x, to.y), (80, 60, 100, 80));
    // Off the near corner, which a negative origin reaches.
    let [from, to] = clamp(&frame(-20, -30, 50, 50), extent).expect("partly inside");
    assert_eq!((from.x, from.y, to.x, to.y), (0, 0, 30, 20));
    // Entirely outside, which is nothing to blit rather than an empty
    // rectangle Vulkan would reject.
    assert!(clamp(&frame(500, 500, 10, 10), extent).is_none());
    assert!(clamp(&frame(-50, 0, 10, 10), extent).is_none());
    // A frame with no area of its own is the same answer.
    assert!(clamp(&frame(10, 10, 0, 40), extent).is_none());
}

/// A swapchain whose extent lags its window's size in pixels, as one does
/// between a resize and the rebuild, is still filled by a layer covering the
/// window: scaled, as before logical clipping, not cropped or left unwritten.
#[test]
fn a_covering_layer_fills_a_swapchain_extent_that_lags_its_window() {
    let frame = appkit::view::Frame {
        window: std::ptr::null_mut(),
        x: 0,
        y: 0,
        width: 320,
        height: 240,
        window_width: 320,
        window_height: 240,
    };
    for extent in [(330, 250), (300, 200)] {
        let extent = vk::Extent2D {
            width: extent.0,
            height: extent.1,
        };
        let (from, into) = direct_blit_to(&frame, extent, None, (320, 240)).unwrap();
        let whole = |w: u32, h: u32| {
            [
                vk::Offset3D { x: 0, y: 0, z: 0 },
                vk::Offset3D {
                    x: w as i32,
                    y: h as i32,
                    z: 1,
                },
            ]
        };
        assert_eq!(into, whole(extent.width, extent.height), "{extent:?}");
        assert_eq!(from, whole(320, 240), "{extent:?}");
    }
    // A logical clip in window pixels scales with the frame.
    let allowed = sdl3::Rect {
        x: 0,
        y: 24,
        w: 320,
        h: 216,
    };
    let extent = vk::Extent2D {
        width: 640,
        height: 480,
    };
    let (from, into) = direct_blit_to(&frame, extent, Some(allowed), (320, 240)).unwrap();
    assert_eq!((into[0].y, into[1].y), (48, 480));
    assert_eq!((from[0].y, from[1].y), (24, 240));
}

/// Actual WSI presentation with a shortened AppKit NSWindow inside a full-
/// display SDL container. A pure clipping unit cannot validate Vulkan image
/// layout changes needed for the new black-band clear and cropped transfer.
#[test]
#[cfg_attr(target_os = "macos", ignore = "requires a Fedora X11 Vulkan window")]
fn a_shrunken_guest_fullscreen_is_presented_through_the_vulkan_swapchain() {
    if isolate_vulkan_graphical_test(
        "a_shrunken_guest_fullscreen_is_presented_through_the_vulkan_swapchain",
    ) {
        return;
    }
    if std::env::var_os("ROSEMARY_TEST_VULKAN").is_none() {
        return;
    }
    let Ok(adapters) = crate::vulkan::adapters() else {
        return;
    };
    let Some(gpu) = adapters.first().and_then(|adapter| adapter.gpu().ok()) else {
        return;
    };
    crate::bootstrap();
    let pool = objc::arc::pool_push();
    let class = objc::class::lookup(c"NSWindow").unwrap();
    let object = objc::object::create_instance(class, appkit::window::INSTANCE_SIZE);
    let mut frame = foundation::geometry::NSRect {
        origin: foundation::geometry::NSPoint { x: 80.0, y: 90.0 },
        size: foundation::geometry::NSSize {
            width: 400.0,
            height: 300.0,
        },
    };
    let init: unsafe extern "C" fn(
        objc::object::Id,
        objc::sel::Sel,
        foundation::geometry::NSRect,
        usize,
        usize,
        u8,
    ) -> objc::object::Id = unsafe { std::mem::transmute(objc::msg::entry()) };
    assert_eq!(
        unsafe {
            init(
                object,
                objc::sel::get("initWithContentRect:styleMask:backing:defer:"),
                frame,
                9,
                0,
                0,
            )
        },
        object
    );
    let sdl = unsafe { appkit::window::sdl_window(object) };
    assert!(!sdl.is_null());
    sdl3::video::show_window(sdl);
    let toggle: unsafe extern "C" fn(objc::object::Id, objc::sel::Sel, objc::object::Id) =
        unsafe { std::mem::transmute(objc::msg::entry()) };
    unsafe {
        toggle(
            object,
            objc::sel::get("toggleFullScreen:"),
            std::ptr::null_mut(),
        )
    };
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while sdl3::video::window_flags(sdl) & sdl3::video::flags::FULLSCREEN == 0 {
        appkit::event::pump(std::ptr::null_mut());
        assert!(
            std::time::Instant::now() < deadline,
            "fullscreen did not settle"
        );
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    appkit::event::pump(std::ptr::null_mut());
    let get: unsafe extern "C" fn(
        objc::object::Id,
        objc::sel::Sel,
    ) -> foundation::geometry::NSRect = unsafe { std::mem::transmute(objc::msg::entry_stret()) };
    frame = unsafe { get(object, objc::sel::get("frame")) };
    let host_frame = frame;
    frame.origin.y += 31.0;
    frame.size.height -= 31.0;
    let set: unsafe extern "C" fn(objc::object::Id, objc::sel::Sel, foundation::geometry::NSRect) =
        unsafe { std::mem::transmute(objc::msg::entry()) };
    unsafe { set(object, objc::sel::get("setFrame:"), frame) };
    let bounds = appkit::window::presentation_clip(sdl).unwrap();
    let (w, h) = sdl3::video::window_size_in_pixels(sdl).unwrap();
    assert!(
        bounds.h < h && bounds.h > 0,
        "logical window did not shrink"
    );
    let full = appkit::view::Frame {
        window: sdl,
        x: 0,
        y: 0,
        width: w as u32,
        height: h as u32,
        window_width: w as u32,
        window_height: h as u32,
    };
    let accepted = clamp(
        &full,
        vk::Extent2D {
            width: w as u32,
            height: h as u32,
        },
    )
    .unwrap();
    assert_eq!(accepted[1].y - accepted[0].y, bounds.h);
    let source = gpu
        .create_image(&ImageDescription {
            image_type: vk::ImageType::TYPE_2D,
            view_type: vk::ImageViewType::TYPE_2D,
            format: vk::Format::B8G8R8A8_UNORM,
            width: full.width,
            height: full.height,
            depth: 1,
            levels: 1,
            layers: 1,
            samples: 1,
            usage: gpu::USAGE_RENDER_TARGET,
            cube: false,
            components: vk::ComponentMapping::default(),
        })
        .unwrap();
    // Give the oversized layer an unmistakable red drawable: pixel-level
    // verification must not pass merely because an uninitialised source image
    // happened to be black.
    gpu.one_shot(|device, commands| unsafe {
        device.cmd_clear_color_image(
            commands,
            source.image,
            vk::ImageLayout::GENERAL,
            &vk::ClearColorValue {
                float32: [1.0, 0.0, 0.0, 1.0],
            },
            &[vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            }],
        );
    })
    .expect("red source cleared through Vulkan");
    for _ in 0..5 {
        assert_eq!(present(gpu, 333, &full, &source, &[]), Ok(true));
        // The second layer forces the accumulator/compositor path that GPU
        // DMA-BUF overlays share, rather than only direct layer presentation.
        assert_eq!(present(gpu, 334, &full, &source, &[]), Ok(true));
    }
    assert!(
        SWAPCHAINS
            .lock()
            .unwrap()
            .as_ref()
            .and_then(|all| all.get(&(sdl as usize)))
            .is_some_and(|chain| chain.state.lock().unwrap().composite.is_some())
    );
    // Optional pixel oracle for real X11 Vulkan hosts with ImageMagick.
    // The test otherwise still executes Vulkan's recorded clear/blit/present.
    if std::env::var_os("ROSEMARY_TEST_VULKAN_PIXELS").is_some() {
        std::thread::sleep(std::time::Duration::from_millis(220));
        let image =
            std::env::temp_dir().join(format!("rosemary-vulkan-clip-{}.png", std::process::id()));
        let photographed = std::process::Command::new("import")
            .args(["-window", "root"])
            .arg(&image)
            .status()
            .unwrap();
        assert!(
            photographed.success(),
            "X11 screenshot could not be captured"
        );
        let rgb = std::process::Command::new("magick")
            .arg(&image)
            .args(["-depth", "8", "RGB:-"])
            .output()
            .unwrap();
        let _ = std::fs::remove_file(&image);
        assert!(rgb.status.success());
        assert_eq!(rgb.stdout.len(), w as usize * h as usize * 3);
        let sample = |x: usize, y: usize| {
            let index = (y * w as usize + x) * 3;
            &rgb.stdout[index..index + 3]
        };
        let x = w as usize / 2;
        assert_eq!(
            sample(x, h as usize - 5),
            &[0, 0, 0],
            "pixels beyond logical frame escaped"
        );
        assert_eq!(
            sample(x, h as usize - 60),
            &[255, 0, 0],
            "valid Metal layer pixel not presented"
        );
    }
    // Restoring the full logical NSWindow must reveal the source pixels in
    // the previously excluded region, not leave a permanent black mask.
    unsafe { set(object, objc::sel::get("setFrame:"), host_frame) };
    let restored = appkit::window::presentation_clip(sdl).unwrap();
    assert_eq!(
        (restored.x, restored.y, restored.w, restored.h),
        (0, 0, w, h)
    );
    for _ in 0..3 {
        assert_eq!(present(gpu, 333, &full, &source, &[]), Ok(true));
        assert_eq!(present(gpu, 334, &full, &source, &[]), Ok(true));
    }
    if std::env::var_os("ROSEMARY_TEST_VULKAN_PIXELS").is_some() {
        std::thread::sleep(std::time::Duration::from_millis(220));
        let image = std::env::temp_dir().join(format!(
            "rosemary-vulkan-restored-{}.png",
            std::process::id()
        ));
        let screenshot = std::process::Command::new("import")
            .args(["-window", "root"])
            .arg(&image)
            .status()
            .unwrap();
        assert!(screenshot.success());
        let rgb = std::process::Command::new("magick")
            .arg(&image)
            .args(["-depth", "8", "RGB:-"])
            .output()
            .unwrap();
        let _ = std::fs::remove_file(&image);
        assert!(rgb.status.success());
        let offset = ((h as usize - 5) * w as usize + w as usize / 2) * 3;
        assert_eq!(
            &rgb.stdout[offset..offset + 3],
            &[255, 0, 0],
            "restoring the logical frame did not reveal its lower pixels"
        );
    }
    SWAPCHAINS
        .lock()
        .unwrap()
        .as_mut()
        .and_then(|all| all.remove(&(sdl as usize)));
    gpu.destroy_image(source);
    objc::arc::release(object);
    objc::arc::pool_pop(pool);
}

/// A real fractional AppKit view with a CAMetalLayer must use the same
/// half-open pixel edges as the logical window boundary. Validate the
/// destination-to-source crop rather than merely masking the last pixel.
#[test]
#[cfg_attr(target_os = "macos", ignore = "requires a non-Cocoa SDL host")]
fn a_fractional_metal_layer_samples_exactly_its_rounded_destination_edges() {
    if isolate_vulkan_graphical_test(
        "a_fractional_metal_layer_samples_exactly_its_rounded_destination_edges",
    ) {
        return;
    }
    crate::bootstrap();
    let pool = objc::arc::pool_push();
    let class = objc::class::lookup(c"NSWindow").unwrap();
    let window = objc::object::create_instance(class, appkit::window::INSTANCE_SIZE);
    let rect = foundation::geometry::NSRect {
        origin: foundation::geometry::NSPoint { x: 0.0, y: 0.0 },
        size: foundation::geometry::NSSize {
            width: 320.0,
            height: 240.0,
        },
    };
    let init: unsafe extern "C" fn(
        objc::object::Id,
        objc::sel::Sel,
        foundation::geometry::NSRect,
        usize,
        usize,
        u8,
    ) -> objc::object::Id = unsafe { std::mem::transmute(objc::msg::entry()) };
    if unsafe {
        init(
            window,
            objc::sel::get("initWithContentRect:styleMask:backing:defer:"),
            rect,
            0,
            0,
            0,
        )
    } != window
    {
        return;
    }
    // Only the SDL video thread can build the presentation cache.
    if !sdl3::video::is_video_thread() {
        objc::arc::pool_pop(pool);
        return;
    }
    let sdl = unsafe { appkit::window::sdl_window(window) };
    let points = sdl3::video::window_size(sdl).unwrap();
    let physical = sdl3::video::window_size_in_pixels(sdl).unwrap();
    if points != physical {
        // This fixture explicitly tests 1x; 2x is tested in AppKit.
        objc::arc::release(window);
        objc::arc::pool_pop(pool);
        return;
    }
    assert!(points.0 >= 100 && points.1 >= 100);
    let content: unsafe extern "C" fn(objc::object::Id, objc::sel::Sel) -> objc::object::Id =
        unsafe { std::mem::transmute(objc::msg::entry()) };
    let parent = unsafe { content(window, objc::sel::get("contentView")) };
    assert!(!parent.is_null());
    let child_class = objc::class::lookup(c"NSView").unwrap();
    let child = objc::object::create_instance(child_class, 0);
    let w = f64::from(points.0);
    let h = f64::from(points.1);
    let frame = foundation::geometry::NSRect {
        origin: foundation::geometry::NSPoint { x: 0.5, y: 1.0 },
        size: foundation::geometry::NSSize {
            width: w - 1.5,
            height: h - 1.5,
        },
    };
    let set_frame: unsafe extern "C" fn(
        objc::object::Id,
        objc::sel::Sel,
        foundation::geometry::NSRect,
    ) -> objc::object::Id = unsafe { std::mem::transmute(objc::msg::entry()) };
    assert_eq!(
        unsafe { set_frame(child, objc::sel::get("initWithFrame:"), frame) },
        child
    );
    let add: unsafe extern "C" fn(objc::object::Id, objc::sel::Sel, objc::object::Id) =
        unsafe { std::mem::transmute(objc::msg::entry()) };
    unsafe { add(parent, objc::sel::get("addSubview:"), child) };
    let layer_class = objc::class::lookup(c"CAMetalLayer").unwrap();
    let layer: unsafe extern "C" fn(objc::object::Id, objc::sel::Sel) -> objc::object::Id =
        unsafe { std::mem::transmute(objc::msg::entry()) };
    let drawable = unsafe {
        layer(
            layer_class as *const objc::class::Class as objc::object::Id,
            objc::sel::get("layer"),
        )
    };
    assert!(!drawable.is_null());
    unsafe { add(child, objc::sel::get("setLayer:"), drawable) };
    let mapped = crate::present::layer_frame(drawable).expect("fractional CAMetalLayer placement");
    // The view starts at half a point and ends at (host_width - 1). These
    // independently rounded endpoints form [1,host_width-1), not [1,host_width).
    assert_eq!(
        (mapped.x, mapped.y, mapped.width, mapped.height),
        (1, 1, (points.0 - 2) as u32, (points.1 - 2) as u32)
    );
    let extent = vk::Extent2D {
        width: physical.0 as u32,
        height: physical.1 as u32,
    };
    let destination = clamp(&mapped, extent).expect("fractional view intersects its host");
    assert_eq!(
        (
            destination[0].x,
            destination[0].y,
            destination[1].x,
            destination[1].y
        ),
        (1, 1, points.0 - 1, points.1 - 1)
    );
    let crop = clamp_to(
        &mapped,
        extent,
        Some(sdl3::Rect {
            x: 2,
            y: 2,
            w: points.0 - 3,
            h: points.1 - 3,
        }),
    )
    .expect("fractional clipped Metal layer");
    let source = source_of(crop, &mapped, (mapped.width * 2, mapped.height * 2));
    assert_eq!(
        (source[0].x, source[0].y, source[1].x, source[1].y),
        (2, 2, (mapped.width * 2) as i32, (mapped.height * 2) as i32)
    );
    // Earlier origin+width rounding returned 319 rather than 318 for the
    // 320-point host, incorrectly mapping the final crop to source x=636.
    objc::arc::release(child);
    objc::arc::release(window);
    objc::arc::pool_pop(pool);
}

#[test]
fn fullscreen_logical_clip_preserves_metal_blit_source_coordinates() {
    let extent = vk::Extent2D {
        width: 100,
        height: 80,
    };
    let frame = appkit::view::Frame {
        window: std::ptr::null_mut(),
        x: -10,
        y: -5,
        width: 120,
        height: 90,
        window_width: 100,
        window_height: 80,
    };
    let allowed = sdl3::Rect {
        x: 0,
        y: 0,
        w: 100,
        h: 49,
    };
    let clipped =
        clamp_to(&frame, extent, Some(allowed)).expect("oversized layer intersects logical window");
    assert_eq!(
        (clipped[0].x, clipped[0].y, clipped[1].x, clipped[1].y),
        (0, 0, 100, 49)
    );
    let source = source_of(clipped, &frame, (240, 180));
    assert_eq!(
        (source[0].x, source[0].y, source[1].x, source[1].y),
        (20, 10, 220, 108)
    );
    assert!(
        clamp_to(
            &appkit::view::Frame {
                y: 51,
                height: 25,
                ..frame
            },
            extent,
            Some(allowed)
        )
        .is_none()
    );
    // The clipping region is computed in the destination coordinate space,
    // not by moving or shortening Frame before deriving source coordinates.
    assert_eq!(clamp_to(&frame, extent, None).unwrap()[1].y, 80);
}

/// The window's store is presented around the layers over it: a rectangle a
/// `CAMetalLayer` owns keeps what that layer last presented there, and the
/// rest of the store still reaches the window.
#[test]
#[cfg_attr(
    target_os = "macos",
    ignore = "Cocoa needs the main thread; use rosemary's Metal guest tests"
)]
fn the_store_is_presented_around_a_layers_rectangle() {
    let whole = Region {
        x: 0,
        y: 0,
        width: 100,
        height: 100,
    };
    let hole = Region {
        x: 20,
        y: 30,
        width: 40,
        height: 10,
    };
    let pieces = subtract(whole, &[hole]);
    // The bands above, below, left and right of the hole, and nothing else.
    assert_eq!(
        pieces,
        vec![
            Region {
                x: 0,
                y: 0,
                width: 100,
                height: 30
            },
            Region {
                x: 0,
                y: 40,
                width: 100,
                height: 60
            },
            Region {
                x: 0,
                y: 30,
                width: 20,
                height: 10
            },
            Region {
                x: 60,
                y: 30,
                width: 40,
                height: 10
            },
        ]
    );
    // Every piece is inside the whole and outside the hole, and together they
    // are the whole less the hole.
    let area: i32 = pieces.iter().map(|p| p.width * p.height).sum();
    assert_eq!(area, 100 * 100 - 40 * 10);

    // A layer covering the window leaves nothing of the store, and a hole
    // outside it takes nothing away.
    assert!(subtract(whole, &[whole]).is_empty());
    assert_eq!(
        subtract(
            whole,
            &[Region {
                x: 200,
                y: 0,
                width: 10,
                height: 10
            }]
        ),
        vec![whole]
    );
}

#[test]
fn a_region_of_a_layer_is_blitted_from_the_matching_part_of_its_drawable() {
    let frame = appkit::view::Frame {
        window: std::ptr::null_mut(),
        x: 10,
        y: 20,
        width: 100,
        height: 50,
        window_width: 400,
        window_height: 300,
    };
    let at = |x, y| vk::Offset3D { x, y, z: 0 };
    let to = |x, y| vk::Offset3D { x, y, z: 1 };
    // A drawable at twice the view's size: the lower half of the view comes
    // from the lower half of the drawable, not from the window's coordinates.
    assert_eq!(
        source_of([at(10, 45), to(110, 70)], &frame, (200, 100)),
        [at(0, 50), to(200, 100)]
    );
    // The window's own store is its size, so a region maps to itself.
    let store = appkit::view::Frame {
        x: 0,
        y: 0,
        width: 400,
        height: 300,
        ..frame
    };
    assert_eq!(
        source_of([at(10, 45), to(110, 70)], &store, (400, 300)),
        [at(10, 45), to(110, 70)]
    );
}

#[test]
fn fifo_wsi_error_classes_do_not_retry_terminal_acquisition() {
    assert!(matches!(
        classify_compositor_acquire(Err(vk::Result::TIMEOUT)),
        CompositorAcquire::TimedOut
    ));
    assert!(matches!(
        classify_compositor_acquire(Err(vk::Result::ERROR_OUT_OF_DATE_KHR)),
        CompositorAcquire::Ready(None)
    ));
    assert!(matches!(
        classify_compositor_acquire(Err(vk::Result::ERROR_SURFACE_LOST_KHR)),
        CompositorAcquire::Terminal(vk::Result::ERROR_SURFACE_LOST_KHR)
    ));
    assert_eq!(
        terminal_present_failure(vk::Result::ERROR_SURFACE_LOST_KHR),
        1
    );
    assert_eq!(terminal_present_failure(vk::Result::ERROR_DEVICE_LOST), 2);
}

#[test]
fn a_fifo_window_that_keeps_timing_out_backs_off_to_a_bound() {
    let waits: Vec<u64> = (1..=8)
        .map(|timeouts| compositor_acquire_backoff(timeouts).as_millis() as u64)
        .collect();
    assert_eq!(waits, [1, 2, 4, 8, 16, 32, 32, 32]);
}

/// Minimize a live FIFO window while its WSI reports no presentable area.
/// Injected zero extent ensures deterministic behavior across X11/Wayland
/// compositors, some of which retain the prior dimensions while minimized.
#[test]
#[cfg_attr(target_os = "macos", ignore = "Cocoa needs the main thread")]
fn minimized_compositor_recovers_after_repeated_zero_extent_rebuilds() {
    if isolate_vulkan_graphical_test(
        "minimized_compositor_recovers_after_repeated_zero_extent_rebuilds",
    ) {
        return;
    }
    let _retire_guard = AwaitTestWindowRetirements;
    if std::env::var_os("ROSEMARY_TEST_VULKAN").is_none() {
        return;
    }
    let adapters = crate::vulkan::adapters().expect("Vulkan adapters");
    let gpu = adapters.first().expect("adapter").gpu().expect("GPU");
    let window = sdl3::video::create_window(c"FIFO minimized recovery", 220, 160, 0)
        .expect("SDL test window");
    sdl3::video::show_window(window);
    sdl3::video::presentation_window(window).expect("SDL snapshot");
    let source = gpu
        .create_image(&ImageDescription {
            image_type: vk::ImageType::TYPE_2D,
            view_type: vk::ImageViewType::TYPE_2D,
            format: vk::Format::B8G8R8A8_UNORM,
            width: 220,
            height: 160,
            depth: 1,
            levels: 1,
            layers: 1,
            samples: 1,
            usage: gpu::USAGE_RENDER_TARGET,
            cube: false,
            components: vk::ComponentMapping::default(),
        })
        .expect("source image");
    let frame = appkit::view::Frame {
        window,
        x: 0,
        y: 0,
        width: 220,
        height: 160,
        window_width: 220,
        window_height: 160,
    };
    assert_eq!(
        present(gpu, window as usize, &frame, &source, &[]),
        Ok(true)
    );
    let entry = SWAPCHAINS
        .lock()
        .unwrap()
        .as_ref()
        .unwrap()
        .get(&(window as usize))
        .expect("installed swapchain")
        .clone();
    assert!(uses_layer_compositor(
        entry.state.lock().unwrap().present_mode
    ));
    let before = {
        let _presenting = entry.presenting.lock().unwrap();
        let mut state = entry.state.lock().unwrap();
        assert!(state.composite_dirty);
        state.injected_unavailable_extent = true;
        state.retired = true;
        let before = state.rebuild_attempts;
        ensure_layer_compositor_worker(window as usize, &entry);
        before
    };
    sdl3::video::minimize_window(window);
    let deadline = Instant::now() + Duration::from_secs(4);
    loop {
        let attempts = entry.state.lock().unwrap().rebuild_attempts;
        if attempts >= before + 4 {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "FIFO worker did not retry zero extent"
        );
        std::thread::sleep(Duration::from_millis(5));
    }
    assert_eq!(entry.present_failure.load(Ordering::Acquire), 0);
    assert!(
        entry.state.lock().unwrap().composite_dirty,
        "snapshot lost while minimized"
    );
    sdl3::video::restore_window(window);
    {
        let _presenting = entry.presenting.lock().unwrap();
        entry.state.lock().unwrap().injected_unavailable_extent = false;
    }
    wake_compositor_worker(&entry);
    let deadline = Instant::now() + Duration::from_secs(6);
    loop {
        let state = entry.state.lock().unwrap();
        if !state.composite_dirty {
            assert!(!state.retired, "restoration never rebuilt WSI");
            break;
        }
        assert_eq!(
            entry.present_failure.load(Ordering::Acquire),
            0,
            "temporary zero area permanently disabled FIFO"
        );
        assert!(
            Instant::now() < deadline,
            "restored FIFO window never presented snapshot"
        );
        drop(state);
        std::thread::sleep(Duration::from_millis(5));
    }
    forget_window(window);
    drop(entry);
    sdl3::video::destroy_window(window);
    gpu.destroy_image(source);
}

#[test]
fn unsupported_sdl_surface_falls_back_without_permanent_transient_blacklisting() {
    if isolate_vulkan_graphical_test(
        "unsupported_sdl_surface_falls_back_without_permanent_transient_blacklisting",
    ) {
        return;
    }
    let permanent = sdl3::video::vulkan::SurfaceFailure {
        reason: "SDL Vulkan not supported on window surface".into(),
        transient: false,
    };
    assert!(permanent_sdl_surface_failure(&permanent));
    let temporary = sdl3::video::vulkan::SurfaceFailure {
        reason: "SDL surface not available while minimized".into(),
        transient: true,
    };
    assert!(!permanent_sdl_surface_failure(&temporary));
    let unknown = sdl3::video::vulkan::SurfaceFailure {
        reason: "unknown SDL Vulkan surface error".into(),
        transient: false,
    };
    assert!(!permanent_sdl_surface_failure(&unknown));
    let key = 0x7947a1usize;
    reset_retry(key);
    for _ in 1..MAX_UNCLASSIFIED_SURFACE_FAILURES {
        assert!(schedule_retry(key, true));
    }
    assert!(
        !schedule_retry(key, true),
        "permanently unsupported surface retries forever"
    );
    reset_retry(key);
    for _ in 0..20 {
        assert!(
            schedule_retry(key, false),
            "minimized window must never exhaust a lifetime retry budget"
        );
    }
    reset_retry(key);
    assert!(retry_ready(key));
}

/// A new layer must not join the occlusion set until its first frame can
/// actually be staged after a deferred WSI rebuild.
#[test]
#[cfg_attr(target_os = "macos", ignore = "SDL/Cocoa main thread integration")]
fn failed_async_rebuild_does_not_register_unpresented_layer() {
    if isolate_vulkan_graphical_test("failed_async_rebuild_does_not_register_unpresented_layer") {
        return;
    }
    let _retire_guard = AwaitTestWindowRetirements;
    if std::env::var_os("ROSEMARY_TEST_VULKAN").is_none() {
        return;
    }
    let gpu = crate::vulkan::adapters()
        .unwrap()
        .first()
        .unwrap()
        .gpu()
        .unwrap();
    let window = sdl3::video::create_window(c"deferred layer rebuild", 200, 150, 0).unwrap();
    sdl3::video::show_window(window);
    sdl3::video::presentation_window(window).unwrap();
    let source = gpu
        .create_image(&ImageDescription {
            image_type: vk::ImageType::TYPE_2D,
            view_type: vk::ImageViewType::TYPE_2D,
            format: vk::Format::B8G8R8A8_UNORM,
            width: 200,
            height: 150,
            depth: 1,
            levels: 1,
            layers: 1,
            samples: 1,
            usage: gpu::USAGE_RENDER_TARGET,
            cube: false,
            components: vk::ComponentMapping::default(),
        })
        .unwrap();
    let frame = appkit::view::Frame {
        window,
        x: 0,
        y: 0,
        width: 200,
        height: 150,
        window_width: 200,
        window_height: 150,
    };
    assert_eq!(
        present(gpu, window as usize, &frame, &source, &[]),
        Ok(true)
    );
    let entry = SWAPCHAINS
        .lock()
        .unwrap()
        .as_ref()
        .unwrap()
        .get(&(window as usize))
        .unwrap()
        .clone();
    let layer = window as usize + 0x40;
    sdl3::video::minimize_window(window);
    {
        let mut state = entry.state.lock().unwrap();
        state.injected_unavailable_extent = true;
        state.retired = true;
    }
    let started = Instant::now();
    assert_eq!(
        present_nonblocking(gpu, layer, &frame, &source, &[]),
        Ok(None)
    );
    assert!(
        started.elapsed() < Duration::from_millis(150),
        "nonblocking resize waited for WSI"
    );
    {
        let state = entry.state.lock().unwrap();
        assert!(
            !state.layers.contains(&layer),
            "failed rebuild registered a never-presented layer"
        );
        assert_eq!(
            state.pending_rebuild,
            Some(vk::Extent2D {
                width: 200,
                height: 150
            })
        );
    }
    let deadline = Instant::now() + Duration::from_secs(3);
    loop {
        if entry.state.lock().unwrap().rebuild_attempts >= 4 {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "worker did not retry deferred rebuild"
        );
        std::thread::sleep(Duration::from_millis(5));
    }
    assert!(!entry.state.lock().unwrap().layers.contains(&layer));
    sdl3::video::restore_window(window);
    {
        let _presenting = entry.presenting.lock().unwrap();
        entry.state.lock().unwrap().injected_unavailable_extent = false;
    }
    wake_resize_worker(&entry);
    let deadline = Instant::now() + Duration::from_secs(5);
    while entry.state.lock().unwrap().pending_rebuild.is_some() {
        assert!(
            Instant::now() < deadline,
            "WSI did not recover after restore"
        );
        std::thread::sleep(Duration::from_millis(5));
    }
    assert_eq!(present(gpu, layer, &frame, &source, &[]), Ok(true));
    assert!(entry.state.lock().unwrap().layers.contains(&layer));
    forget_window(window);
    drop(entry);
    sdl3::video::destroy_window(window);
    gpu.destroy_image(source);
}

/// Compositor invalidation can occur without any change to window geometry.
/// In both out-of-date paths the backing-store caller must return before the
/// worker enters idle(), leaving its store available for restoration.
#[cfg_attr(target_os = "macos", ignore = "SDL video thread needed")]
fn nonfifo_out_of_date_is_deferred(invalidate_during_acquire: bool) {
    if std::env::var_os("ROSEMARY_TEST_VULKAN").is_none() {
        return;
    }
    let adapters = crate::vulkan::adapters().expect("Vulkan adapters");
    let gpu = adapters
        .first()
        .expect("Vulkan adapter")
        .gpu()
        .expect("Vulkan GPU");
    let window =
        sdl3::video::create_window(c"non FIFO out-of-date", 220, 160, 0).expect("SDL window");
    sdl3::video::show_window(window);
    sdl3::video::presentation_window(window).expect("SDL snapshot");
    let frame = appkit::view::Frame {
        window,
        x: 0,
        y: 0,
        width: 220,
        height: 160,
        window_width: 220,
        window_height: 160,
    };
    let source = gpu
        .create_image(&ImageDescription {
            image_type: vk::ImageType::TYPE_2D,
            view_type: vk::ImageViewType::TYPE_2D,
            format: vk::Format::B8G8R8A8_UNORM,
            width: 220,
            height: 160,
            depth: 1,
            levels: 1,
            layers: 1,
            samples: 1,
            usage: gpu::USAGE_RENDER_TARGET,
            cube: false,
            components: vk::ComponentMapping::default(),
        })
        .expect("source image");
    assert_eq!(
        present(gpu, window as usize, &frame, &source, &[]),
        Ok(true)
    );
    flush_if_fifo(window);
    let entry = SWAPCHAINS
        .lock()
        .unwrap()
        .as_ref()
        .unwrap()
        .get(&(window as usize))
        .unwrap()
        .clone();
    {
        let mut state = entry.state.lock().unwrap();
        state.present_mode = vk::PresentModeKHR::IMMEDIATE;
        state.injected_unavailable_extent = true;
        if !invalidate_during_acquire {
            state.injected_queue_present_error = Some(vk::Result::ERROR_OUT_OF_DATE_KHR);
        }
    }
    if invalidate_during_acquire {
        *entry.injected_nonfifo_acquire_error.lock().unwrap() =
            Some(vk::Result::ERROR_OUT_OF_DATE_KHR);
    }
    // A healthy nonblocking VkAcquireNextImageKHR may legitimately return
    // NOT_READY before it reaches the deliberately injected failure. Retry
    // until the fault has actually been exercised, without relaxing the
    // per-call AppKit event-thread latency assertion.
    let deadline = Instant::now() + Duration::from_secs(3);
    let attempts_before = loop {
        let start = Instant::now();
        let presented = present_nonblocking(gpu, window as usize, &frame, &source, &[]);
        assert_eq!(presented, Ok(None), "out-of-date store must remain dirty");
        assert!(
            start.elapsed() < Duration::from_millis(250),
            "AppKit backed up waiting for synchronous WSI rebuild"
        );
        let state = entry.state.lock().unwrap();
        if state.pending_rebuild
            == Some(vk::Extent2D {
                width: 220,
                height: 160,
            })
        {
            break state.rebuild_attempts;
        }
        let still_armed = if invalidate_during_acquire {
            entry
                .injected_nonfifo_acquire_error
                .lock()
                .unwrap()
                .is_some()
        } else {
            state.injected_queue_present_error.is_some()
        };
        assert!(
            still_armed,
            "fault consumed without scheduling a WSI rebuild"
        );
        drop(state);
        assert!(
            Instant::now() < deadline,
            "nonblocking acquire never reached the injected fault"
        );
        std::thread::sleep(Duration::from_millis(2));
    };
    let deadline = Instant::now() + Duration::from_secs(3);
    loop {
        if entry.state.lock().unwrap().rebuild_attempts >= attempts_before + 3 {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "deferred WSI worker did not retry"
        );
        std::thread::sleep(Duration::from_millis(5));
    }
    {
        let _presenting = entry.presenting.lock().unwrap();
        entry.state.lock().unwrap().injected_unavailable_extent = false;
    }
    wake_resize_worker(&entry);
    let deadline = Instant::now() + Duration::from_secs(5);
    while entry.state.lock().unwrap().pending_rebuild.is_some() {
        assert!(Instant::now() < deadline, "deferred WSI did not recover");
        std::thread::sleep(Duration::from_millis(5));
    }
    assert_eq!(entry.fifo_failure.load(Ordering::Acquire), 0);
    assert_eq!(
        present(gpu, window as usize, &frame, &source, &[]),
        Ok(true)
    );
    forget_window(window);
    drop(entry);
    gpu.destroy_image(source);
    sdl3::video::destroy_window(window);
}

#[test]
fn nonfifo_acquisition_out_of_date_never_rebuilds_on_appkit_thread() {
    let _retire_guard = AwaitTestWindowRetirements;
    nonfifo_out_of_date_is_deferred(true);
}

#[test]
fn nonfifo_queue_present_out_of_date_never_rebuilds_on_appkit_thread() {
    let _retire_guard = AwaitTestWindowRetirements;
    nonfifo_out_of_date_is_deferred(false);
}

fn v17_async_window_fixture() -> Option<(
    sdl3::Window,
    &'static Gpu,
    GpuImage,
    appkit::view::Frame,
    Arc<WindowSwapchain>,
)> {
    // FIXME (TESTS): this was turned off on macOS, so i'm leaving it stubbed
    if cfg!(target_os = "macos") {
        return None;
    }

    if std::env::var_os("ROSEMARY_TEST_VULKAN").is_none() {
        return None;
    }
    let gpu = crate::vulkan::adapters().ok()?.first()?.gpu().ok()?;
    let window = sdl3::video::create_window(c"WSI asynchronous ownership", 240, 180, 0)?;
    sdl3::video::show_window(window);
    sdl3::video::presentation_window(window)?;
    let frame = appkit::view::Frame {
        window,
        x: 0,
        y: 0,
        width: 240,
        height: 180,
        window_width: 240,
        window_height: 180,
    };
    let source = gpu
        .create_image(&ImageDescription {
            image_type: vk::ImageType::TYPE_2D,
            view_type: vk::ImageViewType::TYPE_2D,
            format: vk::Format::B8G8R8A8_UNORM,
            width: 240,
            height: 180,
            depth: 1,
            levels: 1,
            layers: 1,
            samples: 1,
            usage: gpu::USAGE_RENDER_TARGET,
            cube: false,
            components: vk::ComponentMapping::default(),
        })
        .ok()?;
    assert_eq!(
        present(gpu, window as usize, &frame, &source, &[]),
        Ok(true)
    );
    flush_if_composed(window);
    let entry = SWAPCHAINS
        .lock()
        .unwrap()
        .as_ref()
        .unwrap()
        .get(&(window as usize))
        .unwrap()
        .clone();
    Some((window, gpu, source, frame, entry))
}

#[test]
#[cfg(not(target_os = "macos"))]
fn stalled_resize_cannot_hold_appkit_window_destruction_or_destroy_sdl_early() {
    if isolate_vulkan_graphical_test(
        "stalled_resize_cannot_hold_appkit_window_destruction_or_destroy_sdl_early",
    ) {
        return;
    }
    let Some((window, gpu, source, frame, entry)) = v17_async_window_fixture() else {
        return;
    };
    let key = window as usize;
    let identity = sdl3::video::window_id(window);
    assert_ne!(identity, 0);
    TEST_WINDOW_RETIRED.store(0, Ordering::Release);
    TEST_BLOCK_RESIZE_WORKER.store(key, Ordering::Release);
    let resized = appkit::view::Frame {
        window_width: frame.window_width + 1,
        ..frame
    };
    assert_eq!(
        present_nonblocking(gpu, key, &resized, &source, &[]),
        Ok(None),
    );
    let deadline = Instant::now() + Duration::from_secs(3);
    while TEST_RESIZE_WORKER_PAUSED.load(Ordering::Acquire) != key {
        assert!(
            Instant::now() < deadline,
            "worker never held the WSI presentation lock"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
    let start = Instant::now();
    forget_window(window);
    assert!(
        start.elapsed() < Duration::from_millis(200),
        "AppKit destruction waited for a worker stalled in Vulkan"
    );
    let start = Instant::now();
    sdl3::video::destroy_window(window);
    assert!(
        start.elapsed() < Duration::from_millis(200),
        "SDL window destroy waited for the worker"
    );
    sdl3::video::vulkan::service_surface_requests();
    assert_eq!(
        sdl3::video::window_id(window),
        identity,
        "SDL destroyed the native window while Vulkan still owned its surface"
    );
    assert!(
        SWAPCHAINS
            .lock()
            .unwrap()
            .as_ref()
            .unwrap()
            .get(&key)
            .is_none()
    );
    TEST_BLOCK_RESIZE_WORKER.store(0, Ordering::Release);
    drop(entry);
    let deadline = Instant::now() + Duration::from_secs(5);
    while TEST_WINDOW_RETIRED.load(Ordering::Acquire) != key {
        assert!(
            Instant::now() < deadline,
            "async WSI retirement never completed"
        );
        std::thread::sleep(Duration::from_millis(2));
    }
    sdl3::video::vulkan::service_surface_requests();
    gpu.destroy_image(source);
}

// Isolated Vulkan tests must not exit while a detached swapchain drop
// still calls the host ICD. Production teardown remains asynchronous.
struct AwaitTestWindowRetirements;
impl Drop for AwaitTestWindowRetirements {
    fn drop(&mut self) {
        let deadline = Instant::now() + Duration::from_secs(6);
        while TEST_ACTIVE_RETIREMENTS.load(Ordering::Acquire) != 0 {
            assert!(
                Instant::now() < deadline,
                "test WSI retirement did not finish"
            );
            std::thread::sleep(Duration::from_millis(2));
        }
        sdl3::video::vulkan::service_surface_requests();
    }
}

/// SDL recursively deletes native popups. Recreating an AppKit parent must
/// rehome the guest's popup before retiring the child's old GPU swapchain.
#[test]
fn appkit_parent_remake_retires_gpu_backed_popup_without_dangling_nswindow() {
    if isolate_vulkan_graphical_test(
        "appkit_parent_remake_retires_gpu_backed_popup_without_dangling_nswindow",
    ) {
        return;
    }
    let _retire_guard = AwaitTestWindowRetirements;
    if std::env::var_os("ROSEMARY_TEST_VULKAN").is_none() {
        return;
    }
    let Ok(adapters) = crate::vulkan::adapters() else {
        return;
    };
    let Some(gpu) = adapters.first().and_then(|adapter| adapter.gpu().ok()) else {
        return;
    };
    crate::bootstrap();
    let rect = foundation::geometry::NSRect {
        origin: foundation::geometry::NSPoint { x: 0.0, y: 0.0 },
        size: foundation::geometry::NSSize {
            width: 180.0,
            height: 120.0,
        },
    };
    let class = objc::class::lookup(c"NSWindow").unwrap();
    let make = || {
        let object = objc::object::create_instance(class, appkit::window::INSTANCE_SIZE);
        unsafe {
            let init: unsafe extern "C" fn(
                objc::object::Id,
                objc::sel::Sel,
                foundation::geometry::NSRect,
                usize,
                usize,
                u8,
            ) = std::mem::transmute(objc::msg::entry());
            init(
                object,
                objc::sel::get("initWithContentRect:styleMask:backing:defer:"),
                rect,
                0,
                0,
                0,
            );
        }
        object
    };
    let parent = make();
    let child = make();
    // SAFETY: AppKit instance methods with their registered Objective-C ABI.
    unsafe {
        let order: unsafe extern "C" fn(objc::object::Id, objc::sel::Sel, objc::object::Id) =
            std::mem::transmute(objc::msg::entry());
        order(parent, objc::sel::get("orderFront:"), std::ptr::null_mut());
        let level: unsafe extern "C" fn(objc::object::Id, objc::sel::Sel, isize) =
            std::mem::transmute(objc::msg::entry());
        level(child, objc::sel::get("setLevel:"), 101);
        order(child, objc::sel::get("orderFront:"), std::ptr::null_mut());
    }
    let old_parent = unsafe { appkit::window::sdl_window(parent) };
    let old_child = unsafe { appkit::window::sdl_window(child) };
    assert!(!old_parent.is_null() && !old_child.is_null());
    assert_eq!(sdl3::video::window_parent(old_child), old_parent);
    let old_child_id = sdl3::video::live_window_id(old_child).unwrap();
    sdl3::video::presentation_window(old_child).expect("child snapshot");
    let child_frame = appkit::view::Frame {
        window: old_child,
        x: 0,
        y: 0,
        width: 180,
        height: 120,
        window_width: 180,
        window_height: 120,
    };
    let source = gpu
        .create_image(&ImageDescription {
            image_type: vk::ImageType::TYPE_2D,
            view_type: vk::ImageViewType::TYPE_2D,
            format: vk::Format::B8G8R8A8_UNORM,
            width: 180,
            height: 120,
            depth: 1,
            levels: 1,
            layers: 1,
            samples: 1,
            usage: gpu::USAGE_RENDER_TARGET,
            cube: false,
            components: vk::ComponentMapping::default(),
        })
        .expect("GPU image for child");
    assert_eq!(
        present(gpu, 0x450_aaa, &child_frame, &source, &[]),
        Ok(true),
        "the child must actually own a GPU-backed SDL/Vulkan presentation"
    );
    assert!(
        SWAPCHAINS
            .lock()
            .unwrap()
            .as_ref()
            .unwrap()
            .contains_key(&(old_child as usize)),
        "no GPU swapchain created for child"
    );
    // A GPU-converted child may also have a parked CPU backing-store uploader.
    // Its worker must receive the same descendant destroy notification as WSI.
    let mut store = appkit::backing::Store::new(180, 120);
    store.pixels.fill(91);
    let _ = crate::present::present_store(old_child, &store, &[], &[]);
    assert!(
        crate::present::test_has_store_upload_worker(old_child),
        "GPU-backed child did not create its store upload worker"
    );
    assert!(appkit::window::rebuild_for_opengl(parent), "parent remake");
    let new_parent = unsafe { appkit::window::sdl_window(parent) };
    let new_child = unsafe { appkit::window::sdl_window(child) };
    assert!(!new_parent.is_null() && new_parent != old_parent);
    assert!(!new_child.is_null() && new_child != old_child);
    assert_eq!(sdl3::video::window_parent(new_child), new_parent);
    assert_eq!(appkit::window::window_for(new_child), child);
    assert!(appkit::window::window_for(old_child).is_null());
    assert!(sdl3::video::live_window_id(old_child).is_none());
    // Native deletion waits for the GPU-backed child's retirement token.
    assert!(
        !SWAPCHAINS
            .lock()
            .unwrap()
            .as_ref()
            .unwrap()
            .contains_key(&(old_child as usize)),
        "old popup's Metal state survived recursive native destruction"
    );
    assert!(
        !crate::present::test_has_store_upload_worker(old_child),
        "recursive native teardown left the child backing-store worker parked"
    );
    let deadline = Instant::now() + Duration::from_secs(5);
    while sdl3::video::window_from_id(old_child_id) != std::ptr::null_mut() {
        sdl3::video::vulkan::service_surface_requests();
        assert!(Instant::now() < deadline, "GPU-backed popup never retired");
        std::thread::sleep(Duration::from_millis(2));
    }
    assert!(appkit::window::window_for(old_parent).is_null());
    assert_eq!(appkit::window::window_for(new_child), child);
    // A remade parent also owns this fresh popup; AppKit's normal recursive
    // cleanup must release it before the replacement parent is destroyed.
    // A replacement OpenGL parent cannot itself be used for this Vulkan test.
    gpu.destroy_image(source);
}

/// A converted SDL window cannot fall back to CPU surface readback. A
/// transient stage failure keeps the drawable queued, and a terminal OOM
/// reports completion without attempting an invalid fallback.
#[test]
fn converted_fifo_staging_failure_never_uses_cpu_fallback_or_dirties_first_layer() {
    if isolate_vulkan_graphical_test(
        "converted_fifo_staging_failure_never_uses_cpu_fallback_or_dirties_first_layer",
    ) {
        return;
    }
    assert!(!composite_first_stage_needs_store(true, &Ok(None), 42, 7));
    assert!(!composite_first_stage_needs_store(
        true,
        &Err("staging failed".into()),
        42,
        7
    ));
    assert!(composite_first_stage_needs_store(
        true,
        &Ok(Some(true)),
        42,
        7
    ));
    assert!(!composite_first_stage_needs_store(
        false,
        &Ok(Some(true)),
        42,
        7
    ));
    assert!(!composite_first_stage_needs_store(
        true,
        &Ok(Some(true)),
        7,
        7
    ));

    let _retire_guard = AwaitTestWindowRetirements;
    let Some((window, gpu, source, frame, entry)) = v17_async_window_fixture() else {
        return;
    };
    let child_layer = (window as usize).wrapping_add(0x129);
    {
        let mut state = entry.state.lock().unwrap();
        state.present_mode = vk::PresentModeKHR::FIFO;
        // Prove an error does not transfer the presenter's drawable retain
        // into a GPU frame or complete the ring slot during staging.
        let mut retained = SourceOwner(Some(0xfeed_f001));
        state.injected_stage_vk_error = Some(vk::Result::ERROR_UNKNOWN);
        assert!(
            state
                .stage_composite(&frame, &source, None, &mut retained, true)
                .is_err()
        );
        assert_eq!(retained.0, Some(0xfeed_f001));
        retained.0 = None; // dummy identity, not an Objective-C allocation
        state.injected_stage_vk_error = Some(vk::Result::ERROR_UNKNOWN);
    }
    assert!(sdl3::video::vulkan::owns_window(window));
    assert_eq!(
        present_nonblocking(gpu, child_layer, &frame, &source, &[]),
        Ok(None),
        "retryable FIFO staging failure must not consume a Metal drawable"
    );
    assert!(
        !entry.state.lock().unwrap().layers.contains(&child_layer),
        "failed first stage must not register a layer"
    );
    assert_eq!(entry.present_failure.load(Ordering::Acquire), 0);
    // A source which has not been staged needs no new CPU store damage.
    assert!(!composite_first_stage_needs_store(
        true,
        &Ok(None),
        child_layer,
        window as usize
    ));

    // The terminal error must also wake a clean, parked FIFO worker so it
    // can retire instead of retaining the window indefinitely.
    ensure_layer_compositor_worker(window as usize, &entry);
    {
        let mut state = entry.state.lock().unwrap();
        state.injected_stage_vk_error = Some(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY);
    }
    assert_eq!(
        present_nonblocking(gpu, child_layer, &frame, &source, &[]),
        Ok(Some(true)),
        "terminal FIFO error must not be routed to unusable SDL readback"
    );
    assert_eq!(entry.present_failure.load(Ordering::Acquire), 2);
    let deadline = Instant::now() + Duration::from_secs(3);
    while !entry
        .worker
        .lock()
        .unwrap()
        .as_ref()
        .is_some_and(|h| h.is_finished())
    {
        assert!(
            Instant::now() < deadline,
            "terminal staging left FIFO worker asleep"
        );
        std::thread::sleep(Duration::from_millis(2));
    }
    forget_window(window);
    drop(entry);
    sdl3::video::destroy_window(window);
    gpu.destroy_image(source);
}

#[test]
fn asynchronous_resize_device_loss_notifies_other_windows() {
    if isolate_vulkan_graphical_test("asynchronous_resize_device_loss_notifies_other_windows") {
        return;
    }
    let _retire_guard = AwaitTestWindowRetirements;
    let Some((first, gpu, first_source, _, first_entry)) = v17_async_window_fixture() else {
        return;
    };
    let Some((second, _, second_source, _, second_entry)) = v17_async_window_fixture() else {
        forget_window(first);
        drop(first_entry);
        sdl3::video::destroy_window(first);
        gpu.destroy_image(first_source);
        return;
    };
    assert!(!gpu.lost.load(Ordering::Acquire));
    {
        let mut state = first_entry.state.lock().unwrap();
        state.injected_build_vk_error = Some(vk::Result::ERROR_DEVICE_LOST);
        state.pending_rebuild = Some(state.requested);
    }
    ensure_resize_worker(first as usize, &first_entry);
    wake_resize_worker(&first_entry);
    let deadline = Instant::now() + Duration::from_secs(5);
    while !gpu.lost.load(Ordering::Acquire) {
        assert!(
            Instant::now() < deadline,
            "async Vulkan loss remained window-local"
        );
        std::thread::sleep(Duration::from_millis(2));
    }
    assert_eq!(first_entry.present_failure.load(Ordering::Acquire), 2);
    assert_eq!(
        second_entry.present_failure.load(Ordering::Acquire),
        2,
        "other idle window did not receive device-wide loss notification"
    );
    forget_window(first);
    forget_window(second);
    drop(first_entry);
    drop(second_entry);
    sdl3::video::destroy_window(first);
    sdl3::video::destroy_window(second);
    // Injected loss, not a real dead device; allow test fixture cleanup.
    gpu.lost.store(false, Ordering::Release);
    gpu.destroy_image(first_source);
    gpu.destroy_image(second_source);
}

#[test]
fn compositor_preparation_device_loss_wakes_other_window_workers() {
    if isolate_vulkan_graphical_test(
        "compositor_preparation_device_loss_wakes_other_window_workers",
    ) {
        return;
    }
    let _retire_guard = AwaitTestWindowRetirements;
    let Some((first, gpu, first_source, _, first_entry)) = v17_async_window_fixture() else {
        return;
    };
    let Some((second, _, second_source, _, second_entry)) = v17_async_window_fixture() else {
        forget_window(first);
        drop(first_entry);
        sdl3::video::destroy_window(first);
        gpu.destroy_image(first_source);
        return;
    };
    assert!(!gpu.lost.load(Ordering::Acquire));
    {
        let mut state = first_entry.state.lock().unwrap();
        assert!(uses_layer_compositor(state.present_mode));
        state.retired = true;
        state.composite_dirty = true;
        state.composite_dirty_since = None;
        state.injected_build_vk_error = Some(vk::Result::ERROR_DEVICE_LOST);
    }
    ensure_layer_compositor_worker(first as usize, &first_entry);
    wake_compositor_worker(&first_entry);
    let deadline = Instant::now() + Duration::from_secs(5);
    while !gpu.lost.load(Ordering::Acquire) {
        assert!(
            Instant::now() < deadline,
            "FIFO preparation loss remained local"
        );
        std::thread::sleep(Duration::from_millis(2));
    }
    assert_eq!(first_entry.present_failure.load(Ordering::Acquire), 2);
    assert_eq!(second_entry.present_failure.load(Ordering::Acquire), 2);
    forget_window(first);
    forget_window(second);
    drop(first_entry);
    drop(second_entry);
    sdl3::video::destroy_window(first);
    sdl3::video::destroy_window(second);
    gpu.lost.store(false, Ordering::Release); // injected, device remains healthy
    gpu.destroy_image(first_source);
    gpu.destroy_image(second_source);
}
