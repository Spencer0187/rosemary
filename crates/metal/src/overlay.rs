//! Overlays drawn straight from dma-bufs: a web view's page, rendered on the
//! GPU by its compositor, blended over the window's store without coming
//! back to the CPU.
//!
//! The store is blitted into an image the pass can write, and each overlay
//! is sampled and blended over it, source-over and premultiplied, as
//! `appkit::backing` blends pixel overlays on the CPU. It is a compute pass
//! because every device can write an RGBA8 storage image, and a blit can
//! neither blend nor scale with filtering into one channel order from
//! another.

use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::sync::{Arc, Mutex, Weak};

use appkit::backing::{Dmabuf, GpuOverlay, Importer};
use ash::vk;
use rspirv::binary::Assemble;
use rspirv::dr::{Builder, Operand};
use rspirv::spirv;

use crate::gpu::{self, Gpu, GpuImage, ImageDescription};

/// The shader's workgroup edge.
const GROUP: u32 = 8;

/// The most overlays one window blends in a frame; one descriptor set each.
const MOST: usize = 8;

/// DRM fourccs the pass can sample, the Vulkan format each is read as, and
/// whether its fourth byte is padding rather than alpha.
const FORMATS: [(u32, vk::Format, bool); 4] = [
    (fourcc(b"AR24"), vk::Format::B8G8R8A8_UNORM, false),
    (fourcc(b"XR24"), vk::Format::B8G8R8A8_UNORM, true),
    (fourcc(b"AB24"), vk::Format::R8G8B8A8_UNORM, false),
    (fourcc(b"XB24"), vk::Format::R8G8B8A8_UNORM, true),
];

const fn fourcc(code: &[u8; 4]) -> u32 {
    u32::from_le_bytes(*code)
}

/// The GPU windows are presented with, and the dma-bufs it can sample: see
/// `appkit::backing::set_importer`.
pub(crate) fn importer() -> Option<Importer> {
    // SAFETY: the Metal framework's exported constructor, which has no
    // preconditions.
    let device = unsafe { crate::MTLCreateSystemDefaultDevice() };
    let gpu = crate::gpu_of_device(device)?;
    let support = gpu.dmabuf.as_ref()?;
    let device = support.render_node?;
    let instance = crate::vulkan::instance()?;
    let mut formats = Vec::new();
    for (code, format, _) in FORMATS {
        for modifier in modifiers(instance, gpu.physical, format) {
            formats.push((code, modifier));
        }
    }
    (!formats.is_empty()).then_some(Importer { device, formats })
}

/// The modifiers `format` can be imported and sampled with.
fn modifiers(
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
    format: vk::Format,
) -> Vec<u64> {
    let mut list = vk::DrmFormatModifierPropertiesListEXT::default();
    let mut properties = vk::FormatProperties2::default().push_next(&mut list);
    // SAFETY: the device has the modifier extension, and both structures
    // live across the query.
    unsafe { instance.get_physical_device_format_properties2(physical, format, &mut properties) };
    let count = list.drm_format_modifier_count as usize;
    if count == 0 {
        return Vec::new();
    }
    let mut entries = vec![vk::DrmFormatModifierPropertiesEXT::default(); count];
    let mut list = vk::DrmFormatModifierPropertiesListEXT::default()
        .drm_format_modifier_properties(&mut entries);
    let mut properties = vk::FormatProperties2::default().push_next(&mut list);
    // SAFETY: as above, with room for every entry the first query counted.
    unsafe { instance.get_physical_device_format_properties2(physical, format, &mut properties) };
    let filled = list.drm_format_modifier_count as usize;
    entries.truncate(filled);
    entries
        .iter()
        .filter(|entry| {
            entry
                .drm_format_modifier_tiling_features
                .contains(vk::FormatFeatureFlags::SAMPLED_IMAGE)
        })
        .map(|entry| entry.drm_format_modifier)
        .filter(|&modifier| importable(instance, physical, format, modifier))
        .collect()
}

fn importable(
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
    format: vk::Format,
    modifier: u64,
) -> bool {
    let mut drm = vk::PhysicalDeviceImageDrmFormatModifierInfoEXT::default()
        .drm_format_modifier(modifier)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let mut external = vk::PhysicalDeviceExternalImageFormatInfo::default()
        .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
    let info = vk::PhysicalDeviceImageFormatInfo2::default()
        .format(format)
        .ty(vk::ImageType::TYPE_2D)
        .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
        .usage(vk::ImageUsageFlags::SAMPLED)
        .push_next(&mut drm)
        .push_next(&mut external);
    let mut answer = vk::ExternalImageFormatProperties::default();
    let mut properties = vk::ImageFormatProperties2::default().push_next(&mut answer);
    // SAFETY: every structure lives across the query.
    let supported = unsafe {
        instance.get_physical_device_image_format_properties2(physical, &info, &mut properties)
    };
    supported.is_ok()
        && answer
            .external_memory_properties
            .external_memory_features
            .contains(vk::ExternalMemoryFeatureFlags::IMPORTABLE)
}

/// A dma-buf made into an image, for as long as its picture lives.
struct Imported {
    gpu: &'static Gpu,
    picture: Weak<Dmabuf>,
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
}

impl Imported {
    /// The caller has waited for every use of it.
    fn destroy(self) {
        let device = &self.gpu.device;
        // SAFETY: objects of this device's that no pending work refers to.
        unsafe {
            device.destroy_image_view(self.view, None);
            device.destroy_image(self.image, None);
            device.free_memory(self.memory, None);
        }
    }
}

// SAFETY: Vulkan handles, used under `IMPORTED`.
unsafe impl Send for Imported {}

/// Keyed by the picture's address; an entry whose picture has gone is
/// destroyed at the next blend, once the one that used it last has finished.
static IMPORTED: Mutex<Vec<Imported>> = Mutex::new(Vec::new());

fn import(gpu: &'static Gpu, picture: &Arc<Dmabuf>) -> Result<(vk::Image, vk::ImageView), String> {
    let mut imported = IMPORTED.lock().expect("imported dma-bufs");
    if let Some(found) = imported
        .iter()
        .find(|entry| std::ptr::eq(entry.picture.as_ptr(), Arc::as_ptr(picture)))
    {
        return Ok((found.image, found.view));
    }
    let made = make(gpu, picture)?;
    let handles = (made.image, made.view);
    imported.push(made);
    Ok(handles)
}

fn make(gpu: &'static Gpu, picture: &Arc<Dmabuf>) -> Result<Imported, String> {
    let support = gpu
        .dmabuf
        .as_ref()
        .ok_or("the device cannot import dma-bufs")?;
    let &(_, format, opaque) = FORMATS
        .iter()
        .find(|(code, _, _)| *code == picture.fourcc)
        .ok_or_else(|| format!("fourcc {:#x} is not one the pass samples", picture.fourcc))?;
    let first = picture.planes.first().ok_or("a dma-buf with no planes")?;
    // One allocation holds every plane; planes in separate buffers would
    // need a disjoint image.
    if picture
        .planes
        .iter()
        .any(|plane| !same_file(plane.fd.as_raw_fd(), first.fd.as_raw_fd()))
    {
        return Err("a dma-buf whose planes are in separate buffers".into());
    }
    let layouts: Vec<vk::SubresourceLayout> = picture
        .planes
        .iter()
        .map(|plane| vk::SubresourceLayout {
            offset: u64::from(plane.offset),
            row_pitch: u64::from(plane.stride),
            ..Default::default()
        })
        .collect();
    let mut explicit = vk::ImageDrmFormatModifierExplicitCreateInfoEXT::default()
        .drm_format_modifier(picture.modifier)
        .plane_layouts(&layouts);
    let mut external = vk::ExternalMemoryImageCreateInfo::default()
        .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
    let info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(format)
        .extent(vk::Extent3D {
            width: picture.width.max(1),
            height: picture.height.max(1),
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
        .usage(vk::ImageUsageFlags::SAMPLED)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .push_next(&mut external)
        .push_next(&mut explicit);
    let device = &gpu.device;
    // SAFETY: a well-formed create info on this live device.
    let image = unsafe { device.create_image(&info, None) }
        .map_err(|error| format!("vkCreateImage for a dma-buf failed with {error:?}"))?;
    let image = gpu::Guard::new(image, |image| {
        // SAFETY: an image of this device's, referred to by nothing.
        unsafe { device.destroy_image(image, None) };
    });
    // The import takes a descriptor of its own, which it owns on success.
    let fd = duplicate(first.fd.as_raw_fd()).ok_or("the dma-buf could not be duplicated")?;
    let mut accepted = vk::MemoryFdPropertiesKHR::default();
    // SAFETY: a valid dma-buf descriptor and a structure for the answer.
    unsafe {
        support.memory_fd.get_memory_fd_properties(
            vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT,
            fd.as_raw_fd(),
            &mut accepted,
        )
    }
    .map_err(|error| format!("vkGetMemoryFdPropertiesKHR failed with {error:?}"))?;
    // SAFETY: the image was just created by this device.
    let requirements = unsafe { device.get_image_memory_requirements(*image) };
    let types = requirements.memory_type_bits & accepted.memory_type_bits;
    if types == 0 {
        return Err("no memory type both the image and the dma-buf allow".into());
    }
    let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().image(*image);
    let mut import = vk::ImportMemoryFdInfoKHR::default()
        .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT)
        .fd(fd.as_raw_fd());
    let allocate = vk::MemoryAllocateInfo::default()
        .allocation_size(requirements.size)
        .memory_type_index(types.trailing_zeros())
        .push_next(&mut import)
        .push_next(&mut dedicated);
    // SAFETY: a well-formed allocate info importing a live descriptor.
    let memory = unsafe { device.allocate_memory(&allocate, None) }
        .map_err(|error| format!("importing a dma-buf failed with {error:?}"))?;
    // The allocation owns the descriptor now.
    std::mem::forget(fd);
    let memory = gpu::Guard::new(memory, |memory| {
        // SAFETY: memory of this device's, bound to nothing which survives.
        unsafe { device.free_memory(memory, None) };
    });
    // SAFETY: fresh, unbound image and memory from this device.
    unsafe { device.bind_image_memory(*image, *memory, 0) }
        .map_err(|error| format!("vkBindImageMemory for a dma-buf failed with {error:?}"))?;
    let alpha = match opaque {
        true => vk::ComponentSwizzle::ONE,
        false => vk::ComponentSwizzle::IDENTITY,
    };
    let view_info = vk::ImageViewCreateInfo::default()
        .image(*image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .components(vk::ComponentMapping {
            a: alpha,
            ..Default::default()
        })
        .subresource_range(color_range());
    // SAFETY: a view of an image of this device's, bound to memory.
    let view = unsafe { device.create_image_view(&view_info, None) }
        .map_err(|error| format!("vkCreateImageView for a dma-buf failed with {error:?}"))?;
    Ok(Imported {
        gpu,
        picture: Arc::downgrade(picture),
        image: image.take(),
        memory: memory.take(),
        view,
    })
}

fn duplicate(fd: i32) -> Option<OwnedFd> {
    // SAFETY: duplicating a descriptor the caller holds open.
    let copy = unsafe { libc::fcntl(fd, libc::F_DUPFD_CLOEXEC, 0) };
    // SAFETY: a new descriptor this owns.
    (copy >= 0).then(|| unsafe { OwnedFd::from_raw_fd(copy) })
}

/// Whether two descriptors name the same buffer.
fn same_file(a: i32, b: i32) -> bool {
    let identity = |fd| {
        let mut stat = std::mem::MaybeUninit::<libc::stat>::uninit();
        // SAFETY: a descriptor the caller holds open, and room for the answer.
        (unsafe { libc::fstat(fd, stat.as_mut_ptr()) } == 0).then(|| {
            // SAFETY: filled in by the successful call.
            let stat = unsafe { stat.assume_init() };
            (stat.st_dev, stat.st_ino)
        })
    };
    a == b || identity(a).is_some_and(|found| identity(b) == Some(found))
}

fn color_range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    }
}

/// Where one overlay goes, as the shader's push constants.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Placement {
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
    pub target_width: u32,
    pub target_height: u32,
    /// The dispatched region of the placement, in window pixels. The source
    /// UV stays relative to the *full* placement even when clipped.
    pub clip_x: i32,
    pub clip_y: i32,
    pub clip_width: u32,
    pub clip_height: u32,
    /// Core Animation opacity multiplies the entire premultiplied source.
    pub opacity: f32,
    /// Only the scene compositor uses this to clear a damaged subrectangle.
    /// The DMA-BUF overlay path always uses source-over (false).
    pub clear: bool,
    /// Index and length of consecutive rounded CA ancestor masks in the
    /// scene's per-frame GPU clip buffer (zero for ordinary overlays).
    pub round_start: u32,
    pub round_count: u32,
}

impl Placement {
    pub(crate) fn bytes(&self) -> [u8; 56] {
        let words = [
            self.x as u32,
            self.y as u32,
            self.width,
            self.height,
            self.target_width,
            self.target_height,
            self.clip_x as u32,
            self.clip_y as u32,
            self.clip_width,
            self.clip_height,
            self.opacity.to_bits(),
            u32::from(self.clear),
            self.round_start,
            self.round_count,
        ];
        let mut bytes = [0; 56];
        for (at, word) in words.into_iter().enumerate() {
            bytes[at * 4..at * 4 + 4].copy_from_slice(&word.to_ne_bytes());
        }
        bytes
    }
}

struct Pass {
    gpu: &'static Gpu,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    sampler: vk::Sampler,
    sets: Vec<vk::DescriptorSet>,
    _pool: vk::DescriptorPool,
}

// SAFETY: Vulkan handles, used under `PASSES`.
unsafe impl Send for Pass {}

/// One pass per device, made on first use and kept for the run.
static PASSES: Mutex<Vec<Pass>> = Mutex::new(Vec::new());

impl Pass {
    fn new(gpu: &'static Gpu) -> Result<Self, String> {
        let device = &gpu.device;
        let set_layout = gpu.descriptor_set_layout(
            &[
                (0, vk::DescriptorType::COMBINED_IMAGE_SAMPLER),
                (1, vk::DescriptorType::STORAGE_IMAGE),
            ],
            vk::ShaderStageFlags::COMPUTE,
            false,
        )?;
        let push = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .size(std::mem::size_of::<Placement>() as u32)];
        let set_layouts = [set_layout];
        let info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts)
            .push_constant_ranges(&push);
        // SAFETY: a well-formed create info on this live device.
        let pipeline_layout = unsafe { device.create_pipeline_layout(&info, None) }
            .map_err(|error| format!("vkCreatePipelineLayout failed with {error:?}"))?;
        let pipeline_layout = gpu::Guard::new(pipeline_layout, |layout| {
            // SAFETY: a layout of this device's that nothing uses yet.
            unsafe { device.destroy_pipeline_layout(layout, None) };
        });
        let words = shader();
        let module = gpu.create_shader_module(&words, "rosemary_overlay_blend")?;
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(c"main");
        let info = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(*pipeline_layout);
        // SAFETY: every borrowed structure lives across the call.
        let pipeline =
            unsafe { device.create_compute_pipelines(vk::PipelineCache::null(), &[info], None) };
        // SAFETY: a pipeline keeps nothing of the module it was made from.
        unsafe { device.destroy_shader_module(module, None) };
        let pipeline = pipeline
            .map_err(|(_, error)| format!("vkCreateComputePipelines failed with {error:?}"))?[0];
        let pipeline = gpu::Guard::new(pipeline, |pipeline| {
            // SAFETY: a pipeline of this device's, never bound.
            unsafe { device.destroy_pipeline(pipeline, None) };
        });
        let info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE);
        // SAFETY: a well-formed create info on this live device.
        let sampler = unsafe { device.create_sampler(&info, None) }
            .map_err(|error| format!("vkCreateSampler failed with {error:?}"))?;
        let sampler = gpu::Guard::new(sampler, |sampler| {
            // SAFETY: a sampler of this device's, never used.
            unsafe { device.destroy_sampler(sampler, None) };
        });
        let sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(MOST as u32),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(MOST as u32),
        ];
        let info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(MOST as u32)
            .pool_sizes(&sizes);
        // SAFETY: a well-formed create info on this live device.
        let pool = unsafe { device.create_descriptor_pool(&info, None) }
            .map_err(|error| format!("vkCreateDescriptorPool failed with {error:?}"))?;
        let pool = gpu::Guard::new(pool, |pool| {
            // SAFETY: a pool of this device's; its sets go with it.
            unsafe { device.destroy_descriptor_pool(pool, None) };
        });
        let layouts = [set_layout; MOST];
        let allocate = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(*pool)
            .set_layouts(&layouts);
        // SAFETY: the pool was sized for exactly these sets.
        let sets = unsafe { device.allocate_descriptor_sets(&allocate) }
            .map_err(|error| format!("vkAllocateDescriptorSets failed with {error:?}"))?;
        Ok(Self {
            gpu,
            pipeline_layout: pipeline_layout.take(),
            pipeline: pipeline.take(),
            sampler: sampler.take(),
            sets,
            _pool: pool.take(),
        })
    }
}

/// An image [`blend`] writes: RGBA8, which every device can store to.
pub(crate) fn target_description(width: u32, height: u32) -> ImageDescription {
    ImageDescription {
        image_type: vk::ImageType::TYPE_2D,
        view_type: vk::ImageViewType::TYPE_2D,
        format: vk::Format::R8G8B8A8_UNORM,
        width,
        height,
        depth: 1,
        levels: 1,
        layers: 1,
        samples: 1,
        usage: gpu::USAGE_SHADER_WRITE,
        cube: false,
        components: vk::ComponentMapping::default(),
    }
}

/// Copy `store` into `target`, of the same size, and blend `overlays` over
/// it, waiting until that is done. Overlays that could not be imported are
/// left out and returned.
pub(crate) fn blend<'a>(
    gpu: &'static Gpu,
    store: &GpuImage,
    target: &GpuImage,
    overlays: &'a [GpuOverlay],
) -> Result<Vec<&'a GpuOverlay>, String> {
    let mut passes = PASSES.lock().expect("overlay passes");
    if !passes.iter().any(|pass| std::ptr::eq(pass.gpu, gpu)) {
        passes.push(Pass::new(gpu)?);
    }
    let pass = passes
        .iter()
        .find(|pass| std::ptr::eq(pass.gpu, gpu))
        .expect("made above");
    prune(gpu);

    let mut refused = Vec::new();
    let mut placed = Vec::new();
    for overlay in overlays.iter().take(MOST) {
        match import(gpu, &overlay.picture) {
            Ok((image, view)) => placed.push((overlay, image, view)),
            Err(reason) => {
                gpu::trace_once(format!("a dma-buf overlay was not imported: {reason}"));
                refused.push(overlay);
            }
        }
    }
    let device = &gpu.device;
    for (at, (_, _, view)) in placed.iter().enumerate() {
        let source = [vk::DescriptorImageInfo::default()
            .sampler(pass.sampler)
            .image_view(*view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let written = [vk::DescriptorImageInfo::default()
            .image_view(target.view)
            .image_layout(vk::ImageLayout::GENERAL)];
        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(pass.sets[at])
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&source),
            vk::WriteDescriptorSet::default()
                .dst_set(pass.sets[at])
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(&written),
        ];
        // SAFETY: the last blend waited for its submission, so no set is in
        // use.
        unsafe { device.update_descriptor_sets(&writes, &[]) };
    }
    let external = match gpu.dmabuf.as_ref().is_some_and(|support| support.foreign) {
        true => vk::QUEUE_FAMILY_FOREIGN_EXT,
        false => vk::QUEUE_FAMILY_EXTERNAL,
    };
    let (width, height) = (store.width, store.height);
    gpu.one_shot(|device, command_buffer| {
        // SAFETY: recording into the call's command buffer; the store and
        // target are in GENERAL from creation, and each imported image is
        // taken from its producer and handed back around its read.
        unsafe {
            // The store's upload lands, and the last present's read of the
            // target is done, before the copy.
            general_barrier(
                device,
                command_buffer,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::TRANSFER,
                &[
                    (
                        store.image,
                        vk::AccessFlags::MEMORY_WRITE,
                        vk::AccessFlags::TRANSFER_READ,
                    ),
                    (
                        target.image,
                        vk::AccessFlags::empty(),
                        vk::AccessFlags::TRANSFER_WRITE,
                    ),
                ],
            );
            let whole = vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            };
            let corner = vk::Offset3D {
                x: width as i32,
                y: height as i32,
                z: 1,
            };
            let region = vk::ImageBlit::default()
                .src_subresource(whole)
                .src_offsets([vk::Offset3D::default(), corner])
                .dst_subresource(whole)
                .dst_offsets([vk::Offset3D::default(), corner]);
            device.cmd_blit_image(
                command_buffer,
                store.image,
                vk::ImageLayout::GENERAL,
                target.image,
                vk::ImageLayout::GENERAL,
                &[region],
                vk::Filter::NEAREST,
            );
            general_barrier(
                device,
                command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                &[(
                    target.image,
                    vk::AccessFlags::TRANSFER_WRITE,
                    vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
                )],
            );
            let acquire: Vec<_> = placed
                .iter()
                .map(|(_, image, _)| ownership(*image, external, gpu.queue_family, true))
                .collect();
            device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &acquire,
            );
            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                pass.pipeline,
            );
            for (at, (overlay, _, _)) in placed.iter().enumerate() {
                let rect = overlay.rect;
                let placement = Placement {
                    x: rect.x.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32,
                    y: rect.y.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32,
                    width: rect.width.clamp(0, i64::from(u32::MAX)) as u32,
                    height: rect.height.clamp(0, i64::from(u32::MAX)) as u32,
                    target_width: width,
                    target_height: height,
                    clip_x: rect.x.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32,
                    clip_y: rect.y.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32,
                    clip_width: rect.width.clamp(0, i64::from(u32::MAX)) as u32,
                    clip_height: rect.height.clamp(0, i64::from(u32::MAX)) as u32,
                    opacity: 1.0,
                    clear: false,
                    round_start: 0,
                    round_count: 0,
                };
                if placement.width == 0 || placement.height == 0 {
                    continue;
                }
                device.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    pass.pipeline_layout,
                    0,
                    &[pass.sets[at]],
                    &[],
                );
                device.cmd_push_constants(
                    command_buffer,
                    pass.pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    &placement.bytes(),
                );
                device.cmd_dispatch(
                    command_buffer,
                    placement.width.div_ceil(GROUP),
                    placement.height.div_ceil(GROUP),
                    1,
                );
                // Overlapping overlays blend in order.
                general_barrier(
                    device,
                    command_buffer,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    &[(
                        target.image,
                        vk::AccessFlags::SHADER_WRITE,
                        vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
                    )],
                );
            }
            let release: Vec<_> = placed
                .iter()
                .map(|(_, image, _)| ownership(*image, gpu.queue_family, external, false))
                .collect();
            device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &release,
            );
            // The swapchain blits the target next.
            general_barrier(
                device,
                command_buffer,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                &[(
                    target.image,
                    vk::AccessFlags::SHADER_WRITE,
                    vk::AccessFlags::TRANSFER_READ,
                )],
            );
        }
    })?;
    Ok(refused)
}

/// Destroy the imports whose pictures are gone. The last blend that used
/// them has finished: every blend waits for its own.
fn prune(gpu: &'static Gpu) {
    let mut imported = IMPORTED.lock().expect("imported dma-bufs");
    let (gone, kept): (Vec<_>, Vec<_>) = std::mem::take(&mut *imported)
        .into_iter()
        .partition(|entry| std::ptr::eq(entry.gpu, gpu) && entry.picture.strong_count() == 0);
    *imported = kept;
    drop(imported);
    for entry in gone {
        entry.destroy();
    }
}

/// A barrier taking an imported image from its producer for sampling, or
/// giving it back. The producer keeps it in `GENERAL`.
fn ownership(
    image: vk::Image,
    from: u32,
    to: u32,
    acquire: bool,
) -> vk::ImageMemoryBarrier<'static> {
    let (old, new, source, target) = match acquire {
        true => (
            vk::ImageLayout::GENERAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::AccessFlags::empty(),
            vk::AccessFlags::SHADER_READ,
        ),
        false => (
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::ImageLayout::GENERAL,
            vk::AccessFlags::SHADER_READ,
            vk::AccessFlags::empty(),
        ),
    };
    vk::ImageMemoryBarrier::default()
        .old_layout(old)
        .new_layout(new)
        .src_access_mask(source)
        .dst_access_mask(target)
        .src_queue_family_index(from)
        .dst_queue_family_index(to)
        .image(image)
        .subresource_range(color_range())
}

/// # Safety
///
/// Recording into `command_buffer`; every image belongs to `device` and is
/// in `GENERAL`.
unsafe fn general_barrier(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    from: vk::PipelineStageFlags,
    to: vk::PipelineStageFlags,
    images: &[(vk::Image, vk::AccessFlags, vk::AccessFlags)],
) {
    let barriers: Vec<_> = images
        .iter()
        .map(|&(image, source, target)| {
            vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::GENERAL)
                .new_layout(vk::ImageLayout::GENERAL)
                .src_access_mask(source)
                .dst_access_mask(target)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(color_range())
        })
        .collect();
    // SAFETY: as the caller promises.
    unsafe {
        device.cmd_pipeline_barrier(
            command_buffer,
            from,
            to,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &barriers,
        )
    };
}

/// The pass, as SPIR-V. In GLSL:
///
/// ```glsl
/// layout(local_size_x = 8, local_size_y = 8) in;
/// layout(binding = 0) uniform sampler2D overlay;
/// layout(binding = 1, rgba8) uniform image2D target;
/// layout(push_constant) uniform Placement {
///     int x, y; uint width, height, target_width, target_height;
/// };
/// void main() {
///     uvec2 id = gl_GlobalInvocationID.xy;
///     ivec2 at = ivec2(x, y) + ivec2(id);
///     if (id.x < width && id.y < height && at.x >= 0 && at.y >= 0
///             && at.x < int(target_width) && at.y < int(target_height)) {
///         vec4 over = textureLod(overlay, (vec2(id) + 0.5) / vec2(width, height), 0.0);
///         vec4 under = imageLoad(target, at);
///         imageStore(target, at, over + under * (1.0 - over.a));
///     }
/// }
/// ```
pub(crate) fn shader() -> Vec<u32> {
    build_shader(false).expect("the overlay shader is well formed")
}

/// The scene shader additionally reads variable-length rounded masks. The
/// ordinary DMA-BUF overlay shader keeps the original two bindings.
pub(crate) fn scene_shader() -> Vec<u32> {
    build_shader(true).expect("the rounded scene shader is well formed")
}

fn build_shader(scene: bool) -> Result<Vec<u32>, rspirv::dr::Error> {
    use spirv::{Decoration, StorageClass};
    let mut b = Builder::new();
    b.set_version(1, 0);
    b.capability(spirv::Capability::Shader);
    let glsl = b.ext_inst_import("GLSL.std.450");
    b.memory_model(spirv::AddressingModel::Logical, spirv::MemoryModel::GLSL450);

    let void = b.type_void();
    let boolean = b.type_bool();
    let uint = b.type_int(32, 0);
    let int = b.type_int(32, 1);
    let float = b.type_float(32, None);
    let uvec3 = b.type_vector(uint, 3);
    let ivec2 = b.type_vector(int, 2);
    let vec2 = b.type_vector(float, 2);
    let vec4 = b.type_vector(float, 4);
    let texture = b.type_image(
        float,
        spirv::Dim::Dim2D,
        0,
        0,
        0,
        1,
        spirv::ImageFormat::Unknown,
        None,
    );
    let sampled = b.type_sampled_image(texture);
    let storage = b.type_image(
        float,
        spirv::Dim::Dim2D,
        0,
        0,
        0,
        2,
        spirv::ImageFormat::Rgba8,
        None,
    );
    let sampled_pointer = b.type_pointer(None, StorageClass::UniformConstant, sampled);
    let storage_pointer = b.type_pointer(None, StorageClass::UniformConstant, storage);
    let overlay = b.variable(sampled_pointer, None, StorageClass::UniformConstant, None);
    let target = b.variable(storage_pointer, None, StorageClass::UniformConstant, None);
    let rounded = if scene {
        // A runtime array of pairs of vec4: (left,top,right,bottom) and
        // (radius,0,0,0). The GPU loop handles any number of ancestor masks.
        let array = b.type_runtime_array(vec4);
        b.decorate(array, Decoration::ArrayStride, [Operand::LiteralBit32(16)]);
        let block = b.type_struct([array]);
        b.decorate(block, Decoration::Block, []);
        b.member_decorate(block, 0, Decoration::Offset, [Operand::LiteralBit32(0)]);
        let ptr = b.type_pointer(None, StorageClass::StorageBuffer, block);
        let variable = b.variable(ptr, None, StorageClass::StorageBuffer, None);
        b.decorate(
            variable,
            Decoration::DescriptorSet,
            [Operand::LiteralBit32(0)],
        );
        b.decorate(variable, Decoration::Binding, [Operand::LiteralBit32(2)]);
        Some(variable)
    } else {
        None
    };
    let rounded_vec_pointer = if scene {
        Some(b.type_pointer(None, StorageClass::StorageBuffer, vec4))
    } else {
        None
    };
    for (variable, binding) in [(overlay, 0), (target, 1)] {
        b.decorate(
            variable,
            Decoration::DescriptorSet,
            [Operand::LiteralBit32(0)],
        );
        b.decorate(
            variable,
            Decoration::Binding,
            [Operand::LiteralBit32(binding)],
        );
    }

    let placement = b.type_struct([
        int, int, uint, uint, uint, uint, int, int, uint, uint, float, uint, uint, uint,
    ]);
    b.decorate(placement, Decoration::Block, []);
    for member in 0..14 {
        b.member_decorate(
            placement,
            member,
            Decoration::Offset,
            [Operand::LiteralBit32(member * 4)],
        );
    }
    let placement_pointer = b.type_pointer(None, StorageClass::PushConstant, placement);
    let placement = b.variable(placement_pointer, None, StorageClass::PushConstant, None);
    let int_pointer = b.type_pointer(None, StorageClass::PushConstant, int);
    let uint_pointer = b.type_pointer(None, StorageClass::PushConstant, uint);

    let id_pointer = b.type_pointer(None, StorageClass::Input, uvec3);
    let id = b.variable(id_pointer, None, StorageClass::Input, None);
    b.decorate(
        id,
        Decoration::BuiltIn,
        [Operand::BuiltIn(spirv::BuiltIn::GlobalInvocationId)],
    );

    let zero_int = b.constant_bit32(int, 0);
    let zero = b.constant_bit32(float, 0f32.to_bits());
    let half = b.constant_bit32(float, 0.5f32.to_bits());
    let one = b.constant_bit32(float, 1f32.to_bits());

    let function = b.type_function(void, []);
    let main = b.begin_function(void, None, spirv::FunctionControl::NONE, function)?;
    b.begin_block(None)?;
    let (round_index, coverage) = if scene {
        let uint_var = b.type_pointer(None, StorageClass::Function, uint);
        let float_var = b.type_pointer(None, StorageClass::Function, float);
        (
            Some(b.variable(uint_var, None, StorageClass::Function, None)),
            Some(b.variable(float_var, None, StorageClass::Function, None)),
        )
    } else {
        (None, None)
    };
    let member = |b: &mut Builder, at: u32, pointer: u32, ty: u32| {
        let at = b.constant_bit32(int, at);
        let address = b.access_chain(pointer, None, placement, [at])?;
        b.load(ty, None, address, None, [])
    };
    let x = member(&mut b, 0, int_pointer, int)?;
    let y = member(&mut b, 1, int_pointer, int)?;
    let width = member(&mut b, 2, uint_pointer, uint)?;
    let height = member(&mut b, 3, uint_pointer, uint)?;
    let target_width = member(&mut b, 4, uint_pointer, uint)?;
    let target_height = member(&mut b, 5, uint_pointer, uint)?;
    let clip_x = member(&mut b, 6, int_pointer, int)?;
    let clip_y = member(&mut b, 7, int_pointer, int)?;
    let clip_width = member(&mut b, 8, uint_pointer, uint)?;
    let clip_height = member(&mut b, 9, uint_pointer, uint)?;
    let float_pointer = b.type_pointer(None, StorageClass::PushConstant, float);
    let opacity = member(&mut b, 10, float_pointer, float)?;
    let clear = member(&mut b, 11, uint_pointer, uint)?;
    let round_start = member(&mut b, 12, uint_pointer, uint)?;
    let round_count = member(&mut b, 13, uint_pointer, uint)?;
    let target_width = b.bitcast(int, None, target_width)?;
    let target_height = b.bitcast(int, None, target_height)?;
    let global = b.load(uvec3, None, id, None, [])?;
    let column = b.composite_extract(uint, None, global, [0])?;
    let row = b.composite_extract(uint, None, global, [1])?;
    let at_x = b.bitcast(int, None, column)?;
    let at_x = b.i_add(int, None, clip_x, at_x)?;
    let at_y = b.bitcast(int, None, row)?;
    let at_y = b.i_add(int, None, clip_y, at_y)?;
    let mut inside = b.u_less_than(boolean, None, column, clip_width)?;
    for condition in [
        b.u_less_than(boolean, None, row, clip_height)?,
        b.s_greater_than_equal(boolean, None, at_x, zero_int)?,
        b.s_greater_than_equal(boolean, None, at_y, zero_int)?,
        b.s_less_than(boolean, None, at_x, target_width)?,
        b.s_less_than(boolean, None, at_y, target_height)?,
    ] {
        inside = b.logical_and(boolean, None, inside, condition)?;
    }
    let body = b.id();
    let merge = b.id();
    b.selection_merge(merge, spirv::SelectionControl::NONE)?;
    b.branch_conditional(inside, body, merge, [])?;

    b.begin_block(Some(body))?;
    // The damaged-area clear uses this same storage image and deliberately
    // skips texture sampling. It does not need a separate Vulkan render pass.
    let zero_uint = b.constant_bit32(uint, 0);
    let is_clear = b.i_not_equal(boolean, None, clear, zero_uint)?;
    let clearing = b.id();
    let blending = b.id();
    let complete = b.id();
    b.selection_merge(complete, spirv::SelectionControl::NONE)?;
    b.branch_conditional(is_clear, clearing, blending, [])?;
    b.begin_block(Some(clearing))?;
    let at = b.composite_construct(ivec2, None, [at_x, at_y])?;
    let image = b.load(storage, None, target, None, [])?;
    let transparent = b.constant_composite(vec4, [zero, zero, zero, zero]);
    b.image_write(image, at, transparent, None, [])?;
    b.branch(complete)?;
    b.begin_block(Some(blending))?;
    let source_x = b.i_sub(int, None, at_x, x)?;
    let source_y = b.i_sub(int, None, at_y, y)?;
    let u = b.convert_s_to_f(float, None, source_x)?;
    let v = b.convert_s_to_f(float, None, source_y)?;
    let u = b.f_add(float, None, u, half)?;
    let v = b.f_add(float, None, v, half)?;
    let across = b.convert_u_to_f(float, None, width)?;
    let down = b.convert_u_to_f(float, None, height)?;
    let u = b.f_div(float, None, u, across)?;
    let v = b.f_div(float, None, v, down)?;
    let coordinate = b.composite_construct(vec2, None, [u, v])?;
    let loaded = b.load(sampled, None, overlay, None, [])?;
    let over = b.image_sample_explicit_lod(
        vec4,
        None,
        loaded,
        coordinate,
        spirv::ImageOperands::LOD,
        [Operand::IdRef(zero)],
    )?;
    let over = b.vector_times_scalar(vec4, None, over, opacity)?;
    let over = if let (Some(buffer), Some(ptr), Some(index_var), Some(coverage_var)) =
        (rounded, rounded_vec_pointer, round_index, coverage)
    {
        let zero_uint = b.constant_bit32(uint, 0);
        let one_uint = b.constant_bit32(uint, 1);
        let two_uint = b.constant_bit32(uint, 2);
        let zero_int = b.constant_bit32(int, 0);
        b.store(index_var, zero_uint, None, [])?;
        b.store(coverage_var, one, None, [])?;
        let header = b.id();
        let body = b.id();
        let repeat = b.id();
        let end = b.id();
        b.branch(header)?;
        b.begin_block(Some(header))?;
        let index = b.load(uint, None, index_var, None, [])?;
        let inside = b.u_less_than(boolean, None, index, round_count)?;
        b.loop_merge(end, repeat, spirv::LoopControl::NONE, [])?;
        b.branch_conditional(inside, body, end, [])?;
        b.begin_block(Some(body))?;
        let offset = b.i_add(uint, None, round_start, index)?;
        let offset = b.i_mul(uint, None, offset, two_uint)?;
        let next = b.i_add(uint, None, offset, one_uint)?;
        let bounds_ptr = b.access_chain(ptr, None, buffer, [zero_int, offset])?;
        let params_ptr = b.access_chain(ptr, None, buffer, [zero_int, next])?;
        let bounds = b.load(vec4, None, bounds_ptr, None, [])?;
        let params = b.load(vec4, None, params_ptr, None, [])?;
        let left = b.composite_extract(float, None, bounds, [0])?;
        let top = b.composite_extract(float, None, bounds, [1])?;
        let right = b.composite_extract(float, None, bounds, [2])?;
        let bottom = b.composite_extract(float, None, bounds, [3])?;
        let radius = b.composite_extract(float, None, params, [0])?;
        let cx = b.convert_s_to_f(float, None, at_x)?;
        let cy = b.convert_s_to_f(float, None, at_y)?;
        let cx = b.f_add(float, None, cx, half)?;
        let cy = b.f_add(float, None, cy, half)?;
        let left_radius = b.f_add(float, None, left, radius)?;
        let right_radius = b.f_sub(float, None, right, radius)?;
        let top_radius = b.f_add(float, None, top, radius)?;
        let bottom_radius = b.f_sub(float, None, bottom, radius)?;
        let dx1 = b.f_sub(float, None, left_radius, cx)?;
        let dx2 = b.f_sub(float, None, cx, right_radius)?;
        let dy1 = b.f_sub(float, None, top_radius, cy)?;
        let dy2 = b.f_sub(float, None, cy, bottom_radius)?;
        let dx = b.ext_inst(
            float,
            None,
            glsl,
            40,
            [Operand::IdRef(dx1), Operand::IdRef(dx2)],
        )?;
        let dy = b.ext_inst(
            float,
            None,
            glsl,
            40,
            [Operand::IdRef(dy1), Operand::IdRef(dy2)],
        )?;
        let dx = b.ext_inst(
            float,
            None,
            glsl,
            40,
            [Operand::IdRef(dx), Operand::IdRef(zero)],
        )?;
        let dy = b.ext_inst(
            float,
            None,
            glsl,
            40,
            [Operand::IdRef(dy), Operand::IdRef(zero)],
        )?;
        let delta = b.composite_construct(vec2, None, [dx, dy])?;
        let distance = b.ext_inst(float, None, glsl, 66, [Operand::IdRef(delta)])?;
        let cov = b.f_sub(float, None, radius, distance)?;
        let cov = b.f_add(float, None, cov, half)?;
        let cov = b.ext_inst(
            float,
            None,
            glsl,
            43,
            [
                Operand::IdRef(cov),
                Operand::IdRef(zero),
                Operand::IdRef(one),
            ],
        )?;
        let aggregate = b.load(float, None, coverage_var, None, [])?;
        let aggregate = b.f_mul(float, None, aggregate, cov)?;
        b.store(coverage_var, aggregate, None, [])?;
        b.branch(repeat)?;
        b.begin_block(Some(repeat))?;
        let next = b.i_add(uint, None, index, one_uint)?;
        b.store(index_var, next, None, [])?;
        b.branch(header)?;
        b.begin_block(Some(end))?;
        let aggregate = b.load(float, None, coverage_var, None, [])?;
        b.vector_times_scalar(vec4, None, over, aggregate)?
    } else {
        over
    };
    let at = b.composite_construct(ivec2, None, [at_x, at_y])?;
    let image = b.load(storage, None, target, None, [])?;
    let under = b.image_read(vec4, None, image, at, None, [])?;
    let alpha = b.composite_extract(float, None, over, [3])?;
    let keep = b.f_sub(float, None, one, alpha)?;
    let under = b.vector_times_scalar(vec4, None, under, keep)?;
    let blended = b.f_add(vec4, None, over, under)?;
    b.image_write(image, at, blended, None, [])?;
    b.branch(complete)?;
    b.begin_block(Some(complete))?;
    b.branch(merge)?;

    b.begin_block(Some(merge))?;
    b.ret()?;
    b.end_function()?;
    b.entry_point(spirv::ExecutionModel::GLCompute, main, "main", [id]);
    b.execution_mode(main, spirv::ExecutionMode::LocalSize, [GROUP, GROUP, 1]);
    Ok(b.module().assemble())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_overlay_shader_is_one_compute_entry_point() {
        let words = shader();
        let module = rspirv::dr::load_words(&words).expect("SPIR-V that loads back");
        assert_eq!(module.entry_points.len(), 1);
        assert_eq!(words[0], spirv::MAGIC_NUMBER);
    }

    #[test]
    fn a_placement_is_laid_out_as_the_shader_reads_it() {
        let placement = Placement {
            x: -3,
            y: 4,
            width: 5,
            height: 6,
            target_width: 7,
            target_height: 8,
            clip_x: -2,
            clip_y: 5,
            clip_width: 3,
            clip_height: 4,
            opacity: 0.5,
            clear: false,
            round_start: 0,
            round_count: 0,
        };
        let bytes = placement.bytes();
        assert_eq!(&bytes[0..4], &(-3i32).to_ne_bytes());
        assert_eq!(&bytes[20..24], &8u32.to_ne_bytes());
        assert_eq!(&bytes[24..28], &(-2i32).to_ne_bytes());
        assert_eq!(&bytes[40..44], &0.5f32.to_ne_bytes());
    }

    #[test]
    fn fourccs_are_read_little_endian() {
        // DRM_FORMAT_ARGB8888.
        assert_eq!(fourcc(b"AR24"), 0x3432_5241);
    }
}
