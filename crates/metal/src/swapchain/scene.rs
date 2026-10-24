//! GPU-owned snapshots of independently submitted Metal layers.
//! A drawable may be reused as soon as its GPU copy finishes, but the last
//! image it drew must survive when another layer continues submitting frames.

use super::*;

#[derive(Clone)]
pub(super) struct SceneLayer {
    pub(super) id: usize,
    frame: DirectFrame,
    pub(super) snapshot: Arc<BackingImageLease>,
    spare: Vec<Arc<BackingImageLease>>,
    opacity: f32,
    clip: Option<DirectFrame>,
    regions: Option<Vec<Region>>,
    pub(super) ready: bool,
}

pub(super) struct SceneWork {
    current: usize,
    previous: Option<usize>,
    /// Pre-submission membership/source state, retained only until the VkQueue
    /// accepts the new generation. Recording errors must not poison the scene.
    prior: Option<SceneLayer>,
}

pub(super) enum ScenePrepared {
    SingleLayer,
    Deferred,
    Ready(SceneWork),
}

// The allowance includes currently reusable and busy snapshots; an initial
// snapshot for a newly visible layer is mandatory for correct composition.
const SCENE_IMAGE_BUDGET: u64 = 256 * 1024 * 1024;
pub(super) const MAX_SCENE_IMAGES_PER_LAYER: usize = FRAMES_IN_FLIGHT;

fn scene_image_bytes(source: PresentSource) -> u64 {
    let bytes = match source.format {
        vk::Format::R8G8B8A8_UNORM
        | vk::Format::B8G8R8A8_UNORM
        | vk::Format::R8G8B8A8_SRGB
        | vk::Format::B8G8R8A8_SRGB => 4,
        vk::Format::R16G16B16A16_SFLOAT => 8,
        _ => 16, // Do not undercount an unfamiliar pixel format.
    };
    u64::from(source.width)
        .saturating_mul(u64::from(source.height))
        .saturating_mul(bytes)
}

fn source_matches(image: &GpuImage, source: PresentSource) -> bool {
    image.width == source.width && image.height == source.height && image.format == source.format
}

// Descriptor sets are separate for every GPU in-flight frame. Reusing one
// while an older command buffer samples it is prohibited by Vulkan.
const MAX_SCENE_LAYERS: usize = 32;

/// Per-window GPU source-over blending, using the same tested shader as
/// AppKit's DMA-BUF overlays. Descriptor updates are frame-slot isolated.
pub(super) struct SceneBlender {
    gpu: &'static Gpu,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    sampler: vk::Sampler,
    pool: vk::DescriptorPool,
    sets: Vec<vk::DescriptorSet>,
}

impl SceneBlender {
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
        let layouts = [set_layout];
        let push = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .size(std::mem::size_of::<crate::overlay::Placement>() as u32)];
        let pipeline_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&layouts)
            .push_constant_ranges(&push);
        let layout = unsafe { device.create_pipeline_layout(&pipeline_info, None) }
            .map_err(|e| format!("scene vkCreatePipelineLayout failed: {e:?}"))?;
        let layout = gpu::Guard::new(layout, |handle| unsafe {
            device.destroy_pipeline_layout(handle, None)
        });
        let module = gpu.create_shader_module(&crate::overlay::shader(), "rosemary_scene_blend")?;
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(c"main");
        let info = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(*layout);
        let pipeline =
            unsafe { device.create_compute_pipelines(vk::PipelineCache::null(), &[info], None) };
        unsafe { device.destroy_shader_module(module, None) };
        let pipeline =
            pipeline.map_err(|(_, e)| format!("scene vkCreateComputePipelines: {e:?}"))?[0];
        let pipeline = gpu::Guard::new(pipeline, |handle| unsafe {
            device.destroy_pipeline(handle, None)
        });
        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE);
        let sampler = unsafe { device.create_sampler(&sampler_info, None) }
            .map_err(|e| format!("scene vkCreateSampler: {e:?}"))?;
        let sampler = gpu::Guard::new(sampler, |handle| unsafe {
            device.destroy_sampler(handle, None)
        });
        let count = (FRAMES_IN_FLIGHT * MAX_SCENE_LAYERS) as u32;
        let sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(count),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(count),
        ];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(count)
            .pool_sizes(&sizes);
        let pool = unsafe { device.create_descriptor_pool(&pool_info, None) }
            .map_err(|e| format!("scene vkCreateDescriptorPool: {e:?}"))?;
        let pool = gpu::Guard::new(pool, |handle| unsafe {
            device.destroy_descriptor_pool(handle, None)
        });
        let layouts = vec![set_layout; count as usize];
        let allocate = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(*pool)
            .set_layouts(&layouts);
        let sets = unsafe { device.allocate_descriptor_sets(&allocate) }
            .map_err(|e| format!("scene vkAllocateDescriptorSets: {e:?}"))?;
        Ok(Self {
            gpu,
            layout: layout.take(),
            pipeline: pipeline.take(),
            sampler: sampler.take(),
            pool: pool.take(),
            sets,
        })
    }

    fn bind(
        &self,
        cb: vk::CommandBuffer,
        frame_slot: usize,
        scene_index: usize,
        source: &GpuImage,
        output: &GpuImage,
    ) -> Result<(), String> {
        if scene_index >= MAX_SCENE_LAYERS {
            return Err(format!(
                "the scene exceeds {MAX_SCENE_LAYERS} visible Metal layers"
            ));
        }
        let device = &self.gpu.device;
        let set = self.sets[frame_slot * MAX_SCENE_LAYERS + scene_index];
        let sampled = [vk::DescriptorImageInfo::default()
            .sampler(self.sampler)
            .image_view(source.view)
            .image_layout(vk::ImageLayout::GENERAL)];
        let target = [vk::DescriptorImageInfo::default()
            .image_view(output.view)
            .image_layout(vk::ImageLayout::GENERAL)];
        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&sampled),
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(&target),
        ];
        unsafe {
            device.update_descriptor_sets(&writes, &[]);
            device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            device.cmd_bind_descriptor_sets(
                cb,
                vk::PipelineBindPoint::COMPUTE,
                self.layout,
                0,
                &[set],
                &[],
            );
        }
        Ok(())
    }

    fn draw(&self, cb: vk::CommandBuffer, placement: &crate::overlay::Placement) {
        if placement.clip_width == 0 || placement.clip_height == 0 {
            return;
        }
        let device = &self.gpu.device;
        unsafe {
            device.cmd_push_constants(
                cb,
                self.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                &placement.bytes(),
            );
            device.cmd_dispatch(
                cb,
                placement.clip_width.div_ceil(8),
                placement.clip_height.div_ceil(8),
                1,
            );
        }
    }
}

impl Drop for SceneBlender {
    fn drop(&mut self) {
        let device = &self.gpu.device;
        unsafe {
            device.destroy_descriptor_pool(self.pool, None);
            device.destroy_sampler(self.sampler, None);
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.layout, None);
        }
    }
}

/// Value-only AppKit layer placement. No Objective-C object is dereferenced
/// after a layer has been detached, hidden or deallocated.
#[derive(Clone, Copy)]
pub(super) struct ScenePlacement {
    pub id: usize,
    pub frame: DirectFrame,
    pub opacity: f32,
    pub clip: Option<DirectFrame>,
}

fn scene_image(gpu: &'static Gpu, source: PresentSource) -> Result<Arc<BackingImageLease>, String> {
    let image = gpu.create_image(&ImageDescription {
        image_type: vk::ImageType::TYPE_2D,
        view_type: vk::ImageViewType::TYPE_2D,
        format: source.format,
        width: source.width,
        height: source.height,
        depth: 1,
        levels: 1,
        layers: 1,
        samples: 1,
        usage: gpu::USAGE_RENDER_TARGET | gpu::USAGE_SHADER_READ,
        cube: false,
        components: vk::ComponentMapping::default(),
    })?;
    Ok(BackingImageLease::new(gpu, image))
}

/// Synchronize transfers, sampled images and the storage compositor across
/// the two pipeline stages, including prior WSI/rounding compute reads.
unsafe fn scene_sync(
    device: &ash::Device,
    cb: vk::CommandBuffer,
    from: vk::PipelineStageFlags,
    to: vk::PipelineStageFlags,
    images: &[(vk::Image, vk::AccessFlags, vk::AccessFlags)],
) {
    let barriers = images
        .iter()
        .map(|&(image, source, destination)| {
            vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::GENERAL)
                .new_layout(vk::ImageLayout::GENERAL)
                .src_access_mask(source)
                .dst_access_mask(destination)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })
        })
        .collect::<Vec<_>>();
    unsafe {
        device.cmd_pipeline_barrier(
            cb,
            from,
            to,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &barriers,
        );
    }
}

/// Does a retained ancestor clipping rectangle fully contain the drawable?
/// Only then may a full-window image bypass scene composition.
pub(super) fn scene_clip_contains(clip: &appkit::view::Frame, frame: &appkit::view::Frame) -> bool {
    i64::from(clip.x) <= i64::from(frame.x)
        && i64::from(clip.y) <= i64::from(frame.y)
        && i64::from(clip.x) + i64::from(clip.width) >= i64::from(frame.x) + i64::from(frame.width)
        && i64::from(clip.y) + i64::from(clip.height)
            >= i64::from(frame.y) + i64::from(frame.height)
}

fn intersection_for_scene(a: &appkit::view::Frame, b: &appkit::view::Frame) -> appkit::view::Frame {
    let left = i64::from(a.x).max(i64::from(b.x));
    let top = i64::from(a.y).max(i64::from(b.y));
    let right = (i64::from(a.x) + i64::from(a.width)).min(i64::from(b.x) + i64::from(b.width));
    let bottom = (i64::from(a.y) + i64::from(a.height)).min(i64::from(b.y) + i64::from(b.height));
    appkit::view::Frame {
        x: left.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32,
        y: top.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32,
        width: right.saturating_sub(left).clamp(0, i64::from(u32::MAX)) as u32,
        height: bottom.saturating_sub(top).clamp(0, i64::from(u32::MAX)) as u32,
        ..*a
    }
}

impl Swapchain {
    pub(super) fn note_scene_frame(&mut self, id: usize, frame: &appkit::view::Frame) {
        if let Some((_, previous)) = self.scene_frames.iter_mut().find(|(known, _)| *known == id) {
            *previous = DirectFrame::from_frame(frame);
        } else {
            self.scene_frames.push((id, DirectFrame::from_frame(frame)));
        }
    }

    /// AppKit's live view hierarchy, not historical present order, decides
    /// which cached layers are still visible and how they must be composed.
    /// The window's CPU backing-store pseudo-layer remains below Metal views.
    pub(super) fn synchronize_scene(&mut self, window: usize, live: &[ScenePlacement]) -> bool {
        let mut ordered = Vec::new();
        if self.layers.contains(&window) {
            ordered.push(window);
        }
        for placement in live {
            if self.layers.contains(&placement.id) && !ordered.contains(&placement.id) {
                ordered.push(placement.id);
            }
        }
        let mut changed = self.layers != ordered;
        self.layers = ordered;
        self.scene_placements = live.to_vec();
        if self
            .direct
            .as_ref()
            .is_some_and(|direct| !self.layers.contains(&direct.layer))
        {
            self.retire_direct();
            changed = true;
        }
        self.scene_layers
            .retain(|layer| self.layers.contains(&layer.id));
        self.scene_frames.retain(|(id, _)| self.layers.contains(id));
        for placement in live {
            if let Some(direct) = self
                .direct
                .as_mut()
                .filter(|direct| direct.layer == placement.id)
            {
                if direct.frame != placement.frame {
                    direct.frame = placement.frame;
                    direct.queued_at = Instant::now();
                    self.composite_dirty = true;
                    self.composite_dirty_since = Some(Instant::now());
                    changed = true;
                }
            }
            if let Some(scene) = self
                .scene_layers
                .iter_mut()
                .find(|scene| scene.id == placement.id)
            {
                if scene.frame != placement.frame
                    || scene.opacity != placement.opacity
                    || scene.clip != placement.clip
                {
                    scene.frame = placement.frame;
                    scene.opacity = placement.opacity;
                    scene.clip = placement.clip;
                    changed = true;
                }
            }
            if let Some((_, frame)) = self
                .scene_frames
                .iter_mut()
                .find(|(id, _)| *id == placement.id)
            {
                *frame = placement.frame;
            }
        }
        if changed {
            self.scene_repaint_needed = true;
        }
        changed
    }

    fn scene_clip(&self, id: usize) -> Option<DirectFrame> {
        self.scene_placements
            .iter()
            .find(|entry| entry.id == id)
            .and_then(|entry| entry.clip)
    }

    fn scene_opacity(&self, id: usize) -> f32 {
        self.scene_placements
            .iter()
            .find(|entry| entry.id == id)
            .map_or(1.0, |entry| entry.opacity)
    }

    /// Create/rotate a snapshot allocation before a command references it.
    /// A dropped generation stays alive through Frame::scene_reads until its
    /// last GPU copy fence has signaled.
    pub(super) fn prepare_scene(
        &mut self,
        id: usize,
        frame: &appkit::view::Frame,
        source: PresentSource,
        regions: Option<&[Region]>,
    ) -> Result<ScenePrepared, String> {
        // A clipped single layer cannot take the historical full-frame blit
        // path: the accumulator must remain transparent outside its ancestor.
        let source_is_clipped = self
            .scene_clip(id)
            .is_some_and(|clip| !scene_clip_contains(&clip.frame(), frame));
        if self.layers.len() <= 1 && self.scene_layers.is_empty() && !source_is_clipped {
            return Ok(ScenePrepared::SingleLayer);
        }
        if self.layers.len() > MAX_SCENE_LAYERS {
            return Err(format!("more than {MAX_SCENE_LAYERS} visible Metal layers"));
        }
        if self.scene_blender.is_none() {
            self.scene_blender = Some(SceneBlender::new(self.gpu)?);
        }
        // Fence-completed frames no longer own scene images, so their pool
        // slots become eligible for reuse without allocation or GPU stalls.
        self.retire_copied();
        let previous = self.layers.iter().copied().find(|layer| {
            *layer != id
                && self.scene_frames.iter().any(|(known, _)| *known == *layer)
                && !self.scene_layers.iter().any(|snap| snap.id == *layer)
        });
        // Prepare both required allocations before mutating the live scene.
        // If the current layer's pool is saturated, leave the previous image
        // intact and ask the presenter to retry when a GPU source is free.
        let previous_image = if previous.is_some() && self.composite.is_some() {
            Some(scene_image(
                self.gpu,
                PresentSource::from(self.composite.as_ref().expect("checked above")),
            )?)
        } else {
            None
        };
        let existing = self.scene_layers.iter().position(|layer| layer.id == id);
        let reused = existing.and_then(|index| {
            let layer = &self.scene_layers[index];
            if source_matches(&layer.snapshot.image, source)
                && Arc::strong_count(&layer.snapshot) == 1
            {
                return Some(Arc::clone(&layer.snapshot));
            }
            layer
                .spare
                .iter()
                .find(|image| source_matches(&image.image, source) && Arc::strong_count(image) == 1)
                .map(Arc::clone)
        });
        let fresh = if let Some(image) = reused {
            image
        } else {
            let count = existing.map_or(0, |index| {
                let layer = &self.scene_layers[index];
                1 + layer.spare.len()
            });
            let currently_held = self
                .scene_layers
                .iter()
                .map(|layer| {
                    std::iter::once(&layer.snapshot)
                        .chain(layer.spare.iter())
                        .map(|image| scene_image_bytes(PresentSource::from(&image.image)))
                        .sum::<u64>()
                })
                .sum::<u64>();
            let mandatory = existing.is_none()
                || existing.is_some_and(|index| {
                    !source_matches(&self.scene_layers[index].snapshot.image, source)
                });
            if !mandatory
                && (count >= MAX_SCENE_IMAGES_PER_LAYER
                    || currently_held.saturating_add(scene_image_bytes(source))
                        > SCENE_IMAGE_BUDGET)
            {
                return Ok(ScenePrepared::Deferred);
            }
            scene_image(self.gpu, source)?
        };
        let prior = existing.map(|index| self.scene_layers[index].clone());
        if let (Some(old), Some(snapshot)) = (previous, previous_image) {
            let old_frame = self
                .scene_frames
                .iter()
                .find(|(known, _)| *known == old)
                .expect("only previously submitted layers are captured")
                .1;
            let full = DirectFrame {
                window: old_frame.window,
                x: 0,
                y: 0,
                width: self.extent.width,
                height: self.extent.height,
                window_width: self.extent.width,
                window_height: self.extent.height,
            };
            self.scene_layers.push(SceneLayer {
                id: old,
                frame: full,
                snapshot,
                spare: Vec::new(),
                opacity: self.scene_opacity(old),
                clip: self.scene_clip(old),
                regions: None,
                ready: false,
            });
        }
        let opacity = self.scene_opacity(id);
        let clip = self.scene_clip(id);
        if let Some(existing) = existing {
            let layer = &mut self.scene_layers[existing];
            if !Arc::ptr_eq(&layer.snapshot, &fresh) {
                // If dimensions changed, don't keep incompatible spare images.
                let resized = !source_matches(&layer.snapshot.image, source);
                if resized {
                    layer.spare.clear();
                }
                if let Some(at) = layer
                    .spare
                    .iter()
                    .position(|image| Arc::ptr_eq(image, &fresh))
                {
                    layer.spare.swap_remove(at);
                }
                let prior = std::mem::replace(&mut layer.snapshot, fresh);
                if !resized {
                    layer.spare.push(prior);
                }
            }
            layer.frame = DirectFrame::from_frame(frame);
            layer.opacity = opacity;
            layer.clip = clip;
            layer.regions = regions.map(|r| r.to_vec());
            layer.ready = false;
        } else {
            self.scene_layers.push(SceneLayer {
                id,
                frame: DirectFrame::from_frame(frame),
                snapshot: fresh,
                spare: Vec::new(),
                opacity,
                clip,
                regions: regions.map(|r| r.to_vec()),
                ready: false,
            });
        }
        Ok(ScenePrepared::Ready(SceneWork {
            current: id,
            previous,
            prior,
        }))
    }

    /// Vulkan may reject a command buffer or its queue submission. The
    /// unsubmitted image must not replace the previously visible generation.
    pub(super) fn rollback_scene(&mut self, work: SceneWork) {
        if work.previous.is_some() {
            self.scene_layers
                .retain(|scene| Some(scene.id) != work.previous);
        }
        match work.prior {
            Some(prior) => {
                if let Some(scene) = self
                    .scene_layers
                    .iter_mut()
                    .find(|scene| scene.id == work.current)
                {
                    *scene = prior;
                } else {
                    self.scene_layers.push(prior);
                }
            }
            None => self.scene_layers.retain(|scene| scene.id != work.current),
        }
    }

    /// Record a full scene recompose within the *same* GPU submission as the
    /// new layer's snapshot copy. This preserves a static foreground when its
    /// background submits again without allocating extra WSI frames.
    pub(super) unsafe fn record_scene(
        &self,
        device: &ash::Device,
        command_buffer: vk::CommandBuffer,
        composite: vk::Image,
        incoming: PresentSource,
        work: &SceneWork,
    ) -> Result<(), String> {
        let frame_extents = |w: u32, h: u32| {
            [
                vk::Offset3D::default(),
                vk::Offset3D {
                    x: w as i32,
                    y: h as i32,
                    z: 1,
                },
            ]
        };
        // A previously single-layer FIFO window only retained the flattened
        // accumulator. Capture it once when a second layer first joins.
        if let Some(old) = work.previous {
            if let Some(old) = self.scene_layers.iter().find(|scene| scene.id == old) {
                let dest = old.snapshot.image.image;
                unsafe {
                    blit(
                        device,
                        command_buffer,
                        composite,
                        frame_extents(self.extent.width, self.extent.height),
                        dest,
                        vk::ImageLayout::GENERAL,
                        frame_extents(self.extent.width, self.extent.height),
                    );
                }
            }
        }
        if let Some(current) = self
            .scene_layers
            .iter()
            .find(|scene| scene.id == work.current)
        {
            let snapshot = current.snapshot.image.image;
            unsafe {
                scene_sync(
                    device,
                    command_buffer,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::PipelineStageFlags::TRANSFER,
                    &[(
                        snapshot,
                        vk::AccessFlags::TRANSFER_READ
                            | vk::AccessFlags::TRANSFER_WRITE
                            | vk::AccessFlags::SHADER_READ,
                        vk::AccessFlags::TRANSFER_WRITE,
                    )],
                );
                blit(
                    device,
                    command_buffer,
                    incoming.image,
                    frame_extents(incoming.width, incoming.height),
                    snapshot,
                    vk::ImageLayout::GENERAL,
                    frame_extents(incoming.width, incoming.height),
                );
            }
        }
        let reads = self
            .scene_layers
            .iter()
            .map(|scene| {
                (
                    scene.snapshot.image.image,
                    vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::SHADER_WRITE,
                    vk::AccessFlags::SHADER_READ,
                )
            })
            .collect::<Vec<_>>();
        unsafe {
            scene_sync(
                device,
                command_buffer,
                vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                &reads,
            );
        }
        unsafe { self.record_scene_output(device, command_buffer, composite, Some(work))? };
        Ok(())
    }

    /// Repaint every currently visible layer in hierarchy order. A scene
    /// update can do this without waiting for a fresh guest Metal drawable.
    unsafe fn record_scene_output(
        &self,
        device: &ash::Device,
        command_buffer: vk::CommandBuffer,
        composite: vk::Image,
        pending: Option<&SceneWork>,
    ) -> Result<(), String> {
        let Some(output) = self.composite.as_ref() else {
            return Err("retained scene missing GPU accumulator".into());
        };
        let blender = self
            .scene_blender
            .as_ref()
            .ok_or("retained scene blend pipeline absent")?;
        let slot = self.frame % self.frames.len().max(1);
        let clear = vk::ClearColorValue { float32: [0.0; 4] };
        let range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };
        unsafe {
            scene_sync(
                device,
                command_buffer,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::TRANSFER,
                &[(
                    composite,
                    vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE,
                    vk::AccessFlags::TRANSFER_WRITE,
                )],
            );
            device.cmd_clear_color_image(
                command_buffer,
                composite,
                vk::ImageLayout::GENERAL,
                &clear,
                &[range],
            );
            scene_sync(
                device,
                command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                &[(
                    composite,
                    vk::AccessFlags::TRANSFER_WRITE,
                    vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
                )],
            );
        }
        let mut scene_index = 0;
        for id in &self.layers {
            let Some(scene) = self.scene_layers.iter().find(|scene| {
                scene.id == *id
                    && (scene.ready
                        || pending.is_some_and(|work| {
                            scene.id == work.current || Some(scene.id) == work.previous
                        }))
            }) else {
                continue;
            };
            blender.bind(
                command_buffer,
                slot,
                scene_index,
                &scene.snapshot.image,
                output,
            )?;
            scene_index += 1;
            let frame = scene.frame.frame();
            let window_clip = appkit::view::Frame {
                x: 0,
                y: 0,
                width: self.extent.width,
                height: self.extent.height,
                ..frame
            };
            let draw = |area: &appkit::view::Frame| {
                // Clamp dispatch dimensions to actual physical window pixels,
                // rather than wasting an unbounded number of GPU workgroups
                // for an offscreen or oversized guest layout rectangle.
                let clip = intersection_for_scene(area, &window_clip);
                let clip = scene.clip.map_or(clip, |ancestor| {
                    intersection_for_scene(&clip, &ancestor.frame())
                });
                let placement = crate::overlay::Placement {
                    x: frame.x,
                    y: frame.y,
                    width: frame.width,
                    height: frame.height,
                    target_width: self.extent.width,
                    target_height: self.extent.height,
                    clip_x: clip.x,
                    clip_y: clip.y,
                    clip_width: clip.width,
                    clip_height: clip.height,
                    opacity: scene.opacity,
                };
                blender.draw(command_buffer, &placement);
            };
            if let Some(clips) = &scene.regions {
                for region in clips {
                    draw(&appkit::view::Frame {
                        x: region.x,
                        y: region.y,
                        width: region.width.max(0) as u32,
                        height: region.height.max(0) as u32,
                        ..frame
                    });
                }
            } else {
                draw(&frame);
            }
            unsafe {
                scene_sync(
                    device,
                    command_buffer,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    &[(
                        composite,
                        vk::AccessFlags::SHADER_WRITE,
                        vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
                    )],
                );
            }
        }
        unsafe {
            scene_sync(
                device,
                command_buffer,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::ALL_COMMANDS,
                &[(
                    composite,
                    vk::AccessFlags::SHADER_WRITE,
                    vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::SHADER_READ,
                )],
            );
        }
        Ok(())
    }

    /// Queue one retained-scene repaint after visibility, geometry or order
    /// changes. No SDL/Objective-C operation is performed under a Vulkan lock.
    pub(super) fn redisplay_scene(&mut self, window: usize) -> Result<bool, String> {
        if self.layers.len() == 1 {
            if self
                .direct
                .as_ref()
                .is_some_and(|direct| direct.layer == self.layers[0])
            {
                self.composite_dirty = true;
                self.composite_dirty_since = Some(Instant::now());
                return Ok(true);
            }
            let layer = self.layers[0];
            if let Some(scene) = self
                .scene_layers
                .iter()
                .find(|scene| scene.id == layer && scene.ready)
            {
                let frame = scene.frame.frame();
                if fast_direct_layer(
                    self.present_mode,
                    1,
                    &frame,
                    scene.regions.as_deref(),
                    self.corners.is_some(),
                    true,
                ) && scene
                    .clip
                    .is_none_or(|clip| scene_clip_contains(&clip.frame(), &frame))
                {
                    let snapshot = Arc::clone(&scene.snapshot);
                    let source = PresentSource::from(&snapshot.image);
                    self.queue_direct(
                        layer,
                        &frame,
                        source,
                        &mut SourceOwner(None),
                        Some(snapshot),
                    );
                    return Ok(true);
                }
            }
        }
        if self.scene_blender.is_none() {
            self.scene_blender = Some(SceneBlender::new(self.gpu)?);
        }
        let slot = self.frame % self.frames.len();
        let (cb, done, submitted) = {
            let frame = &self.frames[slot];
            (frame.command_buffer, frame.done, frame.submitted)
        };
        if submitted && !self.gpu.fence_signaled(done)? {
            return Ok(false);
        }
        self.frames[slot].submitted = false;
        finish_source(self.frames[slot].source.take());
        self.frames[slot].backing.take();
        self.frames[slot].scene_reads.clear();
        let composite = self.composite()?;
        let device = &self.gpu.device;
        unsafe {
            device
                .reset_command_buffer(cb, vk::CommandBufferResetFlags::empty())
                .map_err(|e| format!("vkResetCommandBuffer: {e:?}"))?;
            device
                .begin_command_buffer(
                    cb,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .map_err(|e| format!("vkBeginCommandBuffer: {e:?}"))?;
            barrier(
                device,
                cb,
                composite,
                vk::ImageLayout::GENERAL,
                vk::ImageLayout::GENERAL,
                vk::AccessFlags::TRANSFER_READ
                    | vk::AccessFlags::TRANSFER_WRITE
                    | vk::AccessFlags::SHADER_READ,
                vk::AccessFlags::TRANSFER_WRITE,
            );
            self.record_scene_output(device, cb, composite, None)?;
            device
                .end_command_buffer(cb)
                .map_err(|e| format!("vkEndCommandBuffer: {e:?}"))?;
        }
        #[cfg(test)]
        if let Some(error) = self.injected_post_scene_record_error.take() {
            return Err(format!("injected post-scene recording failure: {error:?}"));
        }
        self.gpu.submit_synchronized(cb, done, &[], &[])?;
        self.frames[slot].submitted = true;
        self.frames[slot].scene_reads = self
            .scene_layers
            .iter()
            .filter(|layer| layer.ready)
            .map(|layer| Arc::clone(&layer.snapshot))
            .collect();
        self.frame = self.frame.wrapping_add(1);
        if !self.composite_dirty {
            self.composite_dirty_since = Some(Instant::now());
        }
        self.composite_dirty = true;
        let _ = window; // Window-local owner schedules WSI publication.
        Ok(true)
    }

    pub(super) fn complete_scene(&mut self, slot: usize, work: SceneWork) {
        for layer in &mut self.scene_layers {
            if layer.id == work.current || Some(layer.id) == work.previous {
                layer.ready = true;
            }
        }
        self.frames[slot].scene_reads = self
            .scene_layers
            .iter()
            .filter(|layer| layer.ready)
            .map(|layer| Arc::clone(&layer.snapshot))
            .collect();
    }
}
