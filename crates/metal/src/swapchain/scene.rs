//! GPU-owned snapshots of independently submitted Metal layers.
//! A drawable may be reused as soon as its GPU copy finishes, but the last
//! image it drew must survive when another layer continues submitting frames.

use super::*;
use crate::gpu::GpuBuffer;

#[derive(Clone)]
pub(super) struct SceneLayer {
    pub(super) id: usize,
    frame: DirectFrame,
    pub(super) snapshot: Arc<BackingImageLease>,
    spare: Vec<Arc<BackingImageLease>>,
    opacity: f32,
    opaque: bool,
    clip: Option<DirectFrame>,
    rounded: Vec<RoundedClip>,
    regions: Option<Vec<Region>>,
    pub(super) native_revision: Option<u64>,
    /// Original AppKit publication revision for the *retained GPU image*.
    /// The GPU revision may instead hash clipping or DMA mapping details.
    pub(super) native_published_revision: Option<u64>,
    pub(super) ready: bool,
}

pub(super) struct SceneWork {
    current: usize,
    previous: Option<usize>,
    /// A safe incremental source-content update. Changes to visibility,
    /// geometry, opacity, order and fallback staging redraw the full scene.
    damage: Option<DirectFrame>,
    /// Pre-submission membership/source state, retained only until the VkQueue
    /// accepts the new generation. Recording errors must not poison the scene.
    prior: Option<SceneLayer>,
}

pub(super) enum ScenePrepared {
    SingleLayer,
    Deferred,
    Ready(SceneWork),
}

// Hard per-window owned compositor GPU image ceiling, including the window
// accumulator, scene snapshots, in-flight retired leases, and transient native
// upload/blend images. Does not cover guest-owned drawable or WSI images.
// A 6016x3384 DMA source can require four RGBA images during initial staging
// (transparent base, blend output, retained layer, and window accumulator).
const SCENE_IMAGE_BUDGET: u64 = 512 * 1024 * 1024;

pub(super) fn fits_scene_budget(held: u64, incoming: u64) -> bool {
    incoming <= SCENE_IMAGE_BUDGET && held <= SCENE_IMAGE_BUDGET - incoming
}
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
const INITIAL_SCENE_LAYERS: usize = 32;

/// Per-window GPU source-over blending, using the same tested shader as
/// AppKit's DMA-BUF overlays. Descriptor updates are frame-slot isolated.
pub(super) struct SceneBlender {
    gpu: &'static Gpu,
    window: usize,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    sampler: vk::Sampler,
    pools: Vec<vk::DescriptorPool>,
    set_layout: vk::DescriptorSetLayout,
    /// One descriptor vector per in-flight frame; growing a scene appends a
    /// new VkDescriptorPool without invalidating any previously bound sets.
    pub(super) sets: Vec<Vec<vk::DescriptorSet>>,
    /// Each clip buffer belongs to one fence-protected frame slot.
    masks: Vec<Mutex<Option<GpuBuffer>>>,
    mask_charges: Vec<Mutex<Option<GpuCharge>>>,
}

impl SceneBlender {
    fn new(gpu: &'static Gpu, window: usize) -> Result<Self, String> {
        let device = &gpu.device;
        let set_layout = gpu.descriptor_set_layout(
            &[
                (0, vk::DescriptorType::COMBINED_IMAGE_SAMPLER),
                (1, vk::DescriptorType::STORAGE_IMAGE),
                (2, vk::DescriptorType::STORAGE_BUFFER),
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
        let module =
            gpu.create_shader_module(&crate::overlay::scene_shader(), "rosemary_scene_blend")?;
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
        let mut result = Self {
            gpu,
            window,
            layout: layout.take(),
            pipeline: pipeline.take(),
            sampler: sampler.take(),
            pools: Vec::new(),
            set_layout,
            sets: vec![Vec::new(); FRAMES_IN_FLIGHT],
            masks: (0..FRAMES_IN_FLIGHT).map(|_| Mutex::new(None)).collect(),
            mask_charges: (0..FRAMES_IN_FLIGHT).map(|_| Mutex::new(None)).collect(),
        };
        result.reserve_layers(INITIAL_SCENE_LAYERS)?;
        Ok(result)
    }

    fn reserve_layers(&mut self, needed: usize) -> Result<(), String> {
        let have = self.sets[0].len();
        if needed <= have {
            return Ok(());
        }
        // Grow geometrically so deep trees do not allocate one descriptor
        // pool per new view. Every old pool remains live for in-flight frames.
        let total = needed.max(have.saturating_mul(2)).max(INITIAL_SCENE_LAYERS);
        let extra = total - have;
        let count = u32::try_from(
            extra
                .checked_mul(FRAMES_IN_FLIGHT)
                .ok_or("scene descriptor count overflow")?,
        )
        .map_err(|_| "scene descriptor count exceeds Vulkan u32")?;
        let sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(count),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(count),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(count),
        ];
        let info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(count)
            .pool_sizes(&sizes);
        let pool = unsafe { self.gpu.device.create_descriptor_pool(&info, None) }
            .map_err(|error| format!("scene descriptor pool growth failed: {error:?}"))?;
        let guard = gpu::Guard::new(pool, |pool| unsafe {
            self.gpu.device.destroy_descriptor_pool(pool, None);
        });
        let layouts = vec![self.set_layout; count as usize];
        let alloc = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(*guard)
            .set_layouts(&layouts);
        let additional = unsafe { self.gpu.device.allocate_descriptor_sets(&alloc) }
            .map_err(|error| format!("scene descriptor set growth failed: {error:?}"))?;
        for (frame, sets) in self.sets.iter_mut().enumerate() {
            sets.extend_from_slice(&additional[frame * extra..(frame + 1) * extra]);
        }
        self.pools.push(guard.take());
        Ok(())
    }

    /// Upload all rounded ancestor masks in source order. A frame slot is
    /// fenced before this is called, including when its buffer must grow.
    fn write_masks(
        &self,
        slot: usize,
        layers: &[SceneLayer],
    ) -> Result<Vec<(usize, u32, u32)>, String> {
        let mut floats = Vec::<f32>::new();
        let mut spans = Vec::with_capacity(layers.len());
        for layer in layers {
            let first = u32::try_from(floats.len() / 8).map_err(|_| "too many masks")?;
            let count = u32::try_from(layer.rounded.len()).map_err(|_| "too many nested masks")?;
            spans.push((layer.id, first, count));
            for mask in &layer.rounded {
                let f = mask.frame;
                floats.extend_from_slice(&[
                    f.x as f32,
                    f.y as f32,
                    (i64::from(f.x) + i64::from(f.width)) as f32,
                    (i64::from(f.y) + i64::from(f.height)) as f32,
                    mask.radius,
                    0.0,
                    0.0,
                    0.0,
                ]);
            }
        }
        if floats.is_empty() {
            floats.extend_from_slice(&[0.0; 8]);
        }
        let bytes = floats.len().checked_mul(4).ok_or("mask size overflow")?;
        let mut guard = self.masks[slot].lock().expect("mask frame");
        if guard
            .as_ref()
            .is_none_or(|buffer| buffer.length < bytes as u64)
        {
            let charge = GpuCharge::reserve(self.window, bytes as u64)?;
            let fresh = self.gpu.create_buffer(bytes as u64)?;
            if let Some(old) = guard.replace(fresh) {
                self.gpu.destroy_buffer(old);
            }
            *self.mask_charges[slot].lock().expect("clip budget") = Some(charge);
        }
        let buffer = guard.as_ref().expect("allocated mask buffer");
        unsafe {
            std::ptr::copy_nonoverlapping(floats.as_ptr().cast::<u8>(), buffer.contents, bytes)
        };
        Ok(spans)
    }

    fn bind(
        &self,
        cb: vk::CommandBuffer,
        frame_slot: usize,
        scene_index: usize,
        source: &GpuImage,
        output: &GpuImage,
    ) -> Result<(), String> {
        let set = *self
            .sets
            .get(frame_slot)
            .and_then(|frame| frame.get(scene_index))
            .ok_or("scene descriptors have not been reserved for the live layer count")?;
        let device = &self.gpu.device;
        let sampled = [vk::DescriptorImageInfo::default()
            .sampler(self.sampler)
            .image_view(source.view)
            .image_layout(vk::ImageLayout::GENERAL)];
        let target = [vk::DescriptorImageInfo::default()
            .image_view(output.view)
            .image_layout(vk::ImageLayout::GENERAL)];
        let masks = self.masks[frame_slot].lock().expect("mask frame");
        let masks = masks.as_ref().ok_or("scene mask data not prepared")?;
        let mask_info = [vk::DescriptorBufferInfo::default()
            .buffer(masks.buffer)
            .offset(0)
            .range(masks.length.max(4))];
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
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&mask_info),
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
            for pool in self.pools.drain(..) {
                device.destroy_descriptor_pool(pool, None);
            }
            for (slot, charge) in self.masks.iter().zip(&self.mask_charges) {
                if let Some(buffer) = slot.lock().expect("clip frame").take() {
                    self.gpu.destroy_buffer(buffer);
                }
                *charge.lock().expect("clip budget") = None;
            }
            device.destroy_sampler(self.sampler, None);
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.layout, None);
        }
    }
}

/// Value-only AppKit layer placement. No Objective-C object is dereferenced
/// after a layer has been detached, hidden or deallocated.
#[derive(Clone, Copy, PartialEq)]
pub(super) struct RoundedClip {
    pub frame: DirectFrame,
    pub radius: f32,
}

#[derive(Clone)]
pub(super) struct NativeCpuSceneSource {
    pub bytes: Arc<Vec<u8>>,
    pub revision: u64,
    pub dmabuf: Option<Arc<appkit::backing::Dmabuf>>,
}

#[derive(Clone)]
pub(super) struct ScenePlacement {
    pub id: usize,
    pub frame: DirectFrame,
    pub opacity: f32,
    pub opaque: bool,
    pub clip: Option<DirectFrame>,
    pub rounded: Vec<RoundedClip>,
    /// Some means a live AppKit-computed native occlusion set. Synthetic
    /// fixtures use None because they supply clipping with drawable holes.
    pub occluded: Option<Vec<Region>>,
    pub native: Option<NativeCpuSceneSource>,
}

fn scene_image(
    gpu: &'static Gpu,
    window: usize,
    source: PresentSource,
) -> Result<Arc<BackingImageLease>, String> {
    BackingImageLease::create_for_window(
        gpu,
        window,
        &ImageDescription {
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
        },
    )
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
        if self.layers.len() == 1 && self.scene_layers.is_empty() {
            let id = self.layers[0];
            let old = self.scene_placements.iter().find(|p| p.id == id);
            if live.iter().find(|p| p.id == id).is_some_and(|p| {
                p.opacity < 1.0
                    || !p.opaque
                    || !p.rounded.is_empty()
                    || p.occluded.as_ref().is_some_and(|r| !r.is_empty())
                    || p.clip
                        .is_some_and(|clip| !scene_clip_contains(&clip.frame(), &p.frame.frame()))
                    || old.is_some_and(|old| {
                        old.frame != p.frame
                            || old.opacity != p.opacity
                            || old.opaque != p.opaque
                            || old.clip != p.clip
                            || old.rounded != p.rounded
                            || old.occluded != p.occluded
                            || old.native.as_ref().map(|n| n.revision)
                                != p.native.as_ref().map(|n| n.revision)
                    })
            }) {
                changed = true;
            }
        }
        if let Some(direct) = self.direct.as_ref() {
            let new = live.iter().find(|p| p.id == direct.layer);
            let old = self.scene_placements.iter().find(|p| p.id == direct.layer);
            if new.zip(old).is_some_and(|(new, old)| {
                new.opacity != old.opacity
                    || new.opaque != old.opaque
                    || new.frame != old.frame
                    || new.clip != old.clip
                    || new.rounded != old.rounded
                    || new.occluded != old.occluded
                    || new.native.as_ref().map(|n| n.revision)
                        != old.native.as_ref().map(|n| n.revision)
            }) || new.is_some_and(|p| {
                p.opacity < 1.0
                    || !p.opaque
                    || !p.rounded.is_empty()
                    || p.occluded.as_ref().is_some_and(|r| !r.is_empty())
                    || p.clip
                        .is_some_and(|clip| !scene_clip_contains(&clip.frame(), &p.frame.frame()))
            }) {
                changed = true;
            }
        }
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
                    || scene.opaque != placement.opaque
                    || scene.clip != placement.clip
                    || scene.rounded != placement.rounded
                    || placement.occluded.as_ref().is_some_and(|coverage| {
                        let updated = (!coverage.is_empty()).then(|| {
                            super::subtract(
                                Region {
                                    x: placement.frame.x,
                                    y: placement.frame.y,
                                    width: placement.frame.width.min(i32::MAX as u32) as i32,
                                    height: placement.frame.height.min(i32::MAX as u32) as i32,
                                },
                                coverage,
                            )
                        });
                        scene.regions != updated
                    })
                {
                    scene.frame = placement.frame;
                    scene.opacity = placement.opacity;
                    scene.opaque = placement.opaque;
                    scene.clip = placement.clip;
                    scene.rounded.clone_from(&placement.rounded);
                    if let Some(coverage) = &placement.occluded {
                        scene.regions = (!coverage.is_empty()).then(|| {
                            super::subtract(
                                Region {
                                    x: placement.frame.x,
                                    y: placement.frame.y,
                                    width: placement.frame.width.min(i32::MAX as u32) as i32,
                                    height: placement.frame.height.min(i32::MAX as u32) as i32,
                                },
                                coverage,
                            )
                        });
                    }
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

    pub(super) fn live_scene_regions(
        &self,
        id: usize,
        frame: &appkit::view::Frame,
    ) -> Option<Vec<Region>> {
        let coverage = self
            .scene_placements
            .iter()
            .find(|p| p.id == id)
            .and_then(|p| p.occluded.as_deref())?;
        if coverage.is_empty() {
            return None;
        }
        Some(super::subtract(
            Region {
                x: frame.x,
                y: frame.y,
                width: frame.width.min(i32::MAX as u32) as i32,
                height: frame.height.min(i32::MAX as u32) as i32,
            },
            coverage,
        ))
    }

    fn scene_clip(&self, id: usize) -> Option<DirectFrame> {
        self.scene_placements
            .iter()
            .find(|entry| entry.id == id)
            .and_then(|entry| entry.clip)
    }

    pub(super) fn scene_rounded(&self, id: usize) -> &[RoundedClip] {
        self.scene_placements
            .iter()
            .find(|entry| entry.id == id)
            .map_or(&[], |entry| entry.rounded.as_slice())
    }

    pub(super) fn scene_opaque(&self, id: usize) -> bool {
        self.scene_placements
            .iter()
            .find(|entry| entry.id == id)
            .is_none_or(|entry| entry.opaque)
    }

    pub(super) fn scene_opacity(&self, id: usize) -> f32 {
        self.scene_placements
            .iter()
            .find(|entry| entry.id == id)
            .map_or(1.0, |entry| entry.opacity)
    }

    /// Count distinct *owned* GPU images, not Arc references: copies held by
    /// the current scene and submitted frame leases refer to one allocation.
    /// External guest drawables and WSI-owned images are deliberately excluded.
    pub(super) fn scene_gpu_bytes(&self) -> u64 {
        let mut seen = std::collections::HashSet::new();
        let mut bytes = 0u64;
        let mut count = |lease: &Arc<BackingImageLease>| {
            if seen.insert(Arc::as_ptr(lease) as usize) {
                bytes = bytes.saturating_add(scene_image_bytes(PresentSource::from(&lease.image)));
            }
        };
        for layer in &self.scene_layers {
            count(&layer.snapshot);
            for spare in &layer.spare {
                count(spare);
            }
        }
        for frame in &self.frames {
            if let Some(lease) = &frame.backing {
                count(lease);
            }
            for lease in &frame.scene_reads {
                count(lease);
            }
        }
        if let Some(direct) = &self.direct {
            if let Some(lease) = &direct.backing {
                count(lease);
            }
        }
        if let Some(composite) = &self.composite {
            bytes = bytes.saturating_add(scene_image_bytes(PresentSource::from(composite)));
        }
        bytes
    }

    /// Lower bound on scene capacity with *all* previous submissions retired.
    /// This is the non-evictable set; a new native source which cannot fit
    /// alongside it will not become stageable by polling fences or sleeping.
    pub(super) fn committed_scene_bytes(&self) -> u64 {
        self.scene_layers
            .iter()
            .filter(|layer| layer.ready)
            .map(|layer| scene_image_bytes(PresentSource::from(&layer.snapshot.image)))
            .sum::<u64>()
            .saturating_add(
                self.composite
                    .as_ref()
                    .map_or(0, |image| scene_image_bytes(PresentSource::from(image))),
            )
    }

    /// Spares are only an optimization. Drop them on budget pressure; Arc
    /// leases held by unfinished GPU commands remain alive and stay charged
    /// through Frame::scene_reads until their completion fences retire.
    pub(super) fn shed_scene_spares(&mut self) {
        for layer in &mut self.scene_layers {
            layer.spare.clear();
        }
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
        if self.layers.len() <= 1
            && self.scene_layers.is_empty()
            // A partial drawable is not interchangeable with the flattened
            // full-window accumulator. Its original layer-space pixels must
            // survive an NSView move, resize or a later sibling arrival.
            && frame.x == 0
            && frame.y == 0
            && frame.width == self.extent.width
            && frame.height == self.extent.height
            && source.width == frame.width
            && source.height == frame.height
            && regions.is_none()
            && !source_is_clipped
            && self.scene_rounded(id).is_empty()
            && self.scene_opacity(id) >= 1.0
            && self.scene_opaque(id)
        {
            return Ok(ScenePrepared::SingleLayer);
        }
        if self.scene_blender.is_none() {
            self.scene_blender = Some(SceneBlender::new(self.gpu, self.window)?);
        }
        self.scene_blender
            .as_mut()
            .expect("initialized")
            .reserve_layers(self.layers.len().saturating_add(1))?;
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
        let previous_image_source = previous
            .and_then(|_| self.composite.as_ref())
            .map(PresentSource::from);
        if let Some(source) = previous_image_source {
            if !fits_scene_budget(self.scene_gpu_bytes(), scene_image_bytes(source)) {
                self.shed_scene_spares();
                if !fits_scene_budget(self.scene_gpu_bytes(), scene_image_bytes(source)) {
                    return Err(
                        "retained scene GPU budget exhausted capturing previous layer".into(),
                    );
                }
            }
        }
        let previous_image = previous_image_source
            .map(|source| scene_image(self.gpu, self.window, source))
            .transpose()?;
        let previous_image_bytes = previous_image_source.map_or(0, scene_image_bytes);
        let existing = self.scene_layers.iter().position(|layer| layer.id == id);
        // Incremental repaint is only safe when a fully committed scene and
        // this layer's previous geometry/masking match the new submission.
        // The output accumulator is persistent, so pixels outside this area
        // must never be cleared on an ordinary stationary foreground update.
        let damage = existing.and_then(|index| {
            let old = &self.scene_layers[index];
            (self.composite.is_some()
                && previous.is_none()
                && !self.scene_repaint_needed
                && self.scene_layers.len() == self.layers.len()
                && self.scene_layers.iter().all(|layer| layer.ready)
                && old.frame == DirectFrame::from_frame(frame)
                && old.regions.as_deref() == regions
                && old.opacity == self.scene_opacity(id)
                && old.opaque == self.scene_opaque(id)
                && old.clip == self.scene_clip(id)
                && old.rounded == self.scene_rounded(id))
            .then_some(DirectFrame::from_frame(frame))
        });
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
            if count >= MAX_SCENE_IMAGES_PER_LAYER {
                return Ok(ScenePrepared::Deferred);
            }
            // Budget all retained images, the full-window accumulator, prior
            // source capture and in-flight/retired leases. No "mandatory"
            // bypass: a new layer or resize can never overcommit GPU memory.
            let incoming = previous_image_bytes.saturating_add(scene_image_bytes(source));
            if !fits_scene_budget(self.scene_gpu_bytes(), incoming) {
                self.shed_scene_spares();
                if !fits_scene_budget(self.scene_gpu_bytes(), incoming) {
                    return Err("retained scene GPU allocation budget exhausted".into());
                }
            }
            scene_image(self.gpu, self.window, source)?
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
                opaque: self.scene_opaque(old),
                clip: self.scene_clip(old),
                rounded: self.scene_rounded(old).to_vec(),
                regions: None,
                native_revision: None,
                native_published_revision: None,
                ready: false,
            });
        }
        let opacity = self.scene_opacity(id);
        let opaque = self.scene_opaque(id);
        let clip = self.scene_clip(id);
        let rounded = self.scene_rounded(id).to_vec();
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
            layer.opaque = opaque;
            layer.clip = clip;
            layer.rounded.clone_from(&rounded);
            layer.regions = regions.map(|r| r.to_vec());
            layer.ready = false;
        } else {
            self.scene_layers.push(SceneLayer {
                id,
                frame: DirectFrame::from_frame(frame),
                snapshot: fresh,
                spare: Vec::new(),
                opacity,
                opaque,
                clip,
                rounded,
                regions: regions.map(|r| r.to_vec()),
                native_revision: None,
                native_published_revision: None,
                ready: false,
            });
        }
        Ok(ScenePrepared::Ready(SceneWork {
            current: id,
            previous,
            prior,
            damage,
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
        let mask_spans = blender.write_masks(slot, &self.scene_layers)?;
        let clear = vk::ClearColorValue { float32: [0.0; 4] };
        let range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };
        let damage = pending
            .and_then(|work| work.damage)
            .map(|frame| frame.frame());
        if let Some(damage) = damage {
            // A scene layer submission does not invalidate the whole window:
            // clear only the source's previous bounds, then recompose every
            // layer intersecting those pixels in the original stacking order.
            // A dedicated extra descriptor avoids overwriting sets referenced
            // by subsequent source-over dispatches in the same command buffer.
            unsafe {
                scene_sync(
                    device,
                    command_buffer,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    &[(
                        composite,
                        vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE,
                        vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
                    )],
                );
            }
            let first = self
                .scene_layers
                .iter()
                .find(|scene| scene.ready || pending.is_some_and(|work| scene.id == work.current))
                .ok_or("damaged scene has no ready source")?;
            blender.bind(
                command_buffer,
                slot,
                self.layers.len(),
                &first.snapshot.image,
                output,
            )?;
            let bounded = intersection_for_scene(
                &damage,
                &appkit::view::Frame {
                    x: 0,
                    y: 0,
                    width: self.extent.width,
                    height: self.extent.height,
                    ..damage
                },
            );
            blender.draw(
                command_buffer,
                &crate::overlay::Placement {
                    x: bounded.x,
                    y: bounded.y,
                    width: bounded.width,
                    height: bounded.height,
                    target_width: self.extent.width,
                    target_height: self.extent.height,
                    clip_x: bounded.x,
                    clip_y: bounded.y,
                    clip_width: bounded.width,
                    clip_height: bounded.height,
                    opacity: 1.0,
                    clear: true,
                    round_start: 0,
                    round_count: 0,
                    sampled_srgb: 0,
                },
            );
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
        } else {
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
                let clip = damage.map_or(clip, |changed| intersection_for_scene(&clip, &changed));
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
                    clear: false,
                    round_start: mask_spans
                        .iter()
                        .find(|(id, _, _)| *id == scene.id)
                        .map_or(0, |(_, first, _)| *first),
                    round_count: scene.rounded.len() as u32,
                    sampled_srgb: u32::from(matches!(
                        scene.snapshot.image.format,
                        vk::Format::R8G8B8A8_SRGB | vk::Format::B8G8R8A8_SRGB
                    )),
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

    /// Materialize a source image from FIFO's last opaque one-layer result
    /// only if a CA-only scene change needs to blend it with opacity.
    fn capture_single_scene_from_composite(&mut self) -> Result<(), String> {
        let id = self.layers[0];
        let accumulated = self
            .composite
            .as_ref()
            .ok_or("single-layer accumulator absent")?;
        let snapshot = scene_image(self.gpu, self.window, PresentSource::from(accumulated))?;
        let src = accumulated.image;
        let dst = snapshot.image.image;
        let extent = self.extent;
        self.gpu.one_shot(|device, cb| unsafe {
            scene_sync(
                device,
                cb,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::TRANSFER,
                &[(
                    src,
                    vk::AccessFlags::MEMORY_WRITE,
                    vk::AccessFlags::TRANSFER_READ,
                )],
            );
            let area = [
                vk::Offset3D::default(),
                vk::Offset3D {
                    x: extent.width as i32,
                    y: extent.height as i32,
                    z: 1,
                },
            ];
            blit(device, cb, src, area, dst, vk::ImageLayout::GENERAL, area);
        })?;
        let window = self
            .scene_frames
            .iter()
            .find(|(known, _)| *known == id)
            .map_or(0, |(_, frame)| frame.window);
        self.scene_layers.push(SceneLayer {
            id,
            frame: DirectFrame {
                window,
                x: 0,
                y: 0,
                width: extent.width,
                height: extent.height,
                window_width: extent.width,
                window_height: extent.height,
            },
            snapshot,
            spare: Vec::new(),
            opacity: self.scene_opacity(id),
            opaque: self.scene_opaque(id),
            clip: self.scene_clip(id),
            rounded: self.scene_rounded(id).to_vec(),
            regions: None,
            native_revision: None,
            native_published_revision: None,
            ready: true,
        });
        Ok(())
    }

    /// Queue one retained-scene repaint after visibility, geometry or order
    /// changes. No SDL/Objective-C operation is performed under a Vulkan lock.
    pub(super) fn redisplay_scene(&mut self, window: usize) -> Result<bool, String> {
        if let Some(direct) = self.direct.as_ref() {
            let source_frame = direct.frame.frame();
            let layer = direct.layer;
            let needs_scene = self.scene_opacity(layer) < 1.0
                || !self.scene_opaque(layer)
                || !self.scene_rounded(layer).is_empty()
                || self.live_scene_regions(layer, &source_frame).is_some()
                || self
                    .scene_clip(layer)
                    .is_some_and(|clip| !scene_clip_contains(&clip.frame(), &source_frame));
            if needs_scene {
                // Reuse the last retained drawable after *any* CA-only
                // change that makes a direct WSI copy visually invalid.
                // stage_direct_snapshot restores it on a busy-slot retry.
                return self
                    .stage_direct_snapshot(true)
                    .map(|done| done == Some(true));
            }
        }
        if self.layers.len() == 1 && self.scene_layers.is_empty() {
            let id = self.layers[0];
            let clipped = self
                .scene_frames
                .iter()
                .find(|(known, _)| *known == id)
                .is_some_and(|(_, frame)| {
                    self.scene_clip(id)
                        .is_some_and(|clip| !scene_clip_contains(&clip.frame(), &frame.frame()))
                });
            if self.scene_opacity(id) < 1.0
                || !self.scene_opaque(id)
                || !self.scene_rounded(id).is_empty()
                || clipped
            {
                // FIFO's opaque flattened single-layer result needs a
                // retained original before any CA-only alpha/mask change.
                self.capture_single_scene_from_composite()?;
            }
        }
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
                ) && scene.opacity >= 1.0
                    && scene.opaque
                    && scene.rounded.is_empty()
                    && scene
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
            self.scene_blender = Some(SceneBlender::new(self.gpu, self.window)?);
        }
        self.scene_blender
            .as_mut()
            .expect("initialized")
            .reserve_layers(self.layers.len().saturating_add(1))?;
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

#[cfg(test)]
mod budget_regressions {
    use super::*;

    #[test]
    fn large_dma_staging_respects_a_hard_window_budget_across_sources_and_resize() {
        // Four simultaneous BGRA images for an initial 6016x3384 native
        // source (accumulator, transparent base, blend, scene snapshot).
        let big = 6016u64 * 3384 * 4;
        assert!(fits_scene_budget(0, 4 * big));
        // The next visible DMA source requires additional base, blend and
        // retained snapshot allocations; no "mandatory" bypass is allowed.
        assert!(!fits_scene_budget(4 * big, 3 * big));
        // Resize churn holds the old generation until a frame fence retires.
        let replacement = 8192u64 * 4096 * 4;
        assert!(!fits_scene_budget(4 * big, replacement * 2));
        // After retirement releases old snapshot references, new allocation
        // can progress rather than permanently poisoning the scene.
        assert!(fits_scene_budget(0, replacement * 2));
        assert!(!fits_scene_budget(SCENE_IMAGE_BUDGET - 1, 2));
    }
}
