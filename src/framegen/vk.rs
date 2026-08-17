//! Vulkan compute-based frame generation — port of the GL 4.3 framegen pipeline.
//!
//! Three compute passes:
//! 1. Hierarchical block-matching optical flow (coarse → fine)
//! 2. 3×3 weighted median filter + temporal dampening on MV field
//! 3. Motion-compensated frame synthesis (warp)

use ash::vk;
use std::ffi::CStr;

use super::{FrameGenMode, FrameGenQuality, FrameGenStats};

const NUM_PYRAMID_LEVELS: u32 = 4;

const FG_FLOW_SPV: &[u8] = include_bytes!("../shaders/fg_flow.spv");
const FG_MV_FILTER_SPV: &[u8] = include_bytes!("../shaders/fg_mv_filter.spv");
const FG_SYNTH_SPV: &[u8] = include_bytes!("../shaders/fg_synth.spv");

struct ImgRes {
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
}

pub struct VkFrameGen {
    device: ash::Device,
    width: u32,
    height: u32,

    // Frame textures (RGBA8, mipmapped)
    img_prev: ImgRes,
    img_curr: ImgRes,
    // Output (RGBA8, no mipmaps)
    img_out: ImgRes,

    // MV textures per pyramid level (RGBA16F)
    mv_levels: Vec<ImgRes>,
    // MV filter ping-pong
    mv_filtered: ImgRes,
    mv_prev: ImgRes,
    has_prev_mv: bool,

    // Pipeline resources
    sampler: vk::Sampler,
    desc_layout: vk::DescriptorSetLayout,
    pipe_layout: vk::PipelineLayout,
    flow_pipeline: vk::Pipeline,
    filter_pipeline: vk::Pipeline,
    synth_pipeline: vk::Pipeline,
    desc_pool: vk::DescriptorPool,
    // [0..3] = block match per level, [4] = filter, [5] = synth
    desc_sets: Vec<vk::DescriptorSet>,

    level_dims: Vec<(u32, u32)>,
    frame_count: u64,
    mv_valid: bool,
    stats: FrameGenStats,
}

impl VkFrameGen {
    /// Create the Vulkan framegen pipeline. Returns None on failure.
    pub fn new(
        device: &ash::Device,
        mem_props: &vk::PhysicalDeviceMemoryProperties,
        w: u32,
        h: u32,
        debug: bool,
    ) -> Option<Self> {
        // Compute pyramid level dimensions
        let mut level_dims = Vec::with_capacity(NUM_PYRAMID_LEVELS as usize);
        let (mut lw, mut lh) = (w, h);
        for _ in 0..NUM_PYRAMID_LEVELS {
            level_dims.push((lw, lh));
            lw = (lw + 1) / 2;
            lh = (lh + 1) / 2;
        }

        // Create images
        let img_prev = create_mipmapped_image(device, mem_props, w, h)?;
        let img_curr = create_mipmapped_image(device, mem_props, w, h)?;
        let img_out = create_output_image(device, mem_props, w, h)?;

        let mv_levels: Vec<_> = level_dims
            .iter()
            .map(|&(lw, lh)| create_mv_image(device, mem_props, (lw + 15) / 16, (lh + 15) / 16))
            .collect::<Option<Vec<_>>>()?;

        let mv0_w = (w + 15) / 16;
        let mv0_h = (h + 15) / 16;
        let mv_filtered = create_mv_image(device, mem_props, mv0_w, mv0_h)?;
        let mv_prev = create_mv_image(device, mem_props, mv0_w, mv0_h)?;

        // Sampler: linear + mipmap for all textures
        let sampler_ci = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .max_lod(NUM_PYRAMID_LEVELS as f32);
        let sampler = unsafe { device.create_sampler(&sampler_ci, None).ok()? };

        // Descriptor set layout: 3 combined image samplers + 1 storage image
        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(3)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];
        let desc_layout_ci = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let desc_layout = unsafe { device.create_descriptor_set_layout(&desc_layout_ci, None).ok()? };

        // Pipeline layout: 24 bytes push constants (max across all 3 shaders)
        let pc_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(24);
        let pipe_layout_ci = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&desc_layout))
            .push_constant_ranges(std::slice::from_ref(&pc_range));
        let pipe_layout = unsafe { device.create_pipeline_layout(&pipe_layout_ci, None).ok()? };

        // Compile pipelines
        let flow_mod = create_shader_module(device, FG_FLOW_SPV)?;
        let filter_mod = create_shader_module(device, FG_MV_FILTER_SPV)?;
        let synth_mod = create_shader_module(device, FG_SYNTH_SPV)?;

        let entry = unsafe { CStr::from_bytes_with_nul_unchecked(b"main\0") };
        let make_ci = |module| {
            vk::ComputePipelineCreateInfo::default()
                .stage(
                    vk::PipelineShaderStageCreateInfo::default()
                        .stage(vk::ShaderStageFlags::COMPUTE)
                        .module(module)
                        .name(entry),
                )
                .layout(pipe_layout)
        };

        let pipelines = unsafe {
            device.create_compute_pipelines(
                vk::PipelineCache::null(),
                &[make_ci(flow_mod), make_ci(filter_mod), make_ci(synth_mod)],
                None,
            )
        };

        unsafe {
            device.destroy_shader_module(flow_mod, None);
            device.destroy_shader_module(filter_mod, None);
            device.destroy_shader_module(synth_mod, None);
        }

        let pipelines = match pipelines {
            Ok(p) => p,
            Err((_, e)) => {
                eprintln!("vk framegen: pipeline creation failed: {}", e);
                unsafe {
                    device.destroy_pipeline_layout(pipe_layout, None);
                    device.destroy_descriptor_set_layout(desc_layout, None);
                    device.destroy_sampler(sampler, None);
                }
                return None;
            }
        };
        let (flow_pipeline, filter_pipeline, synth_pipeline) =
            (pipelines[0], pipelines[1], pipelines[2]);

        // Descriptor pool: 6 sets × (3 samplers + 1 storage)
        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(18),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(6),
        ];
        let pool_ci = vk::DescriptorPoolCreateInfo::default()
            .max_sets(6)
            .pool_sizes(&pool_sizes);
        let desc_pool = unsafe { device.create_descriptor_pool(&pool_ci, None).ok()? };

        // Allocate 6 descriptor sets
        let layouts = [desc_layout; 6];
        let alloc_ci = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(desc_pool)
            .set_layouts(&layouts);
        let desc_sets = unsafe { device.allocate_descriptor_sets(&alloc_ci).ok()? };

        let fg = Self {
            device: device.clone(),
            width: w,
            height: h,
            img_prev,
            img_curr,
            img_out,
            mv_levels,
            mv_filtered,
            mv_prev,
            has_prev_mv: false,
            sampler,
            desc_layout,
            pipe_layout,
            flow_pipeline,
            filter_pipeline,
            synth_pipeline,
            desc_pool,
            desc_sets,
            level_dims,
            frame_count: 0,
            mv_valid: false,
            stats: FrameGenStats::default(),
        };

        fg.write_all_descriptors();

        if debug {
            eprintln!(
                "vk framegen: init {}x{}, {} pyramid levels",
                w, h, NUM_PYRAMID_LEVELS
            );
        }

        Some(fg)
    }

    pub fn can_generate(&self) -> bool {
        self.frame_count >= 2
    }

    pub fn output_image(&self) -> vk::Image {
        self.img_out.image
    }

    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    pub fn stats(&self) -> &FrameGenStats {
        &self.stats
    }

    // ── Command recording ──────────────────────────────────────────

    /// Record commands to capture frame_image for framegen.
    /// frame_image must be in TRANSFER_SRC_OPTIMAL when called.
    /// After return, frame_image is still TRANSFER_SRC_OPTIMAL.
    pub fn record_push_frame(
        &mut self,
        cb: vk::CommandBuffer,
        frame_image: vk::Image,
        w: u32,
        h: u32,
    ) {
        // Swap prev/curr
        std::mem::swap(&mut self.img_prev, &mut self.img_curr);

        unsafe {
            // Transition img_curr mip 0 to TRANSFER_DST
            transition_mip(
                &self.device, cb, self.img_curr.image, 0, 1,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::AccessFlags::empty(),
                vk::AccessFlags::TRANSFER_WRITE,
            );

            // Blit frame_image → img_curr mip 0
            let blit = vk::ImageBlit {
                src_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                src_offsets: [
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D { x: w as i32, y: h as i32, z: 1 },
                ],
                dst_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                dst_offsets: [
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D {
                        x: self.width as i32,
                        y: self.height as i32,
                        z: 1,
                    },
                ],
            };
            self.device.cmd_blit_image(
                cb,
                frame_image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                self.img_curr.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[blit],
                vk::Filter::LINEAR,
            );

            // Generate mipmaps for img_curr
            self.record_generate_mipmaps(cb, self.img_curr.image, self.width, self.height);
        }

        self.frame_count += 1;
        self.mv_valid = false;

        // Refresh only descriptors that reference prev/curr (swapped above).
        // Sets 0-3 (flow) bind prev+curr, set 5 (synth) binds prev+curr.
        // Set 4 (filter) only binds MV textures — no update needed.
        for level in 0..NUM_PYRAMID_LEVELS as usize {
            self.write_flow_descriptors(level, NUM_PYRAMID_LEVELS as usize);
        }
        self.write_synth_descriptors();
    }

    /// Record commands to generate a synthetic frame at parameter t.
    /// After return, img_out is in GENERAL layout.
    pub fn record_generate(
        &mut self,
        cb: vk::CommandBuffer,
        t: f32,
        mode: FrameGenMode,
        quality: FrameGenQuality,
    ) {
        if !self.can_generate() {
            self.stats.miss_count += 1;
            return;
        }

        if !self.mv_valid {
            self.record_compute_mvs(cb, quality);
            self.mv_valid = true;
        }

        self.record_synthesis(cb, t, mode);
        self.stats.synth_count += 1;
    }

    /// Hierarchical block matching + MV filtering.
    fn record_compute_mvs(&mut self, cb: vk::CommandBuffer, quality: FrameGenQuality) {
        let active_levels = quality.levels().min(NUM_PYRAMID_LEVELS as usize);
        let radii = quality.radii();

        unsafe {
            // Ensure prev/curr are readable
            transition_mip(
                &self.device, cb, self.img_prev.image, 0, NUM_PYRAMID_LEVELS,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::AccessFlags::TRANSFER_WRITE,
                vk::AccessFlags::SHADER_READ,
            );
            // img_curr already in SHADER_READ_ONLY after mipmap gen

            self.device.cmd_bind_pipeline(
                cb,
                vk::PipelineBindPoint::COMPUTE,
                self.flow_pipeline,
            );

            // Coarse to fine
            for level in (0..active_levels).rev() {
                let (lw, lh) = self.level_dims[level];
                let mv_w = (lw + 15) / 16;
                let mv_h = (lh + 15) / 16;

                // Transition MV output to GENERAL
                transition_mip(
                    &self.device, cb, self.mv_levels[level].image, 0, 1,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::GENERAL,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::AccessFlags::empty(),
                    vk::AccessFlags::SHADER_WRITE,
                );

                // If not coarsest, transition hint MV to SHADER_READ
                if level < active_levels - 1 {
                    transition_mip(
                        &self.device, cb, self.mv_levels[level + 1].image, 0, 1,
                        vk::ImageLayout::GENERAL,
                        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::AccessFlags::SHADER_WRITE,
                        vk::AccessFlags::SHADER_READ,
                    );
                }

                // Update descriptor set for this level
                self.write_flow_descriptors(level, active_levels);

                self.device.cmd_bind_descriptor_sets(
                    cb,
                    vk::PipelineBindPoint::COMPUTE,
                    self.pipe_layout,
                    0,
                    &[self.desc_sets[level]],
                    &[],
                );

                // Push constants: uSize(8) + uRadius(4) + uHasHint(4) + uLevel(4) = 20 bytes
                let mut pc = [0u32; 6];
                pc[0] = lw;
                pc[1] = lh;
                pc[2] = radii[level] as u32;
                pc[3] = if level < active_levels - 1 { 1 } else { 0 };
                pc[4] = level as u32;
                let pc_bytes = std::slice::from_raw_parts(pc.as_ptr() as *const u8, 24);
                self.device.cmd_push_constants(
                    cb,
                    self.pipe_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    pc_bytes,
                );

                self.device.cmd_dispatch(cb, mv_w, mv_h, 1);

                // Barrier between levels
                let barrier = vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ);
                self.device.cmd_pipeline_barrier(
                    cb,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[barrier],
                    &[],
                    &[],
                );
            }

            // MV filter pass
            let mv0_w = (self.width + 15) / 16;
            let mv0_h = (self.height + 15) / 16;

            // mv_levels[0] → SHADER_READ for filter input
            transition_mip(
                &self.device, cb, self.mv_levels[0].image, 0, 1,
                vk::ImageLayout::GENERAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::AccessFlags::SHADER_WRITE,
                vk::AccessFlags::SHADER_READ,
            );

            // mv_prev → SHADER_READ (if has_prev_mv, else content doesn't matter)
            transition_mip(
                &self.device, cb, self.mv_prev.image, 0, 1,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::AccessFlags::empty(),
                vk::AccessFlags::SHADER_READ,
            );

            // mv_filtered → GENERAL for write
            transition_mip(
                &self.device, cb, self.mv_filtered.image, 0, 1,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::GENERAL,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::AccessFlags::empty(),
                vk::AccessFlags::SHADER_WRITE,
            );

            self.write_filter_descriptors();

            self.device.cmd_bind_pipeline(
                cb,
                vk::PipelineBindPoint::COMPUTE,
                self.filter_pipeline,
            );
            self.device.cmd_bind_descriptor_sets(
                cb,
                vk::PipelineBindPoint::COMPUTE,
                self.pipe_layout,
                0,
                &[self.desc_sets[4]],
                &[],
            );

            let mut pc = [0u32; 6];
            pc[0] = mv0_w;
            pc[1] = mv0_h;
            pc[2] = if self.has_prev_mv { 1 } else { 0 };
            let pc_bytes = std::slice::from_raw_parts(pc.as_ptr() as *const u8, 24);
            self.device.cmd_push_constants(
                cb,
                self.pipe_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                pc_bytes,
            );

            let fg_x = (mv0_w + 7) / 8;
            let fg_y = (mv0_h + 7) / 8;
            self.device.cmd_dispatch(cb, fg_x, fg_y, 1);

            // Barrier after filter
            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ);
            self.device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
        }

        // Swap filtered → prev for temporal dampening next frame
        std::mem::swap(&mut self.mv_filtered, &mut self.mv_prev);
        self.has_prev_mv = true;
    }

    /// Warp synthesis — uses cached MVs to produce output at parameter t.
    fn record_synthesis(&mut self, cb: vk::CommandBuffer, t: f32, mode: FrameGenMode) {
        unsafe {
            // mv_prev (holds filtered result after swap) → SHADER_READ
            transition_mip(
                &self.device, cb, self.mv_prev.image, 0, 1,
                vk::ImageLayout::GENERAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::AccessFlags::SHADER_WRITE,
                vk::AccessFlags::SHADER_READ,
            );

            // img_out → GENERAL for storage write
            transition_mip(
                &self.device, cb, self.img_out.image, 0, 1,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::GENERAL,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::AccessFlags::empty(),
                vk::AccessFlags::SHADER_WRITE,
            );

            self.write_synth_descriptors();

            self.device.cmd_bind_pipeline(
                cb,
                vk::PipelineBindPoint::COMPUTE,
                self.synth_pipeline,
            );
            self.device.cmd_bind_descriptor_sets(
                cb,
                vk::PipelineBindPoint::COMPUTE,
                self.pipe_layout,
                0,
                &[self.desc_sets[5]],
                &[],
            );

            let mode_int = match mode {
                FrameGenMode::Extrapolate => 0u32,
                FrameGenMode::Interpolate => 1u32,
                _ => return,
            };

            // Push constants: uSize(8) + uT(4) + uMode(4) = 16 bytes, padded to 24
            let mut pc = [0u32; 6];
            pc[0] = self.width;
            pc[1] = self.height;
            pc[2] = t.to_bits();
            pc[3] = mode_int;
            let pc_bytes = std::slice::from_raw_parts(pc.as_ptr() as *const u8, 24);
            self.device.cmd_push_constants(
                cb,
                self.pipe_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                pc_bytes,
            );

            let groups_x = (self.width + 7) / 8;
            let groups_y = (self.height + 7) / 8;
            self.device.cmd_dispatch(cb, groups_x, groups_y, 1);
        }
    }

    /// Record mipmap generation for img (mip 0 must be in TRANSFER_DST).
    /// After return, all mip levels are in SHADER_READ_ONLY.
    unsafe fn record_generate_mipmaps(
        &self,
        cb: vk::CommandBuffer,
        image: vk::Image,
        w: u32,
        h: u32,
    ) {
        let levels = NUM_PYRAMID_LEVELS;
        let mut mip_w = w as i32;
        let mut mip_h = h as i32;

        for i in 1..levels {
            // Mip i-1: TRANSFER_DST → TRANSFER_SRC
            transition_mip(
                &self.device, cb, image, i - 1, 1,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::TRANSFER,
                vk::AccessFlags::TRANSFER_WRITE,
                vk::AccessFlags::TRANSFER_READ,
            );

            // Mip i: UNDEFINED → TRANSFER_DST
            transition_mip(
                &self.device, cb, image, i, 1,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::TRANSFER,
                vk::AccessFlags::empty(),
                vk::AccessFlags::TRANSFER_WRITE,
            );

            let new_w = (mip_w / 2).max(1);
            let new_h = (mip_h / 2).max(1);

            let blit = vk::ImageBlit {
                src_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: i - 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                src_offsets: [
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D { x: mip_w, y: mip_h, z: 1 },
                ],
                dst_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: i,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                dst_offsets: [
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D { x: new_w, y: new_h, z: 1 },
                ],
            };

            self.device.cmd_blit_image(
                cb,
                image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[blit],
                vk::Filter::LINEAR,
            );

            mip_w = new_w;
            mip_h = new_h;
        }

        // Transition all levels to SHADER_READ_ONLY
        let mut barriers = Vec::with_capacity(levels as usize);
        for i in 0..levels {
            let old_layout = if i < levels - 1 {
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL
            } else {
                vk::ImageLayout::TRANSFER_DST_OPTIMAL
            };
            barriers.push(
                vk::ImageMemoryBarrier::default()
                    .old_layout(old_layout)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(image)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: i,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::TRANSFER_READ)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ),
            );
        }
        self.device.cmd_pipeline_barrier(
            cb,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &barriers,
        );
    }

    // ── Descriptor writes ──────────────────────────────────────────

    fn write_all_descriptors(&self) {
        // Block match level descriptors (sets 0..3)
        for level in 0..NUM_PYRAMID_LEVELS as usize {
            self.write_flow_descriptors(level, NUM_PYRAMID_LEVELS as usize);
        }
        self.write_filter_descriptors();
        self.write_synth_descriptors();
    }

    fn write_flow_descriptors(&self, level: usize, active_levels: usize) {
        let prev_info = vk::DescriptorImageInfo::default()
            .sampler(self.sampler)
            .image_view(self.img_prev.view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

        let curr_info = vk::DescriptorImageInfo::default()
            .sampler(self.sampler)
            .image_view(self.img_curr.view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

        // Hint MV: next coarser level, or self if coarsest (unused but must be valid)
        let hint_level = if level < active_levels - 1 {
            level + 1
        } else {
            level
        };
        let hint_info = vk::DescriptorImageInfo::default()
            .sampler(self.sampler)
            .image_view(self.mv_levels[hint_level].view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

        let mv_out_info = vk::DescriptorImageInfo::default()
            .image_view(self.mv_levels[level].view)
            .image_layout(vk::ImageLayout::GENERAL);

        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(self.desc_sets[level])
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(&prev_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(self.desc_sets[level])
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(&curr_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(self.desc_sets[level])
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(&hint_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(self.desc_sets[level])
                .dst_binding(3)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(std::slice::from_ref(&mv_out_info)),
        ];
        unsafe { self.device.update_descriptor_sets(&writes, &[]) };
    }

    fn write_filter_descriptors(&self) {
        let mv_in_info = vk::DescriptorImageInfo::default()
            .sampler(self.sampler)
            .image_view(self.mv_levels[0].view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

        let mv_prev_info = vk::DescriptorImageInfo::default()
            .sampler(self.sampler)
            .image_view(self.mv_prev.view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

        // Binding 2: dummy (unused by filter shader but layout requires it)
        let dummy_info = vk::DescriptorImageInfo::default()
            .sampler(self.sampler)
            .image_view(self.mv_levels[0].view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

        let mv_out_info = vk::DescriptorImageInfo::default()
            .image_view(self.mv_filtered.view)
            .image_layout(vk::ImageLayout::GENERAL);

        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(self.desc_sets[4])
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(&mv_in_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(self.desc_sets[4])
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(&mv_prev_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(self.desc_sets[4])
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(&dummy_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(self.desc_sets[4])
                .dst_binding(3)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(std::slice::from_ref(&mv_out_info)),
        ];
        unsafe { self.device.update_descriptor_sets(&writes, &[]) };
    }

    fn write_synth_descriptors(&self) {
        let prev_info = vk::DescriptorImageInfo::default()
            .sampler(self.sampler)
            .image_view(self.img_prev.view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

        let curr_info = vk::DescriptorImageInfo::default()
            .sampler(self.sampler)
            .image_view(self.img_curr.view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

        // After MV swap, mv_prev holds the filtered result
        let mv_info = vk::DescriptorImageInfo::default()
            .sampler(self.sampler)
            .image_view(self.mv_prev.view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

        let out_info = vk::DescriptorImageInfo::default()
            .image_view(self.img_out.view)
            .image_layout(vk::ImageLayout::GENERAL);

        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(self.desc_sets[5])
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(&prev_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(self.desc_sets[5])
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(&curr_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(self.desc_sets[5])
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(&mv_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(self.desc_sets[5])
                .dst_binding(3)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(std::slice::from_ref(&out_info)),
        ];
        unsafe { self.device.update_descriptor_sets(&writes, &[]) };
    }
}

impl Drop for VkFrameGen {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();

            destroy_img_res(&self.device, &self.img_prev);
            destroy_img_res(&self.device, &self.img_curr);
            destroy_img_res(&self.device, &self.img_out);
            for mv in &self.mv_levels {
                destroy_img_res(&self.device, mv);
            }
            destroy_img_res(&self.device, &self.mv_filtered);
            destroy_img_res(&self.device, &self.mv_prev);

            self.device.destroy_sampler(self.sampler, None);
            self.device.destroy_descriptor_pool(self.desc_pool, None);
            self.device.destroy_pipeline(self.flow_pipeline, None);
            self.device.destroy_pipeline(self.filter_pipeline, None);
            self.device.destroy_pipeline(self.synth_pipeline, None);
            self.device.destroy_pipeline_layout(self.pipe_layout, None);
            self.device
                .destroy_descriptor_set_layout(self.desc_layout, None);
        }
    }
}

// ── Helpers ─────────────────────────────────────────────────────────

unsafe fn destroy_img_res(device: &ash::Device, res: &ImgRes) {
    device.destroy_image_view(res.view, None);
    device.destroy_image(res.image, None);
    device.free_memory(res.memory, None);
}

fn create_shader_module(device: &ash::Device, spv: &[u8]) -> Option<vk::ShaderModule> {
    assert!(spv.len() % 4 == 0, "SPIR-V not aligned");
    let code = unsafe { std::slice::from_raw_parts(spv.as_ptr() as *const u32, spv.len() / 4) };
    let ci = vk::ShaderModuleCreateInfo::default().code(code);
    unsafe { device.create_shader_module(&ci, None).ok() }
}

fn create_mipmapped_image(
    device: &ash::Device,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    w: u32,
    h: u32,
) -> Option<ImgRes> {
    let ci = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .extent(vk::Extent3D {
            width: w,
            height: h,
            depth: 1,
        })
        .mip_levels(NUM_PYRAMID_LEVELS)
        .array_layers(1)
        .format(vk::Format::R8G8B8A8_UNORM)
        .tiling(vk::ImageTiling::OPTIMAL)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .usage(
            vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST,
        )
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .samples(vk::SampleCountFlags::TYPE_1);

    let image = unsafe { device.create_image(&ci, None).ok()? };
    let memory = alloc_bind_image(device, mem_props, image)?;

    let view_ci = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(vk::Format::R8G8B8A8_UNORM)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: NUM_PYRAMID_LEVELS,
            base_array_layer: 0,
            layer_count: 1,
        });
    let view = unsafe { device.create_image_view(&view_ci, None).ok()? };

    Some(ImgRes { image, memory, view })
}

fn create_mv_image(
    device: &ash::Device,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    w: u32,
    h: u32,
) -> Option<ImgRes> {
    let ci = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .extent(vk::Extent3D {
            width: w,
            height: h,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .format(vk::Format::R16G16B16A16_SFLOAT)
        .tiling(vk::ImageTiling::OPTIMAL)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .usage(
            vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::STORAGE
                | vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST,
        )
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .samples(vk::SampleCountFlags::TYPE_1);

    let image = unsafe { device.create_image(&ci, None).ok()? };
    let memory = alloc_bind_image(device, mem_props, image)?;

    let view_ci = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(vk::Format::R16G16B16A16_SFLOAT)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        });
    let view = unsafe { device.create_image_view(&view_ci, None).ok()? };

    Some(ImgRes { image, memory, view })
}

fn create_output_image(
    device: &ash::Device,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    w: u32,
    h: u32,
) -> Option<ImgRes> {
    let ci = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .extent(vk::Extent3D {
            width: w,
            height: h,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .format(vk::Format::R8G8B8A8_UNORM)
        .tiling(vk::ImageTiling::OPTIMAL)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .usage(
            vk::ImageUsageFlags::STORAGE
                | vk::ImageUsageFlags::TRANSFER_SRC,
        )
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .samples(vk::SampleCountFlags::TYPE_1);

    let image = unsafe { device.create_image(&ci, None).ok()? };
    let memory = alloc_bind_image(device, mem_props, image)?;

    let view_ci = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(vk::Format::R8G8B8A8_UNORM)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        });
    let view = unsafe { device.create_image_view(&view_ci, None).ok()? };

    Some(ImgRes { image, memory, view })
}

fn alloc_bind_image(
    device: &ash::Device,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    image: vk::Image,
) -> Option<vk::DeviceMemory> {
    let reqs = unsafe { device.get_image_memory_requirements(image) };
    let mem_type = find_memory_type(
        mem_props,
        reqs.memory_type_bits,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;

    let alloc = vk::MemoryAllocateInfo::default()
        .allocation_size(reqs.size)
        .memory_type_index(mem_type);

    let memory = unsafe { device.allocate_memory(&alloc, None).ok()? };
    unsafe { device.bind_image_memory(image, memory, 0).ok()? };
    Some(memory)
}

fn find_memory_type(
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    type_filter: u32,
    properties: vk::MemoryPropertyFlags,
) -> Option<u32> {
    for i in 0..mem_props.memory_type_count {
        if (type_filter & (1 << i)) != 0
            && mem_props.memory_types[i as usize]
                .property_flags
                .contains(properties)
        {
            return Some(i);
        }
    }
    None
}

unsafe fn transition_mip(
    device: &ash::Device,
    cb: vk::CommandBuffer,
    image: vk::Image,
    base_mip: u32,
    level_count: u32,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
    src_stage: vk::PipelineStageFlags,
    dst_stage: vk::PipelineStageFlags,
    src_access: vk::AccessFlags,
    dst_access: vk::AccessFlags,
) {
    let barrier = vk::ImageMemoryBarrier::default()
        .old_layout(old_layout)
        .new_layout(new_layout)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: base_mip,
            level_count,
            base_array_layer: 0,
            layer_count: 1,
        })
        .src_access_mask(src_access)
        .dst_access_mask(dst_access);

    device.cmd_pipeline_barrier(
        cb,
        src_stage,
        dst_stage,
        vk::DependencyFlags::empty(),
        &[],
        &[],
        &[barrier],
    );
}
