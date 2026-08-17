use ash::vk;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;

use crate::capture::{V4L2_PIX_FMT_NV12, V4L2_PIX_FMT_YUYV, V4L2_PIX_FMT_UYVY, V4L2_PIX_FMT_XRGB32, V4L2_PIX_FMT_P010, PIXFMT_RGB24};
use crate::gl_renderer::{fsr_easu_con, fsr_rcas_con};

const CAS_SPV: &[u8] = include_bytes!("shaders/cas.spv");
const FSR_EASU_SPV: &[u8] = include_bytes!("shaders/fsr_easu.spv");
const FSR_RCAS_SPV: &[u8] = include_bytes!("shaders/fsr_rcas.spv");
const OSD_BLEND_SPV: &[u8] = include_bytes!("shaders/osd_blend.spv");

extern "C" {
    fn SDL_Vulkan_GetInstanceExtensions(
        window: *mut sdl2_sys::SDL_Window,
        pCount: *mut u32,
        pNames: *mut *const c_char,
    ) -> sdl2_sys::SDL_bool;

    fn SDL_Vulkan_CreateSurface(
        window: *mut sdl2_sys::SDL_Window,
        instance: vk::Instance,
        surface: *mut vk::SurfaceKHR,
    ) -> sdl2_sys::SDL_bool;

    fn SDL_Vulkan_GetDrawableSize(
        window: *mut sdl2_sys::SDL_Window,
        w: *mut i32,
        h: *mut i32,
    );
}

/// Minimal bitmap font: 8×8 glyphs for ASCII 32..127 (same data as OSD).
mod font {
    // 8×8 bitmap font covering ASCII 32–126 (space through tilde).
    // Each entry is 8 bytes — one byte per row, MSB-left.
    pub const GLYPH_W: u32 = 8;
    pub const GLYPH_H: u32 = 8;
    pub const FIRST_CHAR: u8 = 32;
    pub const LAST_CHAR: u8 = 126;

    // Compact font data — each glyph = 8 bytes (rows top-to-bottom, MSB = left pixel).
    pub static DATA: &[u8] = include_bytes!("font8x8.bin");

    pub fn glyph(ch: u8) -> &'static [u8] {
        if ch < FIRST_CHAR || ch > LAST_CHAR {
            return &DATA[0..8]; // space
        }
        let idx = (ch - FIRST_CHAR) as usize * 8;
        &DATA[idx..idx + 8]
    }
}

pub struct VkRenderer {
    _entry: ash::Entry,
    instance: ash::Instance,
    surface: vk::SurfaceKHR,
    surface_fn: ash::khr::surface::Instance,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    graphics_queue: vk::Queue,
    present_queue: vk::Queue,
    _gfx_family: u32,
    _present_family: u32,
    swapchain_fn: ash::khr::swapchain::Device,
    swapchain: vk::SwapchainKHR,
    swapchain_images: Vec<vk::Image>,
    swapchain_format: vk::Format,
    swapchain_extent: vk::Extent2D,
    raw_window: *mut sdl2_sys::SDL_Window,

    // Frame upload
    staging_buf: vk::Buffer,
    staging_mem: vk::DeviceMemory,
    _staging_size: vk::DeviceSize,
    frame_image: vk::Image,
    frame_mem: vk::DeviceMemory,
    _frame_extent: vk::Extent2D,

    // OSD overlay
    osd_image: vk::Image,
    osd_mem: vk::DeviceMemory,
    osd_staging_buf: vk::Buffer,
    osd_staging_mem: vk::DeviceMemory,
    osd_cpu: Vec<u8>,
    osd_extent: vk::Extent2D,
    osd_dirty: bool,       // OSD has content to composite this frame
    osd_uploaded: bool,     // OSD CPU buffer has been uploaded to GPU (skip re-upload)
    osd_staging_ptr: *mut u8,
    osd_regions: Vec<vk::ImageBlit>,

    // Command recording
    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,

    // Sync
    image_available_sem: vk::Semaphore,
    render_finished_sem: vk::Semaphore,
    in_flight_fence: vk::Fence,

    // State
    frame_w: u32,
    frame_h: u32,
    pixfmt: u32,
    present_mode: vk::PresentModeKHR,
    mailbox: bool,
    needs_recreate: bool,
    pub aspect_mode: crate::config::AspectMode,

    // Temporary RGB buffer for CPU conversion
    rgb_buf: Vec<u8>,
    // Persistent staging memory mapping (avoids map/unmap per frame)
    staging_ptr: *mut u8,
    // Cached brightness/contrast/gamma LUT
    cached_lut: Option<(f32, f32, f32, [u8; 256])>,

    // ── Compute scaling pipeline ──────────────────────────────────────
    pub scale_mode: crate::gl_renderer::ScaleMode,
    sharpness: f32,
    compute_desc_layout: vk::DescriptorSetLayout,
    compute_pipe_layout: vk::PipelineLayout,
    compute_cas_pipeline: vk::Pipeline,
    compute_easu_pipeline: vk::Pipeline,
    compute_rcas_pipeline: vk::Pipeline,
    compute_desc_pool: vk::DescriptorPool,
    // Descriptor sets: [0] = frame→A, [1] = A→B (for FSR two-pass)
    compute_desc_sets: Vec<vk::DescriptorSet>,
    frame_image_view: vk::ImageView,
    compute_sampler: vk::Sampler,
    // Intermediate image A (output res) — CAS output or EASU output
    compute_a: vk::Image,
    compute_a_mem: vk::DeviceMemory,
    compute_a_view: vk::ImageView,
    // Intermediate image B (output res) — RCAS output
    compute_b: vk::Image,
    compute_b_mem: vk::DeviceMemory,
    compute_b_view: vk::ImageView,
    compute_extent: vk::Extent2D,

    // OSD alpha-blend composite pipeline
    osd_blend_pipeline: vk::Pipeline,
    osd_image_view: vk::ImageView,
    // Descriptor sets: [0] = OSD→compute_a, [1] = OSD→compute_b
    osd_blend_desc_sets: Vec<vk::DescriptorSet>,

    // ── Frame generation ──────────────────────────────────────────────
    fg: Option<crate::framegen::vk::VkFrameGen>,
}

impl Drop for VkRenderer {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();

            // Drop framegen before device resources
            self.fg = None;

            self.device.destroy_semaphore(self.image_available_sem, None);
            self.device.destroy_semaphore(self.render_finished_sem, None);
            self.device.destroy_fence(self.in_flight_fence, None);
            self.device.destroy_command_pool(self.command_pool, None);

            self.device.unmap_memory(self.osd_staging_mem);
            self.device.destroy_buffer(self.osd_staging_buf, None);
            self.device.free_memory(self.osd_staging_mem, None);
            self.device.destroy_image(self.osd_image, None);
            self.device.free_memory(self.osd_mem, None);

            self.device.unmap_memory(self.staging_mem);
            self.device.destroy_buffer(self.staging_buf, None);
            self.device.free_memory(self.staging_mem, None);
            self.device.destroy_image_view(self.frame_image_view, None);
            self.device.destroy_image(self.frame_image, None);
            self.device.free_memory(self.frame_mem, None);

            // Compute resources
            self.device.destroy_image_view(self.compute_a_view, None);
            self.device.destroy_image(self.compute_a, None);
            self.device.free_memory(self.compute_a_mem, None);
            self.device.destroy_image_view(self.compute_b_view, None);
            self.device.destroy_image(self.compute_b, None);
            self.device.free_memory(self.compute_b_mem, None);
            self.device.destroy_sampler(self.compute_sampler, None);
            self.device.destroy_descriptor_pool(self.compute_desc_pool, None);
            self.device.destroy_pipeline(self.compute_cas_pipeline, None);
            self.device.destroy_pipeline(self.compute_easu_pipeline, None);
            self.device.destroy_pipeline(self.compute_rcas_pipeline, None);
            self.device.destroy_pipeline(self.osd_blend_pipeline, None);
            self.device.destroy_image_view(self.osd_image_view, None);
            self.device.destroy_pipeline_layout(self.compute_pipe_layout, None);
            self.device.destroy_descriptor_set_layout(self.compute_desc_layout, None);

            self.swapchain_fn.destroy_swapchain(self.swapchain, None);
            self.surface_fn.destroy_surface(self.surface, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

impl VkRenderer {
    /// Create a new Vulkan renderer on the given SDL window.
    /// The window MUST have been created with SDL_WINDOW_VULKAN.
    pub fn new(
        window: &sdl2::video::Window,
        frame_w: u32,
        frame_h: u32,
        pixfmt: u32,
        preferred_present_mode: Option<vk::PresentModeKHR>,
        debug: bool,
    ) -> anyhow::Result<Self> {
        let raw_window = window.raw();

        // --- Entry + Instance ---
        #[cfg(target_os = "macos")]
        let entry = {
            // ash::Entry::load() may not find MoltenVK's libvulkan in Homebrew paths.
            // Try known locations before falling back to default search.
            let paths = [
                "/opt/homebrew/lib/libvulkan.1.dylib",  // Apple Silicon
                "/usr/local/lib/libvulkan.1.dylib",      // Intel
                "/opt/homebrew/lib/libvulkan.dylib",
                "/usr/local/lib/libvulkan.dylib",
            ];
            let mut loaded = None;
            for path in &paths {
                if std::path::Path::new(path).exists() {
                    if let Ok(e) = unsafe { ash::Entry::load_from(path) } {
                        loaded = Some(e);
                        break;
                    }
                }
            }
            match loaded {
                Some(e) => e,
                None => unsafe { ash::Entry::load() }
                    .map_err(|e| anyhow::anyhow!("vulkan: failed to load: {}", e))?,
            }
        };
        #[cfg(not(target_os = "macos"))]
        let entry = unsafe { ash::Entry::load() }
            .map_err(|e| anyhow::anyhow!("vulkan: failed to load: {}", e))?;

        // Get required extensions from SDL
        let extensions = unsafe {
            let mut count: u32 = 0;
            if SDL_Vulkan_GetInstanceExtensions(raw_window, &mut count, std::ptr::null_mut())
                == sdl2_sys::SDL_bool::SDL_FALSE
            {
                anyhow::bail!("vulkan: SDL_Vulkan_GetInstanceExtensions count failed");
            }
            let mut names = vec![std::ptr::null::<c_char>(); count as usize];
            if SDL_Vulkan_GetInstanceExtensions(raw_window, &mut count, names.as_mut_ptr())
                == sdl2_sys::SDL_bool::SDL_FALSE
            {
                anyhow::bail!("vulkan: SDL_Vulkan_GetInstanceExtensions names failed");
            }
            names
        };

        // MoltenVK requires portability enumeration to expose non-conformant devices
        #[cfg(target_os = "macos")]
        let _portability_ext = CString::new("VK_KHR_portability_enumeration").unwrap();
        #[cfg(target_os = "macos")]
        extensions.push(_portability_ext.as_ptr());

        let app_name = CString::new("capview").unwrap();
        let engine_name = CString::new("capview-vk").unwrap();
        let app_info = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .application_version(vk::make_api_version(0, 0, 1, 0))
            .engine_name(&engine_name)
            .engine_version(vk::make_api_version(0, 0, 1, 0))
            .api_version(vk::API_VERSION_1_0);

        let create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&extensions);
        #[cfg(target_os = "macos")]
        {
            create_info.flags |= vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR;
        }

        let instance = unsafe { entry.create_instance(&create_info, None) }
            .map_err(|e| anyhow::anyhow!("vulkan: create_instance: {}", e))?;

        // --- Surface ---
        let surface = unsafe {
            let mut s = vk::SurfaceKHR::null();
            if SDL_Vulkan_CreateSurface(raw_window, instance.handle(), &mut s)
                == sdl2_sys::SDL_bool::SDL_FALSE
            {
                instance.destroy_instance(None);
                anyhow::bail!("vulkan: SDL_Vulkan_CreateSurface failed");
            }
            s
        };

        let surface_fn = ash::khr::surface::Instance::new(&entry, &instance);

        // --- Physical device ---
        let (physical_device, gfx_family, present_family) = unsafe {
            let devices = instance.enumerate_physical_devices()
                .map_err(|e| anyhow::anyhow!("vulkan: enumerate_physical_devices: {}", e))?;
            if devices.is_empty() {
                anyhow::bail!("vulkan: no physical devices found");
            }
            pick_physical_device(&instance, &surface_fn, surface, &devices, debug)?
        };

        if debug {
            let props = unsafe { instance.get_physical_device_properties(physical_device) };
            let name = unsafe { CStr::from_ptr(props.device_name.as_ptr()) };
            eprintln!("vulkan: using device: {:?} (type {:?})", name, props.device_type);
        }

        // --- Logical device + queues ---
        let unique_families: Vec<u32> = if gfx_family == present_family {
            vec![gfx_family]
        } else {
            vec![gfx_family, present_family]
        };

        let queue_priorities = [1.0_f32];
        let queue_cis: Vec<vk::DeviceQueueCreateInfo> = unique_families
            .iter()
            .map(|&fam| {
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(fam)
                    .queue_priorities(&queue_priorities)
            })
            .collect();

        let swapchain_ext = ash::khr::swapchain::NAME;
        #[cfg(target_os = "macos")]
        let portability_subset_ext = CString::new("VK_KHR_portability_subset").unwrap();
        #[cfg(target_os = "macos")]
        let device_extensions = [swapchain_ext.as_ptr(), portability_subset_ext.as_ptr()];
        #[cfg(not(target_os = "macos"))]
        let device_extensions = [swapchain_ext.as_ptr()];

        let device_ci = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_cis)
            .enabled_extension_names(&device_extensions);

        let device = unsafe { instance.create_device(physical_device, &device_ci, None) }
            .map_err(|e| anyhow::anyhow!("vulkan: create_device: {}", e))?;

        let graphics_queue = unsafe { device.get_device_queue(gfx_family, 0) };
        let present_queue = unsafe { device.get_device_queue(present_family, 0) };

        // --- Swapchain ---
        let swapchain_fn = ash::khr::swapchain::Device::new(&instance, &device);

        let (win_w, win_h) = unsafe {
            let mut w: i32 = 0;
            let mut h: i32 = 0;
            SDL_Vulkan_GetDrawableSize(raw_window, &mut w, &mut h);
            (w as u32, h as u32)
        };

        let (swapchain, swapchain_images, swapchain_format, swapchain_extent, present_mode) =
            create_swapchain(
                &surface_fn, &swapchain_fn, &device, physical_device,
                surface, win_w, win_h, vk::SwapchainKHR::null(), preferred_present_mode, debug,
            )?;

        let mailbox = present_mode == vk::PresentModeKHR::MAILBOX;
        if debug {
            eprintln!("vulkan: present mode: {:?} (mailbox={})", present_mode, mailbox);
            eprintln!("vulkan: swapchain images: {}, format: {:?}, extent: {}x{}",
                swapchain_images.len(), swapchain_format,
                swapchain_extent.width, swapchain_extent.height);
        }

        // --- Command pool ---
        let pool_ci = vk::CommandPoolCreateInfo::default()
            .queue_family_index(gfx_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let command_pool = unsafe { device.create_command_pool(&pool_ci, None) }
            .map_err(|e| anyhow::anyhow!("vulkan: create_command_pool: {}", e))?;

        let alloc_ci = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let command_buffers = unsafe { device.allocate_command_buffers(&alloc_ci) }
            .map_err(|e| anyhow::anyhow!("vulkan: allocate_command_buffers: {}", e))?;

        // --- Sync objects ---
        let sem_ci = vk::SemaphoreCreateInfo::default();
        let fence_ci = vk::FenceCreateInfo::default()
            .flags(vk::FenceCreateFlags::SIGNALED);
        let image_available_sem = unsafe { device.create_semaphore(&sem_ci, None)? };
        let render_finished_sem = unsafe { device.create_semaphore(&sem_ci, None)? };
        let in_flight_fence = unsafe { device.create_fence(&fence_ci, None)? };

        // --- Frame upload resources ---
        let mem_props = unsafe { instance.get_physical_device_memory_properties(physical_device) };

        // Staging buffer (host-visible, for CPU → GPU transfer)
        let rgba_size = (frame_w * frame_h * 4) as vk::DeviceSize;
        let (staging_buf, staging_mem) = create_buffer(
            &device, &mem_props, rgba_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        // Persistently map staging buffer (HOST_COHERENT — no flush needed)
        let staging_ptr = unsafe {
            device.map_memory(staging_mem, 0, rgba_size, vk::MemoryMapFlags::empty())
        }? as *mut u8;

        // Device-local frame image (TRANSFER_DST + TRANSFER_SRC for blit + SAMPLED for compute)
        let (frame_image, frame_mem) = create_image(
            &device, &mem_props,
            frame_w, frame_h,
            vk::Format::B8G8R8A8_UNORM,
            vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::SAMPLED,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        // Transition frame image to TRANSFER_DST_OPTIMAL once
        {
            let cb = command_buffers[0];
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            unsafe {
                device.begin_command_buffer(cb, &begin)?;
                transition_image_layout(&device, cb, frame_image,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::AccessFlags::empty(),
                    vk::AccessFlags::TRANSFER_WRITE,
                );
                device.end_command_buffer(cb)?;
                let submit = vk::SubmitInfo::default()
                    .command_buffers(&command_buffers);
                // Wait for initial fence (created signaled)
                device.wait_for_fences(&[in_flight_fence], true, u64::MAX)?;
                device.reset_fences(&[in_flight_fence])?;
                device.queue_submit(graphics_queue, &[submit], in_flight_fence)?;
                device.wait_for_fences(&[in_flight_fence], true, u64::MAX)?;
            }
        }

        // --- OSD overlay resources ---
        let osd_w = win_w.max(1);
        let osd_h = win_h.max(1);
        let osd_rgba_size = (osd_w * osd_h * 4) as vk::DeviceSize;
        let (osd_staging_buf, osd_staging_mem) = create_buffer(
            &device, &mem_props, osd_rgba_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        let (osd_image, osd_mem) = create_image(
            &device, &mem_props,
            osd_w, osd_h,
            vk::Format::B8G8R8A8_UNORM,
            vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        let osd_staging_ptr = unsafe {
            device.map_memory(osd_staging_mem, 0, osd_rgba_size, vk::MemoryMapFlags::empty())
        }? as *mut u8;

        // --- Compute scaling pipeline ---
        let (compute_desc_layout, compute_pipe_layout, compute_cas_pipeline,
             compute_easu_pipeline, compute_rcas_pipeline, compute_desc_pool,
             compute_desc_sets, frame_image_view, compute_sampler,
             compute_a, compute_a_mem, compute_a_view,
             compute_b, compute_b_mem, compute_b_view)
            = create_compute_pipeline(
                &device, &mem_props, frame_image, frame_w, frame_h,
                swapchain_extent.width, swapchain_extent.height,
            )?;
        let compute_extent = vk::Extent2D { width: swapchain_extent.width, height: swapchain_extent.height };

        // --- OSD alpha-blend composite pipeline ---
        let (osd_blend_pipeline, osd_image_view, osd_blend_desc_sets) = create_osd_blend_pipeline(
            &device, compute_desc_layout, compute_pipe_layout, compute_desc_pool,
            compute_sampler, osd_image, compute_a_view, compute_b_view,
        )?;

        eprintln!("vulkan: renderer initialized ({}x{} frame, {}x{} window, compute scaling ready)",
            frame_w, frame_h, win_w, win_h);

        Ok(VkRenderer {
            _entry: entry,
            instance,
            surface,
            surface_fn,
            physical_device,
            device,
            graphics_queue,
            present_queue,
            _gfx_family: gfx_family,
            _present_family: present_family,
            swapchain_fn,
            swapchain,
            swapchain_images,
            swapchain_format,
            swapchain_extent,
            raw_window,
            staging_buf,
            staging_mem,
            _staging_size: rgba_size,
            frame_image,
            frame_mem,
            _frame_extent: vk::Extent2D { width: frame_w, height: frame_h },
            osd_image,
            osd_mem,
            osd_staging_buf,
            osd_staging_mem,
            osd_cpu: vec![0u8; (osd_w * osd_h * 4) as usize],
            osd_extent: vk::Extent2D { width: osd_w, height: osd_h },
            osd_dirty: false,
            osd_uploaded: false,
            osd_staging_ptr,
            osd_regions: Vec::new(),
            command_pool,
            command_buffers,
            image_available_sem,
            render_finished_sem,
            in_flight_fence,
            frame_w,
            frame_h,
            pixfmt,
            present_mode,
            mailbox,
            needs_recreate: false,
            aspect_mode: crate::config::AspectMode::Preserve,
            rgb_buf: vec![0u8; (frame_w * frame_h * 4) as usize],
            staging_ptr,
            cached_lut: None,
            scale_mode: crate::gl_renderer::ScaleMode::Bilinear,
            sharpness: 0.5,
            compute_desc_layout,
            compute_pipe_layout,
            compute_cas_pipeline,
            compute_easu_pipeline,
            compute_rcas_pipeline,
            compute_desc_pool,
            compute_desc_sets,
            frame_image_view,
            compute_sampler,
            compute_a,
            compute_a_mem,
            compute_a_view,
            compute_b,
            compute_b_mem,
            compute_b_view,
            compute_extent,
            osd_blend_pipeline,
            osd_image_view,
            osd_blend_desc_sets,
            fg: None,
        })
    }

    /// Returns true if MAILBOX present mode is active.
    pub fn is_mailbox(&self) -> bool {
        self.mailbox
    }

    /// Returns the active present mode.
    pub fn present_mode(&self) -> vk::PresentModeKHR {
        self.present_mode
    }

    /// Convert a config VkPresentMode to a Vulkan PresentModeKHR.
    pub fn config_to_vk_present_mode(mode: crate::config::VkPresentMode) -> vk::PresentModeKHR {
        match mode {
            crate::config::VkPresentMode::Mailbox => vk::PresentModeKHR::MAILBOX,
            crate::config::VkPresentMode::Immediate => vk::PresentModeKHR::IMMEDIATE,
            crate::config::VkPresentMode::Fifo => vk::PresentModeKHR::FIFO,
        }
    }

    /// Returns true if IMMEDIATE present mode is available (blocked on Wayland).
    pub fn immediate_available() -> bool {
        !std::env::var("WAYLAND_DISPLAY").is_ok()
    }

    pub fn set_scale_mode(&mut self, mode: crate::gl_renderer::ScaleMode) {
        self.scale_mode = mode;
    }

    pub fn set_sharpness(&mut self, level: u32) {
        self.sharpness = (level.min(10) as f32) / 10.0;
    }

    fn use_compute(&self) -> bool {
        use crate::gl_renderer::ScaleMode;
        matches!(self.scale_mode, ScaleMode::Cas | ScaleMode::Fsr | ScaleMode::IntegerFsr)
    }

    // ── Frame generation ──────────────────────────────────────────

    pub fn enable_framegen(&mut self, debug: bool) -> bool {
        let mem_props = unsafe {
            self.instance.get_physical_device_memory_properties(self.physical_device)
        };
        self.fg = crate::framegen::vk::VkFrameGen::new(
            &self.device, &mem_props, self.frame_w, self.frame_h, debug,
        );
        self.fg.is_some()
    }

    pub fn disable_framegen(&mut self) {
        unsafe { let _ = self.device.device_wait_idle(); }
        self.fg = None;
    }

    pub fn fg_can_generate(&self) -> bool {
        self.fg.as_ref().map_or(false, |fg| fg.can_generate())
    }

    pub fn fg_stats(&self) -> Option<&crate::framegen::FrameGenStats> {
        self.fg.as_ref().map(|fg| fg.stats())
    }

    /// Render a synthetic frame and present it.
    pub fn render_synth_and_present(
        &mut self,
        win_w: u32,
        win_h: u32,
        brightness: f32,
        contrast: f32,
        gamma: f32,
        t: f32,
        mode: crate::framegen::FrameGenMode,
        quality: crate::framegen::FrameGenQuality,
    ) -> bool {
        self.render_and_present_inner(win_w, win_h, brightness, contrast, gamma, Some((t, mode, quality)))
    }

    /// Upload a YUV frame (CPU conversion to RGBA, then staging buffer copy).
    /// Brightness and contrast are applied during conversion (1.0 = normal).
    pub fn upload(&mut self, data: &[u8], brightness: f32, contrast: f32, gamma: f32) {
        // XRGB fast path: BGRX→BGRA directly into staging, skip rgb_buf intermediate
        let skip_adjust = self.use_compute()
            || ((brightness - 1.0).abs() <= 0.001
                && (contrast - 1.0).abs() <= 0.001
                && (gamma - 1.0).abs() <= 0.001);
        if self.pixfmt == V4L2_PIX_FMT_XRGB32 && skip_adjust {
            let n = (self.frame_w * self.frame_h * 4) as usize;
            unsafe {
                std::ptr::copy_nonoverlapping(data.as_ptr(), self.staging_ptr, n);
                // Set alpha bytes to 255 (BGRX → BGRA)
                let staging = std::slice::from_raw_parts_mut(self.staging_ptr, n);
                for i in (3..n).step_by(4) { staging[i] = 255; }
            }
            // Also update rgb_buf for screenshot/dimming
            self.rgb_buf[..n].copy_from_slice(unsafe { std::slice::from_raw_parts(self.staging_ptr, n) });
            return;
        }
        // Convert YUV to RGBA on CPU
        yuv_to_rgba(data, self.frame_w, self.frame_h, self.pixfmt, &mut self.rgb_buf);

        // Apply brightness, contrast and gamma via cached LUT (256 entries)
        // Skip if compute scaling is active (shader handles adjustments)
        let need_adjust = !self.use_compute()
            && ((brightness - 1.0).abs() > 0.001
            || (contrast - 1.0).abs() > 0.001
            || (gamma - 1.0).abs() > 0.001);
        if need_adjust {
            let lut = match self.cached_lut {
                Some((b, c, g, ref lut)) if b == brightness && c == contrast && g == gamma => lut,
                _ => {
                    let inv_gamma = 1.0 / gamma;
                    let mut lut = [0u8; 256];
                    for i in 0..256 {
                        let v = ((i as f32 - 128.0) * contrast + 128.0) * brightness;
                        let v = (v / 255.0).clamp(0.0, 1.0);
                        let v = v.powf(inv_gamma) * 255.0;
                        lut[i] = v.clamp(0.0, 255.0) as u8;
                    }
                    self.cached_lut = Some((brightness, contrast, gamma, lut));
                    &self.cached_lut.as_ref().unwrap().3
                }
            };
            for pixel in self.rgb_buf.chunks_exact_mut(4) {
                pixel[0] = lut[pixel[0] as usize];
                pixel[1] = lut[pixel[1] as usize];
                pixel[2] = lut[pixel[2] as usize];
            }
        }

        // Copy to persistently-mapped staging buffer
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.rgb_buf.as_ptr(),
                self.staging_ptr,
                self.rgb_buf.len(),
            );
        }
    }

    /// Write a dimmed copy of rgb_buf to staging. Does not modify rgb_buf.
    pub fn dim_staging(&mut self, factor: f32) {
        let len = self.rgb_buf.len();
        let staging = unsafe { std::slice::from_raw_parts_mut(self.staging_ptr, len) };
        for (dst, src) in staging.chunks_exact_mut(4).zip(self.rgb_buf.chunks_exact(4)) {
            dst[0] = (src[0] as f32 * factor) as u8;
            dst[1] = (src[1] as f32 * factor) as u8;
            dst[2] = (src[2] as f32 * factor) as u8;
            dst[3] = src[3];
        }
    }

    /// Restore staging from rgb_buf (undoes dim).
    pub fn restore_staging(&mut self) {
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.rgb_buf.as_ptr(),
                self.staging_ptr,
                self.rgb_buf.len(),
            );
        }
    }

    /// Render the frame to the swapchain and present.
    /// Returns true on success, false if unrecoverable error.
    pub fn render_and_present(&mut self, win_w: u32, win_h: u32, brightness: f32, contrast: f32, gamma: f32) -> bool {
        self.render_and_present_inner(win_w, win_h, brightness, contrast, gamma, None)
    }

    fn render_and_present_inner(
        &mut self,
        win_w: u32,
        win_h: u32,
        brightness: f32,
        contrast: f32,
        gamma: f32,
        synth: Option<(f32, crate::framegen::FrameGenMode, crate::framegen::FrameGenQuality)>,
    ) -> bool {
        unsafe {
            // Wait for previous frame to finish
            let _ = self.device.wait_for_fences(&[self.in_flight_fence], true, u64::MAX);

            // Recreate swapchain if flagged (from previous present or external request)
            if self.needs_recreate {
                self.needs_recreate = false;
                if let Err(e) = self.recreate_swapchain(win_w, win_h, Some(self.present_mode), false) {
                    eprintln!("vulkan: swapchain recreate failed: {}", e);
                    return false;
                }
            }

            // Acquire swapchain image
            let acquire = self.swapchain_fn.acquire_next_image(
                self.swapchain,
                u64::MAX,
                self.image_available_sem,
                vk::Fence::null(),
            );
            let image_index = match acquire {
                Ok((idx, _)) => idx,
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) |
                Err(vk::Result::SUBOPTIMAL_KHR) => {
                    // Recreate and skip this frame
                    self.needs_recreate = true;
                    return true;
                }
                Err(e) => {
                    eprintln!("vulkan: acquire_next_image: {}", e);
                    return false;
                }
            };

            // Reset fence only after successful acquire
            self.device.reset_fences(&[self.in_flight_fence]).ok();

            let swapchain_image = self.swapchain_images[image_index as usize];
            let cb = self.command_buffers[0];

            self.device.reset_command_buffer(cb, vk::CommandBufferResetFlags::empty()).ok();

            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.device.begin_command_buffer(cb, &begin).ok();

            if let Some((t, mode, quality)) = synth {
                // ── Synth frame: generate → blit fg_out → frame_image ──
                if let Some(ref mut fg) = self.fg {
                    fg.record_generate(cb, t, mode, quality);

                    // fg_out is in GENERAL after synthesis → TRANSFER_SRC
                    transition_image_layout(&self.device, cb, fg.output_image(),
                        vk::ImageLayout::GENERAL,
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::AccessFlags::SHADER_WRITE,
                        vk::AccessFlags::TRANSFER_READ,
                    );

                    // frame_image → TRANSFER_DST
                    transition_image_layout(&self.device, cb, self.frame_image,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::AccessFlags::TRANSFER_WRITE,
                        vk::AccessFlags::TRANSFER_WRITE,
                    );

                    let (fg_w, fg_h) = fg.dimensions();
                    let blit = vk::ImageBlit {
                        src_subresource: vk::ImageSubresourceLayers {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            mip_level: 0, base_array_layer: 0, layer_count: 1,
                        },
                        src_offsets: [
                            vk::Offset3D { x: 0, y: 0, z: 0 },
                            vk::Offset3D { x: fg_w as i32, y: fg_h as i32, z: 1 },
                        ],
                        dst_subresource: vk::ImageSubresourceLayers {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            mip_level: 0, base_array_layer: 0, layer_count: 1,
                        },
                        dst_offsets: [
                            vk::Offset3D { x: 0, y: 0, z: 0 },
                            vk::Offset3D { x: self.frame_w as i32, y: self.frame_h as i32, z: 1 },
                        ],
                    };
                    self.device.cmd_blit_image(
                        cb,
                        fg.output_image(), vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        self.frame_image, vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &[blit], vk::Filter::LINEAR,
                    );
                }
            } else {
                // ── Real frame: staging → frame_image ──
                transition_image_layout(&self.device, cb, self.frame_image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::AccessFlags::TRANSFER_WRITE,
                    vk::AccessFlags::TRANSFER_WRITE,
                );

                let region = vk::BufferImageCopy::default()
                    .buffer_offset(0)
                    .buffer_row_length(0)
                    .buffer_image_height(0)
                    .image_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                    .image_extent(vk::Extent3D {
                        width: self.frame_w,
                        height: self.frame_h,
                        depth: 1,
                    });

                self.device.cmd_copy_buffer_to_image(
                    cb,
                    self.staging_buf,
                    self.frame_image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[region],
                );

                // Framegen: capture frame for motion estimation
                if self.fg.is_some() {
                    // frame_image: TRANSFER_DST → TRANSFER_SRC for blit to fg_curr
                    transition_image_layout(&self.device, cb, self.frame_image,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::AccessFlags::TRANSFER_WRITE,
                        vk::AccessFlags::TRANSFER_READ,
                    );
                    if let Some(ref mut fg) = self.fg {
                        fg.record_push_frame(cb, self.frame_image, self.frame_w, self.frame_h);
                    }
                    // Restore to TRANSFER_DST so the rest of the path works unchanged
                    transition_image_layout(&self.device, cb, self.frame_image,
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::AccessFlags::TRANSFER_READ,
                        vk::AccessFlags::TRANSFER_WRITE,
                    );
                }
            }

            let use_compute = self.use_compute();

            if use_compute {
                // ── Compute scaling path ──────────────────────────────

                // frame_image: TRANSFER_DST → SHADER_READ_ONLY (for sampler)
                transition_image_layout(&self.device, cb, self.frame_image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::AccessFlags::TRANSFER_WRITE,
                    vk::AccessFlags::SHADER_READ,
                );

                let use_fsr = matches!(self.scale_mode,
                    crate::gl_renderer::ScaleMode::Fsr | crate::gl_renderer::ScaleMode::IntegerFsr);

                // Compute content area for aspect-fit
                let (dst_x, dst_y, content_w, content_h) = fit_rect(
                    self.frame_w, self.frame_h,
                    self.swapchain_extent.width, self.swapchain_extent.height,
                    self.aspect_mode,
                );

                let out_w = self.compute_extent.width;
                let out_h = self.compute_extent.height;

                if use_fsr {
                    // ── FSR: EASU + RCAS ──────────────────────────────
                    // compute_a → GENERAL for EASU output
                    transition_image_layout(&self.device, cb, self.compute_a,
                        vk::ImageLayout::UNDEFINED,
                        vk::ImageLayout::GENERAL,
                        vk::PipelineStageFlags::TOP_OF_PIPE,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::AccessFlags::empty(),
                        vk::AccessFlags::SHADER_WRITE,
                    );

                    // EASU dispatch
                    self.device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, self.compute_easu_pipeline);
                    self.device.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::COMPUTE,
                        self.compute_pipe_layout, 0, &[self.compute_desc_sets[0]], &[]);

                    // EASU push constants: con0..con3 matching the GL path
                    let easu_con = fsr_easu_con(
                        self.frame_w as f32, self.frame_h as f32,
                        out_w as f32, out_h as f32,
                    );
                    let mut pc = [0u32; 16];
                    pc[0] = easu_con.0[0]; pc[1] = easu_con.0[1]; pc[2] = easu_con.0[2]; pc[3] = easu_con.0[3];
                    pc[4] = easu_con.1[0]; pc[5] = easu_con.1[1]; pc[6] = easu_con.1[2]; pc[7] = easu_con.1[3];
                    pc[8] = easu_con.2[0]; pc[9] = easu_con.2[1]; pc[10] = easu_con.2[2]; pc[11] = easu_con.2[3];
                    pc[12] = easu_con.3[0]; pc[13] = easu_con.3[1]; pc[14] = easu_con.3[2]; pc[15] = easu_con.3[3];
                    let pc_bytes = std::slice::from_raw_parts(pc.as_ptr() as *const u8, 64);
                    self.device.cmd_push_constants(cb, self.compute_pipe_layout,
                        vk::ShaderStageFlags::COMPUTE, 0, pc_bytes);

                    self.device.cmd_dispatch(cb, (out_w + 7) / 8, (out_h + 7) / 8, 1);

                    // compute_a: GENERAL → SHADER_READ_ONLY for RCAS input
                    transition_image_layout(&self.device, cb, self.compute_a,
                        vk::ImageLayout::GENERAL,
                        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::AccessFlags::SHADER_WRITE,
                        vk::AccessFlags::SHADER_READ,
                    );

                    // compute_b → GENERAL for RCAS output
                    transition_image_layout(&self.device, cb, self.compute_b,
                        vk::ImageLayout::UNDEFINED,
                        vk::ImageLayout::GENERAL,
                        vk::PipelineStageFlags::TOP_OF_PIPE,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::AccessFlags::empty(),
                        vk::AccessFlags::SHADER_WRITE,
                    );

                    // RCAS dispatch
                    self.device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, self.compute_rcas_pipeline);
                    self.device.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::COMPUTE,
                        self.compute_pipe_layout, 0, &[self.compute_desc_sets[1]], &[]);

                    let rcas_strength = self.sharpness * 1.2;
                    let rcas_con = fsr_rcas_con(rcas_strength);
                    let mut rpc = [0u32; 16];
                    rpc[0] = rcas_con[0]; // packed sharpness
                    rpc[1] = out_w; rpc[2] = out_h; rpc[3] = 0;
                    rpc[4] = brightness.to_bits(); rpc[5] = contrast.to_bits();
                    rpc[6] = (1.0_f32 / gamma).to_bits(); rpc[7] = 0;
                    let rpc_bytes = std::slice::from_raw_parts(rpc.as_ptr() as *const u8, 64);
                    self.device.cmd_push_constants(cb, self.compute_pipe_layout,
                        vk::ShaderStageFlags::COMPUTE, 0, rpc_bytes);

                    self.device.cmd_dispatch(cb, (out_w + 7) / 8, (out_h + 7) / 8, 1);

                    // compute_b → TRANSFER_SRC for copy to swapchain
                    transition_image_layout(&self.device, cb, self.compute_b,
                        vk::ImageLayout::GENERAL,
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::AccessFlags::SHADER_WRITE,
                        vk::AccessFlags::TRANSFER_READ,
                    );
                } else {
                    // ── CAS: single pass ──────────────────────────────
                    // compute_a → GENERAL for CAS output
                    transition_image_layout(&self.device, cb, self.compute_a,
                        vk::ImageLayout::UNDEFINED,
                        vk::ImageLayout::GENERAL,
                        vk::PipelineStageFlags::TOP_OF_PIPE,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::AccessFlags::empty(),
                        vk::AccessFlags::SHADER_WRITE,
                    );

                    self.device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, self.compute_cas_pipeline);
                    self.device.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::COMPUTE,
                        self.compute_pipe_layout, 0, &[self.compute_desc_sets[0]], &[]);

                    // CAS push constants
                    let mut pc = [0u32; 16];
                    pc[0] = self.frame_w; pc[1] = self.frame_h;
                    pc[2] = out_w; pc[3] = out_h;
                    pc[4] = (self.sharpness * 1.2).to_bits();
                    pc[5] = brightness.to_bits();
                    pc[6] = contrast.to_bits();
                    pc[7] = (1.0_f32 / gamma).to_bits();
                    let pc_bytes = std::slice::from_raw_parts(pc.as_ptr() as *const u8, 64);
                    self.device.cmd_push_constants(cb, self.compute_pipe_layout,
                        vk::ShaderStageFlags::COMPUTE, 0, pc_bytes);

                    self.device.cmd_dispatch(cb, (out_w + 7) / 8, (out_h + 7) / 8, 1);
                }

                // Determine which image has the result and composite OSD
                let (result_image, osd_desc_idx) = if use_fsr {
                    (self.compute_b, 1usize)
                } else {
                    (self.compute_a, 0usize)
                };

                // OSD alpha-blend composite onto result image (while still in GENERAL)
                if self.osd_dirty {
                    // Result image is still in GENERAL from compute dispatch — perfect for read+write
                    self.record_osd_composite(cb, osd_desc_idx);
                    // Barrier: compute write → transfer read
                    transition_image_layout(&self.device, cb, result_image,
                        vk::ImageLayout::GENERAL,
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::AccessFlags::SHADER_WRITE,
                        vk::AccessFlags::TRANSFER_READ,
                    );
                } else {
                    // No OSD — just transition result to TRANSFER_SRC
                    transition_image_layout(&self.device, cb, result_image,
                        vk::ImageLayout::GENERAL,
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::AccessFlags::SHADER_WRITE,
                        vk::AccessFlags::TRANSFER_READ,
                    );
                }

                // Transition swapchain → TRANSFER_DST, clear, then copy result
                transition_image_layout(&self.device, cb, swapchain_image,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::AccessFlags::empty(),
                    vk::AccessFlags::TRANSFER_WRITE,
                );
                let clear_range = vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                };
                self.device.cmd_clear_color_image(
                    cb, swapchain_image, vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 1.0] },
                    &[clear_range],
                );

                // Blit compute result → swapchain (handles letterboxing)
                let blit = vk::ImageBlit {
                    src_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0, base_array_layer: 0, layer_count: 1,
                    },
                    src_offsets: [
                        vk::Offset3D { x: 0, y: 0, z: 0 },
                        vk::Offset3D { x: out_w as i32, y: out_h as i32, z: 1 },
                    ],
                    dst_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0, base_array_layer: 0, layer_count: 1,
                    },
                    dst_offsets: [
                        vk::Offset3D { x: dst_x.max(0), y: dst_y.max(0), z: 0 },
                        vk::Offset3D {
                            x: (dst_x + content_w as i32).min(self.swapchain_extent.width as i32),
                            y: (dst_y + content_h as i32).min(self.swapchain_extent.height as i32),
                            z: 1,
                        },
                    ],
                };
                self.device.cmd_blit_image(
                    cb, result_image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    swapchain_image, vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[blit], vk::Filter::NEAREST,
                );

                // frame_image → TRANSFER_DST for next upload
                transition_image_layout(&self.device, cb, self.frame_image,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::AccessFlags::SHADER_READ,
                    vk::AccessFlags::TRANSFER_WRITE,
                );

            } else {
                // ── Legacy blit path (Nearest/Bilinear) ───────────────
                // Route through compute_a for OSD compositing support.

                // Transition frame image to TRANSFER_SRC for blit
                transition_image_layout(&self.device, cb, self.frame_image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::AccessFlags::TRANSFER_WRITE,
                    vk::AccessFlags::TRANSFER_READ,
                );

                // Transition compute_a to TRANSFER_DST
                transition_image_layout(&self.device, cb, self.compute_a,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::AccessFlags::empty(),
                    vk::AccessFlags::TRANSFER_WRITE,
                );

                // Clear compute_a to black (letterboxing)
                let clear_range = vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                };
                self.device.cmd_clear_color_image(
                    cb, self.compute_a, vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 1.0] },
                    &[clear_range],
                );

                // Blit frame → compute_a (with scaling and aspect-fit)
                let (dst_x, dst_y, dst_w, dst_h) = fit_rect(
                    self.frame_w, self.frame_h,
                    self.swapchain_extent.width, self.swapchain_extent.height,
                    self.aspect_mode,
                );

                let sw = self.swapchain_extent.width as i32;
                let sh = self.swapchain_extent.height as i32;
                let fw = self.frame_w as f64;
                let fh = self.frame_h as f64;
                let dw = dst_w as f64;
                let dh = dst_h as f64;

                let (src_x0, src_y0, src_x1, src_y1, dst_x0, dst_y0, dst_x1, dst_y1) =
                    if dst_x < 0 || dst_y < 0 || dst_x + dst_w as i32 > sw || dst_y + dst_h as i32 > sh {
                        let sx_per_dx = fw / dw;
                        let sy_per_dy = fh / dh;
                        let cdx0 = dst_x.max(0);
                        let cdy0 = dst_y.max(0);
                        let cdx1 = (dst_x + dst_w as i32).min(sw);
                        let cdy1 = (dst_y + dst_h as i32).min(sh);
                        let csx0 = ((cdx0 - dst_x) as f64 * sx_per_dx) as i32;
                        let csy0 = ((cdy0 - dst_y) as f64 * sy_per_dy) as i32;
                        let csx1 = (fw - ((dst_x + dst_w as i32 - cdx1) as f64 * sx_per_dx)) as i32;
                        let csy1 = (fh - ((dst_y + dst_h as i32 - cdy1) as f64 * sy_per_dy)) as i32;
                        (csx0, csy0, csx1, csy1, cdx0, cdy0, cdx1, cdy1)
                    } else {
                        (0, 0, self.frame_w as i32, self.frame_h as i32,
                         dst_x, dst_y, dst_x + dst_w as i32, dst_y + dst_h as i32)
                    };

                let blit = vk::ImageBlit {
                    src_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0, base_array_layer: 0, layer_count: 1,
                    },
                    src_offsets: [
                        vk::Offset3D { x: src_x0, y: src_y0, z: 0 },
                        vk::Offset3D { x: src_x1, y: src_y1, z: 1 },
                    ],
                    dst_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0, base_array_layer: 0, layer_count: 1,
                    },
                    dst_offsets: [
                        vk::Offset3D { x: dst_x0, y: dst_y0, z: 0 },
                        vk::Offset3D { x: dst_x1, y: dst_y1, z: 1 },
                    ],
                };

                self.device.cmd_blit_image(
                    cb, self.frame_image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    self.compute_a, vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[blit], vk::Filter::LINEAR,
                );

                // Transition frame image back to TRANSFER_DST for next upload
                transition_image_layout(&self.device, cb, self.frame_image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::AccessFlags::TRANSFER_READ,
                    vk::AccessFlags::TRANSFER_WRITE,
                );

                // OSD composite on compute_a
                if self.osd_dirty {
                    transition_image_layout(&self.device, cb, self.compute_a,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        vk::ImageLayout::GENERAL,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::AccessFlags::TRANSFER_WRITE,
                        vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
                    );
                    self.record_osd_composite(cb, 0);
                    transition_image_layout(&self.device, cb, self.compute_a,
                        vk::ImageLayout::GENERAL,
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::AccessFlags::SHADER_WRITE,
                        vk::AccessFlags::TRANSFER_READ,
                    );
                } else {
                    transition_image_layout(&self.device, cb, self.compute_a,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::AccessFlags::TRANSFER_WRITE,
                        vk::AccessFlags::TRANSFER_READ,
                    );
                }

                // Blit compute_a → swapchain
                transition_image_layout(&self.device, cb, swapchain_image,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::AccessFlags::empty(),
                    vk::AccessFlags::TRANSFER_WRITE,
                );
                let final_blit = vk::ImageBlit {
                    src_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0, base_array_layer: 0, layer_count: 1,
                    },
                    src_offsets: [
                        vk::Offset3D { x: 0, y: 0, z: 0 },
                        vk::Offset3D { x: self.compute_extent.width as i32, y: self.compute_extent.height as i32, z: 1 },
                    ],
                    dst_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0, base_array_layer: 0, layer_count: 1,
                    },
                    dst_offsets: [
                        vk::Offset3D { x: 0, y: 0, z: 0 },
                        vk::Offset3D { x: self.swapchain_extent.width as i32, y: self.swapchain_extent.height as i32, z: 1 },
                    ],
                };
                self.device.cmd_blit_image(
                    cb, self.compute_a, vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    swapchain_image, vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[final_blit], vk::Filter::NEAREST,
                );
            }

            // Transition swapchain image to PRESENT_SRC
            transition_image_layout(&self.device, cb, swapchain_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::PRESENT_SRC_KHR,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::AccessFlags::TRANSFER_WRITE,
                vk::AccessFlags::empty(),
            );

            self.device.end_command_buffer(cb).ok();

            // Submit
            let wait_sems = [self.image_available_sem];
            let wait_stages = [vk::PipelineStageFlags::TRANSFER];
            let signal_sems = [self.render_finished_sem];
            let cbs = [cb];

            let submit = vk::SubmitInfo::default()
                .wait_semaphores(&wait_sems)
                .wait_dst_stage_mask(&wait_stages)
                .command_buffers(&cbs)
                .signal_semaphores(&signal_sems);

            if let Err(e) = self.device.queue_submit(self.graphics_queue, &[submit], self.in_flight_fence) {
                eprintln!("vulkan: queue_submit: {}", e);
                return false;
            }

            // Present
            let swapchains = [self.swapchain];
            let image_indices = [image_index];
            let present_info = vk::PresentInfoKHR::default()
                .wait_semaphores(&signal_sems)
                .swapchains(&swapchains)
                .image_indices(&image_indices);

            match self.swapchain_fn.queue_present(self.present_queue, &present_info) {
                Ok(suboptimal) => {
                    if suboptimal {
                        self.needs_recreate = true;
                    }
                    true
                }
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) |
                Err(vk::Result::SUBOPTIMAL_KHR) => {
                    self.needs_recreate = true;
                    true
                }
                Err(e) => {
                    eprintln!("vulkan: queue_present: {}", e);
                    false
                }
            }
        }
    }

    /// Recreate the swapchain (e.g. after window resize or present mode change).
    pub fn recreate_swapchain(&mut self, win_w: u32, win_h: u32, preferred_mode: Option<vk::PresentModeKHR>, debug: bool) -> anyhow::Result<()> {
        // Wait only for our in-flight frame (cheaper than device_wait_idle)
        unsafe { self.device.wait_for_fences(&[self.in_flight_fence], true, u64::MAX)? };

        let mem_props = unsafe { self.instance.get_physical_device_memory_properties(self.physical_device) };

        // Destroy old swapchain and surface, then recreate surface.
        // This avoids Wayland wp_tearing_control_v1 conflicts — the Mesa WSI
        // layer attaches tearing control per surface, and destroying just the
        // swapchain doesn't always release it before the new one is created.
        unsafe { self.swapchain_fn.destroy_swapchain(self.swapchain, None) };
        unsafe { self.surface_fn.destroy_surface(self.surface, None) };

        let new_surface = unsafe {
            let mut s = vk::SurfaceKHR::null();
            if SDL_Vulkan_CreateSurface(self.raw_window, self.instance.handle(), &mut s)
                == sdl2_sys::SDL_bool::SDL_FALSE
            {
                anyhow::bail!("vulkan: SDL_Vulkan_CreateSurface failed during recreate");
            }
            s
        };
        self.surface = new_surface;

        let (swapchain, images, format, extent, mode) = create_swapchain(
            &self.surface_fn, &self.swapchain_fn, &self.device,
            self.physical_device, self.surface,
            win_w, win_h, vk::SwapchainKHR::null(), preferred_mode, debug,
        )?;

        self.swapchain = swapchain;
        self.swapchain_images = images;
        self.swapchain_format = format;
        self.swapchain_extent = extent;
        self.present_mode = mode;
        self.mailbox = mode == vk::PresentModeKHR::MAILBOX;

        // Recreate OSD resources at new window size
        let osd_w = extent.width.max(1);
        let osd_h = extent.height.max(1);
        if osd_w != self.osd_extent.width || osd_h != self.osd_extent.height {
            unsafe {
                self.device.unmap_memory(self.osd_staging_mem);
                self.device.destroy_buffer(self.osd_staging_buf, None);
                self.device.free_memory(self.osd_staging_mem, None);
                self.device.destroy_image(self.osd_image, None);
                self.device.free_memory(self.osd_mem, None);
            }
            let osd_size = (osd_w * osd_h * 4) as vk::DeviceSize;
            let (osd_staging_buf, osd_staging_mem) = create_buffer(
                &self.device, &mem_props, osd_size,
                vk::BufferUsageFlags::TRANSFER_SRC,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;
            let (osd_image, osd_mem) = create_image(
                &self.device, &mem_props,
                osd_w, osd_h,
                vk::Format::B8G8R8A8_UNORM,
                vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )?;
            let osd_staging_ptr = unsafe {
                self.device.map_memory(osd_staging_mem, 0, osd_size, vk::MemoryMapFlags::empty())
            }? as *mut u8;
            self.osd_staging_buf = osd_staging_buf;
            self.osd_staging_mem = osd_staging_mem;
            self.osd_staging_ptr = osd_staging_ptr;
            self.osd_image = osd_image;
            self.osd_mem = osd_mem;
            self.osd_cpu = vec![0u8; (osd_w * osd_h * 4) as usize];
            self.osd_extent = vk::Extent2D { width: osd_w, height: osd_h };
            self.osd_dirty = false;
            self.osd_uploaded = false;

            // Recreate OSD image view
            unsafe { self.device.destroy_image_view(self.osd_image_view, None) };
            let osd_view_ci = vk::ImageViewCreateInfo::default()
                .image(osd_image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(vk::Format::B8G8R8A8_UNORM)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });
            self.osd_image_view = unsafe { self.device.create_image_view(&osd_view_ci, None)? };
        }

        // Recreate compute images if output size changed
        if extent.width != self.compute_extent.width || extent.height != self.compute_extent.height {
            unsafe {
                self.device.destroy_image_view(self.compute_a_view, None);
                self.device.destroy_image(self.compute_a, None);
                self.device.free_memory(self.compute_a_mem, None);
                self.device.destroy_image_view(self.compute_b_view, None);
                self.device.destroy_image(self.compute_b, None);
                self.device.free_memory(self.compute_b_mem, None);
            }
            let (a, a_mem, a_view) = create_storage_image(&self.device, &mem_props, extent.width, extent.height)?;
            let (b, b_mem, b_view) = create_storage_image(&self.device, &mem_props, extent.width, extent.height)?;
            self.compute_a = a;
            self.compute_a_mem = a_mem;
            self.compute_a_view = a_view;
            self.compute_b = b;
            self.compute_b_mem = b_mem;
            self.compute_b_view = b_view;
            self.compute_extent = extent;

            // Update descriptor sets to point to new images
            let a_storage_info = vk::DescriptorImageInfo::default()
                .image_view(a_view)
                .image_layout(vk::ImageLayout::GENERAL);
            let a_sampler_info = vk::DescriptorImageInfo::default()
                .sampler(self.compute_sampler)
                .image_view(a_view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
            let b_storage_info = vk::DescriptorImageInfo::default()
                .image_view(b_view)
                .image_layout(vk::ImageLayout::GENERAL);
            let writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(self.compute_desc_sets[0])
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(std::slice::from_ref(&a_storage_info)),
                vk::WriteDescriptorSet::default()
                    .dst_set(self.compute_desc_sets[1])
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(&a_sampler_info)),
                vk::WriteDescriptorSet::default()
                    .dst_set(self.compute_desc_sets[1])
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(std::slice::from_ref(&b_storage_info)),
            ];
            unsafe { self.device.update_descriptor_sets(&writes, &[]) };
        }

        // Update OSD blend descriptor sets (OSD image or compute targets may have changed)
        {
            let osd_sampler_info = vk::DescriptorImageInfo::default()
                .sampler(self.compute_sampler)
                .image_view(self.osd_image_view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
            let a_storage_info = vk::DescriptorImageInfo::default()
                .image_view(self.compute_a_view)
                .image_layout(vk::ImageLayout::GENERAL);
            let b_storage_info = vk::DescriptorImageInfo::default()
                .image_view(self.compute_b_view)
                .image_layout(vk::ImageLayout::GENERAL);
            let writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(self.osd_blend_desc_sets[0])
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(&osd_sampler_info)),
                vk::WriteDescriptorSet::default()
                    .dst_set(self.osd_blend_desc_sets[0])
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(std::slice::from_ref(&a_storage_info)),
                vk::WriteDescriptorSet::default()
                    .dst_set(self.osd_blend_desc_sets[1])
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(&osd_sampler_info)),
                vk::WriteDescriptorSet::default()
                    .dst_set(self.osd_blend_desc_sets[1])
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(std::slice::from_ref(&b_storage_info)),
            ];
            unsafe { self.device.update_descriptor_sets(&writes, &[]) };
        }

        if debug {
            eprintln!("vulkan: swapchain recreated {}x{}", extent.width, extent.height);
        }
        Ok(())
    }

    // --- OSD rendering (software rasterized, then blitted) ---

    /// Clear the OSD overlay buffer.
    pub fn osd_clear(&mut self) {
        self.osd_cpu.fill(0);
        self.osd_dirty = false;
        self.osd_uploaded = false;
    }

    /// Whether OSD currently has content to composite.
    pub fn osd_has_content(&self) -> bool {
        self.osd_dirty
    }

    /// Mark OSD as having content without re-rasterizing (reuse previous GPU texture).
    pub fn mark_osd_content(&mut self) {
        self.osd_dirty = true;
    }

    /// Begin an OSD frame: clears the overlay and region list.
    pub fn begin_osd(&mut self, _win_w: u32, _win_h: u32) {
        self.osd_cpu.fill(0);
        self.osd_regions.clear();
        self.osd_uploaded = false; // new content will need upload
    }

    /// Draw a filled rectangle on the OSD overlay.
    pub fn osd_rect(&mut self, x: i32, y: i32, w: u32, h: u32, color: [f32; 4], _win_w: u32, _win_h: u32) {
        let ow = self.osd_extent.width as i32;
        let oh = self.osd_extent.height as i32;
        let r = (color[0] * 255.0) as u8;
        let g = (color[1] * 255.0) as u8;
        let b = (color[2] * 255.0) as u8;
        let a = (color[3] * 255.0) as u8;

        let x0 = x.max(0) as u32;
        let y0 = y.max(0) as u32;
        let x1 = ((x + w as i32) as u32).min(ow as u32);
        let y1 = ((y + h as i32) as u32).min(oh as u32);

        if x0 >= x1 || y0 >= y1 { return; }
        self.push_osd_region(x0, y0, x1, y1);

        for row in y0..y1 {
            for col in x0..x1 {
                let idx = ((row * ow as u32 + col) * 4) as usize;
                if idx + 3 < self.osd_cpu.len() {
                    // Alpha blend (BGRA byte order)
                    if a == 255 {
                        self.osd_cpu[idx] = b;
                        self.osd_cpu[idx + 1] = g;
                        self.osd_cpu[idx + 2] = r;
                        self.osd_cpu[idx + 3] = a;
                    } else {
                        let af = a as f32 / 255.0;
                        let inv = 1.0 - af;
                        self.osd_cpu[idx] = (b as f32 * af + self.osd_cpu[idx] as f32 * inv) as u8;
                        self.osd_cpu[idx + 1] = (g as f32 * af + self.osd_cpu[idx + 1] as f32 * inv) as u8;
                        self.osd_cpu[idx + 2] = (r as f32 * af + self.osd_cpu[idx + 2] as f32 * inv) as u8;
                        self.osd_cpu[idx + 3] = (self.osd_cpu[idx + 3] as f32 * inv + a as f32) as u8;
                    }
                }
            }
        }
    }

    /// Draw text on the OSD overlay using the built-in 8×8 bitmap font.
    pub fn osd_text(&mut self, text: &str, x: i32, y: i32, scale: u32, color: [f32; 4], _win_w: u32, _win_h: u32) {
        let ow = self.osd_extent.width as i32;
        let oh = self.osd_extent.height as i32;
        let r = (color[0] * 255.0) as u8;
        let g = (color[1] * 255.0) as u8;
        let b = (color[2] * 255.0) as u8;
        let a = (color[3] * 255.0) as u8;
        let scale = scale.max(1);
        let gw = font::GLYPH_W * scale;
        let gh = font::GLYPH_H * scale;

        // Track text bounding box
        let tx0 = x.max(0) as u32;
        let ty0 = y.max(0) as u32;
        let tx1 = ((x + (text.len() as u32 * gw) as i32) as u32).min(ow as u32);
        let ty1 = ((y + gh as i32) as u32).min(oh as u32);
        if tx0 < tx1 && ty0 < ty1 {
            self.push_osd_region(tx0, ty0, tx1, ty1);
        }

        for (ci, ch) in text.bytes().enumerate() {
            let gx = x + (ci as u32 * gw) as i32;
            if gx + gw as i32 <= 0 || gx >= ow { continue; }

            let glyph = font::glyph(ch);
            for row in 0..font::GLYPH_H {
                let bits = glyph[row as usize];
                if bits == 0 { continue; }
                for col in 0..font::GLYPH_W {
                    if bits & (0x80 >> col) == 0 { continue; }
                    // Scale the pixel
                    for sy in 0..scale {
                        for sx in 0..scale {
                            let px = gx + (col * scale + sx) as i32;
                            let py = y + (row * scale + sy) as i32;
                            if px < 0 || px >= ow || py < 0 || py >= oh { continue; }
                            let idx = ((py as u32 * ow as u32 + px as u32) * 4) as usize;
                            if idx + 3 < self.osd_cpu.len() {
                                self.osd_cpu[idx] = b;
                                self.osd_cpu[idx + 1] = g;
                                self.osd_cpu[idx + 2] = r;
                                self.osd_cpu[idx + 3] = a;
                            }
                        }
                    }
                }
            }
        }
    }

    /// Finalize OSD: mark as dirty so it gets uploaded on next present.
    pub fn end_osd(&mut self) {
        self.osd_dirty = !self.osd_regions.is_empty();
    }

    /// Track a dirty rectangle for region-based OSD blitting.
    fn push_osd_region(&mut self, x0: u32, y0: u32, x1: u32, y1: u32) {
        let layers = vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        };
        self.osd_regions.push(vk::ImageBlit {
            src_subresource: layers,
            src_offsets: [
                vk::Offset3D { x: x0 as i32, y: y0 as i32, z: 0 },
                vk::Offset3D { x: x1 as i32, y: y1 as i32, z: 1 },
            ],
            dst_subresource: layers,
            dst_offsets: [
                vk::Offset3D { x: x0 as i32, y: y0 as i32, z: 0 },
                vk::Offset3D { x: x1 as i32, y: y1 as i32, z: 1 },
            ],
        });
    }

    /// Record OSD blit commands into the given command buffer.
    /// The swapchain image must be in TRANSFER_DST_OPTIMAL layout.
    /// Upload OSD to GPU and alpha-composite it onto the given intermediate image
    /// using a compute shader. The target_image must be in GENERAL layout.
    /// `desc_set_index`: 0 for compute_a, 1 for compute_b.
    unsafe fn record_osd_composite(&mut self, cb: vk::CommandBuffer, desc_set_index: usize) {
        if !self.osd_uploaded {
            // Upload OSD CPU buffer to persistently-mapped staging
            std::ptr::copy_nonoverlapping(
                self.osd_cpu.as_ptr(),
                self.osd_staging_ptr,
                self.osd_cpu.len(),
            );

            // Transition OSD image to TRANSFER_DST
            transition_image_layout(&self.device, cb, self.osd_image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::AccessFlags::empty(),
                vk::AccessFlags::TRANSFER_WRITE,
            );

            // Copy staging → OSD image
            let region = vk::BufferImageCopy::default()
                .buffer_offset(0)
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_extent(vk::Extent3D {
                    width: self.osd_extent.width,
                    height: self.osd_extent.height,
                    depth: 1,
                });

            self.device.cmd_copy_buffer_to_image(
                cb,
                self.osd_staging_buf,
                self.osd_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region],
            );

            // Transition OSD to SHADER_READ_ONLY for sampling
            transition_image_layout(&self.device, cb, self.osd_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::AccessFlags::TRANSFER_WRITE,
                vk::AccessFlags::SHADER_READ,
            );

            self.osd_uploaded = true;
        }
        // else: OSD already on GPU in SHADER_READ_ONLY — no barrier needed

        // Dispatch OSD blend compute shader
        let out_w = self.compute_extent.width;
        let out_h = self.compute_extent.height;
        self.device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, self.osd_blend_pipeline);
        self.device.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::COMPUTE,
            self.compute_pipe_layout, 0, &[self.osd_blend_desc_sets[desc_set_index]], &[]);

        let mut pc = [0u32; 16];
        pc[0] = out_w; pc[1] = out_h;
        let pc_bytes = std::slice::from_raw_parts(pc.as_ptr() as *const u8, 64);
        self.device.cmd_push_constants(cb, self.compute_pipe_layout,
            vk::ShaderStageFlags::COMPUTE, 0, pc_bytes);

        self.device.cmd_dispatch(cb, (out_w + 7) / 8, (out_h + 7) / 8, 1);
    }

    /// Get current swapchain extent.
    pub fn extent(&self) -> (u32, u32) {
        (self.swapchain_extent.width, self.swapchain_extent.height)
    }
}

// --- Helper functions ---

fn pick_physical_device(
    instance: &ash::Instance,
    surface_fn: &ash::khr::surface::Instance,
    surface: vk::SurfaceKHR,
    devices: &[vk::PhysicalDevice],
    debug: bool,
) -> anyhow::Result<(vk::PhysicalDevice, u32, u32)> {
    // Score devices: discrete > integrated > other
    let mut best: Option<(vk::PhysicalDevice, u32, u32, i32)> = None;

    for &device in devices {
        let props = unsafe { instance.get_physical_device_properties(device) };
        let queue_families = unsafe { instance.get_physical_device_queue_family_properties(device) };

        let mut gfx_family = None;
        let mut present_family = None;

        for (i, qf) in queue_families.iter().enumerate() {
            let i = i as u32;
            if qf.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                gfx_family = Some(i);
            }
            let present_support = unsafe {
                surface_fn.get_physical_device_surface_support(device, i, surface).unwrap_or(false)
            };
            if present_support {
                present_family = Some(i);
            }
            if gfx_family.is_some() && present_family.is_some() {
                break;
            }
        }

        if let (Some(gf), Some(pf)) = (gfx_family, present_family) {
            // Check for swapchain extension
            let exts = unsafe { instance.enumerate_device_extension_properties(device).unwrap_or_default() };
            let has_swapchain = exts.iter().any(|e| {
                let name = unsafe { CStr::from_ptr(e.extension_name.as_ptr()) };
                name == ash::khr::swapchain::NAME
            });
            if !has_swapchain { continue; }

            let score = match props.device_type {
                vk::PhysicalDeviceType::DISCRETE_GPU => 100,
                vk::PhysicalDeviceType::INTEGRATED_GPU => 50,
                vk::PhysicalDeviceType::VIRTUAL_GPU => 25,
                _ => 10,
            };

            if debug {
                let name = unsafe { CStr::from_ptr(props.device_name.as_ptr()) };
                eprintln!("vulkan: candidate device: {:?} score={}", name, score);
            }

            if best.as_ref().map_or(true, |(_, _, _, s)| score > *s) {
                best = Some((device, gf, pf, score));
            }
        }
    }

    best.map(|(d, g, p, _)| (d, g, p))
        .ok_or_else(|| anyhow::anyhow!("vulkan: no suitable device found"))
}

/// Create a shader module from SPIR-V bytes.
unsafe fn create_shader_module(device: &ash::Device, spv: &[u8]) -> anyhow::Result<vk::ShaderModule> {
    assert!(spv.len() % 4 == 0, "SPIR-V not aligned to 4 bytes");
    let code = std::slice::from_raw_parts(spv.as_ptr() as *const u32, spv.len() / 4);
    let ci = vk::ShaderModuleCreateInfo::default().code(code);
    Ok(device.create_shader_module(&ci, None)?)
}

/// Create a storage image with an image view (for compute output).
fn create_storage_image(
    device: &ash::Device,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    w: u32, h: u32,
) -> anyhow::Result<(vk::Image, vk::DeviceMemory, vk::ImageView)> {
    let (image, mem) = create_image(
        device, mem_props, w, h,
        vk::Format::R8G8B8A8_UNORM,
        vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;
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
    let view = unsafe { device.create_image_view(&view_ci, None)? };
    Ok((image, mem, view))
}

/// Set up the compute scaling pipeline: descriptor layout, pipelines, images, descriptor sets.
fn create_compute_pipeline(
    device: &ash::Device,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    frame_image: vk::Image,
    _frame_w: u32, _frame_h: u32,
    out_w: u32, out_h: u32,
) -> anyhow::Result<(
    vk::DescriptorSetLayout,
    vk::PipelineLayout,
    vk::Pipeline, // CAS
    vk::Pipeline, // EASU
    vk::Pipeline, // RCAS
    vk::DescriptorPool,
    Vec<vk::DescriptorSet>,
    vk::ImageView,  // frame_image_view
    vk::Sampler,
    vk::Image, vk::DeviceMemory, vk::ImageView, // compute_a
    vk::Image, vk::DeviceMemory, vk::ImageView, // compute_b
)> {
    // Descriptor set layout: binding 0 = combined image sampler, binding 1 = storage image
    let bindings = [
        vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
        vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
    ];
    let desc_layout_ci = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
    let desc_layout = unsafe { device.create_descriptor_set_layout(&desc_layout_ci, None)? };

    // Push constant range: 64 bytes (enough for all three shaders)
    let pc_range = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .offset(0)
        .size(64);
    let pipe_layout_ci = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(std::slice::from_ref(&desc_layout))
        .push_constant_ranges(std::slice::from_ref(&pc_range));
    let pipe_layout = unsafe { device.create_pipeline_layout(&pipe_layout_ci, None)? };

    // Create shader modules
    let cas_mod = unsafe { create_shader_module(device, CAS_SPV)? };
    let easu_mod = unsafe { create_shader_module(device, FSR_EASU_SPV)? };
    let rcas_mod = unsafe { create_shader_module(device, FSR_RCAS_SPV)? };

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
            &[make_ci(cas_mod), make_ci(easu_mod), make_ci(rcas_mod)],
            None,
        ).map_err(|(_, e)| e)?
    };
    let cas_pipeline = pipelines[0];
    let easu_pipeline = pipelines[1];
    let rcas_pipeline = pipelines[2];

    // Clean up shader modules (no longer needed after pipeline creation)
    unsafe {
        device.destroy_shader_module(cas_mod, None);
        device.destroy_shader_module(easu_mod, None);
        device.destroy_shader_module(rcas_mod, None);
    }

    // Sampler for input texture (bilinear)
    let sampler_ci = vk::SamplerCreateInfo::default()
        .mag_filter(vk::Filter::LINEAR)
        .min_filter(vk::Filter::LINEAR)
        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE);
    let sampler = unsafe { device.create_sampler(&sampler_ci, None)? };

    // Image view for frame_image (so compute shader can sample it)
    let frame_view_ci = vk::ImageViewCreateInfo::default()
        .image(frame_image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(vk::Format::B8G8R8A8_UNORM)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        });
    let frame_image_view = unsafe { device.create_image_view(&frame_view_ci, None)? };

    // Intermediate images at output resolution
    let (compute_a, compute_a_mem, compute_a_view) = create_storage_image(device, mem_props, out_w, out_h)?;
    let (compute_b, compute_b_mem, compute_b_view) = create_storage_image(device, mem_props, out_w, out_h)?;

    // Descriptor pool: 4 sets × (1 sampler + 1 storage)
    // Sets 0-1: compute scaling, Sets 2-3: OSD blend (OSD→compute_a, OSD→compute_b)
    let pool_sizes = [
        vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(4),
        vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_IMAGE)
            .descriptor_count(4),
    ];
    let pool_ci = vk::DescriptorPoolCreateInfo::default()
        .max_sets(4)
        .pool_sizes(&pool_sizes);
    let desc_pool = unsafe { device.create_descriptor_pool(&pool_ci, None)? };

    let layouts = [desc_layout, desc_layout];
    let alloc_ci = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(desc_pool)
        .set_layouts(&layouts);
    let desc_sets = unsafe { device.allocate_descriptor_sets(&alloc_ci)? };

    // Set 0: frame_image → compute_a (for CAS or EASU)
    let frame_sampler_info = vk::DescriptorImageInfo::default()
        .sampler(sampler)
        .image_view(frame_image_view)
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    let a_storage_info = vk::DescriptorImageInfo::default()
        .image_view(compute_a_view)
        .image_layout(vk::ImageLayout::GENERAL);
    // Set 1: compute_a → compute_b (for RCAS)
    let a_sampler_info = vk::DescriptorImageInfo::default()
        .sampler(sampler)
        .image_view(compute_a_view)
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    let b_storage_info = vk::DescriptorImageInfo::default()
        .image_view(compute_b_view)
        .image_layout(vk::ImageLayout::GENERAL);

    let writes = [
        vk::WriteDescriptorSet::default()
            .dst_set(desc_sets[0])
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(std::slice::from_ref(&frame_sampler_info)),
        vk::WriteDescriptorSet::default()
            .dst_set(desc_sets[0])
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .image_info(std::slice::from_ref(&a_storage_info)),
        vk::WriteDescriptorSet::default()
            .dst_set(desc_sets[1])
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(std::slice::from_ref(&a_sampler_info)),
        vk::WriteDescriptorSet::default()
            .dst_set(desc_sets[1])
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .image_info(std::slice::from_ref(&b_storage_info)),
    ];
    unsafe { device.update_descriptor_sets(&writes, &[]) };

    Ok((
        desc_layout, pipe_layout,
        cas_pipeline, easu_pipeline, rcas_pipeline,
        desc_pool, desc_sets,
        frame_image_view, sampler,
        compute_a, compute_a_mem, compute_a_view,
        compute_b, compute_b_mem, compute_b_view,
    ))
}

/// Create OSD alpha-blend composite pipeline and descriptor sets.
/// Reuses the existing descriptor layout and pipeline layout from the compute scaling pipeline.
fn create_osd_blend_pipeline(
    device: &ash::Device,
    desc_layout: vk::DescriptorSetLayout,
    pipe_layout: vk::PipelineLayout,
    desc_pool: vk::DescriptorPool,
    sampler: vk::Sampler,
    osd_image: vk::Image,
    compute_a_view: vk::ImageView,
    compute_b_view: vk::ImageView,
) -> anyhow::Result<(vk::Pipeline, vk::ImageView, Vec<vk::DescriptorSet>)> {
    // OSD image view (B8G8R8A8 — shader will swizzle)
    let osd_view_ci = vk::ImageViewCreateInfo::default()
        .image(osd_image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(vk::Format::B8G8R8A8_UNORM)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        });
    let osd_image_view = unsafe { device.create_image_view(&osd_view_ci, None)? };

    // Create OSD blend compute pipeline
    let osd_mod = unsafe { create_shader_module(device, OSD_BLEND_SPV)? };
    let entry = unsafe { CStr::from_bytes_with_nul_unchecked(b"main\0") };
    let ci = vk::ComputePipelineCreateInfo::default()
        .stage(
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(osd_mod)
                .name(entry),
        )
        .layout(pipe_layout);
    let pipelines = unsafe {
        device.create_compute_pipelines(vk::PipelineCache::null(), &[ci], None)
            .map_err(|(_, e)| e)?
    };
    let osd_blend_pipeline = pipelines[0];
    unsafe { device.destroy_shader_module(osd_mod, None) };

    // Allocate 2 descriptor sets: OSD→compute_a and OSD→compute_b
    let layouts = [desc_layout, desc_layout];
    let alloc_ci = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(desc_pool)
        .set_layouts(&layouts);
    let osd_desc_sets = unsafe { device.allocate_descriptor_sets(&alloc_ci)? };

    // OSD sampler info (nearest filtering for pixel-perfect OSD)
    let osd_sampler_info = vk::DescriptorImageInfo::default()
        .sampler(sampler)
        .image_view(osd_image_view)
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    let a_storage_info = vk::DescriptorImageInfo::default()
        .image_view(compute_a_view)
        .image_layout(vk::ImageLayout::GENERAL);
    let b_storage_info = vk::DescriptorImageInfo::default()
        .image_view(compute_b_view)
        .image_layout(vk::ImageLayout::GENERAL);

    let writes = [
        // Set 0: OSD → compute_a
        vk::WriteDescriptorSet::default()
            .dst_set(osd_desc_sets[0])
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(std::slice::from_ref(&osd_sampler_info)),
        vk::WriteDescriptorSet::default()
            .dst_set(osd_desc_sets[0])
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .image_info(std::slice::from_ref(&a_storage_info)),
        // Set 1: OSD → compute_b
        vk::WriteDescriptorSet::default()
            .dst_set(osd_desc_sets[1])
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(std::slice::from_ref(&osd_sampler_info)),
        vk::WriteDescriptorSet::default()
            .dst_set(osd_desc_sets[1])
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .image_info(std::slice::from_ref(&b_storage_info)),
    ];
    unsafe { device.update_descriptor_sets(&writes, &[]) };

    Ok((osd_blend_pipeline, osd_image_view, osd_desc_sets))
}

fn create_swapchain(
    surface_fn: &ash::khr::surface::Instance,
    swapchain_fn: &ash::khr::swapchain::Device,
    _device: &ash::Device,
    physical_device: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
    win_w: u32,
    win_h: u32,
    old_swapchain: vk::SwapchainKHR,
    preferred_mode: Option<vk::PresentModeKHR>,
    debug: bool,
) -> anyhow::Result<(vk::SwapchainKHR, Vec<vk::Image>, vk::Format, vk::Extent2D, vk::PresentModeKHR)> {
    let caps = unsafe { surface_fn.get_physical_device_surface_capabilities(physical_device, surface)? };
    let formats = unsafe { surface_fn.get_physical_device_surface_formats(physical_device, surface)? };
    let modes = unsafe { surface_fn.get_physical_device_surface_present_modes(physical_device, surface)? };

    // On Wayland, IMMEDIATE triggers wp_tearing_control_v1 which many
    // compositors reject with a fatal protocol error.  Filter it out.
    let on_wayland = std::env::var("WAYLAND_DISPLAY").is_ok();
    let modes: Vec<vk::PresentModeKHR> = if on_wayland {
        modes.into_iter().filter(|m| *m != vk::PresentModeKHR::IMMEDIATE).collect()
    } else {
        modes
    };

    // Prefer B8G8R8A8_UNORM (linear, matches our CPU-converted pixel data).
    // Avoid SRGB — the blit would apply gamma encoding, brightening the image.
    let format = formats.iter()
        .find(|f| f.format == vk::Format::B8G8R8A8_UNORM)
        .or_else(|| formats.iter().find(|f| f.format == vk::Format::R8G8B8A8_UNORM))
        .unwrap_or(&formats[0]);

    // Use preferred mode if available, otherwise fall back: MAILBOX → IMMEDIATE → FIFO
    let present_mode = if let Some(pref) = preferred_mode {
        if modes.contains(&pref) {
            pref
        } else if modes.contains(&vk::PresentModeKHR::MAILBOX) {
            vk::PresentModeKHR::MAILBOX
        } else if modes.contains(&vk::PresentModeKHR::IMMEDIATE) {
            vk::PresentModeKHR::IMMEDIATE
        } else {
            vk::PresentModeKHR::FIFO
        }
    } else if modes.contains(&vk::PresentModeKHR::MAILBOX) {
        vk::PresentModeKHR::MAILBOX
    } else if modes.contains(&vk::PresentModeKHR::IMMEDIATE) {
        vk::PresentModeKHR::IMMEDIATE
    } else {
        vk::PresentModeKHR::FIFO
    };

    if debug {
        eprintln!("vulkan: available present modes: {:?}", modes);
    }

    // Extent
    let extent = if caps.current_extent.width != u32::MAX {
        caps.current_extent
    } else {
        vk::Extent2D {
            width: win_w.clamp(caps.min_image_extent.width, caps.max_image_extent.width),
            height: win_h.clamp(caps.min_image_extent.height, caps.max_image_extent.height),
        }
    };

    // Image count: min+1 for MAILBOX, clamped to max
    let mut image_count = caps.min_image_count + 1;
    if caps.max_image_count > 0 && image_count > caps.max_image_count {
        image_count = caps.max_image_count;
    }

    let ci = vk::SwapchainCreateInfoKHR::default()
        .surface(surface)
        .min_image_count(image_count)
        .image_format(format.format)
        .image_color_space(format.color_space)
        .image_extent(extent)
        .image_array_layers(1)
        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST)
        .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
        .pre_transform(caps.current_transform)
        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
        .present_mode(present_mode)
        .clipped(true)
        .old_swapchain(old_swapchain);

    let swapchain = unsafe { swapchain_fn.create_swapchain(&ci, None)? };
    let images = unsafe { swapchain_fn.get_swapchain_images(swapchain)? };

    Ok((swapchain, images, format.format, extent, present_mode))
}

fn create_buffer(
    device: &ash::Device,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
    properties: vk::MemoryPropertyFlags,
) -> anyhow::Result<(vk::Buffer, vk::DeviceMemory)> {
    let ci = vk::BufferCreateInfo::default()
        .size(size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let buffer = unsafe { device.create_buffer(&ci, None)? };
    let reqs = unsafe { device.get_buffer_memory_requirements(buffer) };

    let mem_type = find_memory_type(mem_props, reqs.memory_type_bits, properties)
        .ok_or_else(|| anyhow::anyhow!("vulkan: no suitable memory type for buffer"))?;

    let alloc = vk::MemoryAllocateInfo::default()
        .allocation_size(reqs.size)
        .memory_type_index(mem_type);

    let memory = unsafe { device.allocate_memory(&alloc, None)? };
    unsafe { device.bind_buffer_memory(buffer, memory, 0)? };

    Ok((buffer, memory))
}

fn create_image(
    device: &ash::Device,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    width: u32,
    height: u32,
    format: vk::Format,
    usage: vk::ImageUsageFlags,
    properties: vk::MemoryPropertyFlags,
) -> anyhow::Result<(vk::Image, vk::DeviceMemory)> {
    let ci = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .extent(vk::Extent3D { width, height, depth: 1 })
        .mip_levels(1)
        .array_layers(1)
        .format(format)
        .tiling(vk::ImageTiling::OPTIMAL)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .samples(vk::SampleCountFlags::TYPE_1);

    let image = unsafe { device.create_image(&ci, None)? };
    let reqs = unsafe { device.get_image_memory_requirements(image) };

    let mem_type = find_memory_type(mem_props, reqs.memory_type_bits, properties)
        .ok_or_else(|| anyhow::anyhow!("vulkan: no suitable memory type for image"))?;

    let alloc = vk::MemoryAllocateInfo::default()
        .allocation_size(reqs.size)
        .memory_type_index(mem_type);

    let memory = unsafe { device.allocate_memory(&alloc, None)? };
    unsafe { device.bind_image_memory(image, memory, 0)? };

    Ok((image, memory))
}

fn find_memory_type(
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    type_filter: u32,
    properties: vk::MemoryPropertyFlags,
) -> Option<u32> {
    for i in 0..mem_props.memory_type_count {
        if (type_filter & (1 << i)) != 0
            && mem_props.memory_types[i as usize].property_flags.contains(properties)
        {
            return Some(i);
        }
    }
    None
}

unsafe fn transition_image_layout(
    device: &ash::Device,
    cb: vk::CommandBuffer,
    image: vk::Image,
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
            base_mip_level: 0,
            level_count: 1,
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

fn fit_rect(src_w: u32, src_h: u32, dst_w: u32, dst_h: u32, mode: crate::config::AspectMode) -> (i32, i32, u32, u32) {
    use crate::config::AspectMode;
    match mode {
        AspectMode::Stretch => (0, 0, dst_w, dst_h),
        AspectMode::Zoom => {
            let scale_w = dst_w as f64 / src_w as f64;
            let scale_h = dst_h as f64 / src_h as f64;
            let scale = scale_w.max(scale_h);
            let w = (src_w as f64 * scale) as u32;
            let h = (src_h as f64 * scale) as u32;
            let x = (dst_w as i32 - w as i32) / 2;
            let y = (dst_h as i32 - h as i32) / 2;
            (x, y, w, h)
        }
        AspectMode::Preserve => {
            let scale_w = dst_w as f64 / src_w as f64;
            let scale_h = dst_h as f64 / src_h as f64;
            let scale = scale_w.min(scale_h);
            let w = (src_w as f64 * scale) as u32;
            let h = (src_h as f64 * scale) as u32;
            let x = ((dst_w - w) / 2) as i32;
            let y = ((dst_h - h) / 2) as i32;
            (x, y, w, h)
        }
    }
}

// --- YUV → RGBA CPU conversion ---

fn yuv_to_rgba(src: &[u8], w: u32, h: u32, pixfmt: u32, dst: &mut [u8]) {
    match pixfmt {
        V4L2_PIX_FMT_NV12 => nv12_to_rgba(src, w, h, dst),
        V4L2_PIX_FMT_YUYV => yuyv_to_rgba(src, w, h, dst),
        V4L2_PIX_FMT_UYVY => uyvy_to_rgba(src, w, h, dst),
        V4L2_PIX_FMT_XRGB32 => xrgb_to_rgba(src, w, h, dst),
        V4L2_PIX_FMT_P010 => p010_to_rgba(src, w, h, dst),
        PIXFMT_RGB24 => rgb24_to_rgba(src, w, h, dst),
        _ => nv12_to_rgba(src, w, h, dst),
    }
}

fn yuv_to_rgb(y: u8, u: u8, v: u8) -> (u8, u8, u8) {
    // BT.601 limited range (matches GL shader: y-=0.0625, *1.164)
    let y = (y as i32 - 16).max(0) as i32;
    let u = u as i32 - 128;
    let v = v as i32 - 128;
    // 1.164 * y ≈ (y * 298) >> 8
    // 1.596 * v ≈ (v * 409) >> 8
    // 0.392 * u ≈ (u * 100) >> 8
    // 0.813 * v ≈ (v * 208) >> 8
    // 2.017 * u ≈ (u * 516) >> 8
    let c = y * 298;
    let r = ((c + v * 409 + 128) >> 8).clamp(0, 255) as u8;
    let g = ((c - u * 100 - v * 208 + 128) >> 8).clamp(0, 255) as u8;
    let b = ((c + u * 516 + 128) >> 8).clamp(0, 255) as u8;
    (r, g, b)
}

fn nv12_to_rgba(src: &[u8], w: u32, h: u32, dst: &mut [u8]) {
    let w = w as usize;
    let h = h as usize;
    let y_plane = &src[..w * h];
    let uv_plane = &src[w * h..];

    for row in 0..h {
        for col in 0..w {
            let yi = row * w + col;
            let uvi = (row / 2) * w + (col & !1);
            let y = y_plane[yi];
            let u = uv_plane.get(uvi).copied().unwrap_or(128);
            let v = uv_plane.get(uvi + 1).copied().unwrap_or(128);
            let (r, g, b) = yuv_to_rgb(y, u, v);
            let di = (row * w + col) * 4;
            dst[di] = b;
            dst[di + 1] = g;
            dst[di + 2] = r;
            dst[di + 3] = 255;
        }
    }
}

fn yuyv_to_rgba(src: &[u8], w: u32, h: u32, dst: &mut [u8]) {
    let w = w as usize;
    let h = h as usize;
    for row in 0..h {
        for col in (0..w).step_by(2) {
            let si = (row * w + col) * 2;
            let y0 = src[si];
            let u = src[si + 1];
            let y1 = src.get(si + 2).copied().unwrap_or(y0);
            let v = src.get(si + 3).copied().unwrap_or(128);

            let (r0, g0, b0) = yuv_to_rgb(y0, u, v);
            let (r1, g1, b1) = yuv_to_rgb(y1, u, v);

            let di = (row * w + col) * 4;
            dst[di] = b0; dst[di + 1] = g0; dst[di + 2] = r0; dst[di + 3] = 255;
            if col + 1 < w {
                dst[di + 4] = b1; dst[di + 5] = g1; dst[di + 6] = r1; dst[di + 7] = 255;
            }
        }
    }
}

fn uyvy_to_rgba(src: &[u8], w: u32, h: u32, dst: &mut [u8]) {
    let w = w as usize;
    let h = h as usize;
    for row in 0..h {
        for col in (0..w).step_by(2) {
            let si = (row * w + col) * 2;
            let u = src[si];
            let y0 = src[si + 1];
            let v = src.get(si + 2).copied().unwrap_or(128);
            let y1 = src.get(si + 3).copied().unwrap_or(y0);

            let (r0, g0, b0) = yuv_to_rgb(y0, u, v);
            let (r1, g1, b1) = yuv_to_rgb(y1, u, v);

            let di = (row * w + col) * 4;
            dst[di] = b0; dst[di + 1] = g0; dst[di + 2] = r0; dst[di + 3] = 255;
            if col + 1 < w {
                dst[di + 4] = b1; dst[di + 5] = g1; dst[di + 6] = r1; dst[di + 7] = 255;
            }
        }
    }
}

fn xrgb_to_rgba(src: &[u8], w: u32, h: u32, dst: &mut [u8]) {
    let n = (w * h * 4) as usize;
    dst[..n].copy_from_slice(&src[..n]);
    for i in (3..n).step_by(4) { dst[i] = 255; }
}

fn p010_to_rgba(src: &[u8], w: u32, h: u32, dst: &mut [u8]) {
    let w = w as usize;
    let h = h as usize;
    let y_plane = &src[..w * h * 2];
    let uv_plane = &src[w * h * 2..];
    for row in 0..h {
        for col in 0..w {
            let yi = (row * w + col) * 2;
            let y8 = y_plane.get(yi + 1).copied().unwrap_or(0);
            let uvi = (row / 2) * w * 2 + (col & !1) * 2;
            let u8v = uv_plane.get(uvi + 1).copied().unwrap_or(128);
            let v8v = uv_plane.get(uvi + 3).copied().unwrap_or(128);
            let (r, g, b) = yuv_to_rgb(y8, u8v, v8v);
            let di = (row * w + col) * 4;
            dst[di] = b; dst[di + 1] = g; dst[di + 2] = r; dst[di + 3] = 255;
        }
    }
}

fn rgb24_to_rgba(src: &[u8], _w: u32, _h: u32, dst: &mut [u8]) {
    for (i, chunk) in src.chunks_exact(3).enumerate() {
        let di = i * 4;
        dst[di] = chunk[2]; dst[di + 1] = chunk[1]; dst[di + 2] = chunk[0]; dst[di + 3] = 255;
    }
}
