use crate::metal_atlas::MetalAtlas;
use anyhow::{Context as _, Result};
use block::ConcreteBlock;
use cocoa::{
    base::{NO, YES},
    foundation::{NSSize, NSUInteger},
    quartzcore::AutoresizingMask,
};
use gpui::{
    AtlasTextureId, Background, Bounds, ContentMask, Corners, DevicePixels, PaintSurface, Path,
    Point, PrimitiveBatch, ScaledPixels, Scene, Size, point, size,
};
#[cfg(any(test, feature = "test-support"))]
use image::RgbaImage;

use core_foundation::base::TCFType;
use core_video::{
    metal_texture::CVMetalTextureGetTexture, metal_texture_cache::CVMetalTextureCache,
    pixel_buffer::kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
};
use foreign_types::{ForeignType, ForeignTypeRef};
use metal::{
    CAMetalLayer, CommandQueue, MTLGPUFamily, MTLPixelFormat, MTLResourceOptions, NSRange,
};
use objc::{self, msg_send, sel, sel_impl};
use parking_lot::Mutex;

use std::{
    cell::Cell,
    ffi::{CStr, c_char, c_void},
    mem,
    mem::MaybeUninit,
    ops::Range,
    ptr, slice,
    sync::Arc,
};

// Exported to metal
pub(crate) type PointF = gpui::Point<f32>;

#[cfg(not(feature = "runtime_shaders"))]
const SHADERS_METALLIB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/shaders.metallib"));
#[cfg(feature = "runtime_shaders")]
const SHADERS_SOURCE_FILE: &str = include_str!(concat!(env!("OUT_DIR"), "/stitched_shaders.metal"));
// Use 4x MSAA, all devices support it.
// https://developer.apple.com/documentation/metal/mtldevice/1433355-supportstexturesamplecount
const PATH_SAMPLE_COUNT: u32 = 4;
/// Metal requires the offset a buffer is bound at to be 256-byte aligned.
const INSTANCE_BUFFER_ALIGNMENT: usize = 256;
const MAX_INSTANCE_BUFFER_SIZE: usize = 256 * 1024 * 1024;

fn trace_external_image_render(detail: std::fmt::Arguments<'_>) {
    if std::env::var_os("EXTERNAL_IMAGE_LEASE_TRACE").is_some() {
        static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
        let elapsed = EPOCH.get_or_init(std::time::Instant::now).elapsed();
        eprintln!("[GPUI Metal +{}us] {detail}", elapsed.as_micros());
    }
}

pub type Context = Arc<Mutex<InstanceBufferPool>>;
pub type Renderer = MetalRenderer;

pub unsafe fn new_renderer(
    context: self::Context,
    _native_window: *mut c_void,
    _native_view: *mut c_void,
    _bounds: gpui::Size<f32>,
    transparent: bool,
) -> Renderer {
    MetalRenderer::new(context, transparent)
}

pub struct InstanceBufferPool {
    buffer_size: usize,
    buffers: Vec<metal::Buffer>,
}

impl Default for InstanceBufferPool {
    fn default() -> Self {
        Self {
            buffer_size: 2 * 1024 * 1024,
            buffers: Vec::new(),
        }
    }
}

pub(crate) struct InstanceBuffer {
    metal_buffer: metal::Buffer,
    size: usize,
}

impl InstanceBufferPool {
    pub(crate) fn reset(&mut self, buffer_size: usize) {
        self.buffer_size = buffer_size;
        self.buffers.clear();
    }

    pub(crate) fn acquire(
        &mut self,
        device: &metal::Device,
        unified_memory: bool,
    ) -> InstanceBuffer {
        let buffer = self.buffers.pop().unwrap_or_else(|| {
            let options = if unified_memory {
                MTLResourceOptions::StorageModeShared
                    // Buffers are write only which can benefit from the combined cache
                    // https://developer.apple.com/documentation/metal/mtlresourceoptions/cpucachemodewritecombined
                    | MTLResourceOptions::CPUCacheModeWriteCombined
            } else {
                MTLResourceOptions::StorageModeManaged
            };

            device.new_buffer(self.buffer_size as u64, options)
        });
        InstanceBuffer {
            metal_buffer: buffer,
            size: self.buffer_size,
        }
    }

    pub(crate) fn release(&mut self, buffer: InstanceBuffer) {
        if buffer.size == self.buffer_size {
            self.buffers.push(buffer.metal_buffer)
        }
    }
}

pub struct MetalRenderer {
    device: metal::Device,
    layer: Option<metal::MetalLayer>,
    is_apple_gpu: bool,
    is_unified_memory: bool,
    presents_with_transaction: bool,
    /// For headless rendering, tracks whether output should be opaque
    opaque: bool,
    command_queue: CommandQueue,
    paths_rasterization_pipeline_state: metal::RenderPipelineState,
    path_sprites_pipeline_state: metal::RenderPipelineState,
    shadows_pipeline_state: metal::RenderPipelineState,
    quads_pipeline_state: metal::RenderPipelineState,
    underlines_pipeline_state: metal::RenderPipelineState,
    monochrome_sprites_pipeline_state: metal::RenderPipelineState,
    polychrome_sprites_pipeline_state: metal::RenderPipelineState,
    surfaces_pipeline_state: metal::RenderPipelineState,
    bgra_surfaces_pipeline_state: metal::RenderPipelineState,
    unit_vertices: metal::Buffer,
    #[allow(clippy::arc_with_non_send_sync)]
    instance_buffer_pool: Arc<Mutex<InstanceBufferPool>>,
    sprite_atlas: Arc<MetalAtlas>,
    core_video_texture_cache: core_video::metal_texture_cache::CVMetalTextureCache,
    external_surface_textures: std::collections::HashMap<
        (u64, u64, usize),
        core_video::metal_texture::CVMetalTexture,
    >,
    external_surface_texture_imports: usize,
    path_intermediate_texture: Option<metal::Texture>,
    path_intermediate_msaa_texture: Option<metal::Texture>,
    path_sample_count: u32,
    /// Offscreen render target reused across `render_scene` calls when
    /// rendering headlessly without reading pixels back.
    #[cfg(any(test, feature = "test-support"))]
    headless_render_target: Option<metal::Texture>,
}

#[repr(C)]
pub struct PathRasterizationVertex {
    pub xy_position: Point<ScaledPixels>,
    pub st_position: Point<f32>,
    pub color: Background,
    pub bounds: Bounds<ScaledPixels>,
}

impl MetalRenderer {
    /// Returns the identity of the IOSurface referenced by a CoreVideo pixel
    /// buffer. Backing IDs may be recycled when a WebContent backing pool is
    /// replaced, so texture caches must include the actual IOSurface identity.
    pub fn iosurface_identity(image_buffer: &core_video::pixel_buffer::CVPixelBuffer) -> Result<usize> {
        use core_video::pixel_buffer_io_surface::CVPixelBufferGetIOSurface;

        let iosurface = unsafe { CVPixelBufferGetIOSurface(image_buffer.as_concrete_TypeRef()) };
        anyhow::ensure!(!iosurface.is_null(), "pixel buffer has no IOSurface backing");
        let identity = unsafe { io_surface::IOSurfaceGetID(iosurface) };
        anyhow::ensure!(identity != 0, "IOSurface has an invalid identity");
        Ok(identity as usize)
    }

    /// Builds the persistent, IPC-facing descriptor for an IOSurface backing.
    /// The returned Mach send right is owned by the descriptor and must be
    /// transferred as an attachment (never as its numeric port name).
    pub fn describe_iosurface_backing(
        image_buffer: &core_video::pixel_buffer::CVPixelBuffer,
        backing_id: u64,
        generation: u64,
    ) -> Result<gpui::MacGpuBackingDescriptor> {
        use core_video::pixel_buffer_io_surface::CVPixelBufferGetIOSurface;

        anyhow::ensure!(
            backing_id != 0 && generation != 0,
            "invalid backing identity"
        );
        let iosurface = unsafe { CVPixelBufferGetIOSurface(image_buffer.as_concrete_TypeRef()) };
        anyhow::ensure!(
            !iosurface.is_null(),
            "pixel buffer has no IOSurface backing"
        );
        let port = unsafe { io_surface::IOSurfaceCreateMachPort(iosurface) };
        anyhow::ensure!(
            port != mach2::port::MACH_PORT_NULL,
            "IOSurfaceCreateMachPort failed"
        );
        let iosurface_port = unsafe { gpui::MacIOSurfaceSendRight::from_owned_raw(port) };
        Ok(gpui::MacGpuBackingDescriptor {
            backing_id,
            generation,
            width: image_buffer.get_width() as u32,
            height: image_buffer.get_height() as u32,
            pixel_format: image_buffer.get_pixel_format(),
            iosurface_port,
        })
    }

    /// Imports the backing using a received IOSurface Mach send right.
    pub fn import_iosurface_backing(
        descriptor: &gpui::MacGpuBackingDescriptor,
    ) -> Result<core_video::pixel_buffer::CVPixelBuffer> {
        anyhow::ensure!(
            descriptor.backing_id != 0 && descriptor.generation != 0,
            "invalid backing identity"
        );
        anyhow::ensure!(
            descriptor.pixel_format == core_video::pixel_buffer::kCVPixelFormatType_32BGRA,
            "unsupported IOSurface pixel format"
        );
        let raw_surface =
            unsafe { io_surface::IOSurfaceLookupFromMachPort(descriptor.iosurface_port.as_raw()) };
        anyhow::ensure!(!raw_surface.is_null(), "IOSurfaceLookupFromMachPort failed");
        let iosurface = io_surface::IOSurface { obj: raw_surface };
        anyhow::ensure!(
            unsafe { io_surface::IOSurfaceGetWidth(raw_surface) } == descriptor.width as usize,
            "IOSurface width differs from descriptor"
        );
        anyhow::ensure!(
            unsafe { io_surface::IOSurfaceGetHeight(raw_surface) } == descriptor.height as usize,
            "IOSurface height differs from descriptor"
        );
        let pixel_buffer = core_video::pixel_buffer::CVPixelBuffer::from_io_surface(
            &iosurface, None,
        )
        .map_err(|status| anyhow::anyhow!("CVPixelBufferCreateWithIOSurface failed: {status}"))?;
        Ok(pixel_buffer)
    }

    /// Creates a GPU-rendered IOSurface test frame using the same device
    /// selection policy as the GPUI Metal renderer.
    pub fn create_standalone_test_frame(
        width: usize,
        height: usize,
        backing_id: u64,
        generation: u64,
        frame_id: u64,
    ) -> anyhow::Result<gpui::MacExternalImageFrame> {
        let (frame, command_buffer) = Self::create_standalone_test_frame_deferred(
            width, height, backing_id, generation, frame_id,
        )?;
        command_buffer.commit();
        Ok(frame)
    }

    /// Creates a GPU-rendered test frame but leaves its producer command
    /// buffer uncommitted so cross-process tests can deliver the frame
    /// descriptor before GPU completion.
    pub fn create_standalone_test_frame_deferred(
        width: usize,
        height: usize,
        backing_id: u64,
        generation: u64,
        frame_id: u64,
    ) -> anyhow::Result<(gpui::MacExternalImageFrame, metal::CommandBuffer)> {
        use core_foundation::{
            base::{CFType, TCFType},
            dictionary::CFDictionary,
            string::CFString,
        };
        use core_video::pixel_buffer::{
            CVPixelBuffer, kCVPixelBufferIOSurfacePropertiesKey, kCVPixelFormatType_32BGRA,
        };
        use core_video::pixel_buffer_io_surface::CVPixelBufferGetIOSurface;

        let device = Self::create_device();
        let empty_surface_properties: CFDictionary<CFString, CFType> =
            CFDictionary::from_CFType_pairs(&[]);
        let iosurface_key =
            unsafe { CFString::wrap_under_get_rule(kCVPixelBufferIOSurfacePropertiesKey) };
        let attributes: CFDictionary<CFString, CFType> = CFDictionary::from_CFType_pairs(&[(
            iosurface_key,
            empty_surface_properties.as_CFType(),
        )]);
        let image_buffer =
            CVPixelBuffer::new(kCVPixelFormatType_32BGRA, width, height, Some(&attributes))
                .map_err(|error| {
                    anyhow::anyhow!("failed to allocate BGRA IOSurface pixel buffer: {error}")
                })?;
        // CVPixelBufferGetIOSurface returns a borrowed reference. Do not wrap it
        // using io-surface's create-rule wrapper: that would release CoreVideo's
        // owned IOSurface reference when the temporary wrapper drops.
        let iosurface = unsafe { CVPixelBufferGetIOSurface(image_buffer.as_concrete_TypeRef()) };
        anyhow::ensure!(
            !iosurface.is_null(),
            "CoreVideo did not allocate an IOSurface"
        );
        let iosurface_identity = iosurface as usize;

        let texture_cache = CVMetalTextureCache::new(None, device.clone(), None)
            .map_err(|error| anyhow::anyhow!("failed to create producer texture cache: {error}"))?;
        let cv_texture = texture_cache
            .create_texture_from_image(
                image_buffer.as_concrete_TypeRef(),
                None,
                MTLPixelFormat::BGRA8Unorm,
                width,
                height,
                0,
            )
            .map_err(|error| {
                anyhow::anyhow!("failed to create Metal texture for IOSurface: {error}")
            })?;
        let raw_texture = unsafe { CVMetalTextureGetTexture(cv_texture.as_concrete_TypeRef()) };
        anyhow::ensure!(
            !raw_texture.is_null(),
            "Metal failed to create producer IOSurface texture"
        );
        let texture = unsafe { metal::TextureRef::from_ptr(raw_texture as *mut _) };

        let source = r#"
            #include <metal_stdlib>
            using namespace metal;
            struct Out { float4 position [[position]]; };
            vertex Out v(uint id [[vertex_id]]) {
                constexpr float2 p[3] = { float2(-1.0, -1.0), float2(3.0, -1.0), float2(-1.0, 3.0) };
                return { float4(p[id], 0.0, 1.0) };
            }
            fragment float4 f(Out in [[stage_in]],
                              constant float2 *dimensions [[buffer(0)]],
                              constant uint *marker_top_left [[buffer(1)]]) {
                float2 p = in.position.xy;
                bool marker = *marker_top_left
                    ? (p.x < 36.0 && p.y < 36.0)
                    : (p.x > dimensions->x - 36.0 && p.y > dimensions->y - 36.0);
                if (marker) return float4(0.0, 0.0, 0.0, 1.0);
                if (p.y < dimensions->y * 0.5)
                    return p.x < dimensions->x * 0.5 ? float4(1, 0, 0, 1) : float4(0, 1, 0, 1);
                return p.x < dimensions->x * 0.5 ? float4(0, 0, 1, 1) : float4(1, 1, 1, 1);
            }
        "#;
        let library = device
            .new_library_with_source(source, &metal::CompileOptions::new())
            .map_err(|error| anyhow::anyhow!("failed to compile IOSurface test shader: {error}"))?;
        let pipeline_descriptor = metal::RenderPipelineDescriptor::new();
        let vertex_function = library.get_function("v", None).map_err(|error| {
            anyhow::anyhow!("missing IOSurface producer vertex shader: {error}")
        })?;
        let fragment_function = library.get_function("f", None).map_err(|error| {
            anyhow::anyhow!("missing IOSurface producer fragment shader: {error}")
        })?;
        pipeline_descriptor.set_vertex_function(Some(vertex_function.as_ref()));
        pipeline_descriptor.set_fragment_function(Some(fragment_function.as_ref()));
        pipeline_descriptor
            .color_attachments()
            .object_at(0)
            .unwrap()
            .set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        let pipeline = device
            .new_render_pipeline_state(&pipeline_descriptor)
            .map_err(|error| {
                anyhow::anyhow!("failed to create IOSurface producer pipeline: {error}")
            })?;

        let event = device.new_shared_event();
        let queue = device.new_command_queue();
        let command_buffer = queue.new_command_buffer().to_owned();
        let render_pass = metal::RenderPassDescriptor::new();
        let color_attachment = render_pass.color_attachments().object_at(0).unwrap();
        color_attachment.set_texture(Some(texture));
        color_attachment.set_load_action(metal::MTLLoadAction::DontCare);
        color_attachment.set_store_action(metal::MTLStoreAction::Store);
        let encoder = command_buffer.new_render_command_encoder(render_pass);
        encoder.set_render_pipeline_state(&pipeline);
        let dimensions = [width as f32, height as f32];
        let marker_top_left: u32 = u32::from(frame_id == 1);
        encoder.set_fragment_bytes(
            0,
            mem::size_of_val(&dimensions) as u64,
            dimensions.as_ptr().cast(),
        );
        encoder.set_fragment_bytes(
            1,
            mem::size_of_val(&marker_top_left) as u64,
            &marker_top_left as *const _ as *const c_void,
        );
        encoder.draw_primitives(metal::MTLPrimitiveType::Triangle, 0, 3);
        encoder.end_encoding();
        command_buffer.encode_signal_event(&event, 1);

        let completion_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let completed = completion_count.clone();
        let frame = gpui::MacExternalImageFrame::new(
            backing_id,
            generation,
            frame_id,
            iosurface_identity,
            kCVPixelFormatType_32BGRA,
            image_buffer,
            event,
            1,
            move || {
                completed.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
            },
        );
        Ok((frame, command_buffer))
    }

    /// Creates a new MetalRenderer with a CAMetalLayer for window-based rendering.
    pub fn new(instance_buffer_pool: Arc<Mutex<InstanceBufferPool>>, transparent: bool) -> Self {
        let device = Self::create_device();

        let layer = metal::MetalLayer::new();
        layer.set_device(&device);
        layer.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        // Support direct-to-display rendering if the window is not transparent
        // https://developer.apple.com/documentation/metal/managing-your-game-window-for-metal-in-macos
        layer.set_opaque(!transparent);
        layer.set_maximum_drawable_count(3);
        // Allow texture reading for visual tests (captures screenshots without ScreenCaptureKit)
        #[cfg(any(test, feature = "test-support"))]
        layer.set_framebuffer_only(false);
        unsafe {
            let _: () = msg_send![&*layer, setAllowsNextDrawableTimeout: NO];
            let _: () = msg_send![&*layer, setNeedsDisplayOnBoundsChange: YES];
            let _: () = msg_send![
                &*layer,
                setAutoresizingMask: AutoresizingMask::WIDTH_SIZABLE
                    | AutoresizingMask::HEIGHT_SIZABLE
            ];
        }

        Self::new_internal(device, Some(layer), !transparent, instance_buffer_pool)
    }

    /// Creates a new headless MetalRenderer for offscreen rendering without a window.
    ///
    /// This renderer can render scenes to images without requiring a CAMetalLayer,
    /// window, or AppKit. Use `render_scene_to_image()` to render scenes.
    #[cfg(any(test, feature = "test-support"))]
    pub fn new_headless(instance_buffer_pool: Arc<Mutex<InstanceBufferPool>>) -> Self {
        let device = Self::create_device();
        Self::new_internal(device, None, true, instance_buffer_pool)
    }

    fn create_device() -> metal::Device {
        // Prefer low‐power integrated GPUs on Intel Mac. On Apple
        // Silicon, there is only ever one GPU, so this is equivalent to
        // `metal::Device::system_default()`.
        if let Some(d) = metal::Device::all()
            .into_iter()
            .min_by_key(|d| (d.is_removable(), !d.is_low_power()))
        {
            d
        } else {
            // For some reason `all()` can return an empty list, see https://github.com/zed-industries/zed/issues/37689
            // In that case, we fall back to the system default device.
            log::error!(
                "Unable to enumerate Metal devices; attempting to use system default device"
            );
            metal::Device::system_default().unwrap_or_else(|| {
                log::error!("unable to access a compatible graphics device");
                std::process::exit(1);
            })
        }
    }

    /// Returns the identity of the Metal device selected by GPUI's renderer.
    /// The registry ID is stable across processes and can be compared with a
    /// producer device to verify that both target the same physical GPU.
    pub fn device_identity() -> MetalDeviceIdentity {
        let device = Self::create_device();
        MetalDeviceIdentity {
            name: device.name().to_string(),
            registry_id: device.registry_id(),
        }
    }

    fn new_internal(
        device: metal::Device,
        layer: Option<metal::MetalLayer>,
        opaque: bool,
        instance_buffer_pool: Arc<Mutex<InstanceBufferPool>>,
    ) -> Self {
        log::info!(
            "GPUI Metal device: {} (registryID={})",
            device.name(),
            device.registry_id()
        );
        #[cfg(feature = "runtime_shaders")]
        let library = device
            .new_library_with_source(&SHADERS_SOURCE_FILE, &metal::CompileOptions::new())
            .expect("error building metal library");
        #[cfg(not(feature = "runtime_shaders"))]
        let library = device
            .new_library_with_data(SHADERS_METALLIB)
            .expect("error building metal library");

        fn to_float2_bits(point: PointF) -> u64 {
            let mut output = point.y.to_bits() as u64;
            output <<= 32;
            output |= point.x.to_bits() as u64;
            output
        }

        // Shared memory can be used only if CPU and GPU share the same memory space.
        // https://developer.apple.com/documentation/metal/setting-resource-storage-modes
        let is_unified_memory = device.has_unified_memory();
        // Apple GPU families support memoryless textures, which can significantly reduce
        // memory usage by keeping render targets in on-chip tile memory instead of
        // allocating backing store in system memory.
        // https://developer.apple.com/documentation/metal/mtlgpufamily
        let is_apple_gpu = device.supports_family(MTLGPUFamily::Apple1);

        let unit_vertices = [
            to_float2_bits(point(0., 0.)),
            to_float2_bits(point(1., 0.)),
            to_float2_bits(point(0., 1.)),
            to_float2_bits(point(0., 1.)),
            to_float2_bits(point(1., 0.)),
            to_float2_bits(point(1., 1.)),
        ];
        let unit_vertices = device.new_buffer_with_data(
            unit_vertices.as_ptr() as *const c_void,
            mem::size_of_val(&unit_vertices) as u64,
            if is_unified_memory {
                MTLResourceOptions::StorageModeShared
                    | MTLResourceOptions::CPUCacheModeWriteCombined
            } else {
                MTLResourceOptions::StorageModeManaged
            },
        );

        let paths_rasterization_pipeline_state = build_path_rasterization_pipeline_state(
            &device,
            &library,
            "paths_rasterization",
            "path_rasterization_vertex",
            "path_rasterization_fragment",
            MTLPixelFormat::BGRA8Unorm,
            PATH_SAMPLE_COUNT,
        );
        let path_sprites_pipeline_state = build_path_sprite_pipeline_state(
            &device,
            &library,
            "path_sprites",
            "path_sprite_vertex",
            "path_sprite_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );
        let shadows_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "shadows",
            "shadow_vertex",
            "shadow_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );
        let quads_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "quads",
            "quad_vertex",
            "quad_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );
        let underlines_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "underlines",
            "underline_vertex",
            "underline_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );
        let monochrome_sprites_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "monochrome_sprites",
            "monochrome_sprite_vertex",
            "monochrome_sprite_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );
        let polychrome_sprites_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "polychrome_sprites",
            "polychrome_sprite_vertex",
            "polychrome_sprite_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );
        let surfaces_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "surfaces",
            "surface_vertex",
            "surface_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );
        let bgra_surfaces_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "surfaces",
            "surface_vertex",
            "bgra_surface_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );

        let command_queue = device.new_command_queue();
        let sprite_atlas = Arc::new(MetalAtlas::new(device.clone(), is_apple_gpu));
        let core_video_texture_cache =
            CVMetalTextureCache::new(None, device.clone(), None).unwrap();

        Self {
            device,
            layer,
            presents_with_transaction: false,
            is_apple_gpu,
            is_unified_memory,
            opaque,
            command_queue,
            paths_rasterization_pipeline_state,
            path_sprites_pipeline_state,
            shadows_pipeline_state,
            quads_pipeline_state,
            underlines_pipeline_state,
            monochrome_sprites_pipeline_state,
            polychrome_sprites_pipeline_state,
            surfaces_pipeline_state,
            bgra_surfaces_pipeline_state,
            unit_vertices,
            instance_buffer_pool,
            sprite_atlas,
            core_video_texture_cache,
            external_surface_textures: std::collections::HashMap::new(),
            external_surface_texture_imports: 0,
            path_intermediate_texture: None,
            path_intermediate_msaa_texture: None,
            path_sample_count: PATH_SAMPLE_COUNT,
            #[cfg(any(test, feature = "test-support"))]
            headless_render_target: None,
        }
    }

    pub fn layer(&self) -> Option<&metal::MetalLayerRef> {
        self.layer.as_ref().map(|l| l.as_ref())
    }

    pub fn layer_ptr(&self) -> *mut CAMetalLayer {
        self.layer
            .as_ref()
            .map(|l| l.as_ptr())
            .unwrap_or(ptr::null_mut())
    }

    pub fn sprite_atlas(&self) -> &Arc<MetalAtlas> {
        &self.sprite_atlas
    }

    pub fn set_presents_with_transaction(&mut self, presents_with_transaction: bool) {
        self.presents_with_transaction = presents_with_transaction;
        if let Some(layer) = &self.layer {
            layer.set_presents_with_transaction(presents_with_transaction);
        }
    }

    pub fn update_drawable_size(&mut self, size: Size<DevicePixels>) {
        if let Some(layer) = &self.layer {
            let ns_size = NSSize {
                width: size.width.0 as f64,
                height: size.height.0 as f64,
            };
            unsafe {
                let _: () = msg_send![
                    layer.as_ref(),
                    setDrawableSize: ns_size
                ];
            }
        }
        self.update_path_intermediate_textures(size);
    }

    fn update_path_intermediate_textures(&mut self, size: Size<DevicePixels>) {
        // We are uncertain when this happens, but sometimes size can be 0 here. Most likely before
        // the layout pass on window creation. Zero-sized texture creation causes SIGABRT.
        // https://github.com/zed-industries/zed/issues/36229
        if size.width.0 <= 0 || size.height.0 <= 0 {
            self.path_intermediate_texture = None;
            self.path_intermediate_msaa_texture = None;
            return;
        }

        let texture_descriptor = metal::TextureDescriptor::new();
        texture_descriptor.set_width(size.width.0 as u64);
        texture_descriptor.set_height(size.height.0 as u64);
        texture_descriptor.set_pixel_format(metal::MTLPixelFormat::BGRA8Unorm);
        texture_descriptor.set_storage_mode(metal::MTLStorageMode::Private);
        texture_descriptor
            .set_usage(metal::MTLTextureUsage::RenderTarget | metal::MTLTextureUsage::ShaderRead);
        self.path_intermediate_texture = Some(self.device.new_texture(&texture_descriptor));

        if self.path_sample_count > 1 {
            // https://developer.apple.com/documentation/metal/choosing-a-resource-storage-mode-for-apple-gpus
            // Rendering MSAA textures are done in a single pass, so we can use memory-less storage on Apple Silicon
            let storage_mode = if self.is_apple_gpu {
                metal::MTLStorageMode::Memoryless
            } else {
                metal::MTLStorageMode::Private
            };

            let msaa_descriptor = texture_descriptor;
            msaa_descriptor.set_texture_type(metal::MTLTextureType::D2Multisample);
            msaa_descriptor.set_storage_mode(storage_mode);
            msaa_descriptor.set_sample_count(self.path_sample_count as _);
            self.path_intermediate_msaa_texture = Some(self.device.new_texture(&msaa_descriptor));
        } else {
            self.path_intermediate_msaa_texture = None;
        }
    }

    pub fn update_transparency(&mut self, transparent: bool) {
        self.opaque = !transparent;
        if let Some(layer) = &self.layer {
            layer.set_opaque(!transparent);
        }
    }

    pub fn destroy(&self) {
        // nothing to do
    }

    pub fn draw(&mut self, scene: &Scene) {
        let layer = match &self.layer {
            Some(l) => l.clone(),
            None => {
                log::error!(
                    "draw() called on headless renderer - use render_scene_to_image() instead"
                );
                return;
            }
        };
        let viewport_size = layer.drawable_size();
        let viewport_size: Size<DevicePixels> = size(
            (viewport_size.width.ceil() as i32).into(),
            (viewport_size.height.ceil() as i32).into(),
        );
        let drawable = if let Some(drawable) = layer.next_drawable() {
            drawable
        } else {
            log::error!(
                "failed to retrieve next drawable, drawable size: {:?}",
                viewport_size
            );
            return;
        };

        let command_buffer = match self.render_frame(scene, drawable.texture(), viewport_size) {
            Ok(command_buffer) => command_buffer,
            Err(error) => {
                log::error!("failed to render: {error:#}");
                return;
            }
        };

        if self.presents_with_transaction {
            command_buffer.commit();
            command_buffer.wait_until_scheduled();
            drawable.present();
            trace_external_image_render(format_args!(
                "command_buffer={} drawable_present_scheduled=transaction",
                command_buffer.label()
            ));
        } else {
            command_buffer.present_drawable(drawable);
            command_buffer.commit();
            trace_external_image_render(format_args!(
                "command_buffer={} drawable_present_scheduled=command_buffer",
                command_buffer.label()
            ));
        }
        trace_external_image_render(format_args!(
            "command_buffer={} committed status={:?}",
            command_buffer.label(),
            command_buffer.status()
        ));
    }

    fn render_frame(
        &mut self,
        scene: &Scene,
        texture: &metal::TextureRef,
        viewport_size: Size<DevicePixels>,
    ) -> Result<metal::CommandBuffer> {
        static NEXT_RENDER_ID: std::sync::atomic::AtomicU64 =
            std::sync::atomic::AtomicU64::new(1);
        let render_id = NEXT_RENDER_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let render_label = format!("GPUI external render {render_id}");
        let mut writer = InstanceBufferWriter::new(
            &self.device,
            &self.instance_buffer_pool,
            self.is_unified_memory,
        );
        let instance_bindings = write_instances(scene, &mut writer).with_context(|| {
            format!(
                "scene too large: {} paths, {} shadows, {} quads, {} underlines, {} mono, {} poly, {} surfaces",
                scene.paths.len(),
                scene.shadows.len(),
                scene.quads.len(),
                scene.underlines.len(),
                scene.monochrome_sprites.len(),
                scene.polychrome_sprites.len(),
                scene.surfaces.len(),
            )
        })?;
        let command_buffer = self.draw_primitives_to_texture(
            scene,
            &instance_bindings,
            &mut writer,
            texture,
            viewport_size,
            render_id,
        )?;
        command_buffer.set_label(&render_label);
        trace_external_image_render(format_args!(
            "command_buffer={render_label} webview_draw_encoded external_frames={}",
            scene
                .surfaces
                .iter()
                .filter(|surface| surface.external_image.is_some())
                .count()
        ));

        let instance_buffer_pool = self.instance_buffer_pool.clone();
        let instance_buffer = Cell::new(Some(writer.finish()));
        for surface in &scene.surfaces {
            if let Some(frame) = &surface.external_image {
                frame.mark_submitted();
            }
        }
        let completion_frames = scene
            .surfaces
            .iter()
            .filter_map(|surface| surface.external_image.as_ref())
            .cloned()
            .collect::<Vec<_>>();
        let block = ConcreteBlock::new(move |completed_buffer: &metal::CommandBufferRef| {
            if let Some(instance_buffer) = instance_buffer.take() {
                instance_buffer_pool.lock().release(instance_buffer);
            }
            let error: *mut objc::runtime::Object = unsafe { msg_send![completed_buffer, error] };
            let error_description = if error.is_null() {
                "none".to_string()
            } else {
                let description: *mut objc::runtime::Object =
                    unsafe { msg_send![error, localizedDescription] };
                if description.is_null() {
                    "unavailable".to_string()
                } else {
                    let utf8: *const c_char = unsafe { msg_send![description, UTF8String] };
                    if utf8.is_null() {
                        "unavailable".to_string()
                    } else {
                        unsafe { CStr::from_ptr(utf8) }
                            .to_string_lossy()
                            .into_owned()
                    }
                }
            };
            trace_external_image_render(format_args!(
                "command_buffer={} completed status={:?} error={:?}",
                completed_buffer.label(),
                completed_buffer.status(),
                error_description
            ));
            for frame in &completion_frames {
                frame.mark_gpu_complete();
            }
        });
        let block = block.copy();
        command_buffer.add_completed_handler(&block);

        Ok(command_buffer)
    }

    /// Renders the scene to a texture and returns the pixel data as an RGBA image.
    /// This does not present the frame to screen - useful for visual testing
    /// where we want to capture what would be rendered without displaying it.
    ///
    /// Note: This requires a layer-backed renderer. For headless rendering,
    /// use `render_scene_to_image()` instead.
    #[cfg(any(test, feature = "test-support"))]
    pub fn render_to_image(&mut self, scene: &Scene) -> Result<RgbaImage> {
        let layer = self
            .layer
            .clone()
            .ok_or_else(|| anyhow::anyhow!("render_to_image requires a layer-backed renderer"))?;
        let viewport_size = layer.drawable_size();
        let viewport_size: Size<DevicePixels> = size(
            (viewport_size.width.ceil() as i32).into(),
            (viewport_size.height.ceil() as i32).into(),
        );
        let drawable = layer
            .next_drawable()
            .ok_or_else(|| anyhow::anyhow!("Failed to get drawable for render_to_image"))?;

        let command_buffer = self.render_frame(scene, drawable.texture(), viewport_size)?;

        // Commit and wait for completion without presenting
        command_buffer.commit();
        command_buffer.wait_until_completed();

        read_texture_to_image(drawable.texture())
    }

    /// Renders a scene to an image without requiring a window or CAMetalLayer.
    ///
    /// This is the primary method for headless rendering. It creates an offscreen
    /// texture, renders the scene to it, and returns the pixel data as an RGBA image.
    #[cfg(any(test, feature = "test-support"))]
    pub fn render_scene_to_image(
        &mut self,
        scene: &Scene,
        size: Size<DevicePixels>,
    ) -> Result<RgbaImage> {
        self.render_scene_to_image_with_submission_callback(scene, size, || {})
    }

    /// Renders a scene and invokes a callback immediately after the exact
    /// scene command buffer is committed, before waiting for GPU completion.
    /// This is useful for integration tests that need to release a producer
    /// only after the consumer's GPU wait has been submitted.
    #[cfg(any(test, feature = "test-support"))]
    pub fn render_scene_to_image_with_submission_callback(
        &mut self,
        scene: &Scene,
        size: Size<DevicePixels>,
        on_submitted: impl FnOnce(),
    ) -> Result<RgbaImage> {
        if size.width.0 <= 0 || size.height.0 <= 0 {
            anyhow::bail!("Invalid size for render_scene_to_image: {:?}", size);
        }

        // Update path intermediate textures for this size
        self.update_path_intermediate_textures(size);

        // Create an offscreen texture as render target
        let texture_descriptor = metal::TextureDescriptor::new();
        texture_descriptor.set_width(size.width.0 as u64);
        texture_descriptor.set_height(size.height.0 as u64);
        texture_descriptor.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        texture_descriptor
            .set_usage(metal::MTLTextureUsage::RenderTarget | metal::MTLTextureUsage::ShaderRead);
        texture_descriptor.set_storage_mode(metal::MTLStorageMode::Managed);
        let target_texture = self.device.new_texture(&texture_descriptor);

        let command_buffer = self.render_frame(scene, &target_texture, size)?;

        // On discrete GPUs (non-unified memory), Managed textures require an
        // explicit blit synchronize before the CPU can read back the rendered
        // data. Without this, get_bytes returns stale zeros.
        if !self.is_unified_memory {
            let blit = command_buffer.new_blit_command_encoder();
            blit.synchronize_resource(&target_texture);
            blit.end_encoding();
        }

        // Commit and wait for completion
        command_buffer.commit();
        on_submitted();
        command_buffer.wait_until_completed();

        read_texture_to_image(&target_texture)
    }

    /// Renders a scene to a reused offscreen texture without reading pixels
    /// back or blocking on GPU completion.
    ///
    /// This mirrors the CPU cost of presenting a frame to a window (scene
    /// encoding, instance buffer writes, command submission) and is used by
    /// headless benchmark rendering, where the produced pixels are never
    /// inspected.
    #[cfg(any(test, feature = "test-support"))]
    pub fn render_scene(&mut self, scene: &Scene, size: Size<DevicePixels>) -> Result<()> {
        if size.width.0 <= 0 || size.height.0 <= 0 {
            anyhow::bail!("Invalid size for render_scene: {:?}", size);
        }

        self.update_path_intermediate_textures(size);

        let needs_new_target = self.headless_render_target.as_ref().is_none_or(|texture| {
            texture.width() != size.width.0 as u64 || texture.height() != size.height.0 as u64
        });
        if needs_new_target {
            let texture_descriptor = metal::TextureDescriptor::new();
            texture_descriptor.set_width(size.width.0 as u64);
            texture_descriptor.set_height(size.height.0 as u64);
            texture_descriptor.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
            texture_descriptor.set_usage(
                metal::MTLTextureUsage::RenderTarget | metal::MTLTextureUsage::ShaderRead,
            );
            texture_descriptor.set_storage_mode(metal::MTLStorageMode::Private);
            self.headless_render_target = Some(self.device.new_texture(&texture_descriptor));
        }
        let target_texture = self
            .headless_render_target
            .clone()
            .expect("just ensured the render target exists");

        let command_buffer = self.render_frame(scene, &target_texture, size)?;

        // Commit without waiting, mirroring presentation to a real window where
        // the CPU doesn't block on the GPU.
        command_buffer.commit();
        Ok(())
    }

    fn draw_primitives_to_texture(
        &mut self,
        scene: &Scene,
        instance_bindings: &InstanceBindings,
        writer: &mut InstanceBufferWriter,
        texture: &metal::TextureRef,
        viewport_size: Size<DevicePixels>,
        render_id: u64,
    ) -> Result<metal::CommandBuffer> {
        let command_queue = self.command_queue.clone();
        let command_buffer = command_queue.new_command_buffer();
        // These waits are encoded on the same command buffer as the draws that
        // sample the external textures, before any render encoder is created.
        for surface in &scene.surfaces {
            if let Some(frame) = &surface.external_image {
                trace_external_image_render(format_args!(
                    "command_buffer=GPUI external render {render_id} frame={} generation={} IOSurface={:#x} wait_shared_event={} current_signaled_value={}",
                    frame.frame_id,
                    frame.generation,
                    frame.iosurface_identity,
                    frame.producer_value,
                    frame.producer_event.signaled_value()
                ));
                command_buffer.encode_wait_for_event(&frame.producer_event, frame.producer_value);
            }
        }
        let alpha = if self.opaque { 1. } else { 0. };

        let mut command_encoder = new_command_encoder_for_texture(
            command_buffer,
            texture,
            viewport_size,
            Some(metal::MTLClearColor::new(0., 0., 0., alpha)),
        );

        for batch in scene.batches() {
            match batch {
                PrimitiveBatch::Shadows(range) => {
                    self.draw_shadows(range, instance_bindings, viewport_size, command_encoder)
                }
                PrimitiveBatch::Quads(range) => {
                    self.draw_quads(range, instance_bindings, viewport_size, command_encoder)
                }
                PrimitiveBatch::Paths(range) => {
                    let paths = &scene.paths[range];
                    command_encoder.end_encoding();

                    let did_draw = self.draw_paths_to_intermediate(
                        paths,
                        writer,
                        viewport_size,
                        command_buffer,
                    )?;

                    command_encoder = new_command_encoder_for_texture(
                        command_buffer,
                        texture,
                        viewport_size,
                        None,
                    );

                    if did_draw {
                        if let Err(error) = self.draw_paths_from_intermediate(
                            paths,
                            writer,
                            viewport_size,
                            command_encoder,
                        ) {
                            command_encoder.end_encoding();
                            return Err(error);
                        }
                    }
                }
                PrimitiveBatch::Underlines(range) => {
                    self.draw_underlines(range, instance_bindings, viewport_size, command_encoder)
                }
                PrimitiveBatch::MonochromeSprites { texture_id, range } => self
                    .draw_monochrome_sprites(
                        texture_id,
                        range,
                        instance_bindings,
                        viewport_size,
                        command_encoder,
                    ),
                PrimitiveBatch::PolychromeSprites { texture_id, range } => self
                    .draw_polychrome_sprites(
                        texture_id,
                        range,
                        instance_bindings,
                        viewport_size,
                        command_encoder,
                    ),
                PrimitiveBatch::Surfaces(range) => self.draw_surfaces(
                    &scene.surfaces[range.clone()],
                    range.start,
                    instance_bindings,
                    viewport_size,
                    command_encoder,
                ),
                PrimitiveBatch::SubpixelSprites { .. } => unreachable!(),
            }
        }

        command_encoder.end_encoding();

        // Dynamic atlas updates use CPU replacement, so retain the newest
        // submission as a fence until the GPU has finished sampling it.
        self.sprite_atlas.track_submission(command_buffer);
        Ok(command_buffer.to_owned())
    }

    fn draw_paths_to_intermediate(
        &self,
        paths: &[Path<ScaledPixels>],
        writer: &mut InstanceBufferWriter,
        viewport_size: Size<DevicePixels>,
        command_buffer: &metal::CommandBufferRef,
    ) -> Result<bool> {
        if paths.is_empty() {
            return Ok(false);
        }
        let intermediate_texture = self
            .path_intermediate_texture
            .as_ref()
            .context("missing path intermediate texture")?;

        let mut vertices = Vec::new();
        for path in paths {
            vertices.extend(path.vertices.iter().map(|v| PathRasterizationVertex {
                xy_position: v.xy_position,
                st_position: v.st_position,
                color: path.color,
                bounds: path.bounds.intersect(&path.content_mask.bounds),
            }));
        }
        let vertex_instance_bindings = writer.write(&vertices)?;

        let render_pass_descriptor = metal::RenderPassDescriptor::new();
        let color_attachment = render_pass_descriptor
            .color_attachments()
            .object_at(0)
            .unwrap();
        color_attachment.set_load_action(metal::MTLLoadAction::Clear);
        color_attachment.set_clear_color(metal::MTLClearColor::new(0., 0., 0., 0.));

        if let Some(msaa_texture) = &self.path_intermediate_msaa_texture {
            color_attachment.set_texture(Some(msaa_texture));
            color_attachment.set_resolve_texture(Some(intermediate_texture));
            color_attachment.set_store_action(metal::MTLStoreAction::MultisampleResolve);
        } else {
            color_attachment.set_texture(Some(intermediate_texture));
            color_attachment.set_store_action(metal::MTLStoreAction::Store);
        }

        let command_encoder = command_buffer.new_render_command_encoder(render_pass_descriptor);
        command_encoder.set_render_pipeline_state(&self.paths_rasterization_pipeline_state);
        command_encoder.set_vertex_buffer(
            PathRasterizationInputIndex::Vertices as u64,
            Some(&vertex_instance_bindings.buffer),
            vertex_instance_bindings.offset as u64,
        );
        command_encoder.set_vertex_bytes(
            PathRasterizationInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );
        command_encoder.set_fragment_buffer(
            PathRasterizationInputIndex::Vertices as u64,
            Some(&vertex_instance_bindings.buffer),
            vertex_instance_bindings.offset as u64,
        );
        command_encoder.draw_primitives(
            metal::MTLPrimitiveType::Triangle,
            0,
            vertices.len() as u64,
        );

        command_encoder.end_encoding();
        Ok(true)
    }

    fn draw_shadows(
        &self,
        shadows: Range<usize>,
        instance_bindings: &InstanceBindings,
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) {
        if shadows.is_empty() {
            return;
        }

        command_encoder.set_render_pipeline_state(&self.shadows_pipeline_state);
        command_encoder.set_vertex_buffer(
            ShadowInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_buffer(
            ShadowInputIndex::Shadows as u64,
            Some(&instance_bindings.shadows.buffer),
            instance_bindings.shadows.offset as u64,
        );
        command_encoder.set_fragment_buffer(
            ShadowInputIndex::Shadows as u64,
            Some(&instance_bindings.shadows.buffer),
            instance_bindings.shadows.offset as u64,
        );
        command_encoder.set_vertex_bytes(
            ShadowInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );

        command_encoder.draw_primitives_instanced_base_instance(
            metal::MTLPrimitiveType::Triangle,
            0,
            6,
            shadows.len() as u64,
            shadows.start as u64,
        );
    }

    fn draw_quads(
        &self,
        quads: Range<usize>,
        instance_bindings: &InstanceBindings,
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) {
        if quads.is_empty() {
            return;
        }

        command_encoder.set_render_pipeline_state(&self.quads_pipeline_state);
        command_encoder.set_vertex_buffer(
            QuadInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_buffer(
            QuadInputIndex::Quads as u64,
            Some(&instance_bindings.quads.buffer),
            instance_bindings.quads.offset as u64,
        );
        command_encoder.set_fragment_buffer(
            QuadInputIndex::Quads as u64,
            Some(&instance_bindings.quads.buffer),
            instance_bindings.quads.offset as u64,
        );
        command_encoder.set_vertex_bytes(
            QuadInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );

        command_encoder.draw_primitives_instanced_base_instance(
            metal::MTLPrimitiveType::Triangle,
            0,
            6,
            quads.len() as u64,
            quads.start as u64,
        );
    }

    fn draw_paths_from_intermediate(
        &self,
        paths: &[Path<ScaledPixels>],
        writer: &mut InstanceBufferWriter,
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) -> Result<()> {
        let Some(first_path) = paths.first() else {
            return Ok(());
        };
        let intermediate_texture = self
            .path_intermediate_texture
            .as_ref()
            .context("missing path intermediate texture")?;

        command_encoder.set_render_pipeline_state(&self.path_sprites_pipeline_state);
        command_encoder.set_vertex_buffer(
            SpriteInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_bytes(
            SpriteInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );

        command_encoder.set_fragment_texture(
            SpriteInputIndex::AtlasTexture as u64,
            Some(intermediate_texture),
        );

        // When copying paths from the intermediate texture to the drawable,
        // each pixel must only be copied once, in case of transparent paths.
        //
        // If all paths have the same draw order, then their bounds are all
        // disjoint, so we can copy each path's bounds individually. If this
        // batch combines different draw orders, we perform a single copy
        // for a minimal spanning rect.
        let sprites;
        if paths.last().unwrap().order == first_path.order {
            sprites = paths
                .iter()
                .map(|path| PathSprite {
                    bounds: path.clipped_bounds(),
                })
                .collect();
        } else {
            let mut bounds = first_path.clipped_bounds();
            for path in paths.iter().skip(1) {
                bounds = bounds.union(&path.clipped_bounds());
            }
            sprites = vec![PathSprite { bounds }];
        }

        let sprite_instance_bindings = writer.write(&sprites)?;
        command_encoder.set_vertex_buffer(
            SpriteInputIndex::Sprites as u64,
            Some(&sprite_instance_bindings.buffer),
            sprite_instance_bindings.offset as u64,
        );

        command_encoder.draw_primitives_instanced(
            metal::MTLPrimitiveType::Triangle,
            0,
            6,
            sprites.len() as u64,
        );
        Ok(())
    }

    fn draw_underlines(
        &self,
        underlines: Range<usize>,
        instance_bindings: &InstanceBindings,
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) {
        if underlines.is_empty() {
            return;
        }

        command_encoder.set_render_pipeline_state(&self.underlines_pipeline_state);
        command_encoder.set_vertex_buffer(
            UnderlineInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_buffer(
            UnderlineInputIndex::Underlines as u64,
            Some(&instance_bindings.underlines.buffer),
            instance_bindings.underlines.offset as u64,
        );
        command_encoder.set_fragment_buffer(
            UnderlineInputIndex::Underlines as u64,
            Some(&instance_bindings.underlines.buffer),
            instance_bindings.underlines.offset as u64,
        );
        command_encoder.set_vertex_bytes(
            UnderlineInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );

        command_encoder.draw_primitives_instanced_base_instance(
            metal::MTLPrimitiveType::Triangle,
            0,
            6,
            underlines.len() as u64,
            underlines.start as u64,
        );
    }

    fn draw_monochrome_sprites(
        &self,
        texture_id: AtlasTextureId,
        sprites: Range<usize>,
        instance_bindings: &InstanceBindings,
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) {
        if sprites.is_empty() {
            return;
        }

        let texture = self.sprite_atlas.metal_texture(texture_id);
        let texture_size = size(
            DevicePixels(texture.width() as i32),
            DevicePixels(texture.height() as i32),
        );
        command_encoder.set_render_pipeline_state(&self.monochrome_sprites_pipeline_state);
        command_encoder.set_vertex_buffer(
            SpriteInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_buffer(
            SpriteInputIndex::Sprites as u64,
            Some(&instance_bindings.monochrome_sprites.buffer),
            instance_bindings.monochrome_sprites.offset as u64,
        );
        command_encoder.set_vertex_bytes(
            SpriteInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );
        command_encoder.set_vertex_bytes(
            SpriteInputIndex::AtlasTextureSize as u64,
            mem::size_of_val(&texture_size) as u64,
            &texture_size as *const Size<DevicePixels> as *const _,
        );
        command_encoder.set_fragment_buffer(
            SpriteInputIndex::Sprites as u64,
            Some(&instance_bindings.monochrome_sprites.buffer),
            instance_bindings.monochrome_sprites.offset as u64,
        );
        command_encoder.set_fragment_texture(SpriteInputIndex::AtlasTexture as u64, Some(&texture));

        command_encoder.draw_primitives_instanced_base_instance(
            metal::MTLPrimitiveType::Triangle,
            0,
            6,
            sprites.len() as u64,
            sprites.start as u64,
        );
    }

    fn draw_polychrome_sprites(
        &self,
        texture_id: AtlasTextureId,
        sprites: Range<usize>,
        instance_bindings: &InstanceBindings,
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) {
        if sprites.is_empty() {
            return;
        }

        let texture = self.sprite_atlas.metal_texture(texture_id);
        let texture_size = size(
            DevicePixels(texture.width() as i32),
            DevicePixels(texture.height() as i32),
        );
        command_encoder.set_render_pipeline_state(&self.polychrome_sprites_pipeline_state);
        command_encoder.set_vertex_buffer(
            SpriteInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_buffer(
            SpriteInputIndex::Sprites as u64,
            Some(&instance_bindings.polychrome_sprites.buffer),
            instance_bindings.polychrome_sprites.offset as u64,
        );
        command_encoder.set_vertex_bytes(
            SpriteInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );
        command_encoder.set_vertex_bytes(
            SpriteInputIndex::AtlasTextureSize as u64,
            mem::size_of_val(&texture_size) as u64,
            &texture_size as *const Size<DevicePixels> as *const _,
        );
        command_encoder.set_fragment_buffer(
            SpriteInputIndex::Sprites as u64,
            Some(&instance_bindings.polychrome_sprites.buffer),
            instance_bindings.polychrome_sprites.offset as u64,
        );
        command_encoder.set_fragment_texture(SpriteInputIndex::AtlasTexture as u64, Some(&texture));

        command_encoder.draw_primitives_instanced_base_instance(
            metal::MTLPrimitiveType::Triangle,
            0,
            6,
            sprites.len() as u64,
            sprites.start as u64,
        );
    }

    fn draw_surfaces(
        &mut self,
        surfaces: &[PaintSurface],
        first_surface: usize,
        instance_bindings: &InstanceBindings,
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) {
        if surfaces.is_empty() {
            return;
        }

        command_encoder.set_vertex_buffer(
            SurfaceInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_buffer(
            SurfaceInputIndex::Surfaces as u64,
            Some(&instance_bindings.surfaces.buffer),
            instance_bindings.surfaces.offset as u64,
        );
        command_encoder.set_fragment_buffer(
            SurfaceInputIndex::Clips as u64,
            Some(&instance_bindings.surface_clips.buffer),
            instance_bindings.surface_clips.offset as u64,
        );
        command_encoder.set_vertex_bytes(
            SurfaceInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );

        for (index, surface) in surfaces.iter().enumerate() {
            let texture_size = size(
                DevicePixels::from(surface.image_buffer.get_width() as i32),
                DevicePixels::from(surface.image_buffer.get_height() as i32),
            );
            let content_size = surface
                .external_image
                .as_ref()
                .map_or(texture_size, |frame| frame.content_size);
            let texture_dimensions = SurfaceTextureSize {
                backing: texture_size,
                content: content_size,
            };

            if let Some(frame) = &surface.external_image {
                assert_eq!(
                    frame.image_buffer.get_width() as u64,
                    texture_size.width.0 as u64
                );
                assert_eq!(
                    frame.image_buffer.get_height() as u64,
                    texture_size.height.0 as u64
                );
                command_encoder.set_render_pipeline_state(&self.bgra_surfaces_pipeline_state);
                let key = (
                    frame.backing_id,
                    frame.generation,
                    frame.iosurface_identity,
                );
                if !self.external_surface_textures.contains_key(&key) {
                    let import_started = std::time::Instant::now();
                    let imported = self
                        .core_video_texture_cache
                        .create_texture_from_image(
                            frame.image_buffer.as_concrete_TypeRef(),
                            None,
                            MTLPixelFormat::BGRA8Unorm,
                            frame.image_buffer.get_width(),
                            frame.image_buffer.get_height(),
                            0,
                        )
                        .expect("failed to import BGRA IOSurface into Metal");
                    self.external_surface_textures.insert(key, imported);
                    self.external_surface_texture_imports += 1;
                    log::info!(
                        "GPUI imported IOSurface backing={} generation={} identity={:#x} ({}x{}, BGRA8) in {:.3}ms",
                        frame.backing_id,
                        frame.generation,
                        frame.iosurface_identity,
                        texture_size.width.0,
                        texture_size.height.0,
                        import_started.elapsed().as_secs_f64() * 1000.0
                    );
                }
                let imported = self.external_surface_textures.get(&key).unwrap();
                let raw_texture =
                    unsafe { CVMetalTextureGetTexture(imported.as_concrete_TypeRef()) };
                assert!(
                    !raw_texture.is_null(),
                    "BGRA IOSurface import returned null texture"
                );
                command_encoder.set_fragment_texture(
                    SurfaceInputIndex::BgraTexture as u64,
                    Some(unsafe { metal::TextureRef::from_ptr(raw_texture as *mut _) }),
                );
                frame.mark_imported();
                trace_external_image_render(format_args!(
                    "frame={} generation={} IOSurface={:#x} texture_resolved=true",
                    frame.frame_id, frame.generation, frame.iosurface_identity
                ));
                command_encoder.set_vertex_bytes(
                    SurfaceInputIndex::TextureSize as u64,
                    mem::size_of_val(&texture_dimensions) as u64,
                    &texture_dimensions as *const SurfaceTextureSize as *const _,
                );
                command_encoder.draw_primitives_instanced_base_instance(
                    metal::MTLPrimitiveType::Triangle,
                    0,
                    6,
                    1,
                    (first_surface + index) as u64,
                );
                continue;
            }

            command_encoder.set_render_pipeline_state(&self.surfaces_pipeline_state);
            assert_eq!(
                surface.image_buffer.get_pixel_format(),
                kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
            );

            let y_texture = self
                .core_video_texture_cache
                .create_texture_from_image(
                    surface.image_buffer.as_concrete_TypeRef(),
                    None,
                    MTLPixelFormat::R8Unorm,
                    surface.image_buffer.get_width_of_plane(0),
                    surface.image_buffer.get_height_of_plane(0),
                    0,
                )
                .unwrap();
            let cb_cr_texture = self
                .core_video_texture_cache
                .create_texture_from_image(
                    surface.image_buffer.as_concrete_TypeRef(),
                    None,
                    MTLPixelFormat::RG8Unorm,
                    surface.image_buffer.get_width_of_plane(1),
                    surface.image_buffer.get_height_of_plane(1),
                    1,
                )
                .unwrap();

            command_encoder.set_vertex_bytes(
                SurfaceInputIndex::TextureSize as u64,
                mem::size_of_val(&texture_dimensions) as u64,
                &texture_dimensions as *const SurfaceTextureSize as *const _,
            );
            // let y_texture = y_texture.get_texture().unwrap().
            command_encoder.set_fragment_texture(SurfaceInputIndex::YTexture as u64, unsafe {
                let texture = CVMetalTextureGetTexture(y_texture.as_concrete_TypeRef());
                Some(metal::TextureRef::from_ptr(texture as *mut _))
            });
            command_encoder.set_fragment_texture(SurfaceInputIndex::CbCrTexture as u64, unsafe {
                let texture = CVMetalTextureGetTexture(cb_cr_texture.as_concrete_TypeRef());
                Some(metal::TextureRef::from_ptr(texture as *mut _))
            });

            command_encoder.draw_primitives_instanced_base_instance(
                metal::MTLPrimitiveType::Triangle,
                0,
                6,
                1,
                (first_surface + index) as u64,
            );
        }
    }
}

/// Identifies the Metal device selected by GPUI.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MetalDeviceIdentity {
    /// Human-readable Metal device name.
    pub name: String,
    /// Metal registry ID, suitable for comparing devices across processes.
    pub registry_id: u64,
}

fn new_command_encoder_for_texture<'a>(
    command_buffer: &'a metal::CommandBufferRef,
    texture: &'a metal::TextureRef,
    viewport_size: Size<DevicePixels>,
    clear_color: Option<metal::MTLClearColor>,
) -> &'a metal::RenderCommandEncoderRef {
    let render_pass_descriptor = metal::RenderPassDescriptor::new();
    let color_attachment = render_pass_descriptor
        .color_attachments()
        .object_at(0)
        .unwrap();
    color_attachment.set_texture(Some(texture));
    color_attachment.set_store_action(metal::MTLStoreAction::Store);
    if let Some(clear_color) = clear_color {
        color_attachment.set_load_action(metal::MTLLoadAction::Clear);
        color_attachment.set_clear_color(clear_color);
    } else {
        color_attachment.set_load_action(metal::MTLLoadAction::Load);
    }

    let command_encoder = command_buffer.new_render_command_encoder(render_pass_descriptor);
    command_encoder.set_viewport(metal::MTLViewport {
        originX: 0.0,
        originY: 0.0,
        width: i32::from(viewport_size.width) as f64,
        height: i32::from(viewport_size.height) as f64,
        znear: 0.0,
        zfar: 1.0,
    });
    command_encoder
}

#[cfg(any(test, feature = "test-support"))]
fn read_texture_to_image(texture: &metal::TextureRef) -> Result<RgbaImage> {
    let width = texture.width() as u32;
    let height = texture.height() as u32;
    let bytes_per_row = width as usize * 4;
    let mut pixels = vec![0u8; height as usize * bytes_per_row];

    let region = metal::MTLRegion {
        origin: metal::MTLOrigin { x: 0, y: 0, z: 0 },
        size: metal::MTLSize {
            width: width as u64,
            height: height as u64,
            depth: 1,
        },
    };
    texture.get_bytes(
        pixels.as_mut_ptr() as *mut std::ffi::c_void,
        bytes_per_row as u64,
        region,
        0,
    );

    // Convert BGRA to RGBA (swap B and R channels)
    for chunk in pixels.chunks_exact_mut(4) {
        chunk.swap(0, 2);
    }

    RgbaImage::from_raw(width, height, pixels).context("failed to create RgbaImage from pixel data")
}

fn build_pipeline_state(
    device: &metal::DeviceRef,
    library: &metal::LibraryRef,
    label: &str,
    vertex_fn_name: &str,
    fragment_fn_name: &str,
    pixel_format: metal::MTLPixelFormat,
) -> metal::RenderPipelineState {
    let vertex_fn = library
        .get_function(vertex_fn_name, None)
        .expect("error locating vertex function");
    let fragment_fn = library
        .get_function(fragment_fn_name, None)
        .expect("error locating fragment function");

    let descriptor = metal::RenderPipelineDescriptor::new();
    descriptor.set_label(label);
    descriptor.set_vertex_function(Some(vertex_fn.as_ref()));
    descriptor.set_fragment_function(Some(fragment_fn.as_ref()));
    let color_attachment = descriptor.color_attachments().object_at(0).unwrap();
    color_attachment.set_pixel_format(pixel_format);
    color_attachment.set_blending_enabled(true);
    color_attachment.set_rgb_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_alpha_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_source_rgb_blend_factor(metal::MTLBlendFactor::SourceAlpha);
    color_attachment.set_source_alpha_blend_factor(metal::MTLBlendFactor::One);
    color_attachment.set_destination_rgb_blend_factor(metal::MTLBlendFactor::OneMinusSourceAlpha);
    color_attachment.set_destination_alpha_blend_factor(metal::MTLBlendFactor::One);

    device
        .new_render_pipeline_state(&descriptor)
        .expect("could not create render pipeline state")
}

fn build_path_sprite_pipeline_state(
    device: &metal::DeviceRef,
    library: &metal::LibraryRef,
    label: &str,
    vertex_fn_name: &str,
    fragment_fn_name: &str,
    pixel_format: metal::MTLPixelFormat,
) -> metal::RenderPipelineState {
    let vertex_fn = library
        .get_function(vertex_fn_name, None)
        .expect("error locating vertex function");
    let fragment_fn = library
        .get_function(fragment_fn_name, None)
        .expect("error locating fragment function");

    let descriptor = metal::RenderPipelineDescriptor::new();
    descriptor.set_label(label);
    descriptor.set_vertex_function(Some(vertex_fn.as_ref()));
    descriptor.set_fragment_function(Some(fragment_fn.as_ref()));
    let color_attachment = descriptor.color_attachments().object_at(0).unwrap();
    color_attachment.set_pixel_format(pixel_format);
    color_attachment.set_blending_enabled(true);
    color_attachment.set_rgb_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_alpha_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_source_rgb_blend_factor(metal::MTLBlendFactor::One);
    color_attachment.set_source_alpha_blend_factor(metal::MTLBlendFactor::One);
    color_attachment.set_destination_rgb_blend_factor(metal::MTLBlendFactor::OneMinusSourceAlpha);
    color_attachment.set_destination_alpha_blend_factor(metal::MTLBlendFactor::One);

    device
        .new_render_pipeline_state(&descriptor)
        .expect("could not create render pipeline state")
}

fn build_path_rasterization_pipeline_state(
    device: &metal::DeviceRef,
    library: &metal::LibraryRef,
    label: &str,
    vertex_fn_name: &str,
    fragment_fn_name: &str,
    pixel_format: metal::MTLPixelFormat,
    path_sample_count: u32,
) -> metal::RenderPipelineState {
    let vertex_fn = library
        .get_function(vertex_fn_name, None)
        .expect("error locating vertex function");
    let fragment_fn = library
        .get_function(fragment_fn_name, None)
        .expect("error locating fragment function");

    let descriptor = metal::RenderPipelineDescriptor::new();
    descriptor.set_label(label);
    descriptor.set_vertex_function(Some(vertex_fn.as_ref()));
    descriptor.set_fragment_function(Some(fragment_fn.as_ref()));
    if path_sample_count > 1 {
        descriptor.set_raster_sample_count(path_sample_count as _);
        descriptor.set_alpha_to_coverage_enabled(false);
    }
    let color_attachment = descriptor.color_attachments().object_at(0).unwrap();
    color_attachment.set_pixel_format(pixel_format);
    color_attachment.set_blending_enabled(true);
    color_attachment.set_rgb_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_alpha_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_source_rgb_blend_factor(metal::MTLBlendFactor::One);
    color_attachment.set_source_alpha_blend_factor(metal::MTLBlendFactor::One);
    color_attachment.set_destination_rgb_blend_factor(metal::MTLBlendFactor::OneMinusSourceAlpha);
    color_attachment.set_destination_alpha_blend_factor(metal::MTLBlendFactor::OneMinusSourceAlpha);

    device
        .new_render_pipeline_state(&descriptor)
        .expect("could not create render pipeline state")
}

#[derive(Clone)]
struct InstanceBinding {
    buffer: metal::Buffer,
    offset: usize,
}

struct InstanceBindings {
    quads: InstanceBinding,
    shadows: InstanceBinding,
    underlines: InstanceBinding,
    monochrome_sprites: InstanceBinding,
    polychrome_sprites: InstanceBinding,
    surfaces: InstanceBinding,
    surface_clips: InstanceBinding,
}

fn write_instances(scene: &Scene, writer: &mut InstanceBufferWriter) -> Result<InstanceBindings> {
    let mut surface_clips = Vec::new();
    let surfaces = scene
        .surfaces
        .iter()
        .map(|surface| {
            let clip_offset = surface_clips.len() as u32;
            surface_clips.extend(surface.clip_stack.iter().copied().map(|clip| SurfaceClip {
                bounds: clip.bounds,
                corner_radii: clip.corner_radii,
            }));
            SurfaceBounds {
                bounds: surface.bounds,
                content_mask: surface.content_mask,
                corner_radii: surface.corner_radii,
                clip_offset,
                clip_count: surface.clip_stack.len() as u32,
            }
        })
        .collect::<Vec<_>>();
    Ok(InstanceBindings {
        quads: writer.write(&scene.quads)?,
        shadows: writer.write(&scene.shadows)?,
        underlines: writer.write(&scene.underlines)?,
        monochrome_sprites: writer.write(&scene.monochrome_sprites)?,
        polychrome_sprites: writer.write(&scene.polychrome_sprites)?,
        surfaces: writer.write_iter(surfaces.into_iter())?,
        surface_clips: writer.write_iter(surface_clips.into_iter())?,
    })
}

struct InstanceBufferWriter {
    device: metal::Device,
    pool: Arc<Mutex<InstanceBufferPool>>,
    unified_memory: bool,
    filled: Vec<(InstanceBuffer, usize)>,
    current: InstanceBuffer,
    offset: usize,
}

impl InstanceBufferWriter {
    fn new(
        device: &metal::Device,
        pool: &Arc<Mutex<InstanceBufferPool>>,
        unified_memory: bool,
    ) -> Self {
        let current = pool.lock().acquire(device, unified_memory);
        Self {
            device: device.clone(),
            pool: pool.clone(),
            unified_memory,
            filled: Vec::new(),
            current,
            offset: 0,
        }
    }

    fn allocate<T>(&mut self, count: usize) -> Result<(InstanceBinding, &mut [MaybeUninit<T>])> {
        let size = mem::size_of::<T>() * count;
        let mut offset = self.offset.next_multiple_of(INSTANCE_BUFFER_ALIGNMENT);
        if offset + size > self.current.size {
            self.grow(size)?;
            offset = 0;
        }
        self.offset = offset + size;

        let binding = InstanceBinding {
            buffer: self.current.metal_buffer.clone(),
            offset,
        };
        // Safety: the reservation lies within a buffer this frame owns
        // exclusively, and never overlaps one handed out earlier.
        let values = unsafe {
            let start = (self.current.metal_buffer.contents() as *mut u8).add(offset);
            slice::from_raw_parts_mut(start.cast::<MaybeUninit<T>>(), count)
        };
        Ok((binding, values))
    }

    fn write<T>(&mut self, values: &[T]) -> Result<InstanceBinding> {
        let (binding, destination) = self.allocate::<T>(values.len())?;
        unsafe {
            ptr::copy_nonoverlapping(
                values.as_ptr(),
                destination.as_mut_ptr().cast::<T>(),
                values.len(),
            );
        }
        Ok(binding)
    }

    fn write_iter<T>(
        &mut self,
        values: impl ExactSizeIterator<Item = T>,
    ) -> Result<InstanceBinding> {
        let (binding, destination) = self.allocate::<T>(values.len())?;
        for (slot, value) in destination.iter_mut().zip(values) {
            slot.write(value);
        }
        Ok(binding)
    }

    fn grow(&mut self, required: usize) -> Result<()> {
        let mut pool = self.pool.lock();
        let buffer_size = (pool.buffer_size * 2)
            .max(required.next_power_of_two())
            .min(MAX_INSTANCE_BUFFER_SIZE);
        anyhow::ensure!(
            buffer_size >= required,
            "instance buffer needs {required} bytes, above the maximum of {MAX_INSTANCE_BUFFER_SIZE}"
        );
        anyhow::ensure!(
            buffer_size > self.current.size,
            "frame instance data exceeds the {MAX_INSTANCE_BUFFER_SIZE}-byte maximum"
        );
        if buffer_size != pool.buffer_size {
            log::info!("increased instance buffer size to {buffer_size}");
            pool.reset(buffer_size);
        }
        let buffer = pool.acquire(&self.device, self.unified_memory);
        drop(pool);

        let filled = mem::replace(&mut self.current, buffer);
        self.filled.push((filled, self.offset));
        self.offset = 0;
        Ok(())
    }

    fn finish(self) -> InstanceBuffer {
        let Self {
            unified_memory,
            filled,
            current,
            offset,
            ..
        } = self;

        if !unified_memory {
            for (buffer, written) in &filled {
                if *written == 0 {
                    continue;
                }
                buffer.metal_buffer.did_modify_range(NSRange {
                    location: 0,
                    length: *written as NSUInteger,
                });
            }
            if offset > 0 {
                current.metal_buffer.did_modify_range(NSRange {
                    location: 0,
                    length: offset as NSUInteger,
                });
            }
        }

        // Metal retains encoded resources until the command buffer completes.
        // Only the final, largest buffer is worth keeping in the pool.
        drop(filled);
        current
    }
}

#[repr(C)]
enum ShadowInputIndex {
    Vertices = 0,
    Shadows = 1,
    ViewportSize = 2,
}

#[repr(C)]
enum QuadInputIndex {
    Vertices = 0,
    Quads = 1,
    ViewportSize = 2,
}

#[repr(C)]
enum UnderlineInputIndex {
    Vertices = 0,
    Underlines = 1,
    ViewportSize = 2,
}

#[repr(C)]
enum SpriteInputIndex {
    Vertices = 0,
    Sprites = 1,
    ViewportSize = 2,
    AtlasTextureSize = 3,
    AtlasTexture = 4,
}

#[repr(C)]
enum SurfaceInputIndex {
    Vertices = 0,
    Surfaces = 1,
    ViewportSize = 2,
    TextureSize = 3,
    YTexture = 4,
    CbCrTexture = 5,
    BgraTexture = 6,
    Clips = 7,
}

#[repr(C)]
struct SurfaceTextureSize {
    backing: Size<DevicePixels>,
    content: Size<DevicePixels>,
}

#[repr(C)]
enum PathRasterizationInputIndex {
    Vertices = 0,
    ViewportSize = 1,
}

#[derive(Clone, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct PathSprite {
    pub bounds: Bounds<ScaledPixels>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct SurfaceBounds {
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub corner_radii: Corners<ScaledPixels>,
    pub clip_offset: u32,
    pub clip_count: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct SurfaceClip {
    pub bounds: Bounds<ScaledPixels>,
    pub corner_radii: Corners<ScaledPixels>,
}

#[cfg(any(test, feature = "test-support"))]
pub struct MetalHeadlessRenderer {
    renderer: MetalRenderer,
}

/// Standalone GPU producer used by the cross-process presentation stress
/// example. It retains one Metal texture per IOSurface backing and rewrites
/// that same texture only after the matching GPUI lease is released.
#[cfg(all(target_os = "macos", feature = "test-support"))]
pub struct StandaloneMetalPatternProducer {
    command_queue: CommandQueue,
    pipeline: metal::RenderPipelineState,
    texture_cache: CVMetalTextureCache,
    textures: std::collections::HashMap<(u64, u64), metal::Texture>,
    texture_imports: usize,
    event: metal::SharedEvent,
}

#[cfg(all(target_os = "macos", feature = "test-support"))]
impl StandaloneMetalPatternProducer {
    pub fn new(event: metal::SharedEvent) -> anyhow::Result<Self> {
        let device = MetalRenderer::create_device();
        let source = r#"
            #include <metal_stdlib>
            using namespace metal;
            struct Out { float4 position [[position]]; };
            vertex Out v(uint id [[vertex_id]]) {
                constexpr float2 p[3] = { float2(-1.0, -1.0), float2(3.0, -1.0), float2(-1.0, 3.0) };
                return { float4(p[id], 0.0, 1.0) };
            }
            fragment float4 f(Out in [[stage_in]],
                              constant float2 *dimensions [[buffer(0)]],
                              constant uint *marker_top_left [[buffer(1)]]) {
                float2 p = in.position.xy;
                bool marker = *marker_top_left
                    ? (p.x < 36.0 && p.y < 36.0)
                    : (p.x > dimensions->x - 36.0 && p.y > dimensions->y - 36.0);
                if (marker) return float4(0.0, 0.0, 0.0, 1.0);
                if (p.y < dimensions->y * 0.5)
                    return p.x < dimensions->x * 0.5 ? float4(1, 0, 0, 1) : float4(0, 1, 0, 1);
                return p.x < dimensions->x * 0.5 ? float4(0, 0, 1, 1) : float4(1, 1, 1, 1);
            }
        "#;
        let library = device
            .new_library_with_source(source, &metal::CompileOptions::new())
            .map_err(|error| {
                anyhow::anyhow!("could not compile persistent producer shader: {error}")
            })?;
        let descriptor = metal::RenderPipelineDescriptor::new();
        let vertex = library
            .get_function("v", None)
            .map_err(|error| anyhow::anyhow!("missing producer vertex function: {error}"))?;
        let fragment = library
            .get_function("f", None)
            .map_err(|error| anyhow::anyhow!("missing producer fragment function: {error}"))?;
        descriptor.set_vertex_function(Some(vertex.as_ref()));
        descriptor.set_fragment_function(Some(fragment.as_ref()));
        descriptor
            .color_attachments()
            .object_at(0)
            .unwrap()
            .set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        let pipeline = device
            .new_render_pipeline_state(&descriptor)
            .map_err(|error| anyhow::anyhow!("could not create producer pipeline: {error}"))?;
        let texture_cache = CVMetalTextureCache::new(None, device.clone(), None)
            .map_err(|error| anyhow::anyhow!("could not create producer texture cache: {error}"))?;
        Ok(Self {
            command_queue: device.new_command_queue(),
            pipeline,
            texture_cache,
            textures: std::collections::HashMap::new(),
            texture_imports: 0,
            event,
        })
    }

    /// Imports each producer texture once for this generation/backing set.
    pub fn register_backings(
        &mut self,
        frames: &[&gpui::MacExternalImageFrame],
    ) -> anyhow::Result<usize> {
        for frame in frames {
            let key = (frame.backing_id, frame.generation);
            if self.textures.contains_key(&key) {
                continue;
            }
            let cv_texture = self
                .texture_cache
                .create_texture_from_image(
                    frame.image_buffer.as_concrete_TypeRef(),
                    None,
                    MTLPixelFormat::BGRA8Unorm,
                    frame.image_buffer.get_width(),
                    frame.image_buffer.get_height(),
                    0,
                )
                .map_err(|error| {
                    anyhow::anyhow!("could not import producer IOSurface texture: {error}")
                })?;
            let raw_texture = unsafe { CVMetalTextureGetTexture(cv_texture.as_concrete_TypeRef()) };
            anyhow::ensure!(
                !raw_texture.is_null(),
                "producer IOSurface texture import was null"
            );
            let retained_texture = unsafe { msg_send![raw_texture, retain] };
            let texture = unsafe { metal::Texture::from_ptr(retained_texture) };
            self.textures.insert(key, texture);
            self.texture_imports += 1;
        }
        Ok(self.textures.len())
    }

    pub fn encode_frame(
        &self,
        frame: &gpui::MacExternalImageFrame,
        frame_id: u64,
        signal_value: u64,
    ) -> anyhow::Result<metal::CommandBuffer> {
        let key = (frame.backing_id, frame.generation);
        let texture = self
            .textures
            .get(&key)
            .context("producer does not have this backing imported")?;
        let width = frame.image_buffer.get_width();
        let height = frame.image_buffer.get_height();
        let command_buffer = self.command_queue.new_command_buffer().to_owned();
        let render_pass = metal::RenderPassDescriptor::new();
        let attachment = render_pass.color_attachments().object_at(0).unwrap();
        attachment.set_texture(Some(texture));
        attachment.set_load_action(metal::MTLLoadAction::DontCare);
        attachment.set_store_action(metal::MTLStoreAction::Store);
        let encoder = command_buffer.new_render_command_encoder(render_pass);
        encoder.set_render_pipeline_state(&self.pipeline);
        let dimensions = [width as f32, height as f32];
        // Alternate the asymmetry every frame so the consumer can verify
        // producer reuse actually rewrites each IOSurface after its release.
        let marker_top_left = u32::from(frame_id % 2 == 1);
        encoder.set_fragment_bytes(
            0,
            mem::size_of_val(&dimensions) as u64,
            dimensions.as_ptr().cast(),
        );
        encoder.set_fragment_bytes(
            1,
            mem::size_of_val(&marker_top_left) as u64,
            &marker_top_left as *const _ as *const c_void,
        );
        encoder.draw_primitives(metal::MTLPrimitiveType::Triangle, 0, 3);
        encoder.end_encoding();
        command_buffer.encode_signal_event(&self.event, signal_value);
        Ok(command_buffer)
    }

    pub fn retire_generation(&mut self, generation: u64) -> usize {
        let before = self.textures.len();
        self.textures
            .retain(|(_, cached_generation), _| *cached_generation != generation);
        before - self.textures.len()
    }

    pub fn texture_count(&self) -> usize {
        self.textures.len()
    }

    pub fn texture_import_count(&self) -> usize {
        self.texture_imports
    }

    pub fn signaled_value(&self) -> u64 {
        self.event.signaled_value()
    }
}

#[cfg(any(test, feature = "test-support"))]
impl MetalHeadlessRenderer {
    pub fn new() -> Self {
        let instance_buffer_pool = Arc::new(Mutex::new(InstanceBufferPool::default()));
        let renderer = MetalRenderer::new_headless(instance_buffer_pool);
        Self { renderer }
    }

    /// Number of persistent IOSurface textures imported by this renderer.
    /// Exposed only to make the standalone transport stress test verify that
    /// imports stay bounded by backing count.
    pub fn external_surface_texture_count(&self) -> usize {
        self.renderer.external_surface_textures.len()
    }

    /// Lifetime count of distinct backing texture imports, including retired
    /// generations, for standalone resource accounting.
    pub fn external_surface_texture_import_count(&self) -> usize {
        self.renderer.external_surface_texture_imports
    }

    /// Drops the import cache for a generation after its final frame lease has
    /// completed and no scene can reference it again.
    pub fn retire_external_image_generation(&mut self, generation: u64) -> usize {
        let before = self.renderer.external_surface_textures.len();
        self.renderer
            .external_surface_textures
            .retain(|(_, cached_generation, _), _| *cached_generation != generation);
        before - self.renderer.external_surface_textures.len()
    }

    pub fn render_scene_to_image_with_submission_callback(
        &mut self,
        scene: &Scene,
        size: Size<DevicePixels>,
        on_submitted: impl FnOnce(),
    ) -> anyhow::Result<RgbaImage> {
        self.renderer
            .render_scene_to_image_with_submission_callback(scene, size, on_submitted)
    }
}

#[cfg(any(test, feature = "test-support"))]
impl gpui::PlatformHeadlessRenderer for MetalHeadlessRenderer {
    fn render_scene_to_image(
        &mut self,
        scene: &Scene,
        size: Size<DevicePixels>,
    ) -> anyhow::Result<image::RgbaImage> {
        self.renderer.render_scene_to_image(scene, size)
    }

    fn render_scene(&mut self, scene: &Scene, size: Size<DevicePixels>) -> anyhow::Result<()> {
        self.renderer.render_scene(scene, size)
    }

    fn sprite_atlas(&self) -> Arc<dyn gpui::PlatformAtlas> {
        self.renderer.sprite_atlas().clone()
    }
}

#[cfg(all(test, target_os = "macos"))]
mod external_image_tests {
    use super::*;
    use gpui::{
        ContentMask, PaintSurface, PlatformHeadlessRenderer, RoundedClip, ScaledPixels, bounds,
        point, size,
    };

    fn insert_external_frame(scene: &mut Scene, frame: gpui::MacExternalImageFrame) {
        let width = frame.image_buffer.get_width() as f32;
        let height = frame.image_buffer.get_height() as f32;
        let bounds = bounds(
            point(ScaledPixels(0.), ScaledPixels(0.)),
            size(ScaledPixels(width), ScaledPixels(height)),
        );
        scene.insert_primitive(PaintSurface {
            order: 0,
            bounds,
            content_mask: ContentMask { bounds },
            image_buffer: frame.image_buffer.clone(),
            external_image: Some(frame),
            corner_radii: Corners::default(),
            clip_stack: Vec::new(),
            #[cfg(any(target_os = "linux", target_os = "freebsd"))]
            texture: unreachable!(),
            #[cfg(any(target_os = "linux", target_os = "freebsd"))]
            texture_size: size(DevicePixels(width as i32), DevicePixels(height as i32)),
        });
    }

    #[test]
    fn gpu_rendered_bgra_iosurface_is_sampled_by_gpui_scene() {
        let frame = MetalRenderer::create_standalone_test_frame(800, 500, 1, 1, 1).unwrap();
        let lease = frame.clone();
        let mut scene = Scene::default();
        insert_external_frame(&mut scene, frame.clone());
        scene.finish();

        let mut renderer = MetalHeadlessRenderer::new();
        let image = renderer
            .render_scene_to_image(&scene, size(DevicePixels(800), DevicePixels(500)))
            .unwrap();
        lease.retire();
        assert_eq!(image.get_pixel(100, 100).0, [255, 0, 0, 255]);
        assert_eq!(image.get_pixel(700, 100).0, [0, 255, 0, 255]);
        assert_eq!(image.get_pixel(100, 400).0, [0, 0, 255, 255]);
        assert_eq!(image.get_pixel(700, 400).0, [255, 255, 255, 255]);
        assert_eq!(image.get_pixel(10, 10).0, [0, 0, 0, 255]);
        assert!(
            lease.is_released(),
            "GPUI GPU completion did not release the lease"
        );
        let screenshot = std::env::temp_dir().join("gpui-bgra-iosurface.png");
        image.save(&screenshot).unwrap();
        eprintln!("GPUI IOSurface scene capture: {}", screenshot.display());
    }

    #[test]
    fn iosurface_mach_send_right_imports_same_backing() {
        let frame = MetalRenderer::create_standalone_test_frame(320, 200, 21, 1, 1).unwrap();
        let source_surface = unsafe {
            core_video::pixel_buffer_io_surface::CVPixelBufferGetIOSurface(
                frame.image_buffer.as_concrete_TypeRef(),
            )
        };
        let source_surface_id = unsafe { io_surface::IOSurfaceGetID(source_surface) };
        let producer =
            MetalRenderer::new_headless(Arc::new(Mutex::new(InstanceBufferPool::default())));
        let descriptor = MetalRenderer::describe_iosurface_backing(
            &frame.image_buffer,
            frame.backing_id,
            frame.generation,
        )
        .unwrap();
        let imported_buffer = MetalRenderer::import_iosurface_backing(&descriptor).unwrap();
        let imported_surface = unsafe {
            core_video::pixel_buffer_io_surface::CVPixelBufferGetIOSurface(
                imported_buffer.as_concrete_TypeRef(),
            )
        };
        assert_eq!(
            unsafe { io_surface::IOSurfaceGetID(imported_surface) },
            source_surface_id
        );
        assert_eq!(imported_buffer.get_width(), frame.image_buffer.get_width());
        assert_eq!(
            imported_buffer.get_height(),
            frame.image_buffer.get_height()
        );
    }

    #[test]
    fn external_texture_cache_distinguishes_replaced_iosurface_with_same_ids() {
        let first = MetalRenderer::create_standalone_test_frame(64, 64, 90, 1, 1).unwrap();
        let second = MetalRenderer::create_standalone_test_frame(64, 64, 90, 1, 2).unwrap();
        assert_ne!(first.iosurface_identity, second.iosurface_identity);

        let mut renderer = MetalHeadlessRenderer::new();
        for frame in [first, second] {
            let mut scene = Scene::default();
            insert_external_frame(&mut scene, frame);
            scene.finish();
            let _ = renderer
                .render_scene_to_image(&scene, size(DevicePixels(64), DevicePixels(64)))
                .unwrap();
        }

        assert_eq!(renderer.external_surface_texture_import_count(), 2);
        assert_eq!(renderer.external_surface_texture_count(), 2);
    }

    #[test]
    fn gpu_rendered_bgra_iosurface_uses_gpui_rounded_scene_clip() {
        let frame = MetalRenderer::create_standalone_test_frame(800, 500, 10, 1, 1).unwrap();
        let mut scene = Scene::default();
        insert_external_frame(&mut scene, frame);
        scene.surfaces[0].corner_radii = Corners {
            top_left: ScaledPixels(40.),
            top_right: ScaledPixels(40.),
            bottom_right: ScaledPixels(40.),
            bottom_left: ScaledPixels(40.),
        };
        scene.finish();
        let mut renderer = MetalHeadlessRenderer::new();
        let image = renderer
            .render_scene_to_image(&scene, size(DevicePixels(800), DevicePixels(500)))
            .unwrap();
        image.save("/tmp/gpui-bgra-iosurface-rounded.png").unwrap();
        assert_ne!(
            image.get_pixel(0, 499).0,
            [0, 0, 255, 255],
            "rounded corner retained the blue surface pixel"
        );
        assert_eq!(
            image.get_pixel(40, 40).0,
            [255, 0, 0, 255],
            "surface interior was clipped"
        );
    }

    #[test]
    fn nested_rounded_clips_intersect_for_offset_children() {
        let frame = MetalRenderer::create_standalone_test_frame(800, 500, 11, 1, 2).unwrap();
        let mut renderer = MetalHeadlessRenderer::new();
        for (child_x, child_y) in [(-12., -12.), (18., 8.), (42., 30.)] {
            let mut scene = Scene::default();
            insert_external_frame(&mut scene, frame.clone());
            scene.surfaces[0].clip_stack = vec![
                RoundedClip {
                    bounds: bounds(
                        point(ScaledPixels(0.), ScaledPixels(0.)),
                        size(ScaledPixels(800.), ScaledPixels(500.)),
                    ),
                    corner_radii: Corners {
                        top_left: ScaledPixels(48.),
                        top_right: ScaledPixels(48.),
                        bottom_right: ScaledPixels(48.),
                        bottom_left: ScaledPixels(48.),
                    },
                },
                RoundedClip {
                    bounds: bounds(
                        point(ScaledPixels(child_x), ScaledPixels(child_y)),
                        size(ScaledPixels(740.), ScaledPixels(460.)),
                    ),
                    corner_radii: Corners {
                        top_left: ScaledPixels(24.),
                        top_right: ScaledPixels(24.),
                        bottom_right: ScaledPixels(24.),
                        bottom_left: ScaledPixels(24.),
                    },
                },
            ];
            scene.finish();
            let image = renderer
                .render_scene_to_image(&scene, size(DevicePixels(800), DevicePixels(500)))
                .unwrap();
            assert_ne!(
                image.get_pixel(1, 1).0,
                [255, 0, 0, 255],
                "outer rounded corner leaked at offset ({child_x}, {child_y})"
            );
            assert_eq!(
                image.get_pixel(60, 60).0,
                [255, 0, 0, 255],
                "interior incorrectly clipped at offset ({child_x}, {child_y})"
            );
            if child_x < 0.0 {
                assert_ne!(
                    image.get_pixel(1, 30).0,
                    [255, 0, 0, 255],
                    "inner clip hid the outer-corner leak"
                );
                image.save("/tmp/gpui-bgra-iosurface-nested.png").unwrap();
            }
        }
    }

    #[test]
    fn external_image_resize_generations_wait_and_release() {
        let mut renderer = MetalHeadlessRenderer::new();
        let mut scene = Scene::default();
        let mut retained_old_backings = Vec::new();
        for (index, (width, height)) in
            [(800, 500), (1000, 700), (640, 480), (1280, 720), (800, 500)]
                .into_iter()
                .enumerate()
        {
            let width = width as usize;
            let height = height as usize;
            let frame_id = index as u64 + 1;
            let generation = frame_id;
            let frame = MetalRenderer::create_standalone_test_frame(
                width, height, frame_id, generation, frame_id,
            )
            .unwrap();
            let lease = frame.clone();
            retained_old_backings.push(frame.image_buffer.clone());
            scene.clear();
            insert_external_frame(&mut scene, frame);
            scene.finish();
            let image = renderer
                .render_scene_to_image(
                    &scene,
                    size(DevicePixels(width as i32), DevicePixels(height as i32)),
                )
                .unwrap();
            lease.retire();
            assert_eq!(image.get_pixel(100, 100).0, [255, 0, 0, 255]);
            assert_eq!(
                image.get_pixel((width - 100) as u32, 100).0,
                [0, 255, 0, 255]
            );
            assert_eq!(
                image.get_pixel(100, (height - 100) as u32).0,
                [0, 0, 255, 255]
            );
            assert_eq!(
                image
                    .get_pixel((width - 100) as u32, (height - 100) as u32)
                    .0,
                [255, 255, 255, 255]
            );
            let marker = if frame_id == 1 {
                image.get_pixel(10, 10)
            } else {
                image.get_pixel((width - 10) as u32, (height - 10) as u32)
            };
            assert_eq!(marker.0, [0, 0, 0, 255]);
            assert!(
                lease.is_released(),
                "generation {generation} did not release after GPU completion"
            );
            let screenshot =
                std::env::temp_dir().join(format!("gpui-iosurface-generation-{generation}.png"));
            image.save(screenshot).unwrap();
        }
        drop(retained_old_backings);
    }
}
