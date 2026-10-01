#![cfg(target_os = "macos")]

use core_foundation::base::TCFType;
use core_video::pixel_buffer_io_surface::CVPixelBufferGetIOSurface;
use gpui::{
    Bounds, ContentMask, Corners, DevicePixels, MacExternalImageFrame, PaintSurface, ScaledPixels,
    Scene, bounds, point, size,
};
use gpui_apple::{
    metal_renderer::{MetalHeadlessRenderer, MetalRenderer, StandaloneMetalPatternProducer},
    presentation_xpc::{MacPresentationEventChannel, MacPresentationFrame},
};
use metal::Device;
use std::{
    collections::HashMap,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

const DEFAULT_SERVICE: &str = "com.openai.codex.photon.metalprototype";
const BACKING_A: u64 = 41;
const BACKING_B: u64 = 42;
const BACKING_C: u64 = 43;
const BACKING_D: u64 = 44;

fn frame_count() -> u64 {
    std::env::var("PHOTON_PRESENTATION_FRAME_COUNT")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(200)
}

fn main() {
    let mode = std::env::args().nth(1).expect("pass producer or consumer");
    let channel_id = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "photon-standalone-view-1-generation-1".to_owned());
    let device = Device::system_default().expect("Metal device unavailable");
    println!(
        "Metal device={} registryID={}",
        device.name(),
        device.registry_id()
    );
    let service = std::env::var("PHOTON_PRESENTATION_XPC_SERVICE")
        .unwrap_or_else(|_| DEFAULT_SERVICE.to_owned());
    let channel =
        Arc::new(MacPresentationEventChannel::connect(&service).expect("XPC connection failed"));

    match mode.as_str() {
        "producer" => run_producer(&device, &channel, &channel_id),
        "consumer" => run_consumer(&device, &channel, &channel_id),
        other => panic!("unknown mode {other:?}; pass producer or consumer"),
    }
}

fn run_producer(device: &Device, channel: &Arc<MacPresentationEventChannel>, channel_id: &str) {
    let count = frame_count();
    assert!(count >= 4, "run at least four frames to exercise A/B reuse");
    let (backing_a, _initial_a) =
        MetalRenderer::create_standalone_test_frame_deferred(800, 500, BACKING_A, 1, 1)
            .expect("could not create backing A");
    let (mut backing_b, _initial_b) =
        MetalRenderer::create_standalone_test_frame_deferred(800, 500, BACKING_B, 1, 2)
            .expect("could not create backing B");
    let (mut backing_c, _initial_c) =
        MetalRenderer::create_standalone_test_frame_deferred(1000, 700, BACKING_C, 2, 1)
            .expect("could not create resized backing C");
    let (mut backing_d, _initial_d) =
        MetalRenderer::create_standalone_test_frame_deferred(1000, 700, BACKING_D, 2, 2)
            .expect("could not create resized backing D");
    drop((_initial_a, _initial_b, _initial_c, _initial_d));
    // The channel owns one shared event. Both backing render submissions signal
    // its monotonically increasing values.
    backing_b.producer_event = backing_a.producer_event.clone();
    backing_b.producer_value = 2;
    backing_c.producer_event = backing_a.producer_event.clone();
    backing_c.producer_value = count + 1;
    backing_d.producer_event = backing_a.producer_event.clone();
    backing_d.producer_value = count + 2;

    let mut producer = StandaloneMetalPatternProducer::new(backing_a.producer_event.clone())
        .expect("could not create persistent standalone producer");
    assert_eq!(
        producer
            .register_backings(&[&backing_a, &backing_b])
            .expect("could not import generation 1 producer textures"),
        2
    );

    register_backing(channel, channel_id, &backing_a);
    register_backing(channel, channel_id, &backing_b);
    channel
        .register_shared_event(channel_id, device.registry_id(), &backing_a.producer_event)
        .expect("XPC shared-event registration failed");
    let mut producer_backings = HashMap::from([
        ((BACKING_A, 1), backing_a),
        ((BACKING_B, 1), backing_b),
        ((BACKING_C, 2), backing_c),
        ((BACKING_D, 2), backing_d),
    ]);
    println!(
        "registered 2 generation-1 IOSurfaces and 1 shared event; publishing {count} A/B frames before resize"
    );

    for stage in 0..2 {
        let generation = stage + 1;
        let offset = stage * count;
        if stage == 1 {
            channel
                .unregister_iosurface_backing(channel_id, BACKING_A, 1)
                .expect("generation 1 backing A remained leased");
            channel
                .unregister_iosurface_backing(channel_id, BACKING_B, 1)
                .expect("generation 1 backing B remained leased");
            assert_eq!(producer.retire_generation(1), 2);
            drop(producer_backings.remove(&(BACKING_A, 1)).unwrap());
            drop(producer_backings.remove(&(BACKING_B, 1)).unwrap());
            println!("generation 1 retired; generation 2 A/B registered at 1000x700");
        }
        for local_frame_id in 1..=count {
            let frame_id = offset + local_frame_id;
            let backing_id = match (stage, local_frame_id % 2) {
                (0, 1) => BACKING_A,
                (0, 0) => BACKING_B,
                (1, 1) => BACKING_C,
                _ => BACKING_D,
            };
            let frame = MacPresentationFrame {
                backing_id,
                generation,
                frame_id,
                signal_value: frame_id,
            };
            channel
                .publish_frame(channel_id, frame)
                .expect("FrameReady publication failed");
            channel
                .wait_for_consumer_submission(channel_id, frame.frame_id)
                .expect("consumer did not submit its scene command buffer");

            // On the last old-generation submission, prove the broker refuses
            // retirement before GPU completion, while the new generation is
            // already registered and ready to follow immediately.
            if stage == 0 && local_frame_id == count {
                register_backing(
                    channel,
                    channel_id,
                    producer_backings.get(&(BACKING_C, 2)).unwrap(),
                );
                register_backing(
                    channel,
                    channel_id,
                    producer_backings.get(&(BACKING_D, 2)).unwrap(),
                );
                assert_eq!(
                    producer
                        .register_backings(&[
                            producer_backings.get(&(BACKING_C, 2)).unwrap(),
                            producer_backings.get(&(BACKING_D, 2)).unwrap(),
                        ])
                        .expect("could not import generation 2 producer textures"),
                    4
                );
                let leased_backing = frame.backing_id;
                assert!(
                    channel
                        .unregister_iosurface_backing(channel_id, leased_backing, 1)
                        .is_err(),
                    "broker retired a backing with an outstanding frame lease"
                );
            }
            let producer_backing = producer_backings.get(&(backing_id, generation)).unwrap();
            producer
                .encode_frame(producer_backing, frame_id, frame.signal_value)
                .expect("could not encode producer render into persistent backing")
                .commit();
            channel
                .wait_for_frame_release(
                    channel_id,
                    frame.backing_id,
                    frame.generation,
                    frame.frame_id,
                )
                .expect("producer did not receive GPU-completion release");
            assert!(
                channel
                    .release_frame(
                        channel_id,
                        frame.backing_id,
                        frame.generation,
                        frame.frame_id,
                    )
                    .is_err(),
                "broker accepted duplicate release for frame {}",
                frame.frame_id
            );
            assert!(
                producer.signaled_value() >= frame.signal_value,
                "shared event signal value did not advance to frame {}",
                frame.frame_id
            );
        }
    }
    channel
        .unregister_iosurface_backing(channel_id, BACKING_C, 2)
        .expect("generation 2 backing C remained leased");
    channel
        .unregister_iosurface_backing(channel_id, BACKING_D, 2)
        .expect("generation 2 backing D remained leased");
    assert_eq!(producer.retire_generation(2), 2);
    assert_eq!(producer.texture_count(), 0);
    drop(producer_backings.remove(&(BACKING_C, 2)).unwrap());
    drop(producer_backings.remove(&(BACKING_D, 2)).unwrap());
    assert!(producer_backings.is_empty());
    println!(
        "producer completed {} frames; backing registrations=4 across 2 generations, event exports=1, releases={}",
        count * 2,
        count * 2
    );
}

fn register_backing(
    channel: &MacPresentationEventChannel,
    channel_id: &str,
    frame: &MacExternalImageFrame,
) {
    let descriptor = MetalRenderer::describe_iosurface_backing(
        &frame.image_buffer,
        frame.backing_id,
        frame.generation,
    )
    .expect("could not create IOSurface Mach-port descriptor");
    channel
        .register_iosurface_backing(channel_id, &descriptor)
        .expect("XPC IOSurface registration failed");
}

fn run_consumer(device: &Device, channel: &Arc<MacPresentationEventChannel>, channel_id: &str) {
    let count = frame_count();
    let event = channel
        .import_shared_event(channel_id, device)
        .expect("XPC shared-event import failed");
    println!(
        "consumer imported shared event once; initial value={}",
        event.signaled_value()
    );

    // A backing's send right and imported pixel buffer are cached until the
    // generation retires. Per-frame metadata never causes a resource import.
    let mut backings = HashMap::new();
    let mut renderer = MetalHeadlessRenderer::new();
    let mut iosurface_imports = 0usize;
    let mut iosurface_destroys = 0usize;
    let mut gpui_texture_destroys = 0usize;
    let total_frames = count * 2;
    for expected_frame_id in 1..=total_frames {
        let frame = channel
            .wait_for_frame(channel_id)
            .expect("XPC FrameReady delivery failed");
        assert_eq!(frame.frame_id, expected_frame_id, "out-of-order frame");
        assert_eq!(
            frame.signal_value, expected_frame_id,
            "unexpected event value"
        );
        let generation = if expected_frame_id <= count { 1 } else { 2 };
        let local_frame_id = ((expected_frame_id - 1) % count) + 1;
        assert_eq!(frame.generation, generation, "unexpected resize generation");
        assert_eq!(
            frame.backing_id,
            match (generation, local_frame_id % 2) {
                (1, 1) => BACKING_A,
                (1, 0) => BACKING_B,
                (2, 1) => BACKING_C,
                _ => BACKING_D,
            },
            "producer reused the wrong A/B backing"
        );
        if generation == 2 && local_frame_id == 1 {
            assert_eq!(
                backings.len(),
                2,
                "generation 1 backing cache was lost early"
            );
            iosurface_destroys += backings.len();
            backings.clear();
            let retired_textures = renderer.retire_external_image_generation(1);
            gpui_texture_destroys += retired_textures;
            assert_eq!(
                retired_textures,
                2,
                "generation 1 Metal textures were not retired"
            );
            println!("consumer retired generation 1 after its leases released");
        }
        let key = (frame.backing_id, frame.generation);
        if !backings.contains_key(&key) {
            let descriptor = channel
                .import_iosurface_backing(channel_id, frame.backing_id, frame.generation)
                .expect("XPC IOSurface backing import failed");
            let image_buffer = MetalRenderer::import_iosurface_backing(&descriptor)
                .expect("could not import registered IOSurface");
            let expected_size = if generation == 1 {
                (800, 500)
            } else {
                (1000, 700)
            };
            assert_eq!(image_buffer.get_width(), expected_size.0);
            assert_eq!(image_buffer.get_height(), expected_size.1);
            eprintln!(
                "imported persistent backing {}/{} at {}x{}",
                frame.backing_id, frame.generation, descriptor.width, descriptor.height
            );
            backings.insert(
                key,
                (
                    image_buffer,
                    descriptor.pixel_format,
                    descriptor.width,
                    descriptor.height,
                ),
            );
            iosurface_imports += 1;
        }
        let (image_buffer, pixel_format, width, height) =
            backings.get(&key).expect("backing cache miss");
        let iosurface = unsafe { CVPixelBufferGetIOSurface(image_buffer.as_concrete_TypeRef()) };
        assert!(
            !iosurface.is_null(),
            "imported CVPixelBuffer lost its IOSurface"
        );

        let release_channel = channel.clone();
        let release_channel_id = channel_id.to_owned();
        let (backing_id, generation, frame_id) =
            (frame.backing_id, frame.generation, frame.frame_id);
        let released = Arc::new(AtomicBool::new(false));
        let callback_released = released.clone();
        let lease = MacExternalImageFrame::new(
            backing_id,
            generation,
            frame_id,
            iosurface as usize,
            *pixel_format,
            image_buffer.clone(),
            event.clone(),
            frame.signal_value,
            move || {
                release_channel
                    .release_frame(&release_channel_id, backing_id, generation, frame_id)
                    .expect("producer rejected GPU-completion release");
                callback_released.store(true, Ordering::Release);
            },
        );
        let mut scene = Scene::default();
        insert_surface(&mut scene, lease.clone(), *width as f32, *height as f32);
        scene.finish();
        let capture = renderer
            .render_scene_to_image_with_submission_callback(
                &scene,
                size(DevicePixels(*width as i32), DevicePixels(*height as i32)),
                || {
                    channel
                        .notify_consumer_submission(channel_id, frame_id)
                        .expect("could not notify producer of the submitted Metal wait");
                },
            )
            .expect("GPUI failed to render cross-process IOSurface scene");
        assert_eq!(capture.get_pixel(100, 100).0, [255, 0, 0, 255]);
        assert_eq!(capture.get_pixel(700, 100).0, [0, 255, 0, 255]);
        assert_eq!(capture.get_pixel(100, 400).0, [0, 0, 255, 255]);
        assert_eq!(capture.get_pixel(700, 400).0, [255, 255, 255, 255]);
        let marker_x = *width as u32 - 10;
        let marker_y = *height as u32 - 10;
        if frame_id % 2 == 1 {
            assert_eq!(capture.get_pixel(10, 10).0, [0, 0, 0, 255]);
            assert_eq!(
                capture.get_pixel(marker_x, marker_y).0,
                [255, 255, 255, 255]
            );
        } else {
            assert_eq!(capture.get_pixel(10, 10).0, [255, 0, 0, 255]);
            assert_eq!(capture.get_pixel(marker_x, marker_y).0, [0, 0, 0, 255]);
        }
        if expected_frame_id == 1 {
            let screenshot = std::env::temp_dir().join("gpui-crossprocess-iosurface.png");
            capture.save(&screenshot).unwrap();
            println!("GPUI cross-process capture: {}", screenshot.display());
        }
        lease.retire();
        assert!(lease.is_released(), "GPUI completion did not release frame");
        assert!(
            released.load(Ordering::Acquire),
            "release IPC did not complete"
        );
    }
    assert_eq!(
        backings.len(),
        2,
        "expected only the current generation's two imported IOSurfaces"
    );
    assert_eq!(renderer.external_surface_texture_count(), 2);
    iosurface_destroys += backings.len();
    backings.clear();
    let retired_textures = renderer.retire_external_image_generation(2);
    gpui_texture_destroys += retired_textures;
    assert_eq!(
        retired_textures,
        2,
        "generation 2 Metal textures were not destroyed at shutdown"
    );
    assert_eq!(iosurface_imports, 4, "IOSurfaces should import once per backing");
    assert_eq!(iosurface_destroys, 4, "all imported IOSurfaces should retire");
    assert_eq!(renderer.external_surface_texture_import_count(), 4);
    assert_eq!(gpui_texture_destroys, 4);
    println!(
        "consumer completed {total_frames} frames; IOSurface imports/destroys={iosurface_imports}/{iosurface_destroys}, GPUI texture imports/destroys={}/{gpui_texture_destroys}, shared-event imports=1, releases={total_frames}, outstanding leases=0, generations alive=0",
        renderer.external_surface_texture_import_count()
    );
}

fn insert_surface(scene: &mut Scene, frame: MacExternalImageFrame, width: f32, height: f32) {
    let bounds: Bounds<ScaledPixels> = bounds(
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
        texture_size: gpui::Size::new(DevicePixels(width as i32), DevicePixels(height as i32)),
    });
}
