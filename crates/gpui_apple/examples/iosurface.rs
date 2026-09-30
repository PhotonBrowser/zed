#![cfg(target_os = "macos")]

use gpui::{
    App, Context, IntoElement, Render, Window, WindowBounds, WindowOptions, div, prelude::*, px,
    rgb, size,
};
use gpui_apple::metal_renderer::MetalRenderer;
use gpui_platform::application;
use std::time::Duration;

struct ExternalImageDemo {
    frame: gpui::MacExternalImageFrame,
}

impl Render for ExternalImageDemo {
    fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        div()
            .size_full()
            .flex()
            .items_center()
            .justify_center()
            .bg(rgb(0x202020))
            .child(gpui::surface(self.frame.clone()).w(px(800.)).h(px(500.)))
    }
}

fn main() {
    let device = MetalRenderer::device_identity();
    println!(
        "GPUI Metal device: {} (registryID={})",
        device.name, device.registry_id
    );
    application().run(move |cx: &mut App| {
        cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(gpui::Bounds::centered(
                    None,
                    size(px(880.), px(580.)),
                    cx,
                ))),
                ..Default::default()
            },
            |window, cx| {
                let scale = window.scale_factor();
                let width = (800.0 * scale).round() as usize;
                let height = (500.0 * scale).round() as usize;
                println!(
                    "GPUI surface: logical=800x500 scale={scale:.2} physical={width}x{height}"
                );
                let frame = MetalRenderer::create_standalone_test_frame(width, height, 1, 1, 1)
                    .expect("could not create the GPU-rendered IOSurface test image");
                cx.new(|_| ExternalImageDemo { frame })
            },
        )
        .expect("could not open GPUI external-image demo window");
        cx.activate(true);
        cx.spawn(async move |cx| {
            cx.background_executor().timer(Duration::from_secs(3)).await;
            cx.update(|app| app.quit());
        })
        .detach();
    });
}
