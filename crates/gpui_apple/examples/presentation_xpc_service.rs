#![cfg(target_os = "macos")]

fn main() {
    let service_name = std::env::args()
        .nth(1)
        .expect("pass the launchd Mach service name");
    gpui_apple::presentation_xpc::run_presentation_event_service(&service_name)
        .expect("presentation XPC service failed");
}
