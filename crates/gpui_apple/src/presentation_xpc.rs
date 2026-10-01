//! Persistent XPC transport for Metal shared-event handles.
//!
//! Metal's handle is an NSXPC object, not a byte-serializable payload. This
//! channel transports it once for a stable presentation-channel key; frame
//! traffic carries only IDs and monotonically increasing event values.

use anyhow::{Context as _, Result, bail, ensure};
use foreign_types::ForeignType;
use gpui::{MacGpuBackingDescriptor, MacIOSurfaceSendRight};
use mach2::port::mach_port_t;
use metal::{DeviceRef, SharedEvent};
use objc::{msg_send, sel, sel_impl};
use std::{
    ffi::{CString, c_char, c_void},
    ptr::NonNull,
};

unsafe extern "C" {
    fn photon_presentation_xpc_connect(service_name: *const c_char) -> *mut c_void;
    fn photon_presentation_xpc_disconnect(connection: *mut c_void);
    fn photon_presentation_xpc_register_event(
        connection: *mut c_void,
        channel_name: *const c_char,
        registry_id: u64,
        event: *mut c_void,
    ) -> bool;
    fn photon_presentation_xpc_copy_event_handle(
        connection: *mut c_void,
        channel_name: *const c_char,
        registry_id: *mut u64,
    ) -> *mut c_void;
    fn photon_presentation_xpc_wait_for_consumer_submission(
        connection: *mut c_void,
        channel_name: *const c_char,
        frame_id: u64,
    ) -> bool;
    fn photon_presentation_xpc_notify_consumer_submission(
        connection: *mut c_void,
        channel_name: *const c_char,
        frame_id: u64,
    ) -> bool;
    fn photon_presentation_xpc_register_backing(
        connection: *mut c_void,
        channel_name: *const c_char,
        backing_id: u64,
        generation: u64,
        width: u32,
        height: u32,
        pixel_format: u32,
        mach_port: mach_port_t,
    ) -> bool;
    fn photon_presentation_xpc_copy_backing(
        connection: *mut c_void,
        channel_name: *const c_char,
        backing_id: u64,
        generation: u64,
        mach_port: *mut mach_port_t,
        width: *mut u32,
        height: *mut u32,
        pixel_format: *mut u32,
    ) -> bool;
    fn photon_presentation_xpc_unregister_backing(
        connection: *mut c_void,
        channel_name: *const c_char,
        backing_id: u64,
        generation: u64,
    ) -> bool;
    fn photon_presentation_xpc_release_frame(
        connection: *mut c_void,
        channel_name: *const c_char,
        backing_id: u64,
        generation: u64,
        frame_id: u64,
    ) -> bool;
    fn photon_presentation_xpc_wait_for_frame_release(
        connection: *mut c_void,
        channel_name: *const c_char,
        backing_id: u64,
        generation: u64,
        frame_id: u64,
    ) -> bool;
    fn photon_presentation_xpc_publish_frame(
        connection: *mut c_void,
        channel_name: *const c_char,
        backing_id: u64,
        generation: u64,
        frame_id: u64,
        signal_value: u64,
    ) -> bool;
    fn photon_presentation_xpc_wait_for_frame(
        connection: *mut c_void,
        channel_name: *const c_char,
        backing_id: *mut u64,
        generation: *mut u64,
        frame_id: *mut u64,
        signal_value: *mut u64,
    ) -> bool;
    fn photon_presentation_xpc_release_object(object: *mut c_void);
    fn photon_presentation_xpc_run_service(service_name: *const c_char) -> i32;
}

/// Persistent connection to the XPC presentation-event broker.
///
/// Keep one connection per producer or consumer process, not per frame.
pub struct MacPresentationEventChannel {
    connection: NonNull<c_void>,
}

/// Small per-frame message carried after persistent event/backing registration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MacPresentationFrame {
    pub backing_id: u64,
    pub generation: u64,
    pub frame_id: u64,
    pub signal_value: u64,
}

// Both NSXPCConnection and xpc_connection_t are thread-safe. The opaque C
// wrapper owns the connections until its final Rust owner drops.
unsafe impl Send for MacPresentationEventChannel {}
unsafe impl Sync for MacPresentationEventChannel {}

impl MacPresentationEventChannel {
    /// Connects to the named launchd/XPC service.
    pub fn connect(service_name: &str) -> Result<Self> {
        let service_name = CString::new(service_name).context("XPC service name contains NUL")?;
        let connection = unsafe { photon_presentation_xpc_connect(service_name.as_ptr()) };
        let connection =
            NonNull::new(connection).context("could not create presentation XPC connection")?;
        Ok(Self { connection })
    }

    /// Registers a shared event handle once for a stable view/generation key.
    pub fn register_shared_event(
        &self,
        channel_id: &str,
        producer_registry_id: u64,
        event: &SharedEvent,
    ) -> Result<()> {
        let channel_id =
            CString::new(channel_id).context("presentation channel ID contains NUL")?;
        let accepted = unsafe {
            photon_presentation_xpc_register_event(
                self.connection.as_ptr(),
                channel_id.as_ptr(),
                producer_registry_id,
                event.as_ptr().cast(),
            )
        };
        ensure!(accepted, "XPC broker rejected shared-event registration");
        Ok(())
    }

    /// Imports the registered event once and verifies the producer GPU identity.
    pub fn import_shared_event(
        &self,
        channel_id: &str,
        consumer_device: &DeviceRef,
    ) -> Result<SharedEvent> {
        self.import_shared_event_with_identity(channel_id, consumer_device)
            .map(|(event, _)| event)
    }

    /// Imports the persistent producer event and returns the producer device
    /// registry ID advertised with its XPC handle.
    pub fn import_shared_event_with_identity(
        &self,
        channel_id: &str,
        consumer_device: &DeviceRef,
    ) -> Result<(SharedEvent, u64)> {
        let channel_id =
            CString::new(channel_id).context("presentation channel ID contains NUL")?;
        let mut producer_registry_id = 0;
        let handle = unsafe {
            photon_presentation_xpc_copy_event_handle(
                self.connection.as_ptr(),
                channel_id.as_ptr(),
                &mut producer_registry_id,
            )
        };
        let handle =
            NonNull::new(handle).context("XPC broker has no shared event for this channel")?;
        let _handle_guard = SharedEventHandleGuard(handle);
        ensure!(
            producer_registry_id == consumer_device.registry_id(),
            "Metal registry ID mismatch: producer={producer_registry_id}, consumer={}",
            consumer_device.registry_id()
        );
        let event: *mut metal::MTLSharedEvent =
            unsafe { msg_send![consumer_device, newSharedEventWithHandle: handle.as_ptr()] };
        if event.is_null() {
            bail!("GPUI Metal device could not recreate the XPC shared-event handle");
        }
        Ok((unsafe { SharedEvent::from_ptr(event) }, producer_registry_id))
    }

    /// Test/protocol hook: wait until the consumer has submitted a command
    /// buffer that waits on this channel's shared event.
    pub fn wait_for_consumer_submission(&self, channel_id: &str, frame_id: u64) -> Result<()> {
        let channel_id =
            CString::new(channel_id).context("presentation channel ID contains NUL")?;
        ensure!(
            unsafe {
                photon_presentation_xpc_wait_for_consumer_submission(
                    self.connection.as_ptr(),
                    channel_id.as_ptr(),
                    frame_id,
                )
            },
            "consumer did not submit its event wait"
        );
        Ok(())
    }

    /// Test/protocol hook: notify the producer only after a consumer wait has
    /// been submitted to Metal.
    pub fn notify_consumer_submission(&self, channel_id: &str, frame_id: u64) -> Result<()> {
        let channel_id =
            CString::new(channel_id).context("presentation channel ID contains NUL")?;
        ensure!(
            unsafe {
                photon_presentation_xpc_notify_consumer_submission(
                    self.connection.as_ptr(),
                    channel_id.as_ptr(),
                    frame_id,
                )
            },
            "XPC broker has no producer waiting for this channel"
        );
        Ok(())
    }

    /// Registers a persistent backing. The IOSurface Mach send right is
    /// duplicated for XPC ownership; this descriptor keeps and releases its
    /// original right independently.
    pub fn register_iosurface_backing(
        &self,
        channel_id: &str,
        descriptor: &MacGpuBackingDescriptor,
    ) -> Result<()> {
        let channel_id =
            CString::new(channel_id).context("presentation channel ID contains NUL")?;
        ensure!(
            unsafe {
                photon_presentation_xpc_register_backing(
                    self.connection.as_ptr(),
                    channel_id.as_ptr(),
                    descriptor.backing_id,
                    descriptor.generation,
                    descriptor.width,
                    descriptor.height,
                    descriptor.pixel_format,
                    descriptor.iosurface_port.as_raw(),
                )
            },
            "XPC broker rejected IOSurface backing registration"
        );
        Ok(())
    }

    /// Imports one registered IOSurface send right and its stable dimensions.
    pub fn import_iosurface_backing(
        &self,
        channel_id: &str,
        backing_id: u64,
        generation: u64,
    ) -> Result<MacGpuBackingDescriptor> {
        let channel_id =
            CString::new(channel_id).context("presentation channel ID contains NUL")?;
        let mut mach_port = mach2::port::MACH_PORT_NULL;
        let mut width = 0;
        let mut height = 0;
        let mut pixel_format = 0;
        ensure!(
            unsafe {
                photon_presentation_xpc_copy_backing(
                    self.connection.as_ptr(),
                    channel_id.as_ptr(),
                    backing_id,
                    generation,
                    &mut mach_port,
                    &mut width,
                    &mut height,
                    &mut pixel_format,
                )
            },
            "XPC broker has no IOSurface backing registered for this channel/generation"
        );
        ensure!(
            mach_port != mach2::port::MACH_PORT_NULL,
            "XPC returned a null IOSurface port"
        );
        let iosurface_port = unsafe { MacIOSurfaceSendRight::from_owned_raw(mach_port) };
        Ok(MacGpuBackingDescriptor {
            backing_id,
            generation,
            width,
            height,
            pixel_format,
            iosurface_port,
        })
    }

    /// Retires a registered backing after all consumer frame leases have been
    /// released. The broker deallocates its owned Mach send right on success.
    pub fn unregister_iosurface_backing(
        &self,
        channel_id: &str,
        backing_id: u64,
        generation: u64,
    ) -> Result<()> {
        let channel_id =
            CString::new(channel_id).context("presentation channel ID contains NUL")?;
        ensure!(
            unsafe {
                photon_presentation_xpc_unregister_backing(
                    self.connection.as_ptr(),
                    channel_id.as_ptr(),
                    backing_id,
                    generation,
                )
            },
            "XPC broker rejected backing retirement (it may still have leased frames)"
        );
        Ok(())
    }

    /// Acknowledges a frame after GPUI's sampling command buffer completes.
    pub fn release_frame(
        &self,
        channel_id: &str,
        backing_id: u64,
        generation: u64,
        frame_id: u64,
    ) -> Result<()> {
        let channel_id =
            CString::new(channel_id).context("presentation channel ID contains NUL")?;
        ensure!(
            unsafe {
                photon_presentation_xpc_release_frame(
                    self.connection.as_ptr(),
                    channel_id.as_ptr(),
                    backing_id,
                    generation,
                    frame_id,
                )
            },
            "XPC broker rejected frame release"
        );
        Ok(())
    }

    /// Test/protocol hook that waits without polling until the consumer releases
    /// the matching frame lease.
    pub fn wait_for_frame_release(
        &self,
        channel_id: &str,
        backing_id: u64,
        generation: u64,
        frame_id: u64,
    ) -> Result<()> {
        let channel_id =
            CString::new(channel_id).context("presentation channel ID contains NUL")?;
        ensure!(
            unsafe {
                photon_presentation_xpc_wait_for_frame_release(
                    self.connection.as_ptr(),
                    channel_id.as_ptr(),
                    backing_id,
                    generation,
                    frame_id,
                )
            },
            "producer did not receive GPUI frame release"
        );
        Ok(())
    }

    /// Publishes only frame identifiers and the producer event value. The
    /// IOSurface and shared event are registered separately and persist.
    pub fn publish_frame(&self, channel_id: &str, frame: MacPresentationFrame) -> Result<()> {
        let channel_id =
            CString::new(channel_id).context("presentation channel ID contains NUL")?;
        ensure!(
            unsafe {
                photon_presentation_xpc_publish_frame(
                    self.connection.as_ptr(),
                    channel_id.as_ptr(),
                    frame.backing_id,
                    frame.generation,
                    frame.frame_id,
                    frame.signal_value,
                )
            },
            "XPC broker rejected frame-ready message"
        );
        Ok(())
    }

    /// Waits for the next frame-ready descriptor on this channel without
    /// polling or importing per-frame resources.
    pub fn wait_for_frame(&self, channel_id: &str) -> Result<MacPresentationFrame> {
        let channel_id =
            CString::new(channel_id).context("presentation channel ID contains NUL")?;
        let (mut backing_id, mut generation, mut frame_id, mut signal_value) = (0, 0, 0, 0);
        ensure!(
            unsafe {
                photon_presentation_xpc_wait_for_frame(
                    self.connection.as_ptr(),
                    channel_id.as_ptr(),
                    &mut backing_id,
                    &mut generation,
                    &mut frame_id,
                    &mut signal_value,
                )
            },
            "XPC broker did not deliver a frame-ready message"
        );
        Ok(MacPresentationFrame {
            backing_id,
            generation,
            frame_id,
            signal_value,
        })
    }
}

impl Drop for MacPresentationEventChannel {
    fn drop(&mut self) {
        unsafe { photon_presentation_xpc_disconnect(self.connection.as_ptr()) };
    }
}

struct SharedEventHandleGuard(NonNull<c_void>);

impl Drop for SharedEventHandleGuard {
    fn drop(&mut self) {
        unsafe { photon_presentation_xpc_release_object(self.0.as_ptr()) };
    }
}

/// Runs a named mach-service listener. Intended for the standalone XPC helper
/// executable; production bundles should host the same service as an XPC target.
pub fn run_presentation_event_service(service_name: &str) -> Result<()> {
    let service_name = CString::new(service_name).context("XPC service name contains NUL")?;
    let status = unsafe { photon_presentation_xpc_run_service(service_name.as_ptr()) };
    ensure!(status == 0, "XPC event service exited with status {status}");
    Ok(())
}
