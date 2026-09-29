use crate::{DevicePixels, Pixels, Result, SharedString, Size, size};
use smallvec::SmallVec;

use image::{Delay, Frame};
use std::{
    borrow::Cow,
    fmt,
    hash::Hash,
    sync::atomic::{AtomicUsize, Ordering::SeqCst},
};

/// A source of assets for this app to use.
pub trait AssetSource: 'static + Send + Sync {
    /// Load the given asset from the source path.
    fn load(&self, path: &str) -> Result<Option<Cow<'static, [u8]>>>;

    /// List the assets at the given path.
    fn list(&self, path: &str) -> Result<Vec<SharedString>>;
}

impl AssetSource for () {
    fn load(&self, _path: &str) -> Result<Option<Cow<'static, [u8]>>> {
        Ok(None)
    }

    fn list(&self, _path: &str) -> Result<Vec<SharedString>> {
        Ok(vec![])
    }
}

/// A unique identifier for the image cache
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct ImageId(pub usize);

#[derive(PartialEq, Eq, Hash, Clone)]
#[expect(missing_docs)]
pub struct RenderImageParams {
    pub image_id: ImageId,
    pub frame_index: usize,
}

/// A cached and processed image, in BGRA format
pub struct RenderImage {
    /// The ID associated with this image
    pub id: ImageId,
    /// The scale factor of this image on render.
    pub(crate) scale_factor: f32,
    data: SmallVec<[Frame; 1]>,
}

impl PartialEq for RenderImage {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for RenderImage {}

impl RenderImage {
    /// Create a new image from the given data.
    pub fn new(data: impl Into<SmallVec<[Frame; 1]>>) -> Self {
        Self {
            id: next_image_id(),
            scale_factor: 1.0,
            data: data.into(),
        }
    }

    /// Create a single-frame image from tightly packed BGRA pixels.
    pub fn from_bgra(width: u32, height: u32, bytes: Vec<u8>) -> Option<Self> {
        let buffer = image::RgbaImage::from_raw(width, height, bytes)?;
        Some(Self::new(SmallVec::from_elem(Frame::new(buffer), 1)))
    }

    /// Convert this image into a byte slice.
    pub fn as_bytes(&self, frame_index: usize) -> Option<&[u8]> {
        self.data
            .get(frame_index)
            .map(|frame| frame.buffer().as_raw().as_slice())
    }

    /// Get the size of this image, in pixels.
    pub fn size(&self, frame_index: usize) -> Size<DevicePixels> {
        self.data
            .get(frame_index)
            .map(|frame| {
                let (width, height) = frame.buffer().dimensions();
                size(width.into(), height.into())
            })
            .unwrap_or_default()
    }

    /// Get the size of this image, in pixels for display, adjusted for the scale factor.
    pub(crate) fn render_size(&self, frame_index: usize) -> Size<Pixels> {
        self.size(frame_index)
            .map(|v| (v.0 as f32 / self.scale_factor).into())
    }

    /// Get the delay of this frame from the previous
    pub fn delay(&self, frame_index: usize) -> Delay {
        self.data
            .get(frame_index)
            .map(|frame| frame.delay())
            .unwrap_or(Delay::from_numer_denom_ms(100, 1))
    }

    /// Get the number of frames for this image.
    pub fn frame_count(&self) -> usize {
        self.data.len()
    }
}

/// A single-frame BGRA image whose pixels can be replaced while its identity
/// stays stable. Intended for native surfaces that update frequently.
pub struct LiveImage {
    pub(crate) id: ImageId,
    frame: std::sync::Mutex<LiveImageFrame>,
}

struct LiveImageFrame {
    width: u32,
    height: u32,
    pixels: Vec<u8>,
    revision: u64,
}

impl LiveImage {
    /// Create an empty BGRA surface. Pixel storage is allocated once and is
    /// resized only when the surface dimensions change.
    pub fn new(width: u32, height: u32) -> Option<Self> {
        let length = (width as usize).checked_mul(height as usize)?.checked_mul(4)?;
        Some(Self {
            id: next_image_id(),
            frame: std::sync::Mutex::new(LiveImageFrame {
                width,
                height,
                pixels: vec![0; length],
                revision: 0,
            }),
        })
    }

    /// Copy a strided BGRA frame into the existing backing allocation.
    pub fn update_bgra(&self, width: u32, height: u32, stride: usize, bytes: &[u8]) -> bool {
        let Some(row_bytes) = (width as usize).checked_mul(4) else { return false };
        let Some(source_length) = stride.checked_mul(height as usize) else { return false };
        let Some(packed_length) = row_bytes.checked_mul(height as usize) else { return false };
        if stride < row_bytes || bytes.len() < source_length { return false; }
        let mut frame = self.frame.lock().unwrap();
        if frame.width != width || frame.height != height || frame.pixels.len() != packed_length {
            frame.width = width;
            frame.height = height;
            frame.pixels.resize(packed_length, 0);
        }
        for row in 0..height as usize {
            let source_start = row * stride;
            let target_start = row * row_bytes;
            frame.pixels[target_start..target_start + row_bytes]
                .copy_from_slice(&bytes[source_start..source_start + row_bytes]);
        }
        frame.revision = frame.revision.wrapping_add(1);
        true
    }

    /// Capacity of the owned BGRA backing allocation, in bytes.
    pub fn byte_capacity(&self) -> usize {
        self.frame.lock().unwrap().pixels.capacity()
    }

    pub(crate) fn with_pixels<R>(&self, f: impl FnOnce(u32, u32, u64, &[u8]) -> R) -> R {
        let frame = self.frame.lock().unwrap();
        f(frame.width, frame.height, frame.revision, &frame.pixels)
    }
}

fn next_image_id() -> ImageId {
    static NEXT_ID: AtomicUsize = AtomicUsize::new(0);
    ImageId(NEXT_ID.fetch_add(1, SeqCst))
}

impl fmt::Debug for RenderImage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ImageData")
            .field("id", &self.id)
            .field("size", &self.data.first().map(|f| f.buffer().dimensions()))
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use smallvec::SmallVec;

    #[test]
    fn empty_render_image_does_not_panic() {
        let image = RenderImage::new(SmallVec::new());
        assert_eq!(image.frame_count(), 0);
        assert_eq!(image.size(0), Size::default());
        assert_eq!(image.as_bytes(0), None);
        assert_eq!(image.render_size(0), Size::default());
        assert_eq!(image.delay(0), Delay::from_numer_denom_ms(100, 1));
        let _ = format!("{image:?}");
    }

    #[test]
    fn constructs_an_image_from_bgra_bytes() {
        let bytes = vec![0x10, 0x20, 0x30, 0xff, 0xaa, 0xbb, 0xcc, 0xff];
        let image = RenderImage::from_bgra(2, 1, bytes.clone()).expect("valid BGRA image");
        assert_eq!(image.size(0), size(DevicePixels(2), DevicePixels(1)));
        assert_eq!(image.as_bytes(0), Some(bytes.as_slice()));
        assert!(RenderImage::from_bgra(2, 1, vec![0; 4]).is_none());
    }
}
