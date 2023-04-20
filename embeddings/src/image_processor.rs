use std::io::Cursor;
use std::sync::Arc;
use std::thread::available_parallelism;

use image::io::Reader as ImageReader;
use image::GenericImageView;
use libvips::ops::{jpegsave_buffer, resize, smartcrop_with_opts, Interesting};
use libvips::{VipsApp, VipsImage};
use ndarray::{Array, Dim};
use tokio::runtime::Handle;

use crate::utils::{download_image, download_image_sync};

#[allow(dead_code)]
pub struct ImageProcessor {
    // this simply needs to live for the lifetime of the application - silence the rust warning
    // that it is unused.
    libvips_app: Arc<VipsApp>,
}

impl ImageProcessor {
    pub fn new() -> anyhow::Result<Self> {
        let libvips_app = Arc::new(VipsApp::new("libvips-processor", false)?);
        let mut num_threads = 4;
        match available_parallelism() {
            Ok(ap) => num_threads = ap.get().try_into()?,
            Err(err) => {}
        }
        libvips_app.concurrency_set(num_threads);
        Ok(ImageProcessor {
            libvips_app: libvips_app,
        })
    }
    // heavily inspired by:
    // https://github.com/openai/CLIP. MIT License, Copyright (c) 2021 OpenAI
    pub fn uri_to_clip_vec(
        &self,
        uri: &str,
        dimensions: i32,
    ) -> anyhow::Result<Array<f32, Dim<[usize; 3]>>> {
        let res;
        match Handle::try_current() {
            Ok(handle) => {
                handle.enter();
                res = tokio::task::block_in_place(|| handle.block_on(download_image(uri)))?;
            }
            Err(err) => {
                res = download_image_sync(uri)?;
            }
        };
        // match Handle::try_current();
        // handle.enter();
        // try_current
        // let res = download_image_sync(uri)?;
        let img = VipsImage::new_from_buffer(&res, "")?;
        let scale: f64 = dimensions as f64 / i32::min(img.get_height(), img.get_width()) as f64;
        let img_scaled = resize(&img, scale)?;
        let img_cropped = smartcrop_with_opts(
            &img_scaled,
            dimensions,
            dimensions,
            &libvips::ops::SmartcropOptions {
                interesting: Interesting::Centre,
            },
        )?;
        let formatted = jpegsave_buffer(&img_cropped)?;
        // getpoint(..) is *super* slow in libvips - so we are going to
        // use ImageReader
        let img2 = ImageReader::new(Cursor::new(formatted))
            .with_guessed_format()?
            .decode()?;
        let mut a = Array::<f32, _>::zeros((3, dimensions as usize, dimensions as usize));
        for i in 0..(dimensions as usize) {
            for j in 0..(dimensions as usize) {
                let p = img2.get_pixel(i.try_into()?, j.try_into()?);
                a[[0, i as usize, j as usize]] = p[0] as f32 / 255.0;
                a[[1, i as usize, j as usize]] = p[1] as f32 / 255.0;
                a[[2, i as usize, j as usize]] = p[2] as f32 / 255.0;
                a[[0, i as usize, j as usize]] =
                    (a[[0, i as usize, j as usize]] - 0.48145466) / 0.26862954;
                a[[1, i as usize, j as usize]] =
                    (a[[1, i as usize, j as usize]] - 0.4578275) / 0.26130258;
                a[[2, i as usize, j as usize]] =
                    (a[[2, i as usize, j as usize]] - 0.40821073) / 0.27577711;
            }
        }
        Ok(a)
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_clip_resizing() {
        let processor = ImageProcessor::new().expect("unable to create the img processor");
        let uri = "https://images.unsplash.com/photo-1481349518771-20055b2a7b24?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=2439&q=80";
        assert!(
            processor.uri_to_clip_vec(uri, 224).is_ok(),
            "unable to download and create multi-dimensional arr"
        )
    }
}
