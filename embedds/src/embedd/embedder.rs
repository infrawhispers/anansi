use std::path::Path;
use std::sync::Arc;

use super::image_processor::ImageProcessor;
use ort::environment::Environment;

pub struct EmbedderParams<'a> {
    pub model_path: &'a Path,
    pub model_name: &'a str,
    pub num_threads: i16,
    pub parallel_execution: bool,
    pub ort_environment: Arc<Environment>,
    pub img_processor: Arc<ImageProcessor>,
}

#[derive(Debug)]
pub enum CLIPParams<'a> {
    Text { vals: &'a [String] },
    Uri { vals: &'a [String] },
    UriBytes { vals: &'a Vec<Vec<u8>> },
}

#[derive(Debug)]
pub struct InstructorParams<'a> {
    pub text: &'a [String],
    pub instructions: &'a [String],
}

#[derive(Debug)]
pub struct E5Params<'a> {
    pub text: &'a [String],
}

#[derive(Debug)]
pub enum EmebeddingRequest<'a> {
    CLIPRequest { params: CLIPParams<'a> },
    InstructorRequest { params: InstructorParams<'a> },
    E5Request { params: E5Params<'a> },
}

pub trait Embedder: Send + Sync {
    fn new(params: &EmbedderParams) -> anyhow::Result<Self>
    where
        Self: Sized;
    fn encode(&self, req: &EmebeddingRequest) -> anyhow::Result<Vec<Vec<f32>>>;
}
