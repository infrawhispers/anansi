pub mod app_config;
pub mod clip;
pub mod embedder;
pub mod embedder_manager;
pub mod image_processor;
pub mod instructor;
pub mod utils;

pub const API_DESCRIPTOR_SET: &[u8] = tonic::include_file_descriptor_set!("api");

pub mod api {
    tonic::include_proto!("api"); // The string specified here must match the proto package name
}
