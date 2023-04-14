pub mod clip;
pub mod embedder;
pub mod embedder_manager;
pub mod instructor;
pub mod utils;

pub const API_DESCRIPTOR_SET: &[u8] = tonic::include_file_descriptor_set!("api");
