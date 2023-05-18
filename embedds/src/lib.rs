pub mod app_config;
pub mod embedd;
pub mod manager;
mod utils;

pub const API_DESCRIPTOR_SET: &[u8] = tonic::include_file_descriptor_set!("api");

pub mod api {
    tonic::include_proto!("api"); // The string specified here must match the proto package name
}

// #[derive(Debug)]
// pub enum IndexItems {
//     Embedds {
//         data: Vec<Vec<f32>>,
//         ids: Vec<String>,
//         sub_indices: Vec<String>,
//     },
//     ToEmbedd {
//         data: api::EncodeItem,
//         ids: Vec<String>,
//         sub_indices: Vec<String>,
//     },
// }

#[derive(Debug, Clone)]
pub struct IndexItems {
    pub ids: Vec<retrieval::ann::EId>,
    pub sub_indices: Vec<String>,
    pub embedds: Option<Vec<Vec<f32>>>,
    pub to_embedd: Option<api::EncodeBatch>,
    // items: IndexItem,
}
