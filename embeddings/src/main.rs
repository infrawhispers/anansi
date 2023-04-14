use std::path::PathBuf;
use std::sync::Arc;
use std::thread::available_parallelism;

use anyhow::bail;
use ort::ExecutionProvider;
use tokio::task;
use tonic::{transport::Server, Code, Request, Response, Status};

use api::api_server::{Api, ApiServer};
use api::{
    EncodeItem, EncodeRequest, EncodeResponse, EncodeResult, EncodingModel, EncodingModelDevice,
};
use embeddings::embedder;
use embeddings::embedder::{CLIPParams, InstructorParams};
use embeddings::embedder_manager;
use embeddings::API_DESCRIPTOR_SET;
use tracing::Level;

pub mod api {
    tonic::include_proto!("api"); // The string specified here must match the proto package name
}

pub struct ApiServerImpl {
    mgr: Arc<embedder_manager::EmbedderManager>,
}

impl ApiServerImpl {
    fn new(model_path: &PathBuf) -> anyhow::Result<Self> {
        let obj = ApiServerImpl {
            mgr: Arc::new(embedder_manager::EmbedderManager::new(model_path)?),
        };
        Ok(obj)
    }
}

impl ApiServerImpl {
    fn to_execution_providers(&self, devices: Vec<i32>) -> anyhow::Result<Vec<ExecutionProvider>> {
        let mut res: Vec<ExecutionProvider> = Vec::with_capacity(devices.len());
        for i in 0..devices.len() {
            match EncodingModelDevice::from_i32(devices[i]) {
                Some(EncodingModelDevice::MdUnknown) => {
                    bail!("unknown model device: {}", devices[i])
                }
                Some(EncodingModelDevice::MdCpu) => {
                    res.push(ExecutionProvider::cpu());
                }
                Some(EncodingModelDevice::MdCuda) => {
                    res.push(ExecutionProvider::cuda());
                }
                Some(EncodingModelDevice::MdTensor) => {
                    res.push(ExecutionProvider::tensorrt());
                }
                None => {
                    bail!("unknown model device: {}", devices[i])
                }
            }
        }
        Ok(res)
    }

    fn transform_encode_req<'a>(
        &'a self,
        data: &'a Vec<EncodeItem>,
    ) -> Result<Vec<(&str, embedder::EmebeddingRequest)>, Status> {
        let mut req: Vec<(&str, embedder::EmebeddingRequest)> = Vec::new();
        for i in 0..data.len() {
            let item = &data[i];
            let x;
            match EncodingModel::from_i32(item.model) {
                Some(val) => x = val,
                None => {
                    return Err(Status::new(
                        Code::InvalidArgument,
                        format!("unknown model: {} set at item: {}", item.model, i),
                    ));
                }
            }
            match x {
                EncodingModel::MInstructor => {
                    if data[i].text.len() != data[i].instructions.len() {
                        return Err(Status::new(
                            Code::InvalidArgument,
                            format!(
                                "INSTRUCTOR: len text: {} != len instructions {}",
                                data[i].text.len(),
                                data[i].instructions.len()
                            ),
                        ));
                    }
                    req.push((
                        x.as_str_name(),
                        embedder::EmebeddingRequest::InstructorRequest {
                            params: InstructorParams {
                                text: &data[i].text,
                                instructions: &data[i].instructions,
                            },
                        },
                    ));
                }
                EncodingModel::MClipRn50Openai
                | EncodingModel::MClipRn50Yfcc15m
                | EncodingModel::MClipRn50Cc12m
                | EncodingModel::MClipRn101Openai
                | EncodingModel::MClipRn101Yfcc15m
                | EncodingModel::MClipRn50x4Openai
                | EncodingModel::MClipRn50x16Openai
                | EncodingModel::MClipRn50x64Openai => {
                    if data[i].text.len() != 0 {
                        req.push((
                            x.as_str_name(),
                            embedder::EmebeddingRequest::CLIPRequest {
                                params: CLIPParams::Text {
                                    vals: &data[i].text,
                                },
                            },
                        ))
                    }
                    if data[i].image_uri.len() != 0 {
                        req.push((
                            x.as_str_name(),
                            embedder::EmebeddingRequest::CLIPRequest {
                                params: CLIPParams::Uri {
                                    vals: &data[i].image_uri,
                                },
                            },
                        ))
                    }
                    if data[i].image.len() != 0 {
                        req.push((
                            x.as_str_name(),
                            embedder::EmebeddingRequest::CLIPRequest {
                                params: CLIPParams::UriBytes {
                                    vals: &data[i].image,
                                },
                            },
                        ))
                    }
                }
                EncodingModel::MUnknown => {
                    return Err(Status::new(
                        Code::InvalidArgument,
                        format!("unknown model: {} set at item: {}", item.model, i),
                    ))
                }
            }
        }
        Ok(req)
    }
}
#[tonic::async_trait]
impl Api for ApiServerImpl {
    async fn initialize_model(
        &self,
        request: Request<api::InitializeModelRequest>,
    ) -> Result<Response<api::InitializeModelResponse>, Status> {
        let mut req = request.into_inner();
        let m: EncodingModel;
        match EncodingModel::from_i32(req.model) {
            Some(val) => m = val,
            None => {
                return Err(Status::new(
                    Code::InvalidArgument,
                    format!("unknown model: {}", req.model),
                ));
            }
        }
        if req.num_threads == 0 {
            req.num_threads = available_parallelism()?.get() as u32;
        }
        let providers: Vec<ExecutionProvider>;
        match self.to_execution_providers(req.devices) {
            Ok(p) => providers = p,
            Err(err) => {
                return Err(Status::new(
                    Code::InvalidArgument,
                    format!("unable to match execution_provider, reason: {}", err),
                ));
            }
        }
        let mgr = self.mgr.clone();
        let t = task::spawn_blocking(move || {
            mgr.initialize_model(m.as_str_name(), req.num_threads, providers)
        });
        let res;
        match t.await {
            Ok(r) => {
                res = r;
            }
            Err(err) => {
                return Err(Status::new(
                    Code::Internal,
                    format!("unable to match execution_provider, reason: {}", err),
                ));
            }
        }
        match res {
            Ok(()) => {
                let resp = api::InitializeModelResponse {};
                return Ok(Response::new(resp));
            }
            Err(err) => {
                return Err(Status::new(
                    Code::Internal,
                    format!("unable to create the embedder, reason: {}", err),
                ))
            }
        }
    }

    async fn encode(
        &self,
        request: Request<EncodeRequest>,
    ) -> Result<Response<EncodeResponse>, Status> {
        let data: Vec<EncodeItem> = request.into_inner().data;
        let req_pairs: Vec<(&str, embedder::EmebeddingRequest)> =
            self.transform_encode_req(&data)?;
        let mut encoding_results: Vec<EncodeResult> = Vec::with_capacity(req_pairs.len());
        req_pairs.iter().for_each(|(model_name, embedding_req)| {
            match self.mgr.encode(model_name, embedding_req) {
                Ok(results) => {
                    results.iter().for_each(|result| {
                        encoding_results.push(EncodeResult {
                            err_message: "".to_string(),
                            embedding: result.clone(),
                        });
                    });
                }
                Err(err) => encoding_results.push(EncodeResult {
                    err_message: format!("err while encoding message: {}", err),
                    embedding: Vec::new(),
                }),
            }
        });
        let reply = api::EncodeResponse {
            results: encoding_results,
        };
        Ok(Response::new(reply))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let subscriber = tracing_subscriber::fmt()
        .with_max_level(Level::INFO)
        .finish();
    match tracing::subscriber::set_global_default(subscriber) {
        Ok(()) => {}
        Err(err) => {
            panic!("unable to create the tracing subscriber: {}", err)
        }
    }

    let reflection_server = tonic_reflection::server::Builder::configure()
        .register_encoded_file_descriptor_set(API_DESCRIPTOR_SET)
        .build()?;

    let addr = "[::]:50051".parse()?;
    let model_dir = PathBuf::from("./cache");
    let apiserver;
    match ApiServerImpl::new(&model_dir) {
        Ok(server) => {
            apiserver = server;
        }
        Err(err) => panic!("unable to create the apiserver: reason: {}", err),
    }
    // let apiserver = ApiServerImpl::new(&model_dir);

    Server::builder()
        .add_service(ApiServer::new(apiserver))
        .add_service(reflection_server)
        .serve(addr)
        .await?;

    Ok(())
}