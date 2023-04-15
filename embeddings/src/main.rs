use std::path::PathBuf;
use std::sync::Arc;
use std::thread::available_parallelism;

use anyhow::{anyhow, bail};
use clap::{command, Parser};
use clap_port_flag::Port;
use ort::ExecutionProvider;
use tokio::signal::unix::SignalKind;
use tokio::{signal, task};
use tokio_stream::wrappers::TcpListenerStream;
use tokio_util::sync::CancellationToken;
use tonic::{transport::Server, Code, Request, Response, Status};
use tracing::{info, warn, Level};

use embeddings::api::api_server::{Api, ApiServer};
use embeddings::api::{
    EncodeItem, EncodeRequest, EncodeResponse, EncodeResult, EncodingModel, EncodingModelDevice,
    InitializeModelRequest, InitializeModelResponse, ModelInitResult, ModelSettings,
};
use embeddings::app_config;
use embeddings::embedder;
use embeddings::embedder::{CLIPParams, InstructorParams};
use embeddings::embedder_manager;
use embeddings::API_DESCRIPTOR_SET;

#[derive(Debug, Parser)]
#[clap(name = "embedder-managed", version = "0.1.0", author = "getanansi")]
#[command(author, version, about, long_about = None)]
pub struct BinaryArgs {
    #[clap(flatten)]
    port: Port,
    /// configuration for the embedding models to be loaded on startup.
    #[clap(long, short = 'c', default_value = "config.yml")]
    config: PathBuf,
    /// folder in which embedding models will be downloaded and cached.
    #[clap(long, short = 'f', default_value = ".cache")]
    model_folder: PathBuf,
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
    async fn init_model(&self, m: &ModelSettings) -> anyhow::Result<()> {
        let model_name;
        match EncodingModel::from_i32(m.model_name) {
            Some(val) => model_name = val,
            None => {
                bail!("unknown model: {}", m.model_name)
            }
        }
        let num_threads: u32;
        if m.num_threads == 0 {
            num_threads = available_parallelism()?.get() as u32;
        } else {
            num_threads = m.num_threads
        }

        // let providers: Vec<ExecutionProvider>;
        // match self.to_execution_providers(req.devices) {
        //     Ok(p) => providers = p,
        //     Err(err) => {
        //         return Err(Status::new(
        //             Code::InvalidArgument,
        //             format!("unable to match execution_provider, reason: {}", err),
        //         ));
        //     }
        // }

        let mgr = self.mgr.clone();
        let t = task::spawn_blocking(move || {
            mgr.initialize_model(model_name.as_str_name(), num_threads, Vec::new())
        });
        // let res;
        match t.await {
            Ok(r) => match r {
                Ok(()) => return Ok(()),
                Err(err) => {
                    bail!("err while attemping to initialize model: {}", err)
                }
            },
            Err(err) => {
                bail!("err while attemping to initialize model: {}", err)
            }
        }
    }

    #[allow(dead_code)]
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
                | EncodingModel::MClipVitL14336Openai
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
        request: Request<InitializeModelRequest>,
    ) -> Result<Response<InitializeModelResponse>, Status> {
        let req = request.into_inner();
        let mut results: Vec<ModelInitResult> = Vec::new();
        for idx in 0..req.models.len() {
            let m = &req.models[idx];
            match self.init_model(m).await {
                Ok(()) => results.push(ModelInitResult {
                    err_message: "".to_string(),
                    initialized: true,
                    model_name: m.model_name,
                }),
                Err(err) => results.push(ModelInitResult {
                    err_message: format!("unable to init: {}", err),
                    initialized: false,
                    model_name: m.model_name,
                }),
            }
        }
        let resp = InitializeModelResponse { results: results };
        return Ok(Response::new(resp));
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
        let reply = EncodeResponse {
            results: encoding_results,
        };
        Ok(Response::new(reply))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = BinaryArgs::parse();

    // STEP 1 - parse the model configs, iff a config.yaml was
    // not specified then we default to M_CLIP_VIT_L_14_336_OPENAI
    let model_configs: Vec<ModelSettings>;
    match app_config::fetch_initial_models(&args.config) {
        Ok(c) => model_configs = c,
        Err(err) => {
            panic!("unable to create the model_configs: {}", err);
        }
    }
    if model_configs.len() == 0 {
        panic!("at least 1 model should be specified, please check your config")
    }
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
    // STEP 2 - attach to the specified hostname:port
    let listener = args.port.bind_or(50051)?;
    listener.set_nonblocking(true)?;
    let listener = tokio::net::TcpListener::from_std(listener)?;
    info!("listening at {:?}", listener.local_addr()?);

    // STEP 3 - ensure that our .cache folder for the models exists and create
    // it if necessary - warn that we may not be able to load *new* models if
    // the folder is not writable by us.
    std::fs::create_dir_all(args.model_folder.clone())?;
    let md = std::fs::metadata(args.model_folder.clone())?;
    let permissions = md.permissions();
    if permissions.readonly() {
        warn!(
            "folder: {:?} is readonly we will not be able to load *new* models",
            args.model_folder
        )
    }
    let apiserver;
    match ApiServerImpl::new(&args.model_folder) {
        Ok(server) => {
            apiserver = server;
        }
        Err(err) => panic!("unable to create the apiserver: {}", err),
    }
    // STEP 4 - initialize the models that should be preloaded on startup.
    for idx in 0..model_configs.len() {
        let cfg = &model_configs[idx];
        let model_name = EncodingModel::from_i32(cfg.model_name)
            .ok_or(anyhow!("model \"{}\" is not a valid enum", cfg.model_name))?
            .as_str_name();
        info!(model = model_name, "initializing model before startup");
        match apiserver.init_model(&cfg).await {
            Ok(()) => {
                info!(
                    model = model_name,
                    "successfully initialized model at startup",
                );
            }
            Err(err) => {
                panic!(
                    "could not intialize model: {} | err: {}",
                    cfg.model_name, err
                );
            }
        }
    }
    // Server::builder()
    //     .add_service(ApiServer::new(apiserver))
    //     .add_service(reflection_server)
    //     .serve_with_incoming(TcpListenerStream::new(listener))
    //     .await?;
    let token = CancellationToken::new();
    let cloned_token_1 = token.clone();
    let cloned_token_2 = token.clone();
    let grpc_server = Server::builder()
        .add_service(ApiServer::new(apiserver))
        .add_service(reflection_server)
        .serve_with_incoming_shutdown(TcpListenerStream::new(listener), async {
            let _ = cloned_token_1.cancelled().await;
            info!("gracefully shutting down server");
        });
    let mut terminate_await = tokio::signal::unix::signal(SignalKind::terminate()).unwrap();
    tokio::spawn(async move {
        tokio::select! {
            _ = cloned_token_2.cancelled() => {}
            _ = signal::ctrl_c() => {
                info!("recieved SIGINT..exiting");
                cloned_token_2.cancel();
            }
            _ =terminate_await.recv() => {
                info!("recieved SIGTERM..exiting");
                cloned_token_2.cancel();
            }
        }
    });
    // task2_handle.await;
    grpc_server.await?;
    info!("sever shutdown - exiting");
    // task1_handle.await?;
    // let addr = "[::]:50051".parse()?;
    // also replace serve_with_incoming(..) with serve()
    Ok(())
}
