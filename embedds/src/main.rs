use std::path::PathBuf;
use std::sync::Arc;
use std::thread::available_parallelism;

use anyhow::{anyhow, bail, Context};
use clap::{command, ArgAction, Parser};
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
    EncodeBatch, EncodeRequest, EncodeResponse, EncodeResult, EncodingModelDevice,
    InitializeModelRequest, InitializeModelResponse, ModelClass, ModelInitResult, ModelSettings,
};

use embeddings::app_config;
use embeddings::embedd::embedder;
use embeddings::embedd::embedder::{CLIPParams, E5Params, InstructorParams};
use embeddings::embedd::embedder_manager;
use embeddings::API_DESCRIPTOR_SET;

#[derive(Debug, Parser)]
#[clap(name = "embedder-managed", version = "0.1.0", author = "getanansi")]
#[command(author, version, about, long_about = None)]
pub struct BinaryArgs {
    #[clap(flatten)]
    port: Port,
    /// configuration for the embedding models to be loaded on startup
    #[clap(long, short = 'c', default_value = "/app/runtime/config.yaml")]
    config: PathBuf,
    /// folder in which embedding models will be downloaded and cached
    #[clap(long, short = 'f', default_value = ".cache")]
    model_folder: PathBuf,
    /// folder in which indices and metadata is persisted
    #[clap(long, short = 'i', default_value = ".data")]
    index_folder: PathBuf,
    /// allow for administrative actions: [Initialize()]
    #[clap(long, default_value_t = false, action=ArgAction::Set)]
    allow_admin: bool,
}

pub struct ApiServerImpl {
    embedder_mgr: Arc<embedder_manager::EmbedderManager>,
    index_mgr: Arc<embeddings::manager::json_manager::JSONIndexManager>,
    // accepts requests made to: [Initialize()]
    allow_admin: bool,
}

impl ApiServerImpl {
    fn new(model_path: &PathBuf, index_path: &PathBuf, allow_admin: bool) -> anyhow::Result<Self> {
        let obj = ApiServerImpl {
            index_mgr: Arc::new(embeddings::manager::json_manager::JSONIndexManager::new(
                index_path,
            )?),
            embedder_mgr: Arc::new(embedder_manager::EmbedderManager::new(model_path)?),
            allow_admin: allow_admin,
        };
        Ok(obj)
    }
}

impl ApiServerImpl {
    async fn init_model(&self, m: &ModelSettings) -> anyhow::Result<()> {
        let model_class;
        let model_name = m.model_name.clone();
        match ModelClass::from_i32(m.model_class) {
            Some(val) => model_class = val.as_str_name(),
            None => {
                bail!("unknown model_class: {}", m.model_class)
            }
        }
        let num_threads: u32;
        if m.num_threads == 0 {
            num_threads = available_parallelism()?.get() as u32;
        } else {
            num_threads = m.num_threads
        }
        let parallel_execution = m.parallel_execution;

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

        let embedder_mgr = self.embedder_mgr.clone();
        let t = task::spawn_blocking(move || {
            embedder_mgr.initialize_model(model_class, &model_name, num_threads, parallel_execution)
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
        data: &'a Vec<EncodeBatch>,
    ) -> Result<Vec<(&str, &str, embedder::EmebeddingRequest)>, Status> {
        let mut req: Vec<(&str, &str, embedder::EmebeddingRequest)> = Vec::new();
        for i in 0..data.len() {
            let bundle = &data[i];
            let content = bundle.content.as_ref().ok_or_else(|| {
                Status::new(Code::InvalidArgument, format!("content must be specified",))
            })?;
            let model_class;
            match ModelClass::from_i32(bundle.model_class) {
                Some(val) => model_class = val,
                None => {
                    return Err(Status::new(
                        Code::InvalidArgument,
                        format!(
                            "unknown model class: {} set at idx: {}",
                            bundle.model_class, i
                        ),
                    ));
                }
            }
            match model_class {
                ModelClass::Unknown => todo!(),
                ModelClass::Clip => match content {
                    embeddings::api::encode_batch::Content::Text(c) => {
                        let mut text: Vec<String> = Vec::with_capacity(c.data.len());
                        let mut instructions: Vec<String> = Vec::with_capacity(c.data.len());
                        for o in c.data.iter() {
                            instructions.push(o.instruction.to_string());
                            match &o.data {
                                Some(embeddings::api::content::Data::Bytes(_)) | &None => {
                                    return Err(Status::new(
                                        Code::InvalidArgument,
                                        format!("text requires string values"),
                                    ))
                                }
                                Some(embeddings::api::content::Data::Value(v)) => {
                                    text.push(v.to_string())
                                }
                            }
                        }
                        req.push((
                            model_class.as_str_name(),
                            &bundle.model_name,
                            embedder::EmebeddingRequest::CLIPRequest {
                                params: CLIPParams::Text { vals: text },
                            },
                        ))
                    }
                    embeddings::api::encode_batch::Content::ImageUris(c) => {
                        let mut image_uris: Vec<String> = Vec::with_capacity(c.data.len());
                        for o in c.data.iter() {
                            match &o.data {
                                Some(embeddings::api::content::Data::Bytes(_)) | &None => {
                                    return Err(Status::new(
                                        Code::InvalidArgument,
                                        format!("image_uris requires string values"),
                                    ))
                                }
                                Some(embeddings::api::content::Data::Value(v)) => {
                                    image_uris.push(v.to_string())
                                }
                            }
                        }
                        req.push((
                            model_class.as_str_name(),
                            &bundle.model_name,
                            embedder::EmebeddingRequest::CLIPRequest {
                                params: CLIPParams::Uri { vals: image_uris },
                            },
                        ))
                    }
                    &embeddings::api::encode_batch::Content::Images(_) => {
                        return Err(Status::new(
                            Code::InvalidArgument,
                            format!(
                            "model_class: {model_class:?} only supports {{text, image_uri}} encoding"
                        ),
                        ))
                    }
                },
                ModelClass::Instructor => match content {
                    embeddings::api::encode_batch::Content::Text(c) => {
                        let mut text: Vec<String> = Vec::with_capacity(c.data.len());
                        let mut instructions: Vec<String> = Vec::with_capacity(c.data.len());
                        for o in c.data.iter() {
                            instructions.push(o.instruction.to_string());
                            match &o.data {
                                Some(embeddings::api::content::Data::Bytes(_)) | &None => {
                                    return Err(Status::new(
                                        Code::InvalidArgument,
                                        format!("text requires string values"),
                                    ))
                                }
                                Some(embeddings::api::content::Data::Value(v)) => {
                                    text.push(v.to_string())
                                }
                            }
                        }
                        req.push((
                            model_class.as_str_name(),
                            &bundle.model_name,
                            embedder::EmebeddingRequest::InstructorRequest {
                                params: InstructorParams {
                                    text: text,
                                    instructions: instructions,
                                },
                            },
                        ))
                    }
                    &embeddings::api::encode_batch::Content::Images(_)
                    | &embeddings::api::encode_batch::Content::ImageUris(_) => {
                        return Err(Status::new(
                            Code::InvalidArgument,
                            format!("model_class: {model_class:?} only supports text encoding"),
                        ))
                    }
                },
                ModelClass::E5 => match content {
                    embeddings::api::encode_batch::Content::Text(c) => {
                        let mut text: Vec<String> = Vec::with_capacity(c.data.len());
                        for o in c.data.iter() {
                            match &o.data {
                                Some(embeddings::api::content::Data::Bytes(_)) | &None => {
                                    return Err(Status::new(
                                        Code::InvalidArgument,
                                        format!("text requires string values"),
                                    ))
                                }
                                Some(embeddings::api::content::Data::Value(v)) => {
                                    text.push(v.to_string())
                                }
                            }
                        }
                        req.push((
                            model_class.as_str_name(),
                            &bundle.model_name,
                            embedder::EmebeddingRequest::E5Request {
                                params: E5Params { text: text },
                            },
                        ));
                    }
                    &embeddings::api::encode_batch::Content::Images(_)
                    | &embeddings::api::encode_batch::Content::ImageUris(_) => {
                        return Err(Status::new(
                            Code::InvalidArgument,
                            format!("model_class: {model_class:?} only supports text encoding"),
                        ))
                    }
                },
            }
        }
        Ok(req)
    }

    fn _search_index(
        &self,
        req: &embeddings::api::SearchIndexRequest,
    ) -> anyhow::Result<embeddings::api::SearchIndexResponse> {
        let queries = self
            .index_mgr
            .search_preprocess(&req.index_name, &req.queries)
            .with_context(|| "failed to transform the search req")?;
        if req.per_search_limit > 1024 {
            // this prevents us from massive allocations for a pool of available
            // nns in the backends
            bail!(
                "per_search_limit: {0} must be lower than 1024",
                req.per_search_limit
            );
        }
        // we keep these around to hold the embeddings that are _generated_ by the
        // embedding process
        let mut isearch: Vec<embeddings::manager::json_manager::IndexSearch> = Vec::new();
        let attributes: Vec<&str> = req.attributes.iter().map(|s| s.as_ref()).collect();
        queries
            .into_iter()
            .try_for_each(|q| -> anyhow::Result<()> {
                if q.embedds.is_some() {
                    let embedds = q.embedds.ok_or_else(|| {
                        anyhow::anyhow!("unexpectedly, the to_embedd request is None")
                    })?;
                    embedds.into_iter().for_each(|e| {
                        isearch.push(embeddings::manager::json_manager::IndexSearch {
                            embedding: e,
                            attributes: &attributes,
                            weighting: &req.weighting,
                            limit: req.per_search_limit as usize,
                        });
                    });
                    return Ok(());
                }
                if q.to_embedd.is_some() {
                    let req_embedds = q.to_embedd.ok_or_else(|| {
                        anyhow::anyhow!("unexpectedly, the to_embedd request is None")
                    })?;
                    let req_vec = &vec![req_embedds.clone()];
                    let req_pairs: Vec<(&str, &str, embedder::EmebeddingRequest)> =
                        self.transform_encode_req(&req_vec)?;
                    for (model_class, model_name, embedding_req) in req_pairs.iter() {
                        self.embedder_mgr
                            .encode(model_class, model_name, embedding_req)?
                            .into_iter()
                            .for_each(|embedd| {
                                isearch.push(embeddings::manager::json_manager::IndexSearch {
                                    embedding: embedd,
                                    attributes: &attributes,
                                    weighting: &req.weighting,
                                    limit: req.per_search_limit as usize,
                                });
                            });
                    }
                }
                Ok(())
            })?;
        let mut resp = embeddings::api::SearchIndexResponse {
            response: Vec::new(),
        };
        // we now can issue the search request
        let results = isearch
            .iter()
            .map(|req_search| self.index_mgr.search(&req.index_name, req_search))
            .collect::<Vec<anyhow::Result<Vec<embeddings::manager::json_manager::NodeHit>>>>();
        // then form the results object that we send out to the client
        results.into_iter().for_each(|result| match result {
            Ok(nns) => resp.response.push(embeddings::api::SearchResponse {
                search_id: "".to_string(),
                nns: nns
                    .into_iter()
                    .map(|nn| embeddings::api::NearestNeighbor {
                        id: nn.id,
                        distance: nn.distance,
                        document: "".to_string(),
                    })
                    .collect::<Vec<embeddings::api::NearestNeighbor>>(),
                err_message: "".to_string(),
            }),
            Err(err) => resp.response.push(embeddings::api::SearchResponse {
                search_id: "".to_string(),
                nns: Vec::new(),
                err_message: err.to_string(),
            }),
        });
        Ok(resp)
    }

    fn _index_data(&self, index_name: &str, data: &str) -> anyhow::Result<()> {
        let docs = self
            .index_mgr
            .insert_preprocess(index_name, data)
            .with_context(|| "failed to extract the documents from the supplied JSON")?;
        let mut doc_embedds: Vec<embeddings::IndexItems> = Vec::new();
        for item in docs {
            if item.embedds.is_some() {
                doc_embedds.push(item.clone());
            } else if item.to_embedd.is_some() {
                let req = item
                    .to_embedd
                    .ok_or_else(|| anyhow::anyhow!("unexpectedly, the embedd request is None"))?
                    .clone();
                let req_vec = &vec![req];
                let req_pairs: Vec<(&str, &str, embedder::EmebeddingRequest)> =
                    self.transform_encode_req(&req_vec)?;
                for (model_class, model_name, embedding_req) in req_pairs.iter() {
                    match self
                        .embedder_mgr
                        .encode(model_class, model_name, embedding_req)
                    {
                        Ok(res) => doc_embedds.push(embeddings::IndexItems {
                            ids: item.ids.clone(),
                            sub_indices: item.sub_indices.clone(),
                            embedds: Some(res),
                            to_embedd: None,
                        }),
                        Err(err) => {
                            bail!("ran into trouble encoding: {err}")
                        }
                    }
                }
            } else {
                bail!("one of {{embdds, to_embedd}} must be set - this is an internal bug")
            }
        }
        // finally run the insertion process:
        self.index_mgr.insert_data(index_name, doc_embedds)
    }
}
#[tonic::async_trait]
impl Api for ApiServerImpl {
    async fn initialize_model(
        &self,
        request: Request<InitializeModelRequest>,
    ) -> Result<Response<InitializeModelResponse>, Status> {
        if !self.allow_admin {
            return Err(Status::new(
                Code::PermissionDenied,
                format!("server has been configured with allow_admin: {} Initialize() cannot be called without authorization", self.allow_admin),
            ));
        }
        let req = request.into_inner();
        let mut results: Vec<ModelInitResult> = Vec::new();
        for idx in 0..req.models.len() {
            let m = &req.models[idx];
            match self.init_model(m).await {
                Ok(()) => results.push(ModelInitResult {
                    err_message: "".to_string(),
                    initialized: true,
                    model_name: m.model_name.clone(),
                    model_class: m.model_class,
                }),
                Err(err) => results.push(ModelInitResult {
                    err_message: format!("unable to init: {}", err),
                    initialized: false,
                    model_name: m.model_name.clone(),
                    model_class: m.model_class,
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
        let data: Vec<EncodeBatch> = request.into_inner().batches;
        let req_pairs: Vec<(&str, &str, embedder::EmebeddingRequest)> =
            self.transform_encode_req(&data)?;
        let mut encoding_results: Vec<EncodeResult> = Vec::with_capacity(req_pairs.len());
        req_pairs
            .iter()
            .for_each(|(model_class, model_name, embedding_req)| {
                match self
                    .embedder_mgr
                    .encode(model_class, model_name, embedding_req)
                {
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

    async fn index_data(
        &self,
        request: Request<embeddings::api::IndexDataRequest>,
    ) -> Result<Response<embeddings::api::IndexDataResponse>, Status> {
        let req = request.into_inner();
        match self._index_data(&req.index_name, &req.data) {
            Ok(()) => {}
            Err(err) => return Err(Status::new(Code::Internal, format!("{err:?}"))),
        }
        Ok(Response::new(embeddings::api::IndexDataResponse {
            results: Vec::new(),
        }))
    }

    async fn deactivate_index(
        &self,
        _request: Request<embeddings::api::DeactivateIndexRequest>,
    ) -> Result<Response<embeddings::api::DeactivateIndexResponse>, Status> {
        return Err(Status::new(
            Code::Unimplemented,
            format!("this has not been implemented just yet"),
        ));
    }
    async fn get_indices(
        &self,
        _request: Request<embeddings::api::GetIndicesRequest>,
    ) -> Result<Response<embeddings::api::GetIndicesResponse>, Status> {
        return Err(Status::new(
            Code::Unimplemented,
            format!("this has not been implemented just yet"),
        ));
    }
    async fn delete_index(
        &self,
        request: Request<embeddings::api::DeleteIndexRequest>,
    ) -> Result<Response<embeddings::api::DeleteIndexResponse>, Status> {
        let req = request.into_inner();
        match self.index_mgr.delete_index(&req.name) {
            Ok(()) => {}
            Err(err) => {
                return Err(Status::new(
                    Code::Internal,
                    format!("unable to delete index: {err}"),
                ));
            }
        }
        Ok(Response::new(embeddings::api::DeleteIndexResponse {}))
    }
    async fn search_index(
        &self,
        request: Request<embeddings::api::SearchIndexRequest>,
    ) -> Result<Response<embeddings::api::SearchIndexResponse>, Status> {
        let req = request.into_inner();
        match self._search_index(&req) {
            Ok(response) => return Ok(Response::new(response)),
            Err(err) => return Err(Status::new(Code::Internal, format!("{err:?}"))),
        }
    }

    async fn create_index(
        &self,
        request: Request<embeddings::api::CreateIndexRequest>,
    ) -> Result<Response<embeddings::api::CreateIndexResponse>, Status> {
        let req = request.into_inner();
        let metric_type: String;
        match embeddings::api::MetricType::from_i32(req.metric_type) {
            Some(embeddings::api::MetricType::Cosine) => metric_type = "MetricCosine".to_string(),
            Some(embeddings::api::MetricType::L1) => metric_type = "MetricL1".to_string(),
            Some(embeddings::api::MetricType::L2) => metric_type = "MetricL2".to_string(),
            Some(embeddings::api::MetricType::Hamming) => metric_type = "MetricHamming".to_string(),
            Some(embeddings::api::MetricType::Unknown) => {
                return Err(Status::new(
                    Code::InvalidArgument,
                    format!("metric_type must be set"),
                ));
            }
            None => {
                return Err(Status::new(
                    Code::InvalidArgument,
                    format!("metric_type must be set"),
                ));
            }
        }
        let params: retrieval::ann::ANNParams;
        // match the params that we need
        match req.index_config {
            Some(embeddings::api::create_index_request::IndexConfig::FlatParams(p)) => {
                params = retrieval::ann::ANNParams::FlatLite {
                    params: retrieval::flat_lite::FlatLiteParams {
                        dim: p.dimensions as usize,
                        segment_size_kb: p.segment_size_kb as usize,
                    },
                }
            }
            None => {
                params = retrieval::ann::ANNParams::FlatLite {
                    params: retrieval::flat_lite::FlatLiteParams {
                        dim: 768,
                        segment_size_kb: 1024,
                    },
                }
            }
        }

        // default: VIT_L_14_336_OPENAI
        let embedding_model_name = if req.embedding_model_name == "" {
            "VIT_L_14_336_OPENAI".to_string()
        } else {
            req.embedding_model_name
        };
        let embedding_model_class;
        // default: CLIP
        match embeddings::api::ModelClass::from_i32(req.embedding_model_class) {
            Some(embeddings::api::ModelClass::Unknown) => embedding_model_class = ModelClass::Clip,
            Some(x) => embedding_model_class = x,
            None => embedding_model_class = ModelClass::Clip,
        }
        let settings = embeddings::manager::json_manager::IndexSettings {
            metric_type: metric_type,
            index_type: "Flat".to_string(),
            embedding_model_name: embedding_model_name,
            embedding_model_class: embedding_model_class,
            index_params: params,
        };
        match self
            .index_mgr
            .create_index(&req.name, &req.fields, &settings)
        {
            Ok(()) => {}
            Err(err) => {
                return Err(Status::new(
                    Code::Internal,
                    format!("unable to create the index: {err}"),
                ));
            }
        }
        Ok(Response::new(embeddings::api::CreateIndexResponse {}))
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
        panic!(
            "at least 1 model should be specified, please check your config at: {:?}",
            args.config
        );
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
    match ApiServerImpl::new(&args.model_folder, &args.index_folder, args.allow_admin) {
        Ok(server) => {
            apiserver = server;
        }
        Err(err) => panic!("unable to create the apiserver: {:?}", err),
    }
    // STEP 4 - initialize the models that should be preloaded on startup.
    for idx in 0..model_configs.len() {
        let cfg = &model_configs[idx];
        let model_class = ModelClass::from_i32(cfg.model_class)
            .ok_or(anyhow!("model \"{}\" is not a valid enum", cfg.model_class))?
            .as_str_name();
        info!(
            model = cfg.model_name.clone(),
            class = model_class,
            "initializing model before startup"
        );
        match apiserver.init_model(&cfg).await {
            Ok(()) => {
                info!(
                    model = cfg.model_name,
                    class = model_class,
                    "successfully initialized model at startup",
                );
            }
            Err(err) => {
                info!(
                    model = cfg.model_name,
                    class = model_class,
                    "unable to create the model"
                );
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
    let (mut health_reporter, health_service) = tonic_health::server::health_reporter();
    health_reporter
        .set_serving::<ApiServer<ApiServerImpl>>()
        .await;
    let token = CancellationToken::new();
    let cloned_token_1 = token.clone();
    let cloned_token_2 = token.clone();
    let grpc_server = Server::builder()
        .add_service(ApiServer::new(apiserver))
        .add_service(health_service)
        .add_service(reflection_server)
        .serve_with_incoming_shutdown(TcpListenerStream::new(listener), async {
            let _ = cloned_token_1.cancelled().await;
            info!("gracefully shutting down server");
        });
    let mut terminate_await = tokio::signal::unix::signal(SignalKind::terminate()).unwrap();
    tokio::spawn(async move {
        info!("installing signal handlers for [SIGINT, SIGTERM]");
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
    info!("serving");
    grpc_server.await?;
    info!("sever shutdown - exiting");
    Ok(())
}
