use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::bail;
use crossbeam_channel::{unbounded, Receiver, Sender};
use ort::{environment::Environment, ExecutionProvider};
use parking_lot::RwLock;
use tracing::{error, info};

use crate::embedder;
use crate::embedder::EmebeddingRequest;
use crate::instructor;

struct ModelQueue {
    model_name: String,
    send: Sender<Box<dyn embedder::Embedder>>,
    recv: Receiver<Box<dyn embedder::Embedder>>,
}
pub struct EmbedderManager {
    models: Arc<RwLock<HashMap<String, Arc<ModelQueue>>>>,
    ort_environment: Arc<Environment>,
    model_path: PathBuf,
}
impl EmbedderManager {
    pub fn new(model_path: &PathBuf) -> anyhow::Result<Self> {
        let ort_environment = Arc::new(
            Environment::builder()
                .with_name("anansi.managed")
                .with_execution_providers([ExecutionProvider::cpu(), ExecutionProvider::cuda()])
                .build()?,
        );
        let obj = EmbedderManager {
            models: Arc::new(RwLock::new(HashMap::new())),
            model_path: model_path.clone(),
            ort_environment: ort_environment,
        };
        Ok(obj)
    }

    pub fn initialize_model(
        &self,
        model_name: &str,
        num_threads: u32,
        providers: Vec<ExecutionProvider>,
    ) -> anyhow::Result<()> {
        let model_identifier;
        match model_name.split_once('_') {
            Some((_p, s)) => model_identifier = s,
            None => {
                bail!("unable to determine model_identifier from: {}", model_name)
            }
        }
        let m: Box<dyn embedder::Embedder>;
        match model_identifier {
            "INSTRUCTOR" => {
                let mut path = self.model_path.clone();
                path.push("instructor");
                // "../model-transform/instructor_w_ctx_mask.onnx",
                let embedder: instructor::InstructorEmbedder =
                    embedder::Embedder::new(&embedder::EmbedderParams {
                        model_path: path.as_path(),
                        model_name: model_identifier,
                        num_threads: num_threads as i16,
                        providers: &providers,
                        ort_environment: self.ort_environment.clone(),
                    })?;
                m = Box::new(embedder);
            }
            &_ => {
                bail!("unknown model identifier: {}", model_name)
            }
        }
        let mut m_map = self.models.write();
        match m_map.get(model_identifier) {
            Some(mq) => {
                info!(
                    model = model_identifier,
                    "adding embedder to existing queue"
                );
                mq.send.send(m)?;
            }
            None => {
                info!(model = model_identifier, "creating new model queue");
                let (s, r) = unbounded();
                let mq = ModelQueue {
                    model_name: model_identifier.to_string(),
                    send: s,
                    recv: r,
                };
                mq.send.send(m)?;
                m_map.insert(model_identifier.to_string(), Arc::new(mq));
            }
        }
        Ok(())
    }

    fn encode_impl(
        &self,
        queue: &ModelQueue,
        req: &EmebeddingRequest,
    ) -> anyhow::Result<Vec<Vec<f32>>> {
        let embedder: Box<dyn embedder::Embedder>;
        match queue.recv.recv() {
            Ok(e) => embedder = e,
            Err(err) => {
                let msg = format!(
                    "unable to pull model: {} from the channel - channel is likely drained {}",
                    queue.model_name, err
                );
                error!("{}", msg);
                bail!(msg);
            }
        }
        let res = embedder.encode(req);
        // TOOD(infrawhispers) - do we want to add a
        // panic::catch_unwind(..) ?
        match queue.send.send(embedder) {
            Ok(()) => {}
            Err(_) => {
                error!("unable to re-add model: {} to the channel - this may lead to exhaustion of available encoders", queue.model_name);
            }
        }
        res
    }
    pub fn encode(
        &self,
        model_name: &str,
        req: &EmebeddingRequest,
    ) -> anyhow::Result<Vec<Vec<f32>>> {
        let model_identifier;
        match model_name.split_once('_') {
            Some((_p, s)) => model_identifier = s,
            None => {
                bail!("unable to determine model_identifier from: {}", model_name)
            }
        }

        let m_map = self.models.read();
        let embedder: &Arc<ModelQueue>;
        match m_map.get(model_identifier) {
            None => {
                bail!(
                    "no model {} initialized in the embedd manager",
                    model_identifier
                )
            }
            Some(res) => embedder = res,
        }
        let res = self.encode_impl(embedder, req);
        res
    }
}
