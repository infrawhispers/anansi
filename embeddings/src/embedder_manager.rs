use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::bail;
// use crossbeam_channel::{unbounded, Receiver, Sender};
use ort::{environment::Environment, ExecutionProvider};
use parking_lot::RwLock;
use rand::seq::SliceRandom;
use tracing::{error, info};

use crate::clip;
use crate::embedder;
use crate::embedder::EmebeddingRequest;
use crate::instructor;

// struct ModelQueue {
//     model_name: String,
//     send: Sender<Box<dyn embedder::Embedder>>,
//     recv: Receiver<Box<dyn embedder::Embedder>>,
// }

pub struct EmbedderManager {
    // TODO(infrawhispers) - could we get away with using a Vec<Box>?
    models: Arc<RwLock<HashMap<String, Vec<Arc<dyn embedder::Embedder>>>>>,
    ort_environment: Arc<Environment>,
    model_path: PathBuf,
}

impl EmbedderManager {
    pub fn new(model_path: &PathBuf) -> anyhow::Result<Self> {
        let ort_environment = Arc::new(
            Environment::builder()
                .with_name("anansi.managed")
                .with_execution_providers([ExecutionProvider::cuda(), ExecutionProvider::cpu()])
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
        let m: Arc<dyn embedder::Embedder>;
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
                m = Arc::new(embedder);
            }
            "CLIP_VIT_L_14_336_OPENAI" => {
                let mut path = self.model_path.clone();
                path.push("clip");
                path.push(model_identifier);
                let embedder: clip::CLIPEmbedder =
                    clip::CLIPEmbedder::new(&embedder::EmbedderParams {
                        model_path: path.as_path(),
                        model_name: model_identifier,
                        num_threads: num_threads as i16,
                        providers: &providers,
                        ort_environment: self.ort_environment.clone(),
                    })?;
                m = Arc::new(embedder);
            }
            &_ => {
                bail!("unknown model identifier: {}", model_identifier)
            }
        }
        let mut m_map = self.models.write();
        match m_map.get_mut(model_identifier) {
            Some(mq) => {
                info!(
                    model = model_identifier,
                    "adding embedder to existing queue"
                );
                mq.push((m));
            }
            None => {
                info!(model = model_identifier, "creating new model queue");
                m_map.insert(model_identifier.to_string(), vec![m]);
            } // Some(mq) => {
              //     info!(
              //         model = model_identifier,
              //         "adding embedder to existing queue"
              //     );
              //     mq.send.send(m)?;
              // }
              // None => {
              //     info!(model = model_identifier, "creating new model queue");
              //     let (s, r) = unbounded();
              //     let mq = ModelQueue {
              //         model_name: model_identifier.to_string(),
              //         send: s,
              //         recv: r,
              //     };
              //     mq.send.send(m)?;
              //     m_map.insert(model_identifier.to_string(), Arc::new(mq));
              // }
        }
        Ok(())
    }

    fn encode_impl(
        &self,
        // queue: &ModelQueue,
        // queue:
        embedders: &Vec<Arc<dyn embedder::Embedder>>,
        req: &EmebeddingRequest,
    ) -> anyhow::Result<Vec<Vec<f32>>> {
        let embedder: Arc<dyn embedder::Embedder>;
        if embedders.len() == 1 {
            embedder = embedders[0].clone();
        } else {
            // select one at random from the list to spread the load
            match embedders.choose(&mut rand::thread_rng()) {
                Some(choice) => embedder = choice.clone(),
                None => {
                    bail!("unexpected error selecting a random embedder")
                }
            }
        }
        embedder.encode(req)
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
        let embedder: &Vec<Arc<dyn embedder::Embedder>>;
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use std::time::Instant;
    use tracing::Level;

    #[test]
    fn test_manager() {
        let subscriber = tracing_subscriber::fmt()
            .with_max_level(Level::INFO)
            .finish();
        tracing::subscriber::set_global_default(subscriber)
            .expect("unable to create the tracing subscriber");

        let model_path = PathBuf::from("cache");
        let model_name = "M_CLIP_VIT_L_14_336_OPENAI";
        let mgr =
            Arc::new(EmbedderManager::new(&model_path).expect("unable to create the manager"));

        // a single CLIP instance of CLIP_VIT_L_14_336_OPENAI takes ~4Gb of GPU RAM.
        // on a 3090 we can fit ~5 comfortably.
        let now = Instant::now();
        for i in 0..1 {
            mgr.initialize_model(&model_name, 4, Vec::new())
                .expect(&format!(
                    "unable to initialize the {}th {} model",
                    i, &model_name
                ));
        }
        info!(
            model = model_name,
            "took: {:.2?} to initialize",
            now.elapsed()
        );
        // now make an encoding request.
        let text: Vec<String> = vec![
            "First do it".to_string(),
            "then do it right".to_string(),
            "then do it better".to_string(),
        ];
        let expected_embedds: Vec<Vec<f32>> = Vec::new();
        let req = EmebeddingRequest::CLIPRequest {
            params: embedder::CLIPParams::Text { vals: &text },
        };
        let encode_now = Instant::now();
        let num_iters = 1000;
        for _i in (0..num_iters) {
            mgr.encode(model_name, &req)
                .expect("issue during the encode process");
        }
        let time_taken = encode_now.elapsed();
        info!("took: {:.2?} to encode", time_taken);
        info!(
            "encodes per second: {:.2}",
            1000.0 * (num_iters as f32) / (time_taken.as_millis() as f32)
        );
    }
}
