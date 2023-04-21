use crate::api::{EncodingModel, ModelSettings};
use crate::embedder_manager::ModelConfiguration;
use anyhow::{anyhow, bail};
use std::fs::File;
use std::io::Read;
use std::path::PathBuf;
use tracing::warn;
use yaml_rust::YamlLoader;

pub fn fetch_initial_models(config_path: &PathBuf) -> anyhow::Result<Vec<ModelSettings>> {
    if !config_path.exists() {
        warn!(
            "config_path: {:?} does not exist, initalizing the default M_CLIP_VIT_L_14_336_OPENAI",
            config_path
        );
        return Ok(vec![ModelSettings {
            model_name: EncodingModel::from_str_name("M_CLIP_VIT_L_14_336_OPENAI")
                .ok_or(anyhow!(
                    "model \"M_CLIP_VIT_L_14_336_OPENAI\" is not a valid enum"
                ))?
                .into(),
            num_threads: 4,
            parallel_execution: true,
        }]);
    }
    let mut f = File::open(config_path)?;
    let mut buffer = String::new();
    f.read_to_string(&mut buffer)?;
    let docs = YamlLoader::load_from_str(&buffer)?;
    let mut res: Vec<ModelSettings> = Vec::new();
    for idx_doc in 0..docs.len() {
        let doc = &docs[idx_doc];
        let models;
        match doc["models"].as_vec() {
            Some(m) => models = m,
            None => {
                warn!("yaml file has no associated models with it");
                return Ok(Vec::new());
            }
        }
        for idx_model in 0..models.len() {
            let mut config = ModelSettings {
                model_name: EncodingModel::from_str_name("M_CLIP_VIT_L_14_336_OPENAI")
                    .ok_or(anyhow!(
                        "model \"M_CLIP_VIT_L_14_336_OPENAI\" is not a valid enum"
                    ))?
                    .into(),
                num_threads: 4,
                parallel_execution: true,
            };
            match doc["models"][idx_model]["name"].as_str() {
                Some(model_name) => {
                    config.model_name = EncodingModel::from_str_name(model_name)
                        .ok_or(anyhow!("model {} is not a valid enum", model_name))?
                        .into();
                }
                None => {
                    bail!(
                        "[config] model at idx: {} is missing \"model_name\"",
                        idx_model
                    )
                }
            }
            if !doc["models"][idx_model]["num_threads"].is_badvalue() {
                match doc["models"][idx_model]["num_threads"].as_i64() {
                    Some(num_threads) => {
                        config.num_threads = num_threads as u32;
                    }
                    None => {
                        bail!(
                            "[config] model at idx: {} has invalid num_threads",
                            idx_model
                        )
                    }
                }
            }
            if !doc["models"][idx_model]["parallel_execution"].is_badvalue() {
                match doc["models"][idx_model]["parallel_execution"].as_bool() {
                    Some(pe) => {
                        config.parallel_execution = pe;
                    }
                    None => {
                        bail!(
                            "[config] model at idx: {} has invalid parallel_execution",
                            idx_model
                        )
                    }
                }
            }
            res.push(config);
        }
    }
    Ok(res)
}
