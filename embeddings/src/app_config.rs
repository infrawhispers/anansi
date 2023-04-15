use anyhow::bail;
use std::fs::File;
use std::io::Read;
use std::path::PathBuf;
use yaml_rust::YamlLoader;

use crate::embedder_manager::ModelConfiguration;

pub fn fetch_initial_models(config_path: &PathBuf) -> anyhow::Result<Vec<ModelConfiguration>> {
    if !config_path.exists() {
        return Ok(vec![ModelConfiguration {
            model_name: "M_CLIP_VIT_L_14_336_OPENAI".to_string(),
            num_threads: 4,
            devices: Vec::new(),
        }]);
    }
    let mut f = File::open(config_path)?;
    let mut buffer = String::new();
    f.read_to_string(&mut buffer)?;
    let docs = YamlLoader::load_from_str(&buffer)?;
    let mut res: Vec<ModelConfiguration> = Vec::new();
    for idx_doc in 0..docs.len() {
        let doc = &docs[idx_doc];
        let models;
        match doc["models"].as_vec() {
            Some(m) => models = m,
            None => {
                bail!("[config] yaml file is missing a list of models")
            }
        }
        for idx_model in 0..models.len() {
            let mut config = ModelConfiguration::new();
            match doc["models"][idx_model]["name"].as_str() {
                Some(model_name) => config.model_name = model_name.to_string(),
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
            res.push(config);
        }
    }
    Ok(res)
}
