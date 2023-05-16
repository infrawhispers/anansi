use js_sys::Array;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use wasm_bindgen::prelude::*;

use base::ann::{ANNIndex, ANNParams, EId, Node};
use base::diskannv1::{DiskANNParams, DiskANNV1Index};
use base::flat::{FlatIndex, FlatParams};
use base::metric;

#[derive(Serialize, Deserialize)]
pub struct WNode {
    pub vid: usize,
    eid: String,
    pub distance: f32,
}

#[wasm_bindgen]
pub struct IndexManager {
    indices: RwLock<HashMap<String, Arc<dyn ANNIndex<Val = f32>>>>,
}

impl IndexManager {
    fn new() -> Self {
        IndexManager {
            indices: RwLock::new(HashMap::new()),
        }
    }
    fn new_index(
        &self,
        index_name: &str,
        index_type: &str,
        metric_type: &str,
        dimensions: usize,
        max_points: usize,
    ) -> Result<(), JsError> {
        let aidx: Arc<dyn ANNIndex<Val = f32>>;
        match index_name {
            "DiskANNV1" => {
                let params = ANNParams::DiskANN {
                    params: DiskANNParams {
                        dim: dimensions,
                        max_points: max_points,
                        indexing_threads: Some(1),
                        indexing_range: 64,
                        indexing_queue_size: 100,
                        indexing_maxc: 140,
                        indexing_alpha: 1.2,
                        maintenance_period_millis: 500,
                    },
                };
                match metric_type {
                    "MetricL2" => match DiskANNV1Index::new(&params) {
                        Ok(res) => {
                            let idx: DiskANNV1Index<base::metric::MetricL2, f32> = res;
                            aidx = Arc::new(idx);
                        }
                        Err(err) => {
                            return Err(JsError::new(
                                &format!("unable to create the index: {}", err).to_string(),
                            ))
                        }
                    },
                    "MetricL1" => match DiskANNV1Index::new(&params) {
                        Ok(res) => {
                            let idx: DiskANNV1Index<base::metric::MetricL1, f32> = res;
                            aidx = Arc::new(idx);
                        }
                        Err(err) => {
                            return Err(JsError::new(
                                &format!("unable to create the index: {}", err).to_string(),
                            ))
                        }
                    },
                    "MetricCosine" => match DiskANNV1Index::new(&params) {
                        Ok(res) => {
                            let idx: DiskANNV1Index<base::metric::MetricCosine, f32> = res;
                            aidx = Arc::new(idx);
                        }
                        Err(err) => {
                            return Err(JsError::new(
                                &format!("unable to create the index: {}", err).to_string(),
                            ))
                        }
                    },
                    &_ => todo!(),
                }
            }
            "Flat" => {
                let params = ANNParams::Flat {
                    params: FlatParams {
                        dim: dimensions,
                        segment_size_kb: 512,
                    },
                };
                match metric_type {
                    "MetricL2" => match FlatIndex::new(&params) {
                        Ok(res) => {
                            let idx: FlatIndex<base::metric::MetricL2, f32> = res;
                            aidx = Arc::new(idx);
                        }
                        Err(err) => {
                            return Err(JsError::new(
                                &format!("unable to create the index: {}", err).to_string(),
                            ))
                        }
                    },
                    "MetricL1" => match FlatIndex::new(&params) {
                        Ok(res) => {
                            let idx: FlatIndex<base::metric::MetricL1, f32> = res;
                            aidx = Arc::new(idx);
                        }
                        Err(err) => {
                            return Err(JsError::new(
                                &format!("unable to create the index: {}", err).to_string(),
                            ))
                        }
                    },
                    "MetricCosine" => {
                        let idx: FlatIndex<base::metric::MetricCosine, f32>;
                        match FlatIndex::new(&params) {
                            Ok(res) => {
                                let idx: FlatIndex<base::metric::MetricCosine, f32> = res;
                                aidx = Arc::new(idx);
                            }
                            Err(err) => {
                                return Err(JsError::new(
                                    &format!("unable to create the index: {}", err).to_string(),
                                ))
                            }
                        }
                    }
                    &_ => todo!(),
                }
            }
            &_ => todo!(),
        }
        let mut idx_mapping = self.indices.write();
        idx_mapping.insert(index_name.to_string(), aidx);
        Ok(())
    }
}

#[wasm_bindgen]
pub struct Index {
    index: DiskANNV1Index<base::metric::MetricCosine, f32>,
}

#[wasm_bindgen]
impl Index {
    #[wasm_bindgen(constructor)]
    pub fn new(dims: usize, max_points: usize) -> Index {
        let params = ANNParams::DiskANN {
            params: DiskANNParams {
                dim: dims,
                max_points: max_points,
                indexing_threads: Some(1),
                indexing_range: 64,
                indexing_queue_size: 100,
                indexing_maxc: 140,
                indexing_alpha: 1.2,
                maintenance_period_millis: 500,
            },
        };
        Index {
            index: DiskANNV1Index::new(&params).unwrap(),
        }
    }

    fn array_to_eids(&self, arr: js_sys::Array) -> Result<Vec<EId>, JsError> {
        let mut eids_internal: Vec<EId> = Vec::with_capacity(arr.length().try_into().unwrap());
        for idx in 0..arr.length() {
            let el = arr.get(idx).as_string().ok_or(JsError::new(
                &format!("unable to convert value at idx: {} to string", idx).to_string(),
            ))?;
            let mut eid: base::ann::EId = [0u8; 16];
            let el_bytes = el.as_bytes();
            let num_bytes = if el_bytes.len() > 16 {
                16
            } else {
                el_bytes.len()
            };
            eid[0..num_bytes].copy_from_slice(&el_bytes[0..num_bytes]);
            eids_internal.push(eid);
        }
        Ok(eids_internal)
    }

    fn nodes_to_js(&self, result: Vec<Node>) -> Result<Array, JsError> {
        let res = js_sys::Array::new();
        for idx in 0..result.len() {
            let nn = &result[idx];
            // massage it into the typical JS string, trimming the null characters
            // to hide the [0u8; 16] from the user!
            let w_nn = WNode {
                vid: nn.vid,
                eid: String::from_utf8_lossy(&nn.eid)
                    .trim_matches(char::from(0))
                    .to_string(),
                distance: nn.distance,
            };
            match serde_wasm_bindgen::to_value(&w_nn) {
                Ok(val) => {
                    res.push(&val);
                }
                Err(err) => return Err(err.into()),
            }
        }
        Ok(res)
    }

    pub fn search(&self, q: &[f32], k: usize) -> Result<Array, JsError> {
        match self.index.search(base::ann::Points::Values { vals: q }, k) {
            Ok(nns) => {
                return self.nodes_to_js(nns);
            }
            Err(err) => {
                return Err(JsError::new(
                    &format!("unable to issue search: {}", err).to_string(),
                ))
            }
        }
    }

    pub fn delete(&self, eids: js_sys::Array) -> Result<(), JsError> {
        let eids_internal = self.array_to_eids(eids)?;
        match self.index.delete(&eids_internal) {
            Ok(()) => return Ok(()),
            Err(err) => {
                return Err(JsError::new(
                    &format!("unable to issue delete: {}", err).to_string(),
                ))
            }
        }
    }

    pub fn insert(&self, eids: js_sys::Array, data: &[f32]) -> Result<(), JsError> {
        let eids_internal = self.array_to_eids(eids)?;
        // console_log!("[anansi-core] rust: running the insertion");
        match self
            .index
            .insert(&eids_internal, base::ann::Points::Values { vals: data })
        {
            Ok(()) => return Ok(()),
            Err(err) => {
                // console_log!("{}", err);
                return Err(JsError::new(
                    &format!("unable to insert vector: {}", err).to_string(),
                ));
            }
        }
    }
}
