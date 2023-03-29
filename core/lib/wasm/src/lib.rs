use byteorder::{BigEndian, ByteOrder};
use js_sys::Array;
use rand::distributions::{Distribution, Standard};
use rand::thread_rng;
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;
pub use wasm_bindgen_rayon::init_thread_pool;
use wasm_bindgen_test::*;
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

use base::ann::{ANNIndex, ANNParams, EId, Node};
use base::diskannv1::DiskANNParams;
use base::diskannv1::DiskANNV1Index;

#[wasm_bindgen]
extern "C" {
    // Use `js_namespace` here to bind `console.log(..)` instead of just
    // `log(..)`
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}
macro_rules! console_log {
    // Note that this is using the `log` function imported above during
    // `bare_bones`
    ($($t:tt)*) => (log(&format_args!($($t)*).to_string()))
}

#[wasm_bindgen(getter_with_clone)]
pub struct Baz {
    pub field: i32,
}

#[wasm_bindgen]
pub fn get_a_baz() -> Baz {
    Baz { field: 32 }
}

#[derive(Serialize, Deserialize)]
pub struct WNode {
    pub vid: usize,
    eid: String,
    pub distance: f32,
}

#[wasm_bindgen]
pub struct Index {
    // num_points: i32
    index: DiskANNV1Index<base::metric::MetricCosine>,
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
        match self.index.search(q, k) {
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
        console_log!("[anansi-core] rust: running the insertion");
        match self.index.insert(&eids_internal, data) {
            Ok(()) => return Ok(()),
            Err(err) => {
                console_log!("{}", err);
                return Err(JsError::new(
                    &format!("unable to insert vector: {}", err).to_string(),
                ));
            }
        }
    }
}

#[wasm_bindgen_test]
fn test_new_index() {
    let num_points = 10000;
    let _index = Index::new(512, num_points + 10);
    let eids = js_sys::Array::new();
    for i in 0..num_points {
        eids.push(&js_sys::JsString::from_char_code1(i.try_into().unwrap()));
    }

    let mut rng = thread_rng();
    let v: Vec<f32> = Standard
        .sample_iter(&mut rng)
        .take(512 * num_points)
        .collect();
    match _index.insert(eids, &v) {
        Ok(()) => {}
        Err(err) => assert_eq!(1, 2),
    }
}
