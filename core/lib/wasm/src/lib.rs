use wasm_bindgen::prelude::*;
// pub use wasm_bindgen_rayon::init_thread_pool;

use base::ann::{ANNIndex, ANNParams, EId};
use base::diskannv1::DiskANNParams;
use base::diskannv1::DiskANNV1Index;

#[wasm_bindgen]
extern "C" {
    pub fn alert(s: &str);
}

#[wasm_bindgen]
pub struct Index {
    index: DiskANNV1Index<base::metric::MetricL2>,
}

#[wasm_bindgen]
impl Index {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Index {
        let params = ANNParams::DiskANN {
            params: DiskANNParams {
                dim: 128,
                max_points: 10000,
                indexing_threads: 1,
                indexing_range: 64,
                indexing_queue_size: 100,
                indexing_maxc: 140,
                indexing_alpha: 1.2,
            },
        };
        Index {
            index: DiskANNV1Index::new(&params).unwrap(),
        }
    }
    pub fn numPoints(&self) -> u32 {
        return 5;
    }
    fn insert_internal(&self, eid: &str, data: &[f32]) -> Result<bool, Box<dyn std::error::Error>> {
        Ok(true)
    }

    pub fn insert(&self, eid: &str, data: &[f32]) -> Result<bool, JsError> {
        match self.insert_internal(eid, data) {
            Ok(x) => return Ok(x),
            _ => return Err(JsError::new("message")),
        }
    }
}

#[wasm_bindgen]
pub fn greet(name: &str) {
    alert(&format!("Hello, {}!", name));
}

#[wasm_bindgen]
pub fn new_index(name: &str) {
    let params = ANNParams::DiskANN {
        params: DiskANNParams {
            dim: 128,
            max_points: 10000,
            indexing_threads: 1,
            indexing_range: 64,
            indexing_queue_size: 100,
            indexing_maxc: 140,
            indexing_alpha: 1.2,
        },
    };
    // alert(&format!("number of points, {:?}!", params));
}
