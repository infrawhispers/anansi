use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

use tdigest::TDigest;

#[derive(Serialize, Deserialize, Debug)]
struct QuantizerSettings {
    pub updated: bool,
    pub offset: f32,
    pub alpha: f32,
    pub tdigest: TDigest,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ScalarQuantizer {
    quantile: f32,
    settings: Arc<RwLock<QuantizerSettings>>,
    pre_compute_by_vid: Arc<RwLock<HashMap<usize, f32>>>,
}

impl ScalarQuantizer {
    fn gen_quantize_params(&self, arr_a: &[f32]) -> (f32, f32) {
        let mut vals = Vec::with_capacity(arr_a.len());
        arr_a.iter().for_each(|x| {
            vals.push_within_capacity(*x as f64);
        });

        let mut settings_w = self.settings.write();
        let t = settings_w.tdigest.merge_unsorted(vals);
        let p9x = t.estimate_quantile(self.quantile as f64);
        let p0 = t.estimate_quantile(0.0f64);
        let offset = p0 as f32;
        let alpha = ((p9x - p0) / 255.0f64) as f32;

        settings_w.offset = offset;
        settings_w.alpha = alpha;
        settings_w.tdigest = t;
        settings_w.updated = true;
        (offset, alpha)
    }

    pub fn quantize_arr(&self, arr_a: &[f32]) -> (Vec<u8>, f32) {
        let offset: f32;
        let alpha: f32;
        {
            let settings_r = self.settings.read();
            offset = settings_r.offset;
            alpha = settings_r.alpha;
        }
        let result: Vec<u8> = arr_a.iter().map(|x| ((x - offset) / alpha) as u8).collect();
        let sum: f32 = arr_a[..].iter().sum();
        return (result, sum * offset * alpha);
    }

    pub fn quantize(&self, vids: &[usize], arr_a: &[f32], requantize: Option<bool>) -> Vec<u8> {
        let offset: f32;
        let alpha: f32;
        let gen_params: bool;
        match requantize {
            Some(x) => gen_params = x,
            None => gen_params = !self.settings.read().updated,
        }
        if gen_params {
            (offset, alpha) = self.gen_quantize_params(arr_a);
        } else {
            let settings_r = self.settings.read();
            offset = settings_r.offset;
            alpha = settings_r.alpha;
        }

        // now go through the vectors and re-calculate them!
        // TODO(infrawhipsers) - this could be done using SIMD?
        let result: Vec<u8> = arr_a.iter().map(|x| ((x - offset) / alpha) as u8).collect();
        let mut precompute: Vec<f32> = Vec::with_capacity(vids.len());
        let dims = arr_a.len() / vids.len();
        for i in 0..vids.len() {
            let sum: f32 = arr_a[i * dims..i * dims + dims].iter().sum();
            precompute.push(sum * offset * alpha)
        }
        let mut mappings = self.pre_compute_by_vid.write();
        precompute
            .iter()
            .copied()
            .zip(vids.iter().copied())
            .for_each(|(res, vid)| {
                mappings.insert(vid, res);
            });

        return result;
    }

    pub fn new(quantile: f32) -> anyhow::Result<ScalarQuantizer> {
        return Ok(ScalarQuantizer {
            quantile: quantile,
            settings: Arc::new(RwLock::new(QuantizerSettings {
                offset: 0.0f32,
                alpha: 0.0f32,
                tdigest: TDigest::new_with_size(100),
                updated: false,
            })),
            pre_compute_by_vid: Arc::new(RwLock::new(HashMap::new())),
        });
    }
}

#[allow(unused_imports)]
mod tests {
    use super::*;
    use rand::distributions::{Distribution, Uniform};

    #[test]
    fn test_quantizer() {
        let obj = ScalarQuantizer::new(0.99).expect("unable to create the quantizer");
        let dim = 512;
        let num_vectors = 100;
        let mut rng = rand::thread_rng();
        let arr: Vec<f32> = Uniform::from(1..1_000_000)
            .sample_iter(&mut rng)
            .take(dim * num_vectors)
            .map(|x| x as f32)
            .collect();
        let vids: Vec<usize> = (0..num_vectors).collect();
        assert_eq!(arr.len(), obj.quantize(&vids, &arr, None).len())
        // TODO(infrawhispers) - verify that we can reconstruct the vectors!
    }
}
