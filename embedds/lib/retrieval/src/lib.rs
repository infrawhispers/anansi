#![feature(stdsimd)]
#![feature(is_sorted)]
#![feature(vec_push_within_capacity)]
#![cfg_attr(target_arch = "wasm32", feature(simd_wasm64))]

// #[macro_use]
// extern crate serde_derive;

pub mod ann;
mod av_store;
pub mod diskannv1;
mod errors;
pub mod flat_lite;
pub mod manager;
pub mod metric;
mod nn_query_scratch;
mod nn_queue;
pub mod scalar_quantizer;
pub mod utils;
// mod diskannv1_test;

#[cfg(feature = "full")]
mod flat_full;

#[cfg(target_arch = "wasm32")]
pub mod metric_wasm;

#[cfg(target_arch = "aarch64")]
pub mod metric_aarch;

#[cfg(target_arch = "x86_64")]
pub mod metric_avx;

pub fn add(left: usize, right: usize) -> usize {
    left + right
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
