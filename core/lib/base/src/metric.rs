#[cfg(all(target_arch = "aarch64", target_feature = "neon",))]
use crate::metric_aarch;

#[cfg(all(target_arch = "wasm32", target_feature = "simd128",))]
use crate::metric_wasm;

#[cfg(all(target_arch = "x86_64", target_feature = "fma", target_feature = "avx",))]
use crate::metric_avx;

/*
    the core metric type that we implement for everything!
    MetricHamming
    MetricL2
    MetricL1
*/
pub trait Metric<T>: Sync + Send {
    fn compare(arr_a: &[T], arr_b: &[T]) -> f32;
    fn pre_process(arr_a: &[T]) -> Option<Vec<T>>;
}

pub(crate) fn l2_similarity(arr_a: &[f32], arr_b: &[f32]) -> f32 {
    return arr_a
        .iter()
        .copied()
        .zip(arr_b.iter().copied())
        .map(|(a, b)| (a - b).powi(2))
        .sum();
}
// use std::marker::PhantomData;
#[derive(Debug)]
pub struct MetricL2 {}
impl Metric<f32> for MetricL2 {
    #[allow(unused_variables)]
    fn pre_process(arr_a: &[f32]) -> Option<Vec<f32>> {
        None
    }

    #[inline(always)]
    fn compare(arr_a: &[f32], arr_b: &[f32]) -> f32 {
        #[cfg(all(target_feature = "fma", target_feature = "avx",))]
        {
            if is_x86_feature_detected!("avx") && is_x86_feature_detected!("fma") {
                return unsafe { metric_avx::l2_similarity_avx(arr_a, arr_b) };
            }
        }
        #[cfg(all(target_arch = "aarch64", target_feature = "neon",))]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                return unsafe { metric_aarch::l2_similarity_aarch(arr_a, arr_b) };
            }
        }
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128",))]
        {
            return unsafe { metric_wasm::l2_similarity_wasm(arr_a, arr_b) };
        }
        l2_similarity(arr_a, arr_b)
    }
}
impl Metric<u8> for MetricL2 {
    fn pre_process(arr_a: &[u8]) -> Option<Vec<u8>> {
        None
    }
    fn compare(arr_a: &[u8], arr_b: &[u8]) -> f32 {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_l2() {
        let vec1 = vec![
            0u8, 0u8, 1u8, 2u8, 3u8, 4u8, 5u8, 6u8, 7u8, 1u8, 1u8, 2u8, 3u8, 4u8,
        ];
        let vec2 = vec![
            0u8, 0u8, 1u8, 2u8, 3u8, 4u8, 5u8, 6u8, 7u8, 1u8, 1u8, 2u8, 3u8, 4u8,
        ];
        assert_eq!(0.0, MetricL2::compare(&vec1, &vec2));
    }
}

pub(crate) fn l1_similarity(arr_a: &[f32], arr_b: &[f32]) -> f32 {
    return arr_a
        .iter()
        .cloned()
        .zip(arr_b.iter().cloned())
        .map(|(a, b)| (a - b).abs())
        .sum();
}

#[derive(Debug)]
pub struct MetricL1 {}
impl Metric<f32> for MetricL1 {
    #[allow(unused_variables)]
    fn pre_process(arr_a: &[f32]) -> Option<Vec<f32>> {
        None
    }
    #[inline(always)]
    #[allow(unused_variables)]
    fn compare(arr_a: &[f32], arr_b: &[f32]) -> f32 {
        #[cfg(all(target_arch = "x86_64", target_feature = "fma", target_feature = "avx",))]
        {
            if is_x86_feature_detected!("avx") && is_x86_feature_detected!("fma") {
                return unsafe { metric_avx::l1_similarity_avx(arr_a, arr_b) };
            }
        }
        #[cfg(all(target_arch = "aarch64", target_feature = "neon",))]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                return unsafe { metric_aarch::l1_similarity_aarch(arr_a, arr_b) };
            }
        }
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128",))]
        {
            return unsafe { metric_wasm::l1_similarity_wasm(arr_a, arr_b) };
        }
        l1_similarity(arr_a, arr_b)
    }
}

#[derive(Debug)]
pub struct Hamming {}
impl Metric<f32> for Hamming {
    #[allow(unused_variables)]
    fn pre_process(arr_a: &[f32]) -> Option<Vec<f32>> {
        None
    }
    #[inline(always)]
    fn compare(arr_a: &[f32], arr_b: &[f32]) -> f32 {
        #[cfg(all(target_arch = "aarch64", target_feature = "neon",))]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                return unsafe { metric_aarch::hamming_similarity_aarch(arr_a, arr_b) };
            }
        }

        hamming_similarity(arr_a, arr_b)
    }
}

pub(crate) fn hamming_similarity(arr_a: &[f32], arr_b: &[f32]) -> f32 {
    // TODO(infrawhispers) - we use the raw bitwise representations to do hamming
    // the client needs to be aware of this when creating the vectors and
    // sending them out to the caller
    let res: u32 = arr_a
        .iter()
        .cloned()
        .zip(arr_b.iter().cloned())
        .map(|(a, b)| (a.to_bits() ^ b.to_bits()).count_ones())
        .sum();
    res as f32
}
