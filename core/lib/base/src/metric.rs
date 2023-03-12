#[cfg(all(target_arch = "aarch64", target_feature = "neon",))]
use crate::metric_aarch;

#[cfg(all(target_arch = "wasm", target_feature = "simd128",))]
use crate::metric_wasm;

/*
    the core metric type that we implement for everything!
    MetricHamming
    MetricL2
    MetricL1
*/
pub trait Metric: Sync + Send {
    fn compare(arr_a: &[f32], arr_b: &[f32], length: usize) -> f32;
    fn pre_process(arr_a: &[f32]) -> Option<Vec<f32>>;
}

#[cfg(all(target_feature = "fma", target_feature = "avx",))]
#[inline(always)]
unsafe fn _mm256_reduce_add_ps(x: core::arch::x86_64::__m256) -> f32 {
    // this is fine since AVX is a superset of SSE - meaning we are guaranted
    // to have the SSE instructions available to us
    let x128: core::arch::x86_64::__m128 = core::arch::x86_64::_mm_add_ps(
        core::arch::x86_64::_mm256_extractf128_ps(x, 1),
        core::arch::x86_64::_mm256_castps256_ps128(x),
    );
    let x64: core::arch::x86_64::__m128 =
        core::arch::x86_64::_mm_add_ps(x128, core::arch::x86_64::_mm_movehl_ps(x128, x128));
    let x32: core::arch::x86_64::__m128 =
        core::arch::x86_64::_mm_add_ss(x64, core::arch::x86_64::_mm_shuffle_ps(x64, x64, 0x55));
    core::arch::x86_64::_mm_cvtss_f32(x32)
}

#[cfg(all(target_feature = "fma", target_feature = "avx",))]
#[inline(always)]
unsafe fn l2_similarity_avx(arr_a: &[f32], arr_b: &[f32], length: usize) -> f32 {
    let result;
    let niters = (length / 8) as isize;
    let mut sum = core::arch::x86_64::_mm256_setzero_ps();
    let ptr_a = arr_a.as_ptr() as *mut i8;
    let ptr_a_f = arr_a.as_ptr();
    let ptr_b = arr_b.as_ptr() as *mut i8;
    let ptr_b_f = arr_b.as_ptr();

    for j in 0..niters {
        if j < (niters - 1) {
            core::arch::x86_64::_mm_prefetch(
                ptr_a.offset(8 * (j + 1)),
                core::arch::x86_64::_MM_HINT_T0,
            );
            core::arch::x86_64::_mm_prefetch(
                ptr_b.offset(8 * (j + 1)),
                core::arch::x86_64::_MM_HINT_T0,
            );
        }
        let a_vec: core::arch::x86_64::__m256 =
            core::arch::x86_64::_mm256_load_ps(ptr_a_f.offset(8 * j) as *mut f32);
        let b_vec: core::arch::x86_64::__m256 =
            core::arch::x86_64::_mm256_load_ps(ptr_b_f.offset(8 * j) as *mut f32);
        let tmp_vec: core::arch::x86_64::__m256 = core::arch::x86_64::_mm256_sub_ps(a_vec, b_vec);
        sum = core::arch::x86_64::_mm256_fmadd_ps(tmp_vec, tmp_vec, sum);
    }
    result = self::_mm256_reduce_add_ps(sum);
    result
}

#[allow(unused_variables)]
pub(crate) fn l2_similarity(arr_a: &[f32], arr_b: &[f32], length: usize) -> f32 {
    return arr_a
        .iter()
        .copied()
        .zip(arr_b.iter().copied())
        .map(|(a, b)| (a - b).powi(2))
        .sum();
}

#[derive(Debug)]
pub struct MetricL2 {}
impl Metric for MetricL2 {
    #[allow(unused_variables)]
    fn pre_process(arr_a: &[f32]) -> Option<Vec<f32>> {
        None
    }

    #[inline(always)]
    #[allow(unused_variables)]
    fn compare(arr_a: &[f32], arr_b: &[f32], length: usize) -> f32 {
        #[cfg(all(target_arch = "x86_64", target_feature = "fma", target_feature = "avx",))]
        {
            if is_x86_feature_detected!("avx") && is_x86_feature_detected!("fma") {
                return unsafe { l2_similarity_avx(arr_a, arr_b, length) };
            }
        }
        #[cfg(all(target_arch = "aarch64", target_feature = "neon",))]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                return unsafe { metric_aarch::l2_similarity_aarch(arr_a, arr_b, length) };
            }
        }
        #[cfg(all(target_arch = "wasm", target_feature = "simd128",))]
        {
            return unsafe { l2_similarity_wasm(arr_a, arr_b, length) };
        }
        l2_similarity(arr_a, arr_b, length)
    }
}

#[cfg(all(target_feature = "avx",))]
#[inline(always)]
#[allow(unused_variables)]
unsafe fn l1_similarity_avx(arr_a: &[f32], arr_b: &[f32], length: usize) -> f32 {
    let result;
    let niters = (length / 8) as isize;
    let mut sum = core::arch::x86_64::_mm256_setzero_ps();
    let ptr_a = arr_a.as_ptr() as *mut i8;
    let ptr_a_f = arr_a.as_ptr();
    let ptr_b = arr_b.as_ptr() as *mut i8;
    let ptr_b_f = arr_b.as_ptr();

    for j in 0..niters {
        if j < (niters - 1) {
            core::arch::x86_64::_mm_prefetch(
                ptr_a.offset(8 * (j + 1)),
                core::arch::x86_64::_MM_HINT_T0,
            );
            core::arch::x86_64::_mm_prefetch(
                ptr_b.offset(8 * (j + 1)),
                core::arch::x86_64::_MM_HINT_T0,
            );
        }
        let a_vec: core::arch::x86_64::__m256 =
            core::arch::x86_64::_mm256_load_ps(ptr_a_f.offset(8 * j) as *mut f32);
        let b_vec: core::arch::x86_64::__m256 =
            core::arch::x86_64::_mm256_load_ps(ptr_b_f.offset(8 * j) as *mut f32);
        let tmp_vec: core::arch::x86_64::__m256 = core::arch::x86_64::_mm256_sub_ps(a_vec, b_vec);
        sum = core::arch::x86_64::_mm256_add_ps(tmp_vec, sum);
    }
    result = self::_mm256_reduce_add_ps(sum);
    result
}

#[allow(unused_variables)]
pub(crate) fn l1_similarity(arr_a: &[f32], arr_b: &[f32], length: usize) -> f32 {
    return arr_a
        .iter()
        .cloned()
        .zip(arr_b.iter().cloned())
        .map(|(a, b)| (a - b).abs())
        .sum();
}

#[derive(Debug)]
pub struct MetricL1 {}
impl Metric for MetricL1 {
    #[allow(unused_variables)]
    fn pre_process(arr_a: &[f32]) -> Option<Vec<f32>> {
        None
    }
    #[inline(always)]
    #[allow(unused_variables)]
    fn compare(arr_a: &[f32], arr_b: &[f32], length: usize) -> f32 {
        #[cfg(all(target_arch = "x86_64", target_feature = "fma", target_feature = "avx",))]
        {
            if is_x86_feature_detected!("avx") && is_x86_feature_detected!("fma") {
                return unsafe { l1_similarity_avx(arr_a, arr_b, length) };
            }
        }
        #[cfg(all(target_arch = "aarch64", target_feature = "neon",))]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                return unsafe { metric_aarch::l1_similarity_aarch(arr_a, arr_b, length) };
            }
        }
        l1_similarity(arr_a, arr_b, length)
    }
}

// #[cfg(all(target_feature = "neon",))]
// unsafe fn count_bytes(v: core::arch::aarch64::uint32x4_t) -> core::arch::aarch64::uint32x4_t {
//     // 0u8, 1u8, 1u8, 2u8
//     let low_mask: core::arch::aarch64::uint32x4_t = core::arch::aarch64::vdupq_n_u32(0u32);
//     let lo: core::arch::aarch64::uint32x4_t = core::arch::aarch64::vandq_u32(v, low_mask);
//     let hi: core::arch::aarch64::uint32x4_t =
//         core::arch::aarch64::vandq_u32(core::arch::aarch64::vshrq_n_u32(v, 16), low_mask);
// }

#[allow(unused_variables)]
fn hamming_similarity(arr_a: &[f32], arr_b: &[f32], length: usize) -> f32 {
    // TODO(lneath) - we use the raw bitwise representations to do hamming
    // the client needs to be aware of this when creating the vectors and
    // sending them out to the caller
    let res: u32 = arr_a
        .iter()
        .cloned()
        .zip(arr_b.iter().cloned())
        .map(|(a, b)| a.to_bits() & b.to_bits())
        .sum();
    res as f32
}

#[derive(Debug)]
pub struct Hamming {}
impl Metric for Hamming {
    #[allow(unused_variables)]
    fn pre_process(arr_a: &[f32]) -> Option<Vec<f32>> {
        None
    }
    #[inline(always)]
    fn compare(arr_a: &[f32], arr_b: &[f32], length: usize) -> f32 {
        hamming_similarity(arr_a, arr_b, length)
    }
}
