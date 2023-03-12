use wasm_bindgen_test::*;

#[cfg(target_feature = "neon")]
use std::arch::aarch64::*;

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
unsafe fn l2_similiarity_avx(arr_a: &[f32], arr_b: &[f32], length: usize) -> f32 {
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

#[cfg(all(target_feature = "simd128",))]
#[inline(always)]
unsafe fn l2_similiarity_wasm(arr_a: &[f32], arr_b: &[f32], length: usize) -> f32 {
    let n = arr_a.len();
    let m: isize = (n).try_into().unwrap();
    let mut sum1: core::arch::wasm::v128 = core::arch::wasm32::f32x4_splat(0.0f32);
    let mut sum2: core::arch::wasm::v128 = core::arch::wasm32::f32x4_splat(0.0f32);
    let mut sum3: core::arch::wasm::v128 = core::arch::wasm32::f32x4_splat(0.0f32);
    let mut sum4: core::arch::wasm::v128 = core::arch::wasm32::f32x4_splat(0.0f32);
    let mut ptr_a: *const core::arch::wasm32::v128 =
        arr_a.as_ptr() as *const core::arch::wasm32::v128;
    let mut ptr_b: *const core::arch::wasm32::v128 =
        arr_b.as_ptr() as *const core::arch::wasm32::v128;
    let mut i: isize = 0;
    while i < m {
        let temp1: core::arch::wasm::v128 = core::arch::wasm::f32x4_sub(
            core::arch::wasm32::v128_load(ptr_a),
            core::arch::wasm32::v128_load(ptr_b),
        );
        let temp2: core::arch::wasm::v128 = core::arch::wasm::f32x4_sub(
            core::arch::wasm32::v128_load(ptr_a.offset(1)),
            core::arch::wasm32::v128_load(ptr_b.offset(1)),
        );
        let temp3: core::arch::wasm::v128 = core::arch::wasm::f32x4_sub(
            core::arch::wasm32::v128_load(ptr_a.offset(2)),
            core::arch::wasm32::v128_load(ptr_b.offset(2)),
        );
        let temp4: core::arch::wasm::v128 = core::arch::wasm::f32x4_sub(
            core::arch::wasm32::v128_load(ptr_a.offset(3)),
            core::arch::wasm32::v128_load(ptr_b.offset(3)),
        );

        let sum1: core::arch::wasm::v128 =
            core::arch::wasm32::f32x4_add(core::arch::wasm32::f32x4_mul(temp1, temp1), sum1);
        let sum2: core::arch::wasm::v128 =
            core::arch::wasm32::f32x4_add(core::arch::wasm32::f32x4_mul(temp2, temp2), sum2);
        let sum3: core::arch::wasm::v128 =
            core::arch::wasm32::f32x4_add(core::arch::wasm32::f32x4_mul(temp3, temp3), sum3);
        let sum4: core::arch::wasm::v128 =
            core::arch::wasm32::f32x4_add(core::arch::wasm32::f32x4_mul(temp4, temp4), sum4);

        ptr_a = ptr_a.offset(4);
        ptr_b = ptr_b.offset(4);
        i += 16
    }
    // TOOD(infrawhispers) - is there a better way to do this in wasm?
    // https://doc.rust-lang.org/beta/core/arch/wasm32/fn.f32x4_extract_lane.html is the only thing I could find
    let lane_sum = core::arch::wasm32::f32x4_add(
        core::arch::wasm32::f32x4_add(sum1, sum2),
        core::arch::wasm32::f32x4_add(sum3, sum4),
    );
    let result = core::arch::wasm32::f32x4_extract_lane::<0>(lane_sum)
        + core::arch::wasm32::f32x4_extract_lane::<1>(lane_sum)
        + core::arch::wasm32::f32x4_extract_lane::<2>(lane_sum)
        + core::arch::wasm32::f32x4_extract_lane::<3>(lane_sum);
    result
}

#[cfg(all(target_feature = "neon",))]
#[inline(always)]
unsafe fn l2_similiarity_aarch(arr_a: &[f32], arr_b: &[f32], length: usize) -> f32 {
    // 0.0
    // let niters = (length / 4) as isize;
    // let mut sum: core::arch::aarch64::float32x4_t = core::arch::aarch64::vdupq_n_f32(0.0f32);
    // let ptr_a: *const f32 = arr_a.as_ptr();
    // let ptr_b: *const f32 = arr_b.as_ptr();
    // for i in 0..niters {
    //     let vec_a: core::arch::aarch64::float32x4_t = core::arch::aarch64::vld1q_f32(ptr_a.offset(4 * i));
    //     let vec_b: core::arch::aarch64::float32x4_t = core::arch::aarch64::vld1q_f32(ptr_b.offset(4 * i));
    //     let temp: core::arch::aarch64::float32x4_t = core::arch::aarch64::vsubq_f32(vec_a, vec_b);
    //     sum = core::arch::aarch64::vfmaq_f32(sum, temp, temp);
    // }
    // let result = core::arch::aarch64::vaddvq_f32(sum);
    // result

    let n = arr_a.len();
    let m: isize = (n).try_into().unwrap();
    let mut sum1: core::arch::aarch64::float32x4_t = core::arch::aarch64::vdupq_n_f32(0.0f32);
    let mut sum2: core::arch::aarch64::float32x4_t = core::arch::aarch64::vdupq_n_f32(0.0f32);
    let mut sum3: core::arch::aarch64::float32x4_t = core::arch::aarch64::vdupq_n_f32(0.0f32);
    let mut sum4: core::arch::aarch64::float32x4_t = core::arch::aarch64::vdupq_n_f32(0.0f32);
    let mut ptr_a: *const f32 = arr_a.as_ptr();
    let mut ptr_b: *const f32 = arr_b.as_ptr();
    let mut i: isize = 0;
    while i < m {
        let temp1: core::arch::aarch64::float32x4_t = core::arch::aarch64::vsubq_f32(
            core::arch::aarch64::vld1q_f32(ptr_a),
            core::arch::aarch64::vld1q_f32(ptr_b),
        );
        let temp2: core::arch::aarch64::float32x4_t = core::arch::aarch64::vsubq_f32(
            core::arch::aarch64::vld1q_f32(ptr_a.offset(4)),
            core::arch::aarch64::vld1q_f32(ptr_b.offset(4)),
        );
        let temp3: core::arch::aarch64::float32x4_t = core::arch::aarch64::vsubq_f32(
            core::arch::aarch64::vld1q_f32(ptr_a.offset(8)),
            core::arch::aarch64::vld1q_f32(ptr_b.offset(8)),
        );
        let temp4: core::arch::aarch64::float32x4_t = core::arch::aarch64::vsubq_f32(
            core::arch::aarch64::vld1q_f32(ptr_a.offset(12)),
            core::arch::aarch64::vld1q_f32(ptr_b.offset(12)),
        );
        sum1 = core::arch::aarch64::vfmaq_f32(sum1, temp1, temp1);
        sum2 = core::arch::aarch64::vfmaq_f32(sum2, temp2, temp2);
        sum3 = core::arch::aarch64::vfmaq_f32(sum3, temp3, temp3);
        sum4 = core::arch::aarch64::vfmaq_f32(sum4, temp4, temp4);
        ptr_a = ptr_a.offset(16);
        ptr_b = ptr_b.offset(16);
        i += 16
    }
    let result = core::arch::aarch64::vaddvq_f32(sum1)
        + core::arch::aarch64::vaddvq_f32(sum2)
        + core::arch::aarch64::vaddvq_f32(sum3)
        + core::arch::aarch64::vaddvq_f32(sum4);
    result
}

#[allow(unused_variables)]
fn l2_similiarity(arr_a: &[f32], arr_b: &[f32], length: usize) -> f32 {
    return arr_a
        .iter()
        .copied()
        .zip(arr_b.iter().copied())
        .map(|(a, b)| (a - b).powi(2))
        .sum();
}

#[cfg(all(target_feature = "simd128",))]
#[wasm_bindgen_test]
fn test_wasm() {
    let v1: Vec<f32> = vec![
        10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24., 25.,
    ];
    let v2: Vec<f32> = vec![
        40., 41., 42., 43., 44., 45., 46., 47., 48., 49., 50., 51., 52., 53., 54., 55.,
    ];
    let l2 = l2_similiarity(&v1, &v2, v2.len());
    let l2_simd = unsafe { l2_similiarity_wasm(&v1, &v2, v2.len()) };
    assert_eq!(l2, l2_simd);
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_neon() {
        if std::arch::is_aarch64_feature_detected!("neon") {
            let v1: Vec<f32> = vec![
                10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24., 25.,
            ];
            let v2: Vec<f32> = vec![
                40., 41., 42., 43., 44., 45., 46., 47., 48., 49., 50., 51., 52., 53., 54., 55.,
            ];
            let l2 = l2_similiarity(&v1, &v2, v2.len());
            let l2_simd = unsafe { l2_similiarity_aarch(&v1, &v2, v2.len()) };
            assert_eq!(l2, l2_simd);

            let l1 = l1_similiarity(&v1, &v2, v2.len());
            let l1_simd = unsafe { l1_similiarity_aarch(&v1, &v2, v2.len()) };
            assert_eq!(l1, l1_simd);
        }
    }
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
                return unsafe { l2_similiarity_avx(arr_a, arr_b, length) };
            }
        }
        #[cfg(all(target_arch = "aarch64", target_feature = "neon",))]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                return unsafe { l2_similiarity_aarch(arr_a, arr_b, length) };
            }
        }
        #[cfg(all(target_arch = "wasm", target_feature = "simd128",))]
        {
            return unsafe { l2_similiarity_wasm(arr_a, arr_b, length) };
        }

        l2_similiarity(arr_a, arr_b, length)
    }
}

#[cfg(all(target_feature = "avx",))]
#[inline(always)]
#[allow(unused_variables)]
unsafe fn l1_similiarity_avx(arr_a: &[f32], arr_b: &[f32], length: usize) -> f32 {
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
#[cfg(all(target_feature = "neon",))]
#[inline(always)]
unsafe fn l1_similiarity_aarch(arr_a: &[f32], arr_b: &[f32], length: usize) -> f32 {
    let n = arr_a.len();
    let m: isize = (n).try_into().unwrap();
    let mut sum1: core::arch::aarch64::float32x4_t = core::arch::aarch64::vdupq_n_f32(0.0f32);
    let mut sum2: core::arch::aarch64::float32x4_t = core::arch::aarch64::vdupq_n_f32(0.0f32);
    let mut sum3: core::arch::aarch64::float32x4_t = core::arch::aarch64::vdupq_n_f32(0.0f32);
    let mut sum4: core::arch::aarch64::float32x4_t = core::arch::aarch64::vdupq_n_f32(0.0f32);
    let mut ptr_a: *const f32 = arr_a.as_ptr();
    let mut ptr_b: *const f32 = arr_b.as_ptr();
    let mut i: isize = 0;
    while i < m {
        let temp1: core::arch::aarch64::float32x4_t = core::arch::aarch64::vabdq_f32(
            core::arch::aarch64::vld1q_f32(ptr_a),
            core::arch::aarch64::vld1q_f32(ptr_b),
        );
        let temp2: core::arch::aarch64::float32x4_t = core::arch::aarch64::vabdq_f32(
            core::arch::aarch64::vld1q_f32(ptr_a.offset(4)),
            core::arch::aarch64::vld1q_f32(ptr_b.offset(4)),
        );
        let temp3: core::arch::aarch64::float32x4_t = core::arch::aarch64::vabdq_f32(
            core::arch::aarch64::vld1q_f32(ptr_a.offset(8)),
            core::arch::aarch64::vld1q_f32(ptr_b.offset(8)),
        );
        let temp4: core::arch::aarch64::float32x4_t = core::arch::aarch64::vabdq_f32(
            core::arch::aarch64::vld1q_f32(ptr_a.offset(12)),
            core::arch::aarch64::vld1q_f32(ptr_b.offset(12)),
        );
        sum1 = core::arch::aarch64::vaddq_f32(temp1, sum1);
        sum2 = core::arch::aarch64::vaddq_f32(temp2, sum2);
        sum3 = core::arch::aarch64::vaddq_f32(temp3, sum3);
        sum4 = core::arch::aarch64::vaddq_f32(temp4, sum4);
        ptr_a = ptr_a.offset(16);
        ptr_b = ptr_b.offset(16);
        i += 16
    }
    let result = core::arch::aarch64::vaddvq_f32(sum1)
        + core::arch::aarch64::vaddvq_f32(sum2)
        + core::arch::aarch64::vaddvq_f32(sum3)
        + core::arch::aarch64::vaddvq_f32(sum4);
    result
}

#[allow(unused_variables)]
fn l1_similiarity(arr_a: &[f32], arr_b: &[f32], length: usize) -> f32 {
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
                return unsafe { l1_similiarity_avx(arr_a, arr_b, length) };
            }
        }
        #[cfg(all(target_arch = "aarch64", target_feature = "neon",))]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                return unsafe { l1_similiarity_aarch(arr_a, arr_b, length) };
            }
        }
        l1_similiarity(arr_a, arr_b, length)
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

#[cfg(all(target_feature = "neon",))]
#[inline(always)]
unsafe fn hamming_similarity_aarch(arr_a: &[f32], arr_b: &[f32], length: usize) -> f32 {
    let n = arr_a.len();
    let m: isize = (n).try_into().unwrap();
    let mut sum1: core::arch::aarch64::uint32x4_t = core::arch::aarch64::vdupq_n_u32(0u32);
    let mut sum2: core::arch::aarch64::uint32x4_t = core::arch::aarch64::vdupq_n_u32(0u32);
    let mut sum3: core::arch::aarch64::uint32x4_t = core::arch::aarch64::vdupq_n_u32(0u32);
    let mut sum4: core::arch::aarch64::uint32x4_t = core::arch::aarch64::vdupq_n_u32(0u32);
    // we need an explicit cast in order to work with
    // the rust compiler!
    let mut ptr_a: *const u32 = arr_a.as_ptr() as *const u32;
    let mut ptr_b: *const u32 = arr_b.as_ptr() as *const u32;
    let mut i: isize = 0;
    while i < m {
        let temp1: core::arch::aarch64::uint32x4_t = core::arch::aarch64::vandq_u32(
            core::arch::aarch64::vld1q_u32(ptr_a),
            core::arch::aarch64::vld1q_u32(ptr_b),
        );
        let temp2: core::arch::aarch64::uint32x4_t = core::arch::aarch64::vandq_u32(
            core::arch::aarch64::vld1q_u32(ptr_a.offset(4)),
            core::arch::aarch64::vld1q_u32(ptr_b.offset(4)),
        );
        let temp3: core::arch::aarch64::uint32x4_t = core::arch::aarch64::vandq_u32(
            core::arch::aarch64::vld1q_u32(ptr_a.offset(8)),
            core::arch::aarch64::vld1q_u32(ptr_b.offset(8)),
        );
        let temp4: core::arch::aarch64::uint32x4_t = core::arch::aarch64::vandq_u32(
            core::arch::aarch64::vld1q_u32(ptr_a.offset(12)),
            core::arch::aarch64::vld1q_u32(ptr_b.offset(12)),
        );
        // sum1 = count_ones
        // sum1 = core::arch::aarch64::vaddq_f32(temp1, sum1);
        // sum2 = core::arch::aarch64::vaddq_f32(temp2, sum2);
        // sum3 = core::arch::aarch64::vaddq_f32(temp3, sum3);
        // sum4 = core::arch::aarch64::vaddq_f32(temp4, sum4);
        ptr_a = ptr_a.offset(16);
        ptr_b = ptr_b.offset(16);
        i += 16
    }
    // let result = core::arch::aarch64::vaddvq_f32(sum1)
    //     + core::arch::aarch64::vaddvq_f32(sum2)
    //     + core::arch::aarch64::vaddvq_f32(sum3)
    //     + core::arch::aarch64::vaddvq_f32(sum4);
    // result
    0.0
}

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
