use crate::metric::{l1_similarity, l2_similarity};
use core::arch::aarch64::*;

#[cfg(all(target_feature = "neon",))]
#[inline(always)]
pub(crate) unsafe fn l2_similarity_aarch(arr_a: &[f32], arr_b: &[f32]) -> f32 {
    // 0.0
    // let niters = (length / 4) as isize;
    // let mut sum: float32x4_t = vdupq_n_f32(0.0f32);
    // let ptr_a: *const f32 = arr_a.as_ptr();
    // let ptr_b: *const f32 = arr_b.as_ptr();
    // for i in 0..niters {
    //     let vec_a: float32x4_t = vld1q_f32(ptr_a.offset(4 * i));
    //     let vec_b: float32x4_t = vld1q_f32(ptr_b.offset(4 * i));
    //     let temp: float32x4_t = vsubq_f32(vec_a, vec_b);
    //     sum = vfmaq_f32(sum, temp, temp);
    // }
    // let result = vaddvq_f32(sum);
    // result

    let n = arr_a.len();
    let m: isize = (n).try_into().unwrap();
    let mut sum1: float32x4_t = vdupq_n_f32(0.0f32);
    let mut sum2: float32x4_t = vdupq_n_f32(0.0f32);
    let mut sum3: float32x4_t = vdupq_n_f32(0.0f32);
    let mut sum4: float32x4_t = vdupq_n_f32(0.0f32);
    let mut ptr_a: *const f32 = arr_a.as_ptr();
    let mut ptr_b: *const f32 = arr_b.as_ptr();
    let mut i: isize = 0;
    while i < m {
        let temp1: float32x4_t = vsubq_f32(vld1q_f32(ptr_a), vld1q_f32(ptr_b));
        let temp2: float32x4_t = vsubq_f32(vld1q_f32(ptr_a.offset(4)), vld1q_f32(ptr_b.offset(4)));
        let temp3: float32x4_t = vsubq_f32(vld1q_f32(ptr_a.offset(8)), vld1q_f32(ptr_b.offset(8)));
        let temp4: float32x4_t =
            vsubq_f32(vld1q_f32(ptr_a.offset(12)), vld1q_f32(ptr_b.offset(12)));
        sum1 = vfmaq_f32(sum1, temp1, temp1);
        sum2 = vfmaq_f32(sum2, temp2, temp2);
        sum3 = vfmaq_f32(sum3, temp3, temp3);
        sum4 = vfmaq_f32(sum4, temp4, temp4);
        ptr_a = ptr_a.offset(16);
        ptr_b = ptr_b.offset(16);
        i += 16
    }
    let result = vaddvq_f32(sum1) + vaddvq_f32(sum2) + vaddvq_f32(sum3) + vaddvq_f32(sum4);
    result
}

#[cfg(all(target_feature = "neon",))]
#[inline(always)]
pub(crate) unsafe fn l1_similarity_aarch(arr_a: &[f32], arr_b: &[f32]) -> f32 {
    let n = arr_a.len();
    let m: isize = (n).try_into().unwrap();
    let mut sum1: float32x4_t = vdupq_n_f32(0.0f32);
    let mut sum2: float32x4_t = vdupq_n_f32(0.0f32);
    let mut sum3: float32x4_t = vdupq_n_f32(0.0f32);
    let mut sum4: float32x4_t = vdupq_n_f32(0.0f32);
    let mut ptr_a: *const f32 = arr_a.as_ptr();
    let mut ptr_b: *const f32 = arr_b.as_ptr();
    let mut i: isize = 0;
    while i < m {
        let temp1: float32x4_t = vabdq_f32(vld1q_f32(ptr_a), vld1q_f32(ptr_b));
        let temp2: float32x4_t = vabdq_f32(vld1q_f32(ptr_a.offset(4)), vld1q_f32(ptr_b.offset(4)));
        let temp3: float32x4_t = vabdq_f32(vld1q_f32(ptr_a.offset(8)), vld1q_f32(ptr_b.offset(8)));
        let temp4: float32x4_t =
            vabdq_f32(vld1q_f32(ptr_a.offset(12)), vld1q_f32(ptr_b.offset(12)));
        sum1 = vaddq_f32(temp1, sum1);
        sum2 = vaddq_f32(temp2, sum2);
        sum3 = vaddq_f32(temp3, sum3);
        sum4 = vaddq_f32(temp4, sum4);
        ptr_a = ptr_a.offset(16);
        ptr_b = ptr_b.offset(16);
        i += 16
    }
    let result = vaddvq_f32(sum1) + vaddvq_f32(sum2) + vaddvq_f32(sum3) + vaddvq_f32(sum4);
    result
}

#[cfg(all(target_feature = "neon",))]
#[inline(always)]
pub(crate) unsafe fn hamming_similarity_aarch(arr_a: &[f32], arr_b: &[f32]) -> f32 {
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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_neon() {
        #[cfg(all(target_arch = "aarch64", target_feature = "neon",))]
        if std::arch::is_aarch64_feature_detected!("neon") {
            let v1: Vec<f32> = vec![
                10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24., 25.,
            ];
            let v2: Vec<f32> = vec![
                40., 41., 42., 43., 44., 45., 46., 47., 48., 49., 50., 51., 52., 53., 54., 55.,
            ];
            let l2 = l2_similarity(&v1, &v2);
            let l2_simd = unsafe { l2_similarity_aarch(&v1, &v2) };
            assert_eq!(l2, l2_simd);

            let l1 = l1_similarity(&v1, &v2);
            let l1_simd = unsafe { l1_similarity_aarch(&v1, &v2) };
            assert_eq!(l1, l1_simd);
        }
    }
}
