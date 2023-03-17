use core::arch::x86_64::*;

#[cfg(all(target_feature = "fma", target_feature = "avx",))]
#[inline(always)]
unsafe fn _mm256_reduce_add_ps(x: __m256) -> f32 {
    // this is fine since AVX is a superset of SSE - meaning we are guaranted
    // to have the SSE instructions available to us
    let x128: __m128 = _mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
    let x64: __m128 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
    let x32: __m128 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
    _mm_cvtss_f32(x32)
}

#[cfg(all(target_feature = "fma", target_feature = "avx",))]
#[inline(always)]
pub(crate) unsafe fn l2_similarity_avx(arr_a: &[f32], arr_b: &[f32]) -> f32 {
    let result;
    let niters = (length / 8) as isize;
    let mut sum = _mm256_setzero_ps();
    let ptr_a = arr_a.as_ptr() as *mut i8;
    let ptr_a_f = arr_a.as_ptr();
    let ptr_b = arr_b.as_ptr() as *mut i8;
    let ptr_b_f = arr_b.as_ptr();

    for j in 0..niters {
        if j < (niters - 1) {
            _mm_prefetch(ptr_a.offset(8 * (j + 1)), _MM_HINT_T0);
            _mm_prefetch(ptr_b.offset(8 * (j + 1)), _MM_HINT_T0);
        }
        let a_vec: __m256 = _mm256_load_ps(ptr_a_f.offset(8 * j) as *mut f32);
        let b_vec: __m256 = _mm256_load_ps(ptr_b_f.offset(8 * j) as *mut f32);
        let tmp_vec: __m256 = _mm256_sub_ps(a_vec, b_vec);
        sum = _mm256_fmadd_ps(tmp_vec, tmp_vec, sum);
    }
    result = self::_mm256_reduce_add_ps(sum);
    result
}

#[cfg(all(target_feature = "fma", target_feature = "avx",))]
#[inline(always)]
unsafe fn l1_similarity_avx(arr_a: &[f32], arr_b: &[f32]) -> f32 {
    let result;
    let niters = (length / 8) as isize;
    let mut sum = _mm256_setzero_ps();
    let ptr_a = arr_a.as_ptr() as *mut i8;
    let ptr_a_f = arr_a.as_ptr();
    let ptr_b = arr_b.as_ptr() as *mut i8;
    let ptr_b_f = arr_b.as_ptr();

    for j in 0..niters {
        if j < (niters - 1) {
            _mm_prefetch(ptr_a.offset(8 * (j + 1)), _MM_HINT_T0);
            _mm_prefetch(ptr_b.offset(8 * (j + 1)), _MM_HINT_T0);
        }
        let a_vec: __m256 = _mm256_load_ps(ptr_a_f.offset(8 * j) as *mut f32);
        let b_vec: __m256 = _mm256_load_ps(ptr_b_f.offset(8 * j) as *mut f32);
        let tmp_vec: __m256 = _mm256_sub_ps(a_vec, b_vec);
        sum = _mm256_add_ps(tmp_vec, sum);
    }
    result = self::_mm256_reduce_add_ps(sum);
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_euclid() {
        #[cfg(all(target_arch = "avx", target_feature = "fma",))]
        if std::arch::is_x86_feature_detected!("fma") && std::arch::is_x86_feature_detected!("avx")
        {
            let v1: Vec<f32> = vec![
                10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24., 25.,
            ];
            let v2: Vec<f32> = vec![
                40., 41., 42., 43., 44., 45., 46., 47., 48., 49., 50., 51., 52., 53., 54., 55.,
            ];
            let l2 = l2_similarity(&v1, &v2);
            let l2_simd = unsafe { l2_similarity_avx(&v1, &v2) };
            assert_eq!(l2, l2_simd);

            let l1 = l1_similarity(&v1, &v2);
            let l1_simd = unsafe { l1_similarity_avx(&v1, &v2) };
            assert_eq!(l1, l1_simd);
        }
    }
}
