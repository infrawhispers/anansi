use core::arch::wasm32::*;
use wasm_bindgen_test::*;

use crate::metric::{l1_similarity, l2_similarity};

wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

#[cfg(all(target_feature = "simd128",))]
pub(crate) unsafe fn l2_similarity_wasm(arr_a: &[f32], arr_b: &[f32]) -> f32 {
    let n = arr_a.len();
    let m: isize = (n).try_into().unwrap();
    let mut sum1: core::arch::wasm::v128 = f32x4_splat(0.0f32);
    let mut sum2: core::arch::wasm::v128 = f32x4_splat(0.0f32);
    let mut sum3: core::arch::wasm::v128 = f32x4_splat(0.0f32);
    let mut sum4: core::arch::wasm::v128 = f32x4_splat(0.0f32);
    let mut ptr_a: *const v128 = arr_a.as_ptr() as *const v128;
    let mut ptr_b: *const v128 = arr_b.as_ptr() as *const v128;
    let mut i: isize = 0;
    while i < m {
        let temp1: core::arch::wasm::v128 =
            core::arch::wasm::f32x4_sub(v128_load(ptr_a), v128_load(ptr_b));
        let temp2: core::arch::wasm::v128 =
            core::arch::wasm::f32x4_sub(v128_load(ptr_a.offset(1)), v128_load(ptr_b.offset(1)));
        let temp3: core::arch::wasm::v128 =
            core::arch::wasm::f32x4_sub(v128_load(ptr_a.offset(2)), v128_load(ptr_b.offset(2)));
        let temp4: core::arch::wasm::v128 =
            core::arch::wasm::f32x4_sub(v128_load(ptr_a.offset(3)), v128_load(ptr_b.offset(3)));

        sum1 = f32x4_add(f32x4_mul(temp1, temp1), sum1);
        sum2 = f32x4_add(f32x4_mul(temp2, temp2), sum2);
        sum3 = f32x4_add(f32x4_mul(temp3, temp3), sum3);
        sum4 = f32x4_add(f32x4_mul(temp4, temp4), sum4);

        ptr_a = ptr_a.offset(4);
        ptr_b = ptr_b.offset(4);
        i += 16
    }
    // TOOD(infrawhispers) - is there a better way to do this in wasm?
    // https://doc.rust-lang.org/beta/core/arch/wasm32/fn.f32x4_extract_lane.html is the only thing I could find
    let lane_sum = f32x4_add(f32x4_add(sum1, sum2), f32x4_add(sum3, sum4));
    let result = f32x4_extract_lane::<0>(lane_sum)
        + f32x4_extract_lane::<1>(lane_sum)
        + f32x4_extract_lane::<2>(lane_sum)
        + f32x4_extract_lane::<3>(lane_sum);
    result
}

pub(crate) unsafe fn l1_similarity_wasm(arr_a: &[f32], arr_b: &[f32]) -> f32 {
    let n = arr_a.len();
    let m: isize = (n).try_into().unwrap();
    let mut sum1: core::arch::wasm::v128 = f32x4_splat(0.0f32);
    let mut sum2: core::arch::wasm::v128 = f32x4_splat(0.0f32);
    let mut sum3: core::arch::wasm::v128 = f32x4_splat(0.0f32);
    let mut sum4: core::arch::wasm::v128 = f32x4_splat(0.0f32);
    let mut ptr_a: *const v128 = arr_a.as_ptr() as *const v128;
    let mut ptr_b: *const v128 = arr_b.as_ptr() as *const v128;
    let mut i: isize = 0;

    while i < m {
        let temp1: core::arch::wasm::v128 =
            core::arch::wasm::f32x4_sub(v128_load(ptr_a), v128_load(ptr_b));
        let temp2: core::arch::wasm::v128 =
            core::arch::wasm::f32x4_sub(v128_load(ptr_a.offset(1)), v128_load(ptr_b.offset(1)));
        let temp3: core::arch::wasm::v128 =
            core::arch::wasm::f32x4_sub(v128_load(ptr_a.offset(2)), v128_load(ptr_b.offset(2)));
        let temp4: core::arch::wasm::v128 =
            core::arch::wasm::f32x4_sub(v128_load(ptr_a.offset(3)), v128_load(ptr_b.offset(3)));

        sum1 = f32x4_add(f32x4_abs(temp1), sum1);
        sum2 = f32x4_add(f32x4_abs(temp2), sum2);
        sum3 = f32x4_add(f32x4_abs(temp3), sum3);
        sum4 = f32x4_add(f32x4_abs(temp4), sum4);

        ptr_a = ptr_a.offset(4);
        ptr_b = ptr_b.offset(4);
        i += 16
    }
    // TOOD(infrawhispers) - is there a better way to do this in wasm?
    // https://doc.rust-lang.org/beta/core/arch/wasm32/fn.f32x4_extract_lane.html is the only thing I could find
    let lane_sum = f32x4_add(f32x4_add(sum1, sum2), f32x4_add(sum3, sum4));
    let result = f32x4_extract_lane::<0>(lane_sum)
        + f32x4_extract_lane::<1>(lane_sum)
        + f32x4_extract_lane::<2>(lane_sum)
        + f32x4_extract_lane::<3>(lane_sum);
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    #[wasm_bindgen_test]
    fn test_wasm() {
        #[cfg(all(target_feature = "simd128",))]
        let v1: Vec<f32> = vec![
            10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24., 25.,
        ];
        let v2: Vec<f32> = vec![
            40., 41., 42., 43., 44., 45., 46., 47., 48., 49., 50., 51., 52., 53., 54., 55.,
        ];
        let l2 = l2_similarity(&v1, &v2);
        let l2_simd = unsafe { l2_similarity_wasm(&v1, &v2) };
        assert_eq!(l2, l2_simd);
        let l1 = l1_similarity(&v1, &v2);
        let l1_simd = unsafe { l1_similarity_wasm(&v1, &v2) };
        assert_eq!(l1, l1_simd);
    }
}
