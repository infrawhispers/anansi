use crate::metric::{hamming_similarity, l1_similarity, l2_similarity};
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
unsafe fn vpadalq(sum: uint64x2_t, t: uint8x16_t) -> uint64x2_t {
    return vpadalq_u32(sum, vpaddlq_u16(vpaddlq_u8(t)));
}
// this was lifted from Mula - we use SIMD instructions
// if the array is < 64 BYTES, otherwise, we use SIMD via
// sucessfull calls to vcntq_u8(...)
// https://github.com/WojciechMula/sse-popcount/blob/master/popcnt-neon.cpp
#[cfg(all(target_feature = "neon",))]
unsafe fn popcnt_neon_vvnt(data: &[u8]) -> f32 {
    let mut cnt: u64 = 0;
    let mut i: u64 = 0;
    // chunk size is in BYTES!
    let chunk_size = 64;
    if data.len() > chunk_size {
        let iters: u64 = (data.len() / chunk_size).try_into().unwrap();
        let mut ptr = data.as_ptr();
        let mut sum: uint64x2_t = vcombine_u64(vcreate_u64(0), vcreate_u64(0));
        let zero: uint8x16_t = vcombine_u8(vcreate_u8(0), vcreate_u8(0));
        while i < iters {
            let mut t0: uint8x16_t = zero;
            let mut t1: uint8x16_t = zero;
            let mut t2: uint8x16_t = zero;
            let mut t3: uint8x16_t = zero;
            /*
             * After every 31 iterations we need to add the
             * temporary sums (t0, t1, t2, t3) to the total sum.
             * We must ensure that the temporary sums <= 255
             * and 31 * 8 bits = 248 which is OK.
             */
            let limit: u64 = if i + 31 < iters { i + 31 } else { iters };
            while i < limit {
                i += 1;
                let input: uint8x16x4_t = vld4q_u8(ptr);
                ptr = ptr.add(chunk_size);
                t0 = vaddq_u8(t0, vcntq_u8(input.0));
                t1 = vaddq_u8(t1, vcntq_u8(input.1));
                t2 = vaddq_u8(t2, vcntq_u8(input.2));
                t3 = vaddq_u8(t3, vcntq_u8(input.3));
            }
            sum = vpadalq(sum, t0);
            sum = vpadalq(sum, t1);
            sum = vpadalq(sum, t2);
            sum = vpadalq(sum, t3);
        }
        let tmp: [u64; 2] = [0u64, 0u64];
        vst1q_u64(tmp.as_ptr() as *mut u64, sum);
        cnt += tmp[0];
        cnt += tmp[1];
        i *= chunk_size as u64;
    }
    for idx in (i as usize)..data.len() {
        cnt += LOOKUP8BIT[data[idx] as usize] as u64;
    }
    return cnt as f32;
}

// TODO(infrawhispers) - figure this one out!
#[cfg(all(target_feature = "neon",))]
#[inline(always)]
pub(crate) unsafe fn hamming_similarity_aarch(arr_a: &[f32], arr_b: &[f32]) -> f32 {
    let n = arr_a.len();
    let m: isize = (n).try_into().unwrap();
    let mut ptr_a: *const u32 = arr_a.as_ptr() as *const u32;
    let mut ptr_b: *const u32 = arr_b.as_ptr() as *const u32;
    let temp: Vec<u32> = vec![0u32; arr_a.len()];
    let ptr_t: *mut u32 = temp.as_ptr() as *mut u32;
    let mut i: isize = 0;
    while i < m {
        let mut res = vld1q_u32_x4(ptr_a);
        let ld_a = vld1q_u32_x4(ptr_a);
        let ld_b = vld1q_u32_x4(ptr_b);
        res.0 = veorq_u32(ld_a.0, ld_b.0);
        res.1 = veorq_u32(ld_a.1, ld_b.1);
        res.2 = veorq_u32(ld_a.2, ld_b.2);
        res.3 = veorq_u32(ld_a.3, ld_b.3);
        vst1q_u32_x4(ptr_t, res);
        ptr_a = ptr_a.offset(16);
        ptr_b = ptr_b.offset(16);
        i += 16
    }
    let ptr = temp.as_ptr() as *const u8;
    popcnt_neon_vvnt(std::slice::from_raw_parts(ptr, temp.len() * 4))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_popcnt_simd() {
        #[cfg(all(target_arch = "aarch64", target_feature = "neon",))]
        if std::arch::is_aarch64_feature_detected!("neon") {
            let mut hamming_0 = vec![
                0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32,
                0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32,
                0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32,
                0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32,
                0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32,
            ];
            let ptr = hamming_0.as_ptr() as *const u8;
            assert_eq!(0.0, unsafe {
                popcnt_neon_vvnt(std::slice::from_raw_parts(ptr, hamming_0.len() * 4))
            });
            hamming_0[10] = 1u32;
            assert_eq!(1.0, unsafe {
                popcnt_neon_vvnt(std::slice::from_raw_parts(ptr, hamming_0.len() * 4))
            });
            hamming_0[63] = 1u32;
            assert_eq!(2.0, unsafe {
                popcnt_neon_vvnt(std::slice::from_raw_parts(ptr, hamming_0.len() * 4))
            });
        }
    }
    #[test]
    fn test_hamming() {
        #[cfg(all(target_arch = "aarch64", target_feature = "neon",))]
        if std::arch::is_aarch64_feature_detected!("neon") {
            let hamming_0 = vec![
                0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32,
                0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32,
                0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32,
                0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32,
                0u32, 1u32, 1u32, 1u32, 1u32, 1u32, 1u32, 1u32,
            ];
            let hamming_0_f32 = unsafe {
                std::slice::from_raw_parts(hamming_0.as_ptr() as *const f32, hamming_0.len())
            };
            let hamming_1 = vec![
                0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32,
                0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32,
                0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32,
                0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32,
                0u32, 1u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32,
            ];
            let hamming_1_f32 = unsafe {
                std::slice::from_raw_parts(hamming_1.as_ptr() as *const f32, hamming_1.len())
            };
            let hamming_simd = unsafe { hamming_similarity_aarch(&hamming_0_f32, &hamming_1_f32) };
            let hamming = hamming_similarity(hamming_0_f32, hamming_1_f32);
            assert_eq!(hamming_simd, hamming);
        }
    }
    #[test]
    fn test_euclid() {
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

#[cfg(all(target_feature = "neon",))]
const LOOKUP8BIT: &'static [u8; 256] = &[
    /* 0 */ 0, /* 1 */ 1, /* 2 */ 1, /* 3 */ 2, /* 4 */ 1, /* 5 */ 2,
    /* 6 */ 2, /* 7 */ 3, /* 8 */ 1, /* 9 */ 2, /* a */ 2, /* b */ 3,
    /* c */ 2, /* d */ 3, /* e */ 3, /* f */ 4, /* 10 */ 1,
    /* 11 */ 2, /* 12 */ 2, /* 13 */ 3, /* 14 */ 2, /* 15 */ 3,
    /* 16 */ 3, /* 17 */ 4, /* 18 */ 2, /* 19 */ 3, /* 1a */ 3,
    /* 1b */ 4, /* 1c */ 3, /* 1d */ 4, /* 1e */ 4, /* 1f */ 5,
    /* 20 */ 1, /* 21 */ 2, /* 22 */ 2, /* 23 */ 3, /* 24 */ 2,
    /* 25 */ 3, /* 26 */ 3, /* 27 */ 4, /* 28 */ 2, /* 29 */ 3,
    /* 2a */ 3, /* 2b */ 4, /* 2c */ 3, /* 2d */ 4, /* 2e */ 4,
    /* 2f */ 5, /* 30 */ 2, /* 31 */ 3, /* 32 */ 3, /* 33 */ 4,
    /* 34 */ 3, /* 35 */ 4, /* 36 */ 4, /* 37 */ 5, /* 38 */ 3,
    /* 39 */ 4, /* 3a */ 4, /* 3b */ 5, /* 3c */ 4, /* 3d */ 5,
    /* 3e */ 5, /* 3f */ 6, /* 40 */ 1, /* 41 */ 2, /* 42 */ 2,
    /* 43 */ 3, /* 44 */ 2, /* 45 */ 3, /* 46 */ 3, /* 47 */ 4,
    /* 48 */ 2, /* 49 */ 3, /* 4a */ 3, /* 4b */ 4, /* 4c */ 3,
    /* 4d */ 4, /* 4e */ 4, /* 4f */ 5, /* 50 */ 2, /* 51 */ 3,
    /* 52 */ 3, /* 53 */ 4, /* 54 */ 3, /* 55 */ 4, /* 56 */ 4,
    /* 57 */ 5, /* 58 */ 3, /* 59 */ 4, /* 5a */ 4, /* 5b */ 5,
    /* 5c */ 4, /* 5d */ 5, /* 5e */ 5, /* 5f */ 6, /* 60 */ 2,
    /* 61 */ 3, /* 62 */ 3, /* 63 */ 4, /* 64 */ 3, /* 65 */ 4,
    /* 66 */ 4, /* 67 */ 5, /* 68 */ 3, /* 69 */ 4, /* 6a */ 4,
    /* 6b */ 5, /* 6c */ 4, /* 6d */ 5, /* 6e */ 5, /* 6f */ 6,
    /* 70 */ 3, /* 71 */ 4, /* 72 */ 4, /* 73 */ 5, /* 74 */ 4,
    /* 75 */ 5, /* 76 */ 5, /* 77 */ 6, /* 78 */ 4, /* 79 */ 5,
    /* 7a */ 5, /* 7b */ 6, /* 7c */ 5, /* 7d */ 6, /* 7e */ 6,
    /* 7f */ 7, /* 80 */ 1, /* 81 */ 2, /* 82 */ 2, /* 83 */ 3,
    /* 84 */ 2, /* 85 */ 3, /* 86 */ 3, /* 87 */ 4, /* 88 */ 2,
    /* 89 */ 3, /* 8a */ 3, /* 8b */ 4, /* 8c */ 3, /* 8d */ 4,
    /* 8e */ 4, /* 8f */ 5, /* 90 */ 2, /* 91 */ 3, /* 92 */ 3,
    /* 93 */ 4, /* 94 */ 3, /* 95 */ 4, /* 96 */ 4, /* 97 */ 5,
    /* 98 */ 3, /* 99 */ 4, /* 9a */ 4, /* 9b */ 5, /* 9c */ 4,
    /* 9d */ 5, /* 9e */ 5, /* 9f */ 6, /* a0 */ 2, /* a1 */ 3,
    /* a2 */ 3, /* a3 */ 4, /* a4 */ 3, /* a5 */ 4, /* a6 */ 4,
    /* a7 */ 5, /* a8 */ 3, /* a9 */ 4, /* aa */ 4, /* ab */ 5,
    /* ac */ 4, /* ad */ 5, /* ae */ 5, /* af */ 6, /* b0 */ 3,
    /* b1 */ 4, /* b2 */ 4, /* b3 */ 5, /* b4 */ 4, /* b5 */ 5,
    /* b6 */ 5, /* b7 */ 6, /* b8 */ 4, /* b9 */ 5, /* ba */ 5,
    /* bb */ 6, /* bc */ 5, /* bd */ 6, /* be */ 6, /* bf */ 7,
    /* c0 */ 2, /* c1 */ 3, /* c2 */ 3, /* c3 */ 4, /* c4 */ 3,
    /* c5 */ 4, /* c6 */ 4, /* c7 */ 5, /* c8 */ 3, /* c9 */ 4,
    /* ca */ 4, /* cb */ 5, /* cc */ 4, /* cd */ 5, /* ce */ 5,
    /* cf */ 6, /* d0 */ 3, /* d1 */ 4, /* d2 */ 4, /* d3 */ 5,
    /* d4 */ 4, /* d5 */ 5, /* d6 */ 5, /* d7 */ 6, /* d8 */ 4,
    /* d9 */ 5, /* da */ 5, /* db */ 6, /* dc */ 5, /* dd */ 6,
    /* de */ 6, /* df */ 7, /* e0 */ 3, /* e1 */ 4, /* e2 */ 4,
    /* e3 */ 5, /* e4 */ 4, /* e5 */ 5, /* e6 */ 5, /* e7 */ 6,
    /* e8 */ 4, /* e9 */ 5, /* ea */ 5, /* eb */ 6, /* ec */ 5,
    /* ed */ 6, /* ee */ 6, /* ef */ 7, /* f0 */ 4, /* f1 */ 5,
    /* f2 */ 5, /* f3 */ 6, /* f4 */ 5, /* f5 */ 6, /* f6 */ 6,
    /* f7 */ 7, /* f8 */ 5, /* f9 */ 6, /* fa */ 6, /* fb */ 7,
    /* fc */ 6, /* fd */ 7, /* fe */ 7, /* ff */ 8,
];
