#![feature(stdsimd)]
// #[cfg(all(target_arch = "wasm", target_feature = "simd128",))]
#![feature(simd_wasm64)]

pub mod ann;
mod av_store;
pub mod diskannv1;
mod errors;
pub mod flat;
pub mod metric;

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
