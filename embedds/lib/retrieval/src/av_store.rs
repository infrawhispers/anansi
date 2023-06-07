use num::Num;
use serde::{Deserialize, Serialize};
use std::mem;

#[derive(Clone)]
#[repr(align(32))]
pub struct AlignToThirtyTwo([u8; 32]);

#[derive(Serialize, Deserialize, Debug)]
pub struct AlignedDataStore<T> {
    pub data: Vec<T>,
    pub num_vectors: usize,
}

impl<T: Num + std::marker::Copy> AlignedDataStore<T> {
    unsafe fn aligned_vec(n_bytes: usize) -> Vec<T> {
        // Lazy math to ensure we always have enough.
        let n_units = (n_bytes / mem::size_of::<AlignToThirtyTwo>()) + 1;
        let mut aligned: Vec<AlignToThirtyTwo> = vec![AlignToThirtyTwo([0u8; 32]); n_units]; //Vec::with_capacity(n_units);
        let ptr = aligned.as_mut_ptr();
        let len_units = aligned.len();
        let cap_units = aligned.capacity();
        mem::forget(aligned);
        Vec::from_raw_parts(
            ptr as *mut T,
            len_units * mem::size_of::<AlignToThirtyTwo>(),
            cap_units * mem::size_of::<AlignToThirtyTwo>(),
        )
    }
    pub fn aligned_insert(&mut self, id: usize, data: &[T]) {
        let ptr = self.data.as_ptr();
        unsafe {
            let insert_loc = ptr.add(id * data.len()) as *mut _;
            let write_loc: &mut [T] = std::slice::from_raw_parts_mut(insert_loc, data.len());
            write_loc.copy_from_slice(&data[..]);
            if id >= self.num_vectors {
                // this is really the number of _possible_ vectors, and not necessarily
                // the number of actual _live_ vectors the database
                self.num_vectors = id + 1
            }
        }
    }
    pub fn new(total_internal_points: usize, aligned_dim: usize) -> AlignedDataStore<T> {
        let mut data_vec: Vec<T>;
        unsafe {
            // n_bytes: total_points * aligned_dim * [size_of(f32) | size_of(u8)]
            data_vec = AlignedDataStore::<T>::aligned_vec(
                total_internal_points * aligned_dim * std::mem::size_of::<T>(),
            );
            data_vec.set_len(total_internal_points * aligned_dim);
        }
        return AlignedDataStore {
            data: data_vec,
            num_vectors: 0,
        };
    }

    // TODO(infrawhispers) - do we still need this??
    // pub fn aligned_add(&self, id: usize, length: usize, data: &[f32]) {
    //     let ptr = self.data.as_ptr();
    //     unsafe {
    //         let insert_loc = ptr.add(id * length) as *mut _;
    //         let write_loc: &mut [f32] = std::slice::from_raw_parts_mut(insert_loc, length);
    //         write_loc.copy_from_slice(&data[..]);
    //     }
    // }
}
