use std::mem;

#[derive(Clone)]
#[repr(align(32))]
pub struct AlignToThirtyTwo([u8; 32]);

#[derive(Debug)]
pub struct AlignedDataStore {
    pub data: Vec<f32>,
    pub num_vectors: usize,
}

impl AlignedDataStore {
    unsafe fn aligned_vec(n_bytes: usize) -> Vec<f32> {
        // Lazy math to ensure we always have enough.
        let n_units = (n_bytes / mem::size_of::<AlignToThirtyTwo>()) + 1;
        let mut aligned: Vec<AlignToThirtyTwo> = vec![AlignToThirtyTwo([0u8; 32]); n_units]; //Vec::with_capacity(n_units);
        let ptr = aligned.as_mut_ptr();
        let len_units = aligned.len();
        let cap_units = aligned.capacity();
        mem::forget(aligned);
        Vec::from_raw_parts(
            ptr as *mut f32,
            len_units * mem::size_of::<AlignToThirtyTwo>(),
            cap_units * mem::size_of::<AlignToThirtyTwo>(),
        )
    }
    pub fn aligned_insert(&mut self, id: usize, data: &[f32]) {
        let ptr = self.data.as_ptr();
        unsafe {
            let insert_loc = ptr.add(id * data.len()) as *mut _;
            let write_loc: &mut [f32] = std::slice::from_raw_parts_mut(insert_loc, data.len());
            write_loc.copy_from_slice(&data[..]);
            if id >= self.num_vectors {
                self.num_vectors += 1
            }
        }
    }
    pub fn new(aligned_dim: usize, total_internal_points: usize) -> AlignedDataStore {
        let mut data_vec;
        unsafe {
            // we store floats so we always multiple by 4
            // TODO(infrawhispers) - make this configurable if we end up using ints
            data_vec = AlignedDataStore::aligned_vec(total_internal_points * aligned_dim * 4);
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
