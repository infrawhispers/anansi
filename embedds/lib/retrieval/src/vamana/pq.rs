use std::collections::HashMap;
use std::fs;
use std::io::Read;
use std::io::Write;
use std::marker::PhantomData;
use std::path::Path;

use anyhow::Context;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use tracing::info;

use crate::av_store::AlignedDataStore;
use crate::vamana::utils::div_round_up;
use crate::vamana::BLOCK_SIZE;

pub const NUM_PQ_BITS: usize = 8;
pub const NUM_PQ_CENTROIDS: usize = 1 << NUM_PQ_BITS;
pub const MAX_PQ_CHUNKS: usize = 512;

pub struct PQScratch {
    pub pq_coord_scratch: AlignedDataStore<u8>,
    pub pqtable_dist_scratch: AlignedDataStore<f32>,
    pub dist_scratch: AlignedDataStore<f32>,
    pub query: AlignedDataStore<f32>,
    pub rotated_query: AlignedDataStore<f32>,
}
impl PQScratch {
    pub fn new(graph_degree: usize, aligned_dim: usize) -> Self {
        PQScratch {
            pq_coord_scratch: AlignedDataStore::<u8>::new(graph_degree, MAX_PQ_CHUNKS),
            pqtable_dist_scratch: AlignedDataStore::<f32>::new(1, 256 * MAX_PQ_CHUNKS),
            dist_scratch: AlignedDataStore::<f32>::new(1, graph_degree),
            query: AlignedDataStore::<f32>::new(1, aligned_dim),
            rotated_query: AlignedDataStore::<f32>::new(1, aligned_dim),
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct PQResult {
    pub chunks: Vec<usize>,
    pub pivots: Vec<f32>,
    pub pivots_dims: usize,
    pub centroid: Vec<f32>,
}

impl PQResult {
    pub fn from_ref_bin_format(filepath: &std::path::Path) -> anyhow::Result<Self> {
        let res_meta = super::utils::diskann_load_bin(filepath, 0, "u64")
            .with_context(|| "unable to load the pq metadata")?;
        let file_offsets = res_meta
            .buf_usize
            .ok_or_else(|| anyhow::anyhow!("expected [usize] to be populated - logic err"))?;
        let res_pivots = super::utils::diskann_load_bin(filepath, file_offsets[0], "f32")
            .with_context(|| "expected [f32] to be populated - logic err")?;
        let pivots = res_pivots
            .buf_f32
            .ok_or_else(|| anyhow::anyhow!("unable to get the pivot data"))?;
        let res_centroids = super::utils::diskann_load_bin(filepath, file_offsets[1], "f32")
            .with_context(|| "expected [f32] to be populated - logic err")?;
        let centroids = res_centroids
            .buf_f32
            .ok_or_else(|| anyhow::anyhow!("unable to get the centroid data"))?;
        let res_chunks = super::utils::diskann_load_bin(filepath, file_offsets[2], "u32")
            .with_context(|| "unable to load the chunk_offsets")?;
        let chunks = res_chunks
            .buf_usize
            .ok_or_else(|| anyhow::anyhow!("expected [usize] to be populated - logic err"))?;
        Ok(PQResult {
            chunks: chunks,
            pivots: pivots,
            pivots_dims: res_pivots.dims,
            centroid: centroids,
        })
    }
}

#[derive(Serialize, Deserialize)]
pub struct PQMetadata {
    pub num_points: u32,
    pub num_pq_chunks: u32,
}

pub struct PQ<TMetric> {
    metric: PhantomData<TMetric>,
}
impl<TMetric> PQ<TMetric>
where
    TMetric: crate::metric::Metric<f32>,
{
    pub fn new() -> Self {
        PQ {
            metric: PhantomData,
        }
    }

    #[allow(unused_variables)]
    pub fn generate_opq_pivots(
        &self,
        data: &[f32],
        dim: usize,
        num_centers: usize,
        num_pq_chunks: usize,
        make_zero_mean: bool,
    ) {
        unimplemented!()
    }

    pub fn generate_pq_pivots(
        &self,
        data: &[f32],
        dim: usize,
        num_centers: usize,
        num_pq_chunks: usize,
        make_zero_mean: bool,
    ) -> anyhow::Result<()> {
        let num_train = data.len() / dim;
        if num_pq_chunks > dim {
            panic!("num chunks: {} > dimension: {}", num_pq_chunks, dim);
        }
        if std::path::Path::new(".disk/pq_pivots.bin").exists() {
            // let mut f = std::fs::File::open(".disk/pq_pivots.bin")?;
            // let mut buf: Vec<u8> = Vec::new();
            // f.read_to_end(&mut buf)?;
            // _ = rmp_serde::from_slice(&buf)?;
            info!("found existing product quantization record:");
            return Ok(());
        }

        let mut train_data = vec![0.0; data.len()];
        train_data[..].copy_from_slice(data);
        let mut centroid: Vec<f32> = vec![0.0; dim];
        if make_zero_mean {
            for d in 0..dim {
                for p in 0..num_train {
                    centroid[d] += train_data[p * dim + d];
                }
                centroid[d] /= num_train as f32;
            }
            for d in 0..dim {
                for p in 0..num_train {
                    train_data[p * dim + d] -= centroid[d];
                }
            }
        }
        info!("train_data: {:?}", &train_data[0..2]);
        let low_val: usize = (dim as f32 / num_pq_chunks as f32).floor() as usize;
        let high_val: usize = (dim as f32 / num_pq_chunks as f32).ceil() as usize;
        let max_num_high: usize = dim - (low_val * num_pq_chunks);
        let mut curr_num_high: usize = 0;
        let mut curr_bin_threshold: usize = high_val;

        let mut bin_to_dims: Vec<Vec<usize>> = vec![vec![]; num_pq_chunks];
        let dim_to_bin: HashMap<usize, usize> = HashMap::new();
        let bin_loads: Vec<f32> = vec![0.0; num_pq_chunks];

        // process dimensions not inserted by the previous loop
        for d in 0..dim {
            if dim_to_bin.contains_key(&d) {
                continue;
            }
            let mut curr_best: usize = num_pq_chunks + 1;
            let mut curr_best_load: f32 = f32::MAX;
            for b in 0..num_pq_chunks {
                if bin_loads[b] < curr_best_load && bin_to_dims[b].len() < curr_bin_threshold {
                    curr_best = b;
                    curr_best_load = bin_loads[b];
                }
            }
            bin_to_dims[curr_best].push(d);
            if bin_to_dims[curr_best].len() == high_val {
                curr_num_high += 1;
                if curr_num_high == max_num_high {
                    curr_bin_threshold = low_val;
                }
            }
        }
        // info!(".0 -> {} .1 -> {}", train_data[0], train_data[1]);
        // info!("point: (0): {:?}", &train_data[0..128]);
        // info!("point: (1): {:?}", &train_data[128..256]);

        let mut chunk_offsets: Vec<usize> = vec![0; 1];
        for b in 0..num_pq_chunks {
            if b > 0 {
                let one = chunk_offsets[b - 1];
                let two = bin_to_dims[b - 1].len();
                chunk_offsets.push(one + two);
            }
        }
        chunk_offsets.push(dim);
        let full_pivot_data: Mutex<Vec<f32>> = Mutex::new(vec![0.0; num_centers * dim]);
        (0..num_pq_chunks).into_iter().for_each(|i| {
            let curr_chunk_size = chunk_offsets[i + 1] - chunk_offsets[i];
            if curr_chunk_size == 0 {
                return;
            }
            // let curr_pivot_data: Vec<f32> = vec![0.0; num_centers * curr_chunk_size];
            let mut curr_data: Vec<f32> = vec![0.0; num_train * curr_chunk_size];
            info!(
                "cur_data-size: {} | num_centers: {}",
                num_train * curr_chunk_size,
                num_centers
            );
            // let closest_center: Vec<usize> = vec![0; num_train];
            info!(
                "processing chunk {} with dims: ({}, {})",
                i,
                chunk_offsets[i],
                chunk_offsets[i + 1]
            );
            for j in 0..num_train {
                let curr_start = j * curr_chunk_size;
                let curr_end = curr_start + curr_chunk_size;
                let data_start = j * dim + chunk_offsets[i];
                let data_end = data_start + curr_chunk_size;
                curr_data[curr_start..curr_end].copy_from_slice(&train_data[data_start..data_end]);
            }
            info!(
                "kmeans args: {} num_points: {} | dim: {} ",
                curr_data.len(),
                num_train,
                curr_chunk_size
            );
            let kmean = kmeans::KMeans::new(curr_data, num_train, curr_chunk_size);
            let result = kmean.kmeans_lloyd(
                num_centers,
                12,
                kmeans::KMeans::init_random_sample,
                &kmeans::KMeansConfig::default(),
            );
            for j in 0..num_centers {
                let data_start = j * dim + chunk_offsets[i];
                let data_end = data_start + curr_chunk_size;
                let result_start = j * curr_chunk_size;
                let result_end = result_start + curr_chunk_size;
                // info!("result: {:?}", &result.centroids[result_start..result_end]);
                full_pivot_data.lock()[data_start..data_end]
                    .copy_from_slice(&result.centroids[result_start..result_end]);
            }
        });
        let res: PQResult = PQResult {
            chunks: chunk_offsets,
            pivots: full_pivot_data.lock().to_vec(),
            pivots_dims: 0,
            centroid: centroid,
        };
        let res_b = rmp_serde::to_vec(&res)?;
        fs::write(".disk/pq_pivots.bin", res_b)?;
        Ok(())
    }

    pub fn generate_pq_data_from_pivots(
        &self,
        data: &[f32],
        num_centers: usize,
        num_pq_chunks: usize,
        use_opq: bool,
    ) -> anyhow::Result<()> {
        let mut f = std::fs::File::open(".disk/pq_pivots.bin")?;
        let mut buf: Vec<u8> = Vec::new();
        f.read_to_end(&mut buf)?;
        // let res: PQResult = rmp_serde::from_slice(&buf)?;
        let directory = Path::new("../../../../public/DiskANN/build/data/siftsmall/disk_index_sift_learn_R32_L50_A1.2_pq_pivots.bin");
        let res: PQResult = PQResult::from_ref_bin_format(directory)?;

        let pq_compressed_path = ".disk/pq_compressed.bin";
        let mut outputf = std::fs::File::create(pq_compressed_path)?;
        let num_points: usize = data.len() / 128;
        let meta = PQMetadata {
            num_points: num_points as u32,
            num_pq_chunks: num_pq_chunks as u32,
        };
        outputf.write(&rmp_serde::to_vec(&meta)?)?;
        outputf.sync_data()?;
        let block_size: usize = if num_points < crate::vamana::BLOCK_SIZE {
            num_points
        } else {
            BLOCK_SIZE
        };
        let dim = 128;
        let mut block_compressed_base: Vec<usize> = vec![0; block_size * num_pq_chunks];
        // let block_data_t: Vec<f32> = vec![0.0; block_size * dim];
        let mut block_data_f: Vec<f32> = vec![0.0; block_size * dim];
        let mut block_data_tmp: Vec<f32> = vec![0.0; block_size * dim];
        let num_blocks = div_round_up(num_points, block_size);
        for block in 0..num_blocks {
            let start_id = block * block_size;
            let end_id = std::cmp::min((block + 1) * block_size, num_points);
            let curr_block_size = end_id - start_id;
            let idx_start = block * block_size * dim;
            let idx_end = idx_start + (block_size * dim);
            block_data_tmp[..].copy_from_slice(&data[idx_start..idx_end]);
            info!("processing: [{start_id}, {end_id})");
            for p in 0..curr_block_size {
                for d in 0..dim {
                    block_data_tmp[p * dim + d] -= res.centroid[d];
                }
            }
            for p in 0..curr_block_size {
                for d in 0..dim {
                    block_data_f[p * dim + d] = block_data_tmp[p * dim + d];
                }
            }
            if use_opq {
                unimplemented!()
            }
            for i in 0..num_pq_chunks {
                let curr_chunk_size = res.chunks[i + 1] - res.chunks[i];
                if curr_chunk_size == 0 {
                    continue;
                }
                let mut curr_pivot_data: Vec<f32> = vec![0.0; num_centers * curr_chunk_size];
                let mut curr_data: Vec<f32> = vec![0.0; curr_chunk_size * curr_block_size];
                let mut closest_centers: Vec<usize> = vec![0; curr_block_size];
                // this has a pragma omp parallel in the c++ impl
                for j in 0..curr_block_size {
                    for k in 0..curr_chunk_size {
                        curr_data[j * curr_chunk_size + k] =
                            block_data_f[j * dim + res.chunks[i] + k];
                    }
                }
                // this has a pragma omp parallel in the c++ impl
                for j in 0..num_centers {
                    curr_pivot_data[j * curr_chunk_size..j * curr_chunk_size + curr_chunk_size]
                        .copy_from_slice(
                            &res.pivots[j * dim + res.chunks[i]
                                ..j * dim + res.chunks[i] + curr_chunk_size],
                        )
                }
                super::math_utils::compute_closest_centers(
                    &curr_data,
                    curr_block_size,
                    curr_chunk_size,
                    &curr_pivot_data,
                    num_centers,
                    1,
                    &mut closest_centers,
                )?;
                for j in 0..curr_block_size {
                    block_compressed_base[j * num_pq_chunks + i] = closest_centers[j];
                }
            }
            if num_centers > 256 {
                unimplemented!("we don't handle the case where num_centers > 256 just yet")
            } else {
                let block = rmp_serde::to_vec(&block_compressed_base)?;
                outputf.write(&block)?;
            }
        }
        info!(".done.");
        Ok(())
    }

    pub fn generate_quantized_data(
        &self,
        data: &[f32],
        num_pq_chunks: usize,
        use_opq: bool,
    ) -> anyhow::Result<()> {
        let train_dim: usize = 128;
        let train_size: usize = data.len() / train_dim;
        info!("training with {} samples", train_size);
        let mut make_zero_mean = true;
        if use_opq {
            make_zero_mean = false;
        }
        if !use_opq {
            self.generate_pq_pivots(
                data,
                train_dim,
                NUM_PQ_CENTROIDS,
                num_pq_chunks,
                make_zero_mean,
            )
            .with_context(|| "unable to generate the pq pivots")?;
        } else {
            self.generate_opq_pivots(
                data,
                train_dim,
                NUM_PQ_CENTROIDS,
                num_pq_chunks,
                make_zero_mean,
            )
        }
        self.generate_pq_data_from_pivots(data, NUM_PQ_CENTROIDS, num_pq_chunks, use_opq)
    }
}

pub fn aggregate_coords(ids: &[usize], all_coords: &[u8], ndims: usize, out: &mut [u8]) {
    for (i, id) in ids.iter().enumerate() {
        if *id == 25000 {
            // TOOD = REMOVE THIS REALLY BAD HACK!!!!!!
            out[i * ndims..i * ndims + ndims]
                .copy_from_slice(&all_coords[22877 * ndims..22877 * ndims + ndims]);
        } else {
            out[i * ndims..i * ndims + ndims]
                .copy_from_slice(&all_coords[id * ndims..id * ndims + ndims]);
        }
    }
}

pub fn pq_dist_lookup(
    pq_ids: &[u8],
    n_pts: usize,
    pq_nchunks: usize,
    pq_dists: &[f32],
    dists_out: &mut [f32],
) {
    #[cfg(all(target_feature = "x86",))]
    {
        core::arch::x86::_mm_prefetch(
            dists_out.as_ptr() as *const i8,
            core::arch::x86::_MM_HINT_T0::_MM_HINT_T0,
        );
        core::arch::x86::_mm_prefetch(
            pq_ids.as_ptr() as *const i8,
            core::arch::x86::_MM_HINT_T0::_MM_HINT_T0,
        );
        core::arch::x86::_mm_prefetch(
            (pq_ids.as_ptr() as *const i8).add(64),
            core::arch::x86::_MM_HINT_T0::_MM_HINT_T0,
        );
        core::arch::x86::_mm_prefetch(
            (pq_ids.as_ptr() as *const i8).add(128),
            core::arch::x86::_MM_HINT_T0::_MM_HINT_T0,
        );
    }
    dists_out.fill(0.0);
    for chunk in 0..pq_nchunks {
        let chunk_dists_loc: usize = 256 * chunk;
        #[cfg(all(target_feature = "x86",))]
        if (chunk < pq_nchunks - 1) {
            core::arch::x86::_mm_prefetch(
                (chunk_dists.as_ptr() as *const i8).add(256),
                core::arch::x86::_MM_HINT_T0::_MM_HINT_T0,
            );
        }
        for idx in 0..n_pts {
            let pq_centerid = pq_ids[pq_nchunks * idx + chunk];
            dists_out[idx] += pq_dists[pq_centerid as usize + 256 * chunk];
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::path::Path;
    #[test]
    fn read_pq_pivots() {
        let filepath = Path::new("../../../../public/DiskANN/build/data/siftsmall/disk_index_sift_learn_R32_L50_A1.2_pq_pivots.bin");
        PQResult::from_ref_bin_format(filepath).expect("able to load PQResults from file");
    }
}
