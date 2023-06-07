use std::io::Read;
use std::path::Path;

use anyhow::{bail, Context};

use crate::vamana::pq::PQResult;

pub struct FixedChunkPQTable {
    pub ndims: usize,
    pub n_chunks: usize,
    pub chunks: Vec<usize>,
    pub tables: Vec<f32>,
    pub tables_tr: Vec<f32>,
    pub centroids: Vec<f32>,
}

impl FixedChunkPQTable {
    // assumes pre-processed query which has been centered and rotated
    pub fn populate_chunk_distances(&self, query_vec: &[f32], dist_vec: &mut [f32]) {
        dist_vec.fill(0.0);
        for chunk in 0..self.n_chunks {
            let loc = 256 * chunk;
            for j in self.chunks[chunk]..self.chunks[chunk + 1] {
                let centers_dim_vec_loc: usize = 256 * j;
                for idx in 0..256 {
                    let diff = self.tables_tr[centers_dim_vec_loc + idx] - query_vec[j];
                    dist_vec[loc + idx] = diff * diff;
                }
            }
        }
    }
    pub fn preprocess_query(&self, query_vec: &mut [f32]) {
        for d in 0..self.ndims {
            query_vec[d] -= self.centroids[d];
        }
        // TOOD(infrawhispers) - we will need to add some code to
        // handle rotation by doing something of the form:
        // tmp[d] += query_vec[d1] * rotmat_tr[d1 * ndims + d];
    }

    pub fn fr_pivots_bin(folder_path: &Path, num_chunks: usize) -> anyhow::Result<Self> {
        let mut f = std::fs::File::open(folder_path.join("pq_pivots.bin"))
            .with_context(|| "unable to open pq_pivots")?;
        let mut buf: Vec<u8> = Vec::new();
        f.read_to_end(&mut buf)?;
        let res: PQResult = rmp_serde::from_slice(&buf)?;

        //
        let nr = res.pivots.len() / res.pivots_dims;
        if nr != crate::vamana::pq::NUM_PQ_CENTROIDS {
            bail!(
                "error reading pq_pivots file, got: num_centers: {nr} expected: {}",
                crate::vamana::pq::NUM_PQ_CENTROIDS
            )
        }
        let ndims: usize = res.pivots_dims;
        if ndims != res.centroid.len() {
            bail!(
                "error reading pq_pivots file, got: num_centroids: {} expected: {} ",
                res.centroid.len(),
                ndims
            )
        }
        if res.chunks.len() != num_chunks + 1 && num_chunks != 0 {
            bail!(
                "error reading pq_pivots file, got: num_chunks: {}, expected: {}",
                res.chunks.len(),
                num_chunks + 1
            )
        }
        let n_chunks = res.chunks.len() - 1;
        let mut tables_tr: Vec<f32> = vec![0.0; ndims * 256];
        for i in 0..256 {
            for j in 0..ndims {
                tables_tr[j * 256 + i] = res.pivots[i * ndims + j]
            }
        }
        Ok(FixedChunkPQTable {
            ndims: ndims,
            n_chunks: n_chunks,
            chunks: res.chunks,
            tables: res.pivots,
            tables_tr: tables_tr,
            centroids: res.centroid,
        })
    }

    pub fn load_pq_centroid_bin(path: &Path, num_chunks: usize) -> anyhow::Result<Self> {
        let res = PQResult::from_ref_bin_format(path)?;
        let nr = res.pivots.len() / res.pivots_dims;
        if nr != crate::vamana::pq::NUM_PQ_CENTROIDS {
            bail!(
                "error reading pq_pivots file, got: num_centers: {nr} expected: {}",
                crate::vamana::pq::NUM_PQ_CENTROIDS
            )
        }
        let ndims: usize = res.pivots_dims;
        if ndims != res.centroid.len() {
            bail!(
                "error reading pq_pivots file, got: num_centroids: {} expected: {} ",
                res.centroid.len(),
                ndims
            )
        }
        if res.chunks.len() != num_chunks + 1 && num_chunks != 0 {
            bail!(
                "error reading pq_pivots file, got: num_chunks: {}, expected: {}",
                res.chunks.len(),
                num_chunks + 1
            )
        }
        let n_chunks = res.chunks.len() - 1;
        let mut tables_tr: Vec<f32> = vec![0.0; ndims * 256];
        for i in 0..256 {
            for j in 0..ndims {
                tables_tr[j * 256 + i] = res.pivots[i * ndims + j]
            }
        }
        Ok(FixedChunkPQTable {
            ndims: ndims,
            n_chunks: n_chunks,
            chunks: res.chunks,
            tables: res.pivots,
            tables_tr: tables_tr,
            centroids: res.centroid,
        })
    }
}
