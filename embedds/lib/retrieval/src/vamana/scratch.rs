use std::collections::HashSet;

use crate::av_store;
use crate::nn_queue::NNPriorityQueue;
use crate::vamana::pq::PQScratch;
use crate::vamana::utils::round_up;

pub struct SSDQueryScratch {
    pub coord_scratch: av_store::AlignedDataStore<f32>,
    pub coord_idx: usize,

    pub sector_scratch: av_store::AlignedDataStore<u8>,
    pub sector_idx: usize,

    pub aligned_query_t: av_store::AlignedDataStore<f32>,
    pub pq_scratch: PQScratch,

    pub visited: HashSet<usize>,
    pub retset: NNPriorityQueue,
    pub full_retset: Vec<crate::ann::Node>,
}

impl SSDQueryScratch {
    fn new(aligned_dim: usize, visited_reserve: usize) -> anyhow::Result<Self> {
        let coord_alloc_size: usize = round_up(crate::vamana::MAX_N_CMPS * aligned_dim, 256);
        let coord_scratch = av_store::AlignedDataStore::<f32>::new(1, coord_alloc_size);
        let sector_scratch = av_store::AlignedDataStore::<u8>::new(
            1,
            crate::vamana::MAX_N_SECTOR_READS * crate::vamana::SECTOR_LEN,
        );
        let aligned_query_t =
            av_store::AlignedDataStore::<f32>::new(1, aligned_dim * std::mem::size_of::<f32>());

        let pq_scratch = PQScratch::new(crate::vamana::MAX_GRAPH_DEGREE, aligned_dim);

        Ok(SSDQueryScratch {
            coord_scratch: coord_scratch,
            coord_idx: 0,

            sector_scratch: sector_scratch,
            sector_idx: 0,

            aligned_query_t: aligned_query_t,
            pq_scratch: pq_scratch,
            visited: HashSet::with_capacity(visited_reserve),
            retset: NNPriorityQueue::new(visited_reserve),
            full_retset: Vec::new(),
        })
    }
}

pub struct SSDThreadData {
    pub scratch: SSDQueryScratch,
}

impl SSDThreadData {
    pub fn new(aligned_dim: usize, visited_reserve: usize) -> anyhow::Result<Self> {
        let scratch = SSDQueryScratch::new(aligned_dim, visited_reserve)?;
        Ok(SSDThreadData { scratch: scratch })
    }
}
