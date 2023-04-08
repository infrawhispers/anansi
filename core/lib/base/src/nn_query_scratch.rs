use crate::ann;
use crate::av_store::AlignedDataStore;
use crate::diskannv1::DiskANNParamsInternal;
use crate::nn_queue::NNPriorityQueue;

use roaring::RoaringTreemap;

use std::collections::HashSet;

pub struct InMemoryQueryScratch {
    pub q_aligned: AlignedDataStore<f32>,
    pub occlude_factor: Vec<f32>,
    pub inserted_into_pool_rb: RoaringTreemap,
    pub inserted_into_pool_hs: HashSet<usize>,
    pub id_scratch: Vec<usize>,
    pub dist_scratch: Vec<f32>,
    pub best_l_nodes: NNPriorityQueue,
    pub pool: Vec<ann::INode>,
    // _marker: NoCopy
    // curr_l: usize,
    // curr_r: usize,
}

impl InMemoryQueryScratch {
    pub fn new(params: &DiskANNParamsInternal) -> Self {
        InMemoryQueryScratch {
            q_aligned: AlignedDataStore::new(params.aligned_dim, 1),
            occlude_factor: Vec::with_capacity(params.params_e.indexing_maxc),
            inserted_into_pool_rb: RoaringTreemap::new(),
            inserted_into_pool_hs: HashSet::new(),
            id_scratch: Vec::with_capacity(
                (1.5 * (params.params_e.indexing_range as f32) * 1.05).ceil() as usize,
            ),
            dist_scratch: Vec::with_capacity(
                (1.5 * (params.params_e.indexing_range as f32) * 1.05).ceil() as usize,
            ),
            best_l_nodes: NNPriorityQueue::new(params.params_e.indexing_queue_size),
            pool: Vec::with_capacity(
                3 * params.params_e.indexing_queue_size + params.params_e.indexing_range,
            ),
            // curr_l: params.params_e.indexing_queue_size,
            // curr_r: params.params_e.indexing_range,
        }
    }
    pub fn clear(&mut self) {
        self.pool.clear();
        self.best_l_nodes.clear();
        self.occlude_factor.clear();

        self.inserted_into_pool_rb.clear();
        self.inserted_into_pool_hs.clear();

        self.id_scratch.clear();
        self.dist_scratch.clear();
    }
    // pub fn resize_for_new_l(&mut self, l_new: usize) {
    //     if l_new > self.curr_l {
    //         self.curr_l = l_new;
    //         self.pool.reserve(3 * self.curr_l + self.curr_r);
    //         self.inserted_into_pool_hs.reserve(20 * self.curr_l);
    //     }
    // }
}
