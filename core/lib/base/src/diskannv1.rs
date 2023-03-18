use crate::ann;
use crate::ann::EId;
use crate::av_store;
use crate::av_store::AlignedDataStore;
use crate::errors;
use crate::metric;
use crate::nn_query_scratch;
use crate::nn_queue;

use crossbeam_channel::{bounded, Receiver, Sender};
use parking_lot::RwLock;
use rayon::prelude::*;
use roaring::RoaringTreemap;
use std::cmp;
use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;
use std::ops::ControlFlow;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;
use std::time::Instant;

#[derive(Debug, Clone, Copy)]
pub struct DiskANNParams {
    pub dim: usize,
    pub max_points: usize,
    pub indexing_threads: usize,
    pub indexing_range: usize,
    pub indexing_queue_size: usize,
    pub indexing_maxc: usize,
    pub indexing_alpha: f32,
}

#[allow(dead_code)]
pub struct DiskANNParamsInternal {
    pub params_e: DiskANNParams,
    pub aligned_dim: usize,
    pub data_len: usize,
    pub num_frozen_pts: usize,
    pub total_internal_points: usize,
    pub nd: usize,
    pub neighbor_len: usize,
    pub node_size: usize,
    pub start: usize,
    pub saturate_graph: bool,
}

#[allow(dead_code)]
pub struct DiskANNV1Index<TMetric: metric::Metric<f32>> {
    params: Arc<RwLock<DiskANNParamsInternal>>,
    metric: PhantomData<TMetric>,

    data: Arc<RwLock<av_store::AlignedDataStore>>,
    final_graph: Arc<Vec<RwLock<Vec<usize>>>>,
    location_to_tag: Arc<RwLock<HashMap<usize, EId>>>,
    tag_to_location: Arc<RwLock<HashMap<EId, usize>>>,

    id_increment: Arc<AtomicUsize>,
    delete_set: Arc<RwLock<HashSet<usize>>>,
    empty_slots: Arc<RwLock<HashSet<usize>>>,
    s_scratch: Sender<nn_query_scratch::InMemoryQueryScratch>,
    r_scratch: Receiver<nn_query_scratch::InMemoryQueryScratch>,
}

const GRAPH_SLACK_FACTOR: f64 = 1.3;
const MAX_POINTS_FOR_USING_BITSET: usize = 10_000_000;
enum QueryTarget<'a> {
    VId(usize),
    Vector(&'a [f32]),
}

impl<TMetric> ann::ANNIndex for DiskANNV1Index<TMetric>
where
    TMetric: metric::Metric<f32>,
{
    fn new(params: &ann::ANNParams) -> Result<DiskANNV1Index<TMetric>, Box<dyn std::error::Error>> {
        let diskann_params: &DiskANNParams = match params {
            ann::ANNParams::Flat { params: _ } => {
                unreachable!("incorrect params passed for construction")
            }
            ann::ANNParams::DiskANN { params } => params,
        };
        DiskANNV1Index::new(diskann_params)
    }
    fn batch_insert(&self, eids: &[EId], data: &[f32]) -> Result<(), Box<dyn std::error::Error>> {
        self.batch_insert(eids, data)
    }
    fn insert(&self, eid: EId, data: &[f32]) -> Result<(), Box<dyn std::error::Error>> {
        self.insert(eid, data)
    }
    fn search(&self, q: &[f32], k: usize) -> Result<Vec<ann::Node>, Box<dyn std::error::Error>> {
        self.search(q, k)
    }
    fn save(&self) -> Result<(), Box<dyn std::error::Error>> {
        unimplemented!()
    }
}

impl<TMetric> DiskANNV1Index<TMetric>
where
    TMetric: metric::Metric<f32>,
{
    fn calculate_entry_point(
        &self,
        paramsr: &DiskANNParamsInternal,
        data: &AlignedDataStore,
    ) -> usize {
        // let paramsr = self.params.read();
        // let data = self.data.read();

        let aligned_dim: usize = paramsr.aligned_dim;
        let nd: usize = paramsr.nd;
        let mut center: Vec<f32> = vec![0.0; aligned_dim];
        for i in 0..nd {
            for j in 0..aligned_dim {
                center[j] += data.data[i * aligned_dim + j];
            }
        }
        for j in 0..aligned_dim {
            center[j] /= nd as f32;
        }
        let mut distances: Vec<f32> = vec![0.0; nd];
        distances.par_iter_mut().enumerate().for_each(|(i, x)| {
            let s_idx: usize = i * aligned_dim;
            let e_idx: usize = s_idx + aligned_dim;
            let vec_t = &data.data[s_idx..e_idx];
            let mut dist: f32 = 0.0;
            for j in 0..aligned_dim {
                dist += (center[j] - vec_t[j]).powf(2.0)
            }
            *x = dist
        });
        let mut min_idx: usize = 0;
        let mut min_dis: f32 = distances[0];
        for i in 1..nd {
            if distances[i] < min_dis {
                min_idx = i;
                min_dis = distances[i];
            }
        }
        return min_idx;
    }

    fn generate_frozen_point(&self) {
        let params_r = self.params.read();
        let mut data_w = self.data.write();
        // {idx_f, idx_t} are indices into an array
        let idx_f: usize = self.calculate_entry_point(&params_r, &data_w) * params_r.aligned_dim;
        let idx_t: usize = params_r.params_e.max_points * params_r.aligned_dim;
        ann::copy_within_a_slice(&mut data_w.data, idx_f, idx_t, params_r.aligned_dim);
    }
    fn link(&self) {
        let params_r = self.params.read();
        let mut visit_order: Vec<usize> = Vec::with_capacity(params_r.nd + params_r.num_frozen_pts);
        for vid in 0..params_r.nd {
            visit_order.push(vid);
        }
        if params_r.num_frozen_pts > 0 {
            visit_order.push(params_r.params_e.max_points);
        }
        let start = Instant::now();
        visit_order.par_iter().for_each(|vid| {
            let mut pruned_list: Vec<usize> = Vec::new();
            // let mut scratch: nn_query_scratch::InMemoryQueryScratch =
            //     nn_query_scratch::InMemoryQueryScratch::new(&params_r);
            let mut scratch: nn_query_scratch::InMemoryQueryScratch =
                self.r_scratch.recv().unwrap();
            scratch.clear();
            self.search_for_point_and_prune(*vid, &mut pruned_list, &params_r, &mut scratch);
            {
                let mut segment = self.final_graph[*vid].write();
                segment.clear();
                for i in 0..pruned_list.len() {
                    segment.push(pruned_list[i]);
                }
            }
            self.inter_insert(
                *vid,
                &mut pruned_list,
                &params_r,
                &mut scratch,
                &self.data.read(),
            );
            self.s_scratch.send(scratch).unwrap();
        });
        let data = &self.data.read();
        visit_order.par_iter().for_each(|curr_vid| {
            let should_prune: bool;
            let graph_copy: Vec<usize>;
            {
                let nbrs = self.final_graph[*curr_vid].write();
                should_prune = nbrs.len() > params_r.params_e.indexing_range;
                graph_copy = nbrs.clone();
            }
            if should_prune {
                let mut scratch: nn_query_scratch::InMemoryQueryScratch =
                    self.r_scratch.recv().unwrap();
                scratch.clear();

                // let mut scratch: nn_query_scratch::InMemoryQueryScratch =
                //     nn_query_scratch::InMemoryQueryScratch::new(&params_r);
                let mut dummy_visited: HashSet<usize> = HashSet::new();
                let mut dummy_pool: Vec<ann::INode> = Vec::new();
                let mut new_out_neighbors: Vec<usize> = Vec::new();
                graph_copy.iter().for_each(|nbr_vid| {
                    if !dummy_visited.contains(nbr_vid) && *nbr_vid != *curr_vid {
                        let arr_a: &[f32] = &data.data[*curr_vid * params_r.aligned_dim
                            ..(*curr_vid * params_r.aligned_dim) + params_r.aligned_dim];
                        let arr_b: &[f32] = &data.data[nbr_vid * params_r.aligned_dim
                            ..(nbr_vid * params_r.aligned_dim) + params_r.aligned_dim];
                        dummy_pool.push(ann::INode {
                            vid: *nbr_vid,
                            distance: TMetric::compare(arr_a, arr_b),
                            flag: false,
                        });
                        dummy_visited.insert(*nbr_vid);
                    }
                });
                scratch.pool = dummy_pool;
                self.prune_neighbors(
                    *curr_vid,
                    &mut new_out_neighbors,
                    &params_r,
                    &mut scratch,
                    data,
                );
                {
                    let mut segment = self.final_graph[*curr_vid].write();
                    segment.clear();
                    for i in 0..new_out_neighbors.len() {
                        segment.push(new_out_neighbors[i]);
                    }
                }
                self.s_scratch.send(scratch).unwrap();
            }
        });
        println!("link time: {:?}", start.elapsed());
    }

    fn iterate_to_fixed_point(
        &self,
        target: QueryTarget,
        params_r: &DiskANNParamsInternal,
        init_ids: &mut Vec<usize>,
        scratch: &mut nn_query_scratch::InMemoryQueryScratch,
        ret_frozen: bool,
        is_search: bool,
    ) -> (usize, usize) {
        let data = self.data.read();
        // pull out the slice we are comparing against
        let arr_b: &[f32];
        match target {
            QueryTarget::VId(vid) => {
                arr_b = &data.data
                    [vid * params_r.aligned_dim..vid * params_r.aligned_dim + params_r.aligned_dim];
            }
            QueryTarget::Vector(v) => {
                arr_b = v;
            }
        }
        let expanded_nodes: &mut Vec<ann::INode> = &mut scratch.pool;
        let best_l_nodes: &mut nn_queue::NNPriorityQueue = &mut scratch.best_l_nodes;
        let inserted_into_pool_hs: &mut HashSet<usize> = &mut scratch.inserted_into_pool_hs;
        let inserted_into_pool_rb: &mut RoaringTreemap = &mut scratch.inserted_into_pool_rb;
        let id_scratch: &mut Vec<usize> = &mut scratch.id_scratch;
        let dist_scratch: &mut Vec<f32> = &mut scratch.dist_scratch;
        // debug_assert!(id_scratch.len() == 0, "scratch space must be cleared");
        // debug_assert!(expanded_nodes.len() == 0, "scratch space must be cleared");
        let fast_iterate: bool =
            params_r.params_e.max_points + params_r.num_frozen_pts <= MAX_POINTS_FOR_USING_BITSET;

        init_ids.iter().for_each(|nn_id| {
            if *nn_id >= params_r.params_e.max_points + params_r.num_frozen_pts {
                // TODO(infrawhispers) - get rid of this panic, this isn't *really* needed
                panic!(
                    "out of range loc: {} max_points: {}, num_frozen_pts: {}",
                    *nn_id, params_r.params_e.max_points, params_r.num_frozen_pts
                );
            }
            let is_not_visited = if fast_iterate {
                !inserted_into_pool_rb.contains((*nn_id).try_into().unwrap())
            } else {
                !inserted_into_pool_hs.contains(nn_id)
            };
            if is_not_visited {
                if fast_iterate {
                    inserted_into_pool_rb.insert((*nn_id).try_into().unwrap());
                } else {
                    inserted_into_pool_hs.insert(*nn_id);
                }
                let arr_a: &[f32] = &data.data[*nn_id * params_r.aligned_dim
                    ..(*nn_id * params_r.aligned_dim) + params_r.aligned_dim];
                let nn = ann::INode {
                    vid: *nn_id,
                    distance: TMetric::compare(arr_a, arr_b),
                    flag: false,
                };
                best_l_nodes.insert(nn);
            }
        });
        let hops: usize = 0;
        let mut cmps: usize = 0;
        while best_l_nodes.has_unexpanded_node() {
            let nbr = best_l_nodes.closest_unexpanded();
            let nbr_vid = nbr.vid;
            if !is_search
                && (nbr_vid != params_r.start || params_r.num_frozen_pts == 0 || ret_frozen)
            {
                expanded_nodes.push(nbr);
            }
            id_scratch.clear();
            dist_scratch.clear();
            {
                let _final_graph = self.final_graph[nbr_vid].read();
                for m in 0.._final_graph.len() {
                    debug_assert!(
                        _final_graph[m] <= params_r.params_e.max_points + params_r.num_frozen_pts,
                        "out of range edge: {edge} | found at vertex: {vertex}",
                        edge = _final_graph[m],
                        vertex = nbr_vid,
                    );
                    let nn_id = _final_graph[m];
                    let is_not_visited = if fast_iterate {
                        !inserted_into_pool_rb.contains((nn_id).try_into().unwrap())
                    } else {
                        !inserted_into_pool_hs.contains(&nn_id)
                    };
                    if is_not_visited {
                        id_scratch.push(_final_graph[m]);
                    }
                }
            }
            // mark nodes visited
            id_scratch.iter().for_each(|nn| {
                if fast_iterate {
                    inserted_into_pool_rb.insert((*nn).try_into().unwrap());
                } else {
                    inserted_into_pool_hs.insert(*nn);
                }
            });
            debug_assert!(dist_scratch.len() == 0);
            let mut nbrs_potential: Vec<ann::INode> = Vec::with_capacity(id_scratch.len());
            id_scratch.iter().for_each(|nn| {
                // if i + 1 < id_scratch.len() {
                // // TODO(infrawhispers) there is some pre-fetch funny biz that happens in the original implementation
                // }
                let arr_a: &[f32] = &data.data[*nn * params_r.aligned_dim
                    ..(*nn * params_r.aligned_dim) + params_r.aligned_dim];
                let dist: f32 = TMetric::compare(arr_a, arr_b);
                nbrs_potential.push(ann::INode {
                    vid: *nn,
                    distance: dist,
                    flag: false,
                })
            });
            cmps += id_scratch.len();
            nbrs_potential.iter().for_each(|nn| {
                best_l_nodes.insert(*nn);
            });
        }
        (hops, cmps)
    }

    fn occlude_list(
        &self,
        vid: usize,
        pool: &mut Vec<ann::INode>,
        alpha: f32,
        degree: usize,
        maxc: usize,
        result: &mut Vec<usize>,
        params_r: &DiskANNParamsInternal,
        data: &AlignedDataStore,
    ) {
        if pool.len() == 0 {
            return;
        }
        debug_assert!(pool.is_sorted());
        debug_assert!(result.len() == 0);
        if pool.len() > maxc {
            pool.resize(
                maxc,
                ann::INode {
                    vid: 0,
                    distance: std::f32::MAX,
                    flag: false,
                },
            );
        }

        let mut occlude_factor = vec![0.0; pool.len()];
        let mut curr_alpha = 1.0;
        while curr_alpha <= alpha && result.len() < degree {
            for idx_1 in 0..pool.len() {
                let nn = pool[idx_1];
                if occlude_factor[idx_1] > curr_alpha {
                    continue;
                }
                if result.len() >= degree {
                    break;
                }
                occlude_factor[idx_1] = f32::MAX;
                if nn.vid != vid {
                    result.push(nn.vid);
                }
                for idx2 in idx_1 + 1..pool.len() {
                    let nn_2 = pool[idx2];
                    if occlude_factor[idx2] > alpha {
                        continue;
                    }
                    let arr_a: &[f32] = &data.data[nn.vid * params_r.aligned_dim
                        ..(nn.vid * params_r.aligned_dim) + params_r.aligned_dim];
                    let arr_b: &[f32] = &data.data[pool[idx2].vid * params_r.aligned_dim
                        ..(pool[idx2].vid * params_r.aligned_dim) + params_r.aligned_dim];
                    let djk: f32 = TMetric::compare(arr_a, arr_b);
                    occlude_factor[idx2] = if djk == 0.0 {
                        f32::MAX
                    } else {
                        occlude_factor[idx2].max(nn_2.distance / djk)
                    }
                }
            }
            curr_alpha *= 1.2;
        }
    }

    fn prune_neighbors(
        &self,
        vid: usize,
        pruned_list: &mut Vec<usize>,
        params_r: &DiskANNParamsInternal,
        scratch: &mut nn_query_scratch::InMemoryQueryScratch,
        data: &AlignedDataStore,
    ) {
        let pool = &mut scratch.pool;
        if pool.len() == 0 {
            pruned_list.clear();
            return;
        }
        let range = params_r.params_e.indexing_range;
        let alpha = params_r.params_e.indexing_alpha;
        let maxc = params_r.params_e.indexing_maxc;

        pool.sort_by(|a, b| a.cmp(b));
        // println!("pool: {:?} | vid: {:?}", pool, vid);
        pruned_list.clear();
        pruned_list.reserve(range);
        self.occlude_list(vid, pool, alpha, range, maxc, pruned_list, params_r, data);
        debug_assert!(
            pruned_list.len() <= range,
            "pruned_list: {:?}, range: {:?}",
            pruned_list.len(),
            range
        );

        if params_r.saturate_graph && alpha > 1.0 {
            for i in 0..pool.len() {
                if pruned_list.len() >= range {
                    break;
                }
                if !pruned_list.contains(&pool[i].vid) && pool[i].vid != vid {
                    pruned_list.push(pool[i].vid)
                }
            }
        }
    }

    fn inter_insert(
        &self,
        vid: usize,
        pruned_list: &mut Vec<usize>,
        params_r: &DiskANNParamsInternal,
        scratch: &mut nn_query_scratch::InMemoryQueryScratch,
        data: &AlignedDataStore,
    ) {
        debug_assert!(!pruned_list.is_empty(), "inter_insert:: vid: {:?}", vid);
        let range = params_r.params_e.indexing_range;
        for des in pruned_list.iter() {
            debug_assert!(*des < params_r.params_e.max_points + params_r.num_frozen_pts);
            let mut copy_of_neighhbors: Vec<usize> = Vec::new();
            let mut prune_needed: bool = false;
            {
                let mut node_neighbors_f = self.final_graph[*des].write();
                if !node_neighbors_f.iter().any(|&i| i == vid) {
                    if node_neighbors_f.len() < ((GRAPH_SLACK_FACTOR * (range as f64)) as usize) {
                        node_neighbors_f.push(vid);
                        prune_needed = false;
                    } else {
                        copy_of_neighhbors = node_neighbors_f.clone();
                        copy_of_neighhbors.push(vid);
                        prune_needed = true;
                    }
                }
            }

            if prune_needed {
                // println!("prune is needed: {}", vid);
                let reserve_size: usize =
                    ((range as f64) * GRAPH_SLACK_FACTOR * 1.05).ceil() as usize;
                let mut dummy_visited: HashSet<usize> = HashSet::with_capacity(reserve_size);
                let mut dummy_pool: Vec<ann::INode> = Vec::with_capacity(reserve_size);
                for curr_nbr in copy_of_neighhbors.iter() {
                    if !dummy_visited.contains(curr_nbr) && *curr_nbr != *des {
                        let arr_a: &[f32] = &data.data[*des * params_r.aligned_dim
                            ..(*des * params_r.aligned_dim) + params_r.aligned_dim];
                        let arr_b: &[f32] = &data.data[*curr_nbr * params_r.aligned_dim
                            ..(*curr_nbr * params_r.aligned_dim) + params_r.aligned_dim];
                        let dist: f32 = TMetric::compare(arr_a, arr_b);
                        dummy_pool.push(ann::INode {
                            vid: *curr_nbr,
                            distance: dist,
                            flag: true,
                        });
                        dummy_visited.insert(*curr_nbr);
                    }
                }

                let mut new_out_neighbors: Vec<usize> = Vec::new();
                scratch.pool = dummy_pool;
                self.prune_neighbors(
                    *des,
                    &mut new_out_neighbors,
                    params_r,
                    scratch,
                    &self.data.read(),
                );
                {
                    let mut segment = self.final_graph[*des].write();
                    segment.clear();
                    for i in 0..new_out_neighbors.len() {
                        segment.push(new_out_neighbors[i]);
                    }
                }
            }
        }
    }

    fn search_for_point_and_prune(
        &self,
        vid: usize,
        pruned_list: &mut Vec<usize>,
        params_r: &DiskANNParamsInternal,
        scratch: &mut nn_query_scratch::InMemoryQueryScratch,
    ) {
        let mut init_ids: Vec<usize> = Vec::new();
        init_ids.push(params_r.start);
        self.iterate_to_fixed_point(
            QueryTarget::VId(vid),
            params_r,
            &mut init_ids,
            scratch,
            true,
            false,
        );

        {
            let pool = &mut scratch.pool;
            pool.retain(|&nn| nn.vid != vid);
        }
        debug_assert!(pruned_list.len() == 0);
        self.prune_neighbors(vid, pruned_list, params_r, scratch, &self.data.read());
        debug_assert!(
            pruned_list.len() != 0,
            "vid: {:?}, pool is of : {:?}, pruned_list is of: {:?}",
            vid,
            scratch.pool,
            pruned_list,
        );
    }

    fn search(&self, q: &[f32], k: usize) -> Result<Vec<ann::Node>, Box<dyn std::error::Error>> {
        let mut init_ids: Vec<usize> = Vec::new();
        let mut q_aligned;
        let params_r = self.params.read();
        if init_ids.len() == 0 {
            init_ids.push(params_r.start);
        }
        q_aligned = AlignedDataStore::new(params_r.aligned_dim, 1);
        q_aligned.data[0..q.len()].clone_from_slice(q);
        let mut scratch: nn_query_scratch::InMemoryQueryScratch =
            nn_query_scratch::InMemoryQueryScratch::new(&params_r);
        self.iterate_to_fixed_point(
            QueryTarget::Vector(&q_aligned.data),
            &params_r,
            &mut init_ids,
            &mut scratch,
            true,
            true,
        );
        let mut filtered: Vec<ann::Node> = Vec::with_capacity(k + 1);
        scratch.best_l_nodes.data.iter().try_for_each(|nn| {
            if filtered.len() > k {
                return ControlFlow::Break(nn);
            }
            if nn.vid != params_r.start && nn.distance != f32::MAX {
                filtered.push(ann::Node {
                    vid: nn.vid,
                    distance: nn.distance,
                    eid: [0u8; 16],
                });
            }
            ControlFlow::Continue(())
        });
        Ok(filtered)
    }

    fn build(&self) {
        {
            let mut paramsw = self.params.write();
            if paramsw.num_frozen_pts > 0 {
                paramsw.start = paramsw.params_e.max_points;
            } else {
                paramsw.start = self.calculate_entry_point(&paramsw, &self.data.read());
            }
        }
        self.generate_frozen_point();
        self.link();
        let mut max: usize = 0;
        let mut min: usize = usize::MAX;
        let mut total: usize = 0;
        let mut cnt: usize = 0;
        for idx in 0..self.params.read().nd {
            let fg_nbrs = self.final_graph[idx].read();
            max = cmp::max(max, fg_nbrs.len());
            min = cmp::min(min, fg_nbrs.len());
            total += fg_nbrs.len();
            if fg_nbrs.len() < 2 {
                cnt += 1;
            }
        }
        println!(
            "Index Build Stats\nmax: {} | avg: {} | nmin: {} | count(deg<2): {}",
            max,
            total / (self.params.read().nd + 1),
            min,
            cnt,
        );
    }

    fn reserve_location(&self) -> usize {}
    fn insert(&self, eid: EId, data: &[f32]) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }
    fn batch_insert(&self, eids: &[EId], data: &[f32]) -> Result<(), Box<dyn std::error::Error>> {
        {
            // verify that we have the correct arguments
            let mut params_w = self.params.write();
            let expected_len = eids.len() * params_w.aligned_dim;
            if data.len() != expected_len {
                return Err(Box::new(errors::ANNError::GenericError {
                    message: format!(
                        "points.len: {} !=  aligned_dim * eids.len: {}",
                        data.len(),
                        expected_len,
                    )
                    .into(),
                }));
            }
            // batch import the data to the AlignedDataStore
            params_w.nd = eids.len();
            let mut data_w = self.data.write();
            data_w.data[0..data.len()].copy_from_slice(&data[..]);
        }
        {
            // write to the locations
            let mut tl = self.tag_to_location.write();
            let mut lt = self.location_to_tag.write();
            for i in 0..eids.len() {
                tl.insert(eids[i], i);
                lt.insert(i, eids[i]);
            }
        }
        self.build();
        Ok(())
    }

    fn new(params: &DiskANNParams) -> Result<DiskANNV1Index<TMetric>, Box<dyn std::error::Error>> {
        let num_frozen_pts: usize = 1;
        let total_internal_points: usize = params.max_points + num_frozen_pts;
        let aligned_dim: usize = ann::round_up(params.dim.try_into().unwrap()) as usize;
        let data_len: usize = (aligned_dim + 1) * std::mem::size_of::<f32>();
        let paramsi: Arc<RwLock<DiskANNParamsInternal>> =
            Arc::new(RwLock::new(DiskANNParamsInternal {
                params_e: params.clone(),
                aligned_dim: aligned_dim,
                data_len: data_len,
                nd: params.max_points,
                neighbor_len: 0,
                node_size: 0,
                saturate_graph: false,
                num_frozen_pts: num_frozen_pts,
                total_internal_points: total_internal_points,
                start: params.max_points,
            }));

        let reserve_size =
            ((params.indexing_range as f64) * GRAPH_SLACK_FACTOR * 1.05).ceil() as usize;

        let shared: Vec<_> =
            std::iter::repeat_with(|| (RwLock::new(Vec::with_capacity(reserve_size))))
                .take(total_internal_points)
                .collect();
        let final_graph = Arc::new(shared);
        let empty_slots: Arc<RwLock<HashSet<usize>>> = Arc::new(RwLock::new(HashSet::new()));
        let delete_set: Arc<RwLock<HashSet<usize>>> = Arc::new(RwLock::new(HashSet::new()));

        let location_to_tag: Arc<RwLock<HashMap<usize, EId>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let tag_to_location: Arc<RwLock<HashMap<EId, usize>>> =
            Arc::new(RwLock::new(HashMap::new()));

        let data: Arc<RwLock<AlignedDataStore>>;

        {
            let params = paramsi.read();
            let mut lt = location_to_tag.write();
            lt.reserve(params.total_internal_points);
            let mut tl = tag_to_location.write();
            tl.reserve(params.total_internal_points);
            data = Arc::new(RwLock::new(AlignedDataStore::new(
                params.params_e.max_points + 1,
                params.aligned_dim,
            )));
        }
        let num_scratch_spaces = 12;
        let (s, r) = bounded(num_scratch_spaces);
        for _ in 0..num_scratch_spaces {
            let scratch = nn_query_scratch::InMemoryQueryScratch::new(&paramsi.read());
            s.send(scratch).unwrap();
        }

        let id_increment = Arc::new(AtomicUsize::new(0));
        let obj: DiskANNV1Index<TMetric> = DiskANNV1Index::<TMetric> {
            params: paramsi,
            data,
            final_graph,
            location_to_tag,
            tag_to_location,
            id_increment,
            delete_set,
            empty_slots,
            metric: PhantomData,
            s_scratch: s,
            r_scratch: r,
        };
        Ok(obj)
    }
}
