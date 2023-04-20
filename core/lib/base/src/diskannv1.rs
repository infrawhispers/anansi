use crate::ann;
use crate::ann::EId;
use crate::av_store;
use crate::av_store::AlignedDataStore;
use crate::errors;
use crate::metric;
use crate::nn_query_scratch;
use crate::nn_queue;
use crate::scalar_quantizer;

use anyhow::bail;
use crossbeam_channel::{bounded, Receiver, Sender};
use parking_lot::RwLock;
use rand::distributions::{Distribution, Standard};
use rand::thread_rng;
use rayon::prelude::*;
use roaring::RoaringTreemap;
use std::cmp;
use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;
use std::ops::ControlFlow;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;
use std::thread::available_parallelism;
use std::time::Instant;
use std::{thread, time};

#[derive(Debug, Clone, Copy)]
pub struct DiskANNParams {
    pub dim: usize,
    pub max_points: usize,
    pub indexing_threads: Option<usize>,
    pub indexing_range: usize,
    pub indexing_queue_size: usize,
    pub indexing_maxc: usize,
    pub indexing_alpha: f32,
    pub maintenance_period_millis: u64,
}

#[allow(dead_code)]
pub struct DiskANNParamsInternal {
    pub params_e: DiskANNParams,
    pub aligned_dim: usize,
    pub num_frozen_pts: usize,
    pub nd: usize,
    pub start: usize,
    pub saturate_graph: bool,
}

#[allow(dead_code)]
pub struct DiskANNV1Index<TMetric, TVal: ann::ElementVal> {
    params: Arc<RwLock<DiskANNParamsInternal>>,
    metric: PhantomData<TMetric>,

    data: Arc<RwLock<av_store::AlignedDataStore<TVal>>>,
    in_graph: Arc<Vec<RwLock<HashSet<usize>>>>, // all vids going *into* the node: vid -> {vid_1, vid_2, ..}
    final_graph: Arc<Vec<RwLock<Vec<usize>>>>, // all vids that are closest: vid -> [vid_1, vid_2...]
    location_to_tag: Arc<RwLock<HashMap<usize, EId>>>,
    tag_to_location: Arc<RwLock<HashMap<EId, usize>>>,

    id_increment: Arc<AtomicUsize>,
    delete_set: Arc<RwLock<HashSet<usize>>>,
    empty_slots: Arc<RwLock<HashSet<usize>>>,
    s_scratch: Sender<nn_query_scratch::InMemoryQueryScratch>,
    r_scratch: Receiver<nn_query_scratch::InMemoryQueryScratch>,
    quantizer: Arc<scalar_quantizer::ScalarQuantizer>,
    // indexing_pool: rayon::ThreadPool,
    // handle: Option<thread::JoinHandle<()>>,
}

const GRAPH_SLACK_FACTOR: f64 = 1.3;
const MAX_POINTS_FOR_USING_BITSET: usize = 10_000_000;
enum QueryTarget<'a, TVal: ann::ElementVal> {
    VId(usize),
    Vector(&'a [TVal]),
}

impl<TMetric, TVal> ann::ANNIndex for DiskANNV1Index<TMetric, TVal>
where
    TVal: ann::ElementVal,
    TMetric: metric::Metric<TVal>,
{
    type Val = TVal;
    fn new(params: &ann::ANNParams) -> anyhow::Result<DiskANNV1Index<TMetric, TVal>> {
        let diskann_params: &DiskANNParams = match params {
            ann::ANNParams::Flat { params: _ } => {
                unreachable!("incorrect params passed for construction")
            }
            ann::ANNParams::DiskANN { params } => params,
        };
        DiskANNV1Index::new(diskann_params)
    }
    fn insert(&self, eids: &[EId], data: ann::Points<TVal>) -> anyhow::Result<()> {
        self.insert(eids, data)
    }
    fn delete(&self, eids: &[EId]) -> anyhow::Result<()> {
        self.delete(eids)
    }
    fn search(&self, q: ann::Points<TVal>, k: usize) -> anyhow::Result<Vec<ann::Node>> {
        self.search(q, k)
    }
    fn save(&self) -> anyhow::Result<()> {
        unimplemented!()
    }
}

impl<TMetric, TVal> DiskANNV1Index<TMetric, TVal>
where
    TVal: ann::ElementVal,
    TMetric: metric::Metric<TVal>,
{
    fn set_start_point_at_random(&self, radius: f32) {
        let params_r = self.params.read();
        let mut rng = thread_rng();
        let v: Vec<f64> = Standard
            .sample_iter(&mut rng)
            .take(params_r.aligned_dim)
            .collect();

        let mut norm_sq: f64 = 0.0f64;
        let real_vec: Vec<f64> = v
            .iter()
            .map(|r| {
                norm_sq += r * r;
                *r
            })
            .collect();
        let norm: f64 = f64::sqrt(norm_sq);
        let start_vec: Vec<TVal> = real_vec
            .iter()
            .map(|p| {
                TVal::from_f64(*p * (radius as f64) / norm).expect("unexpected cast issue fr: f32")
            })
            .collect();
        // copy the data into our vector!
        let mut data_w = self.data.write();
        let idx_s: usize = params_r.start * params_r.aligned_dim;
        let idx_e: usize = idx_s + params_r.aligned_dim;
        data_w.data[idx_s..idx_e].copy_from_slice(&start_vec[..]);
    }

    fn calculate_entry_point(
        &self,
        paramsr: &DiskANNParamsInternal,
        data: &AlignedDataStore<TVal>,
    ) -> usize {
        // let paramsr = self.params.read();
        // let data = self.data.read();

        let aligned_dim: usize = paramsr.aligned_dim;
        let nd: usize = paramsr.nd;
        let mut center: Vec<TVal> = vec![Default::default(); aligned_dim];
        for i in 0..nd {
            for j in 0..aligned_dim {
                center[j] += data.data[i * aligned_dim + j];
            }
        }
        // for j in 0..aligned_dim {
        //     center[j] /= nd as TVal;
        // }
        let mut distances: Vec<f32> = vec![0.0; nd];
        distances.par_iter_mut().enumerate().for_each(|(i, x)| {
            let s_idx: usize = i * aligned_dim;
            let e_idx: usize = s_idx + aligned_dim;
            let vec_t = &data.data[s_idx..e_idx];
            let mut dist: f32 = 0.0;
            for j in 0..aligned_dim {
                dist += ((center[j] - vec_t[j]) * (center[j] - vec_t[j]))
                    .to_f32()
                    .expect("hello");
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

    #[allow(dead_code)]
    fn generate_frozen_point(&self) {
        let params_r = self.params.read();
        let mut data_w = self.data.write();
        // {idx_f, idx_t} are indices into an array
        let idx_f: usize = self.calculate_entry_point(&params_r, &data_w) * params_r.aligned_dim;
        let idx_t: usize = params_r.start * params_r.aligned_dim;
        ann::copy_within_a_slice(&mut data_w.data, idx_f, idx_t, params_r.aligned_dim);
    }
    fn link(&self, visit_order: Vec<usize>, do_prune: bool) {
        let params_r = self.params.read();
        // TODO(infrawhispers) - WASM + Rayon on M1 macs is broken!
        visit_order.par_iter().for_each(|vid| {
            let mut pruned_list: Vec<usize> = Vec::new();
            // let mut scratch: nn_query_scratch::InMemoryQueryScratch =
            //     nn_query_scratch::InMemoryQueryScratch::new(&params_r);
            let mut scratch: nn_query_scratch::InMemoryQueryScratch =
                self.r_scratch.recv().unwrap();
            scratch.clear();
            self.search_for_point_and_prune(*vid, &mut pruned_list, &params_r, &mut scratch);
            self.update_graph_nbrs(*vid, pruned_list.clone(), true);
            // let old_segment: Vec<usize>;
            // {
            //     let mut segment = self.final_graph[*vid].write();
            //     old_segment = segment.clone();
            //     segment.clear();
            //     for i in 0..pruned_list.len() {
            //         segment.push(pruned_list[i]);
            //     }
            // }
            // {
            //     // populate_in_graph - the regexp to find instances this must
            //     // be tied to is: "final_graph\[.*\].*write\(\)"
            //     old_segment.iter().for_each(|nbr_vid| {
            //         self.in_graph[*nbr_vid].write().remove(vid);
            //     });r
            //         self.in_graph[*nbr_vid].write().insert(*vid);
            //     });
            // }
            self.inter_insert(
                *vid,
                &mut pruned_list,
                &params_r,
                &mut scratch,
                &self.data.read(),
            );
            self.s_scratch.send(scratch).unwrap();
        });
        if !do_prune {
            return;
        }
        let data = &self.data.read();
        // self.indexing_pool.install(|| {
        visit_order.par_iter().for_each(|curr_vid| {
            let should_prune: bool;
            let graph_copy: Vec<usize>;
            {
                let nbrs = self.final_graph[*curr_vid].read();
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
                        let arr_a: &[TVal] = &data.data[*curr_vid * params_r.aligned_dim
                            ..(*curr_vid * params_r.aligned_dim) + params_r.aligned_dim];
                        let arr_b: &[TVal] = &data.data[nbr_vid * params_r.aligned_dim
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
                self.update_graph_nbrs(*curr_vid, new_out_neighbors, true);
                // let old_segment: Vec<usize>;
                // {
                //     let mut segment = self.final_graph[*curr_vid].write();
                //     old_segment = segment.clone();
                //     segment.clear();
                //     for i in 0..new_out_neighbors.len() {
                //         segment.push(new_out_neighbors[i]);
                //     }
                // }
                // {
                //     // populate_in_graph - the regexp to find instances this must
                //     // be tied to is: "final_graph\[.*\].*write\(\)"
                //     old_segment.iter().for_each(|nbr_vid| {
                //         self.in_graph[*nbr_vid].write().remove(curr_vid);
                //     });
                //     new_out_neighbors.iter().for_each(|nbr_vid| {
                //         self.in_graph[*nbr_vid].write().insert(*curr_vid);
                //     });
                // }
                self.s_scratch.send(scratch).unwrap();
            }
        })
        // })
    }

    fn iterate_to_fixed_point(
        &self,
        target: QueryTarget<TVal>,
        params_r: &DiskANNParamsInternal,
        init_ids: &mut Vec<usize>,
        scratch: &mut nn_query_scratch::InMemoryQueryScratch,
        ret_frozen: bool,
        is_search: bool,
    ) -> (usize, usize) {
        let data = self.data.read();
        // pull out the slice we are comparing against
        let arr_b: &[TVal];
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
        debug_assert!(id_scratch.len() == 0, "scratch space must be cleared");
        debug_assert!(expanded_nodes.len() == 0, "scratch space must be cleared");
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
                let arr_a: &[TVal] = &data.data[*nn_id * params_r.aligned_dim
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
                let arr_a: &[TVal] = &data.data[*nn * params_r.aligned_dim
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
        data: &AlignedDataStore<TVal>,
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
                    let arr_a: &[TVal] = &data.data[nn.vid * params_r.aligned_dim
                        ..(nn.vid * params_r.aligned_dim) + params_r.aligned_dim];
                    let arr_b: &[TVal] = &data.data[pool[idx2].vid * params_r.aligned_dim
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
        data: &AlignedDataStore<TVal>,
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
        data: &AlignedDataStore<TVal>,
    ) {
        debug_assert!(!pruned_list.is_empty(), "inter_insert:: vid: {:?}", vid);
        let range = params_r.params_e.indexing_range;
        for des in pruned_list.iter() {
            debug_assert!(*des < params_r.params_e.max_points + params_r.num_frozen_pts);
            let mut copy_of_neighhbors: Vec<usize>;
            let mut prune_needed: bool = false;
            let mut vids_added = Vec::with_capacity(range);
            {
                let mut node_neighbors_f = self.final_graph[*des].write();
                copy_of_neighhbors = node_neighbors_f.clone();
                if !node_neighbors_f.iter().any(|&i| i == vid) {
                    if node_neighbors_f.len() < ((GRAPH_SLACK_FACTOR * (range as f64)) as usize) {
                        node_neighbors_f.push(vid);
                        copy_of_neighhbors.push(vid);
                        vids_added.push(vid);
                        prune_needed = false;
                    } else {
                        copy_of_neighhbors.push(vid);
                        prune_needed = true;
                    }
                }
            }
            if !prune_needed {
                // populate_in_graph - the regexp to find instances this must
                // be tied to is: "final_graph\[.*\].*write\(\)"
                vids_added.iter().for_each(|nbr_vid| {
                    self.in_graph[*nbr_vid].write().insert(vid);
                });
            } else {
                // println!("prune is needed: {}", vid);
                let reserve_size: usize =
                    ((range as f64) * GRAPH_SLACK_FACTOR * 1.05).ceil() as usize;
                let mut dummy_visited: HashSet<usize> = HashSet::with_capacity(reserve_size);
                let mut dummy_pool: Vec<ann::INode> = Vec::with_capacity(reserve_size);
                for curr_nbr in copy_of_neighhbors.iter() {
                    if !dummy_visited.contains(curr_nbr) && *curr_nbr != *des {
                        let arr_a: &[TVal] = &data.data[*des * params_r.aligned_dim
                            ..(*des * params_r.aligned_dim) + params_r.aligned_dim];
                        let arr_b: &[TVal] = &data.data[*curr_nbr * params_r.aligned_dim
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
                self.update_graph_nbrs(*des, new_out_neighbors, true);
                // let old_segment: Vec<usize>;
                // {
                //     let mut segment = self.final_graph[*des].write();
                //     old_segment = segment.clone();
                //     segment.clear();
                //     for i in 0..new_out_neighbors.len() {
                //         segment.push(new_out_neighbors[i]);
                //     }
                // }
                // {
                //     // populate_in_graph - the regexp to find instances this must
                //     // be tied to is: "final_graph\[.*\].*write\(\)"
                //     old_segment.iter().for_each(|nbr_vid| {
                //         self.in_graph[*nbr_vid].write().remove(des);
                //     });
                //     new_out_neighbors.iter().for_each(|nbr_vid| {
                //         self.in_graph[*nbr_vid].write().insert(*des);
                //     });
                // }
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
            let delete_set = self.delete_set.read();
            pool.retain(|&nn| nn.vid != vid);
            pool.retain(|&nn| !delete_set.contains(&nn.vid));
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

    fn search(&self, q: ann::Points<TVal>, k: usize) -> anyhow::Result<Vec<ann::Node>> {
        let mut init_ids: Vec<usize> = Vec::new();
        let params_r = self.params.read();
        if init_ids.len() == 0 {
            init_ids.push(params_r.start);
        }

        let data: &[TVal];
        let quantize_result: Vec<TVal>;
        match q {
            ann::Points::QuantizerIn { vals } => {
                let (res, _) = self.quantizer.quantize_arr(vals);
                quantize_result = res
                    .iter()
                    .map(|x| TVal::from_u8(*x).expect("unable to coerce to u8"))
                    .collect();
                data = &quantize_result[..]
            }
            ann::Points::Values { vals } => data = vals,
        }
        if data.len() > params_r.aligned_dim {
            bail!(
                "query dim: {} > aligned_dim: {}",
                data.len(),
                params_r.aligned_dim
            );
        }
        let padded_vector: Vec<TVal>;
        let padded_points: &[TVal];
        match ann::pad_and_preprocess::<TVal, TMetric>(data, data.len(), params_r.aligned_dim) {
            Some(vec) => {
                padded_vector = vec;
                padded_points = &padded_vector[..]
            }
            None => padded_points = data,
        }

        let mut q_aligned: AlignedDataStore<TVal> =
            AlignedDataStore::<TVal>::new(params_r.aligned_dim, 1);
        q_aligned.data[0..padded_points.len()].clone_from_slice(padded_points);
        if TMetric::uses_preprocessor() {
            match TMetric::pre_process(&q_aligned.data[0..q_aligned.data.len()]) {
                Some(vec) => {
                    q_aligned.data[0..padded_points.len()].copy_from_slice(&vec[0..vec.len()]);
                }
                None => {}
            }
        }

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
        let mapping = self.location_to_tag.read();

        scratch.best_l_nodes.data.iter().try_for_each(|nn| {
            if filtered.len() >= k {
                return ControlFlow::Break(nn);
            }
            let eid;
            match mapping.get(&nn.vid) {
                Some(val) => eid = *val,
                None => return ControlFlow::Continue(()),
            }
            if nn.vid != params_r.start && !(nn.distance.is_infinite() || nn.distance == f32::MAX) {
                filtered.push(ann::Node {
                    vid: nn.vid,
                    distance: nn.distance,
                    eid: eid,
                });
            }
            ControlFlow::Continue(())
        });
        Ok(filtered)
    }

    #[allow(dead_code)]
    fn build(&self) {
        let mut visit_order: Vec<usize>;
        {
            let params_r = self.params.read();
            // if params_w.num_frozen_pts > 0 {
            //     params_w.start = params_w.params_e.max_points;
            // } else {
            //     params_w.start = self.calculate_entry_point(&params_w, &self.data.read());
            // }
            visit_order = Vec::with_capacity(params_r.nd + params_r.num_frozen_pts);
            for vid in 0..params_r.nd {
                visit_order.push(vid);
            }
            visit_order.push(params_r.params_e.max_points);
        }
        self.generate_frozen_point();
        let start = Instant::now();
        self.link(visit_order, true);
        println!("link time: {:?}", start.elapsed());
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

    fn reserve_locations(&self, count: usize) -> anyhow::Result<Vec<usize>> {
        let mut vids: Vec<usize> = Vec::with_capacity(count);
        let mut empty_slots_w = self.empty_slots.write();
        if !empty_slots_w.is_empty() {
            empty_slots_w.iter().try_for_each(|vid| {
                if vids.len() > count {
                    return ControlFlow::Break(vid);
                }
                vids.push(*vid);
                return ControlFlow::Continue(());
            });
            // don't forget to actually remove them!
            vids.iter().for_each(|vid| {
                empty_slots_w.remove(vid);
            });
        }
        // fully satisified by the empty_slots
        if vids.len() == count {
            return Ok(vids);
        }
        let cnt_needed: usize = count - vids.len();
        let params_r = self.params.read();
        if cnt_needed + self.id_increment.load(std::sync::atomic::Ordering::SeqCst)
            > params_r.params_e.max_points
        {
            // push back our vids that we previously had before we roll everything back
            vids.iter().for_each(|vid| {
                empty_slots_w.insert(*vid);
            });
            bail!(
                "reservation ({}) would lead to > max_points ({}) in index, resizing not supported as yet",
                count,
                params_r.params_e.max_points,
            );
        }
        let vid_s = self
            .id_increment
            .fetch_add(cnt_needed, std::sync::atomic::Ordering::SeqCst);

        (vid_s..vid_s + cnt_needed).for_each(|vid| {
            vids.push(vid);
        });
        Ok(vids)
    }

    #[inline(always)]
    fn update_graph_nbrs(
        &self,
        vid: usize,
        new_nbrs: impl IntoIterator<Item = usize>,
        update_in_graph: bool,
    ) {
        let mut old_segment: Option<(Vec<usize>, Vec<usize>)> = None;
        {
            let mut segment = self.final_graph[vid].write();
            if update_in_graph {
                let mut nbrs_clone = Vec::new();
                let segment_clone = segment.clone();
                segment.clear();
                new_nbrs.into_iter().for_each(|nbr_vid| {
                    segment.push(nbr_vid);
                    nbrs_clone.push(nbr_vid);
                });
                old_segment = Some((segment_clone, nbrs_clone));
            } else {
                segment.clear();
                new_nbrs.into_iter().for_each(|nbr_vid| {
                    segment.push(nbr_vid);
                })
            }
        }
        match old_segment {
            Some((old_vids, nbrs_copy)) => {
                // populate_in_graph - the regexp to find instances this must
                // be tied to is: "final_graph\[.*\].*write\(\)"
                old_vids.iter().for_each(|nbr_vid| {
                    self.in_graph[*nbr_vid].write().remove(&vid);
                });
                nbrs_copy.into_iter().for_each(|nbr_vid| {
                    self.in_graph[nbr_vid].write().insert(vid);
                });
            }
            None => {}
        }
    }

    // this is a *lazy* delete, the items just gets marked as removed!
    fn delete(&self, eids: &[EId]) -> anyhow::Result<()> {
        let mut delete_set = self.delete_set.write();
        let mut tag_to_location = self.tag_to_location.write();
        let mut location_to_tag = self.location_to_tag.write();
        eids.iter().for_each(|eid| match tag_to_location.get(eid) {
            Some(vid) => {
                delete_set.insert(*vid);
                location_to_tag.remove(&vid);
                tag_to_location.remove(eid);
            }
            None => {}
        });
        Ok(())
    }

    fn process_delete(
        &self,
        vid: usize,
        delete_set: &HashSet<usize>,
        params_r: &DiskANNParamsInternal,
        data: &AlignedDataStore<TVal>,
    ) {
        // TODO(infrawhispers) - what should this be set as?
        let mut expanded_nodes_set: Vec<usize> = Vec::with_capacity(10);

        // first pool all the items that we care about
        {
            self.final_graph[vid].read().iter().for_each(|nbr_vid| {
                if !delete_set.contains(nbr_vid) && *nbr_vid != vid {
                    expanded_nodes_set.push(*nbr_vid);
                }
            });
        }
        if expanded_nodes_set.len() < params_r.params_e.indexing_range {
            self.update_graph_nbrs(vid, expanded_nodes_set, true);
            return;
        }
        let mut expanded_nbrs_vec: Vec<ann::INode> = Vec::with_capacity(expanded_nodes_set.len());
        expanded_nodes_set.iter().for_each(|nbr_vid| {
            let arr_a: &[TVal] = &data.data[*nbr_vid * params_r.aligned_dim
                ..(*nbr_vid * params_r.aligned_dim) + params_r.aligned_dim];
            let arr_b: &[TVal] = &data.data
                [vid * params_r.aligned_dim..(vid * params_r.aligned_dim) + params_r.aligned_dim];
            expanded_nbrs_vec.push(ann::INode {
                vid: *nbr_vid,
                distance: TMetric::compare(arr_a, arr_b),
                flag: false,
            })
        });
        expanded_nbrs_vec.sort();
        let mut pruned_list: Vec<usize> = Vec::with_capacity(params_r.params_e.indexing_range);
        self.occlude_list(
            vid,
            &mut expanded_nbrs_vec,
            params_r.params_e.indexing_alpha,
            params_r.params_e.indexing_range,
            params_r.params_e.indexing_maxc,
            &mut pruned_list,
            params_r,
            data,
        );
        self.update_graph_nbrs(vid, pruned_list, true);
    }

    fn consolidate_deletes(&self) {
        let old_delete_set: HashSet<usize>;
        {
            old_delete_set = self.delete_set.read().clone();
        }
        let start = Instant::now();
        let params_r = self.params.read();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let data = self.data.read();
        pool.install(|| {
            // build a map of all vids we care about, these are vids that _link into_
            // the nodes that we are about to remove!
            let mut vids_to_visit: HashSet<usize> =
                HashSet::with_capacity(params_r.params_e.indexing_range * old_delete_set.len());
            old_delete_set.iter().for_each(|vid| {
                let nbrs_linked = self.in_graph[*vid].read();
                nbrs_linked
                    .iter()
                    .for_each(|nbr_vid| match old_delete_set.get(nbr_vid) {
                        Some(_) => {}
                        None => {
                            vids_to_visit.insert(*nbr_vid);
                        }
                    });
            });
            vids_to_visit.par_iter().for_each(|vid| {
                self.process_delete(*vid, &old_delete_set, &params_r, &data);
            });
        });
        let mut delete_set = self.delete_set.write();
        let mut empty_slots = self.empty_slots.write();
        let mut tag_to_location = self.tag_to_location.write();
        let mut location_to_tag = self.location_to_tag.write();

        // let empty_slots = self.
        old_delete_set.iter().for_each(|vid| {
            delete_set.remove(vid);
            empty_slots.insert(*vid);
            match location_to_tag.remove_entry(vid) {
                Some((_, eid)) => {
                    tag_to_location.remove(&eid);
                }
                None => {}
            }
        });
        println!("consolidate_delete time: {:?}", start.elapsed());
    }

    fn insert(&self, eids: &[EId], p: ann::Points<TVal>) -> anyhow::Result<()> {
        let data: &[TVal];
        match p {
            ann::Points::QuantizerIn { .. } => {
                unreachable!("incorrect params passed for construction")
            }
            ann::Points::Values { vals } => data = vals,
        }

        // we assume everything is good!
        {
            let params_r = self.params.read();
            let expected_len = eids.len() * params_r.aligned_dim;
            if data.len() != expected_len {
                bail!(
                    "points.len: {} !=  aligned_dim * eids.len: {}",
                    data.len(),
                    expected_len
                );
            }
        }
        let vids = self.reserve_locations(eids.len())?;
        debug_assert!(
            vids.len() == eids.len(),
            "could not get enough vids to map to the eid database",
        );
        let data_processed;
        let mut preprocess_scratch: Vec<TVal>;

        if TMetric::uses_preprocessor() {
            let params_r = self.params.read();
            preprocess_scratch = vec![Default::default(); data.len()];
            for idx in 0..vids.len() {
                let idx_s_fr = idx * params_r.aligned_dim;
                let idx_e_fr = idx_s_fr + params_r.aligned_dim;
                match TMetric::pre_process(&data[idx_s_fr..idx_e_fr]) {
                    Some(vec) => {
                        preprocess_scratch[idx_s_fr..idx_e_fr].copy_from_slice(&vec[0..vec.len()]);
                    }
                    None => {
                        preprocess_scratch[idx_s_fr..idx_e_fr]
                            .copy_from_slice(&data[idx_s_fr..idx_e_fr]);
                    }
                }
            }
            data_processed = &preprocess_scratch[0..preprocess_scratch.len()];
        } else {
            data_processed = data;
        }
        {
            // hold the lock and plop the data into our datastore - switching
            // to segments should allow us to not block _all_ readers during
            // this operation
            let mut data_w = self.data.write();
            let params_r = self.params.read();
            for idx in 0..vids.len() {
                // params copying into our aligned data source
                let idx_s_to = vids[idx] * params_r.aligned_dim;
                let idx_e_to = idx_s_to + params_r.aligned_dim;
                // params copying from our supplied slice
                let idx_s_fr = idx * params_r.aligned_dim;
                let idx_e_fr = idx_s_fr + params_r.aligned_dim;
                data_w.data[idx_s_to..idx_e_to]
                    .copy_from_slice(&data_processed[idx_s_fr..idx_e_fr]);
            }
        }
        {
            // insert items into the tag_to_location + location_to_tag
            let mut tl = self.tag_to_location.write();
            let mut lt = self.location_to_tag.write();
            for idx in 0..eids.len() {
                tl.insert(eids[idx], vids[idx]);
                lt.insert(vids[idx], eids[idx]);
            }
        }
        // finally run the insertion process

        self.link(vids, false);
        Ok(())
    }
    #[allow(dead_code)]
    fn batch_insert(&self, eids: &[EId], data: &[TVal]) -> Result<(), Box<dyn std::error::Error>> {
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

    pub fn maintain(&self) {
        let maintenance_period: u64;
        {
            maintenance_period = self.params.read().params_e.maintenance_period_millis;
        }
        loop {
            thread::sleep(time::Duration::from_millis(maintenance_period));
            self.consolidate_deletes();
        }
    }

    fn new(params: &DiskANNParams) -> anyhow::Result<DiskANNV1Index<TMetric, TVal>> {
        let num_frozen_pts: usize = 1;
        let total_internal_points: usize = params.max_points + num_frozen_pts;
        let aligned_dim: usize = ann::round_up(params.dim.try_into().unwrap()) as usize;
        let mut params_e = params.clone();
        match params_e.indexing_threads {
            Some(_) => {}
            None => {
                let suggested = available_parallelism();
                match suggested {
                    Ok(suggestion) => params_e.indexing_threads = Some(suggestion.into()),
                    Err(_) => params_e.indexing_threads = Some(4),
                }
            }
        }
        // let indexing_pool: rayon::ThreadPool;
        // match params_e.indexing_threads {
        //     Some(cnt_threads) => {
        //         match rayon::ThreadPoolBuilder::new()
        //             .num_threads(cnt_threads)
        //             .build()
        //         {
        //             Ok(pool) => indexing_pool = pool,
        //             Err(err) => {
        //                 bail!("unable to build the threading pool {:?}", err);
        //             }
        //         }
        //     }
        //     None => {
        //         bail!("params_e.indexing_threads unexpectedly is zero");
        //     }
        // }

        let paramsi: Arc<RwLock<DiskANNParamsInternal>> =
            Arc::new(RwLock::new(DiskANNParamsInternal {
                params_e: params.clone(),
                aligned_dim: aligned_dim,
                nd: params.max_points,
                saturate_graph: false,
                num_frozen_pts: num_frozen_pts,
                start: params.max_points,
            }));

        let reserve_size =
            ((params.indexing_range as f64) * GRAPH_SLACK_FACTOR * 1.05).ceil() as usize;

        let shared: Vec<_> =
            std::iter::repeat_with(|| (RwLock::new(Vec::with_capacity(reserve_size))))
                .take(total_internal_points)
                .collect();
        let shared_in: Vec<_> =
            std::iter::repeat_with(|| (RwLock::new(HashSet::with_capacity(reserve_size))))
                .take(total_internal_points)
                .collect();

        let final_graph = Arc::new(shared);
        let in_graph = Arc::new(shared_in);
        let empty_slots: Arc<RwLock<HashSet<usize>>> = Arc::new(RwLock::new(HashSet::new()));
        let delete_set: Arc<RwLock<HashSet<usize>>> = Arc::new(RwLock::new(HashSet::new()));

        let location_to_tag: Arc<RwLock<HashMap<usize, EId>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let tag_to_location: Arc<RwLock<HashMap<EId, usize>>> =
            Arc::new(RwLock::new(HashMap::new()));

        let data: Arc<RwLock<AlignedDataStore<TVal>>>;

        {
            let params = paramsi.read();
            let mut lt = location_to_tag.write();
            lt.reserve(total_internal_points);
            let mut tl = tag_to_location.write();
            tl.reserve(total_internal_points);
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
        let obj: DiskANNV1Index<TMetric, TVal> = DiskANNV1Index::<TMetric, TVal> {
            params: paramsi,
            data,
            in_graph,
            final_graph,
            location_to_tag,
            tag_to_location,
            id_increment,
            delete_set,
            empty_slots,
            metric: PhantomData,
            // indexing_pool: indexing_pool,
            s_scratch: s,
            r_scratch: r,
            quantizer: Arc::new(scalar_quantizer::ScalarQuantizer::new(0.99)?),
        };
        // any additional setup that we need to do _on the instance_
        obj.set_start_point_at_random(5.0);
        Ok(obj)
    }
}
