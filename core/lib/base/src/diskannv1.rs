use crate::ann;
use crate::ann::EId;
use crate::ann::Node;
use crate::av_store;
use crate::av_store::AlignedDataStore;
use crate::errors;
use crate::metric;

use parking_lot::RwLock;
use rayon::prelude::*;
use roaring::RoaringTreemap;
use serde::de::Expected;
use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;
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
    params_e: DiskANNParams,
    aligned_dim: usize,
    data_len: usize,
    num_frozen_pts: usize,
    total_internal_points: usize,
    nd: usize,
    neighbor_len: usize,
    node_size: usize,
    start: usize,
    saturate_graph: bool,
}

#[allow(dead_code)]
pub struct DiskANNV1Index<TMetric: metric::Metric> {
    params: Arc<RwLock<DiskANNParamsInternal>>,
    metric: PhantomData<TMetric>,

    data: Arc<RwLock<av_store::AlignedDataStore>>,
    final_graph: Arc<Vec<RwLock<Vec<usize>>>>,
    in_graph: Arc<Vec<RwLock<Vec<usize>>>>,
    location_to_tag: Arc<RwLock<HashMap<usize, EId>>>,
    tag_to_location: Arc<RwLock<HashMap<EId, usize>>>,

    id_increment: Arc<AtomicUsize>,
    delete_set: Arc<RwLock<HashSet<usize>>>,
    empty_slots: Arc<RwLock<HashSet<usize>>>,
}

const GRAPH_SLACK_FACTOR: f64 = 1.3;
const MAX_POINTS_FOR_USING_BITSET: usize = 10_000_000;
enum QueryTarget<'a> {
    VId(usize),
    Vector(&'a [f32]),
}

impl<TMetric> ann::ANNIndex for DiskANNV1Index<TMetric>
where
    TMetric: metric::Metric,
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
        unimplemented!()
    }
    fn insert(&self, eid: EId, data: &[f32]) -> Result<(), Box<dyn std::error::Error>> {
        unimplemented!()
    }
    fn search(&self, q: &[f32], k: usize) -> Result<Vec<ann::Node>, Box<dyn std::error::Error>> {
        unimplemented!()
    }
    fn save(&self) -> Result<(), Box<dyn std::error::Error>> {
        unimplemented!()
    }
}

impl<TMetric> DiskANNV1Index<TMetric>
where
    TMetric: metric::Metric,
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

        // let num_items_linked = Arc::new(AtomicUsize::new(1));
        // let num_items: usize = visit_order.len();
        // let _num_items_linked = num_items_linked.clone();
        // let _num_items = num_items.clone();
        // let _handle = thread::spawn(move || {
        //     let mut num_indexed = 0;
        //     while num_indexed < _num_items {
        //         thread::sleep(Duration::from_millis(500));
        //         num_indexed = _num_items_linked.load(std::sync::atomic::Ordering::Relaxed);
        //         DiskANNIndex::log_progress((num_indexed as f32) / (num_elements as f32))
        //     }
        // });

        // the initial link which occurs
        visit_order.par_iter_mut().for_each(|x| {
            let vid: usize = *x;
            let mut pool: Vec<ann::INode> =
                Vec::with_capacity(params_r.params_e.indexing_queue_size * 2);
            let mut visited: HashSet<usize> =
                HashSet::with_capacity(params_r.params_e.indexing_queue_size * 2);
            let mut des: Vec<usize> = Vec::with_capacity(
                ((params_r.params_e.indexing_range as f64) * GRAPH_SLACK_FACTOR) as usize,
            );
            let mut best_l_nodes: Vec<ann::INode> = vec![
                ann::INode {
                    vid: 0,
                    flag: false,
                    distance: 0.0
                };
                params_r.params_e.indexing_range + 1
            ];
            let mut inserted_into_pool_hs: HashSet<usize> = HashSet::new();
            let mut inserted_into_pool_rb: RoaringTreemap = RoaringTreemap::new();
            self.search_for_point_and_add_links(
                vid,
                &params_r,
                &mut pool,
                &mut visited,
                &mut des,
                &mut best_l_nodes,
                &mut inserted_into_pool_hs,
                &mut inserted_into_pool_rb,
            );
        });
        // optimizing the graph we just built
        visit_order.par_iter_mut().for_each(|x| {
            let node: usize = *x;
            let mut dummy_visited: HashSet<usize> = HashSet::new();
            let mut dummy_pool: Vec<ann::INode> = Vec::new();
            let mut new_out_neighbors: Vec<usize> = Vec::new();
            let mut node_neighbors_f = self.final_graph[node].write();
            let data = self.data.read();
            if node_neighbors_f.len() > params_r.params_e.indexing_range {
                for curr_nbr in node_neighbors_f.iter() {
                    if !dummy_visited.contains(curr_nbr) && *curr_nbr != node {
                        let arr_a: &[f32] = &data.data[*curr_nbr * params_r.aligned_dim
                            ..(*curr_nbr * params_r.aligned_dim) + params_r.aligned_dim];
                        let arr_b: &[f32] = &data.data[node * params_r.aligned_dim
                            ..(node * params_r.aligned_dim) + params_r.aligned_dim];
                        let nn = ann::INode {
                            vid: *curr_nbr,
                            distance: TMetric::compare(arr_a, arr_b),
                            flag: true,
                        };
                        dummy_pool.push(nn);
                        dummy_visited.insert(*curr_nbr);
                    }
                }
                self.prune_neighbors(
                    node,
                    &mut dummy_pool,
                    &mut new_out_neighbors,
                    &params_r,
                    &data,
                );
                node_neighbors_f.clear();
                node_neighbors_f.resize(new_out_neighbors.len(), 0);
                node_neighbors_f
                    .splice(..new_out_neighbors.len(), new_out_neighbors.iter().cloned());
            }
        });
    }

    // this maintains our sorted invariant using a binary search
    // lifted exactly from the original implementation
    fn insert_into_pool(
        &self,
        best_l_nodes: &mut Vec<ann::INode>,
        k: usize,
        nn: ann::INode,
    ) -> usize {
        let mut left: usize = 0;
        let mut right: usize = k - 1;
        if best_l_nodes[left].distance > nn.distance {
            best_l_nodes.insert(left, nn);
            return left;
        }
        if best_l_nodes[right].distance < nn.distance {
            best_l_nodes[k] = nn;
            return k;
        }
        while right > 1 && left < right - 1 {
            let mid: usize = (left + right) / 2;
            if best_l_nodes[mid].distance > nn.distance {
                right = mid;
            } else {
                left = mid;
            }
        }
        while left > 0 {
            if best_l_nodes[left].distance < nn.distance {
                break;
            }
            if best_l_nodes[left].vid == nn.vid {
                return k + 1;
            }
            left -= 1;
        }
        if best_l_nodes[left].vid == nn.vid || best_l_nodes[right].vid == nn.vid {
            return k + 1;
        }
        best_l_nodes.insert(right, nn);
        return right;
    }

    fn iterate_to_fixed_point(
        &self,
        vid: usize,
        params_r: &DiskANNParamsInternal,
        pool: &mut Vec<ann::INode>,
        visited: &mut HashSet<usize>,
        des: &mut Vec<usize>,
        best_l_nodes: &mut Vec<ann::INode>,
        inserted_into_pool_hs: &mut HashSet<usize>,
        inserted_into_pool_rb: &mut RoaringTreemap,
        // ----
        init_ids: &mut Vec<usize>,
        ret_frozen: bool,
        is_search: bool,
        l_size: usize,
        target: QueryTarget,
    ) -> (usize, usize) {
        // pool -> expanded_nodes_info
        // visited -> expanded_nodes_ids
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
        for i in 0..best_l_nodes.len() {
            best_l_nodes[i].distance = std::f32::INFINITY;
        }
        if !is_search {
            pool.clear();
            visited.clear();
            des.clear();
        }

        let mut l: usize = 0;
        let fast_iterate: bool =
            params_r.params_e.max_points + params_r.num_frozen_pts <= MAX_POINTS_FOR_USING_BITSET;
        // TOOD(infrawhispers) - cpp implementation needs us to reserve space in the bitset  - there is no
        // reserve for the rust impl, look into the bitmap details
        // if (fast_iterate) {
        //     auto total_num_points = _max_points + _num_frozen_pts;
        //     if (inserted_into_pool_bs.size() < total_num_points) {
        //       // hopefully using 2X will reduce the number of allocations.
        //       auto resize_size = 2 * total_num_points > MAX_POINTS_FOR_USING_BITSET
        //                              ? MAX_POINTS_FOR_USING_BITSET
        //                              : 2 * total_num_points;
        //       inserted_into_pool_bs.resize(resize_size);
        //     }
        //   }
        for id in init_ids.iter() {
            if *id >= params_r.params_e.max_points + params_r.num_frozen_pts {
                // TODO(infrawhispers) - get rid of this panic, this isn't *really* needed
                panic!(
                    "out of range loc: {} max_points: {}, num_frozen_pts: {}",
                    *id, params_r.params_e.max_points, params_r.num_frozen_pts
                );
            }
            let arr_a: &[f32] = &data.data
                [*id * params_r.aligned_dim..(*id * params_r.aligned_dim) + params_r.aligned_dim];
            let nn = ann::INode {
                vid: *id,
                distance: TMetric::compare(arr_a, arr_b),
                flag: true,
            };
            if fast_iterate {
                if !inserted_into_pool_rb.contains((*id).try_into().unwrap()) {
                    inserted_into_pool_rb.insert((*id).try_into().unwrap());
                    best_l_nodes[l] = nn;
                    l += 1
                }
            } else {
                if !inserted_into_pool_hs.contains(id) {
                    inserted_into_pool_hs.insert(*id);
                    best_l_nodes[l] = nn;
                    l += 1
                }
            }
            if l == l_size {
                break;
            }
        }
        best_l_nodes.sort();
        let mut k: usize = 0;
        let hops: usize = 0;
        let mut cmps: usize = 0;
        while k < l {
            let mut nk: usize = l;
            if best_l_nodes[k].flag {
                best_l_nodes[k].flag = false;
                let n_vid: usize = best_l_nodes[k].vid;
                if !(best_l_nodes[k].vid != params_r.start
                    && params_r.num_frozen_pts > 0
                    && !ret_frozen)
                {
                    if !is_search {
                        pool.push(best_l_nodes[k]);
                        visited.insert(n_vid);
                    }
                }
                des.clear();
                {
                    let _final_graph = self.final_graph[n_vid].read();
                    for m in 0.._final_graph.len() {
                        debug_assert!(
                            _final_graph[m]
                                <= params_r.params_e.max_points + params_r.num_frozen_pts,
                            "out of range edge: {edge} | found at vertex: {vertex}",
                            edge = _final_graph[m],
                            vertex = n_vid,
                        );
                        des.push(_final_graph[m]);
                    }
                }
                for m in 0..des.len() {
                    let id: usize = des[m];
                    let id_is_missing = if fast_iterate {
                        !inserted_into_pool_rb.contains(id.try_into().unwrap())
                    } else {
                        !inserted_into_pool_hs.contains(&id)
                    };
                    if id_is_missing {
                        if fast_iterate {
                            inserted_into_pool_rb.insert(id.try_into().unwrap());
                        } else {
                            inserted_into_pool_hs.insert(id);
                        }

                        if m + 1 < des.len() {
                            let nextn: usize = des[m + 1];
                            // TODO(infrawhispers) - implement a prefetch that is architecture
                            // dependent, we probably want something like:
                            // TMetric::prefetch(vec: &[f32])
                        }
                        cmps += 1;
                        let arr_a: &[f32] = &data.data[id * params_r.aligned_dim
                            ..(id * params_r.aligned_dim) + params_r.aligned_dim];
                        let dist: f32 = TMetric::compare(arr_a, arr_b);
                        if dist >= best_l_nodes[l - 1].distance && l == l_size {
                            continue;
                        }
                        let nn_new: ann::INode = ann::INode {
                            vid: id,
                            distance: dist,
                            flag: true,
                        };
                        let r: usize = self.insert_into_pool(best_l_nodes, l, nn_new);
                        if l < l_size {
                            l += 1;
                        }
                        if r < nk {
                            nk = r;
                        }
                    }
                }
                if nk <= k {
                    k = nk;
                } else {
                    k += 1;
                }
            } else {
                k += 1;
            }
        }
        (hops, cmps)
    }

    fn get_expanded_nodes(
        &self,
        vid: usize,
        params_r: &DiskANNParamsInternal,
        pool: &mut Vec<ann::INode>,
        visited: &mut HashSet<usize>,
        des: &mut Vec<usize>,
        best_l_nodes: &mut Vec<ann::INode>,
        inserted_into_pool_hs: &mut HashSet<usize>,
        inserted_into_pool_rb: &mut RoaringTreemap,
        // ----
        init_ids: &mut Vec<usize>,
    ) {
        let vid_idx: usize = params_r.aligned_dim * vid;
        if init_ids.len() == 0 {
            init_ids.push(params_r.start);
        }
        let _ = self.iterate_to_fixed_point(
            vid,
            params_r,
            pool,
            visited,
            des,
            best_l_nodes,
            inserted_into_pool_hs,
            inserted_into_pool_rb,
            init_ids,
            true,
            false,
            params_r.params_e.indexing_queue_size,
            QueryTarget::VId(vid),
        );
    }

    fn occlude_list(
        &self,
        vid: usize,
        pool: &mut Vec<ann::INode>,
        alpha: f32,
        degree: usize,
        maxc: usize,
        result: &mut Vec<ann::INode>,
        params_r: &DiskANNParamsInternal,
        data: &AlignedDataStore,
    ) {
        if pool.len() == 0 {
            return;
        }
        if pool.len() > maxc {
            pool.resize(
                maxc,
                ann::INode {
                    vid: 0,
                    distance: 0.0,
                    flag: false,
                },
            );
        }
        let mut occlude_factor = vec![0.0; pool.len()];
        let mut curr_alpha = 1.0;
        while curr_alpha <= alpha && result.len() < degree {
            for (iter1, ele1) in pool.iter().enumerate() {
                if result.len() >= degree {
                    break;
                }
                if occlude_factor[iter1] > curr_alpha {
                    continue;
                }
                occlude_factor[iter1] = f32::MAX;
                result.push(*ele1);
                let space = iter1 + 1;
                if space >= pool.len() {
                    continue;
                }
                for (iter2, ele2) in pool.iter().enumerate() {
                    if iter2 < iter1 + 1 {
                        continue;
                    }
                    if occlude_factor[iter2] > alpha {
                        continue;
                    }
                    let arr_a: &[f32] = &data.data[ele2.vid * params_r.aligned_dim
                        ..(ele2.vid * params_r.aligned_dim) + params_r.aligned_dim];
                    let arr_b: &[f32] = &data.data[ele1.vid * params_r.aligned_dim
                        ..(ele1.vid * params_r.aligned_dim) + params_r.aligned_dim];
                    let djk: f32 = TMetric::compare(arr_a, arr_b);
                    if djk == 0.0 {
                        occlude_factor[iter2] = f32::MAX;
                    } else {
                        occlude_factor[iter2] =
                            occlude_factor[iter2].max(pool[iter2].distance / djk);
                    }
                    // TODO(infrawhispers) - there is more work needed to handle the
                    // inner product calculation
                }
            }
            curr_alpha *= 1.2;
        }
    }

    fn prune_neighbors_impl(
        &self,
        vid: usize,
        pool: &mut Vec<ann::INode>,
        range: usize,
        max_candidate_size: usize,
        alpha: f32,
        pruned_list: &mut Vec<usize>,
        params_r: &DiskANNParamsInternal,
        data: &AlignedDataStore,
    ) {
        if pool.len() == 0 {
            return;
        }
        pool.sort_by(|a, b| a.cmp(b));
        let mut result = Vec::with_capacity(range);
        self.occlude_list(
            vid,
            pool,
            alpha,
            range,
            max_candidate_size,
            &mut result,
            params_r,
            data,
        );
        pruned_list.clear();
        for nn in result.into_iter() {
            if nn.vid != vid {
                pruned_list.push(nn.vid);
            }
        }
        if params_r.saturate_graph && alpha > 1.0 {
            let mut i: usize = 0;
            while i < pool.len() && pruned_list.len() < range {
                // this should be a binary search over the items!
                let node = pool.iter().find(|&&x| x.vid == pool[i].vid);
                match node {
                    Some(_x) => {
                        pruned_list.push(pool[i].vid);
                    }
                    None => (),
                }
                i += 1;
            }
        }
    }

    fn prune_neighbors(
        &self,
        vid: usize,
        pool: &mut Vec<ann::INode>,
        pruned_list: &mut Vec<usize>,
        params_r: &DiskANNParamsInternal,
        data: &AlignedDataStore,
    ) {
        self.prune_neighbors_impl(
            vid,
            pool,
            params_r.params_e.indexing_range,
            params_r.params_e.indexing_maxc,
            params_r.params_e.indexing_alpha,
            pruned_list,
            params_r,
            data,
        )
    }

    fn inter_insert(
        &self,
        vid: usize,
        pruned_list: &mut Vec<usize>,
        params_r: &DiskANNParamsInternal,
        data: &AlignedDataStore,
    ) {
        let update_in_graph = true;
        let range = params_r.params_e.indexing_range;
        for des in pruned_list.iter() {
            let mut copy_of_neighhbors: Vec<usize> = Vec::new();
            let mut add: bool = false;
            let mut prune_needed: bool = false;
            {
                let mut node_neighbors_f = self.final_graph[*des].write();
                add = !node_neighbors_f.iter().any(|&i| i == vid);
                if add {
                    if node_neighbors_f.len() < ((GRAPH_SLACK_FACTOR * (range as f64)) as usize) {
                        node_neighbors_f.push(vid);
                        if update_in_graph {
                            let mut node_neighbors_i = self.in_graph[vid].write();
                            node_neighbors_i.push(*des);
                        }
                        prune_needed = false;
                    } else {
                        copy_of_neighhbors = node_neighbors_f.clone();
                        prune_needed = true;
                    }
                }
            }
            if prune_needed {
                copy_of_neighhbors.push(vid);
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
                self.prune_neighbors(
                    *des,
                    &mut dummy_pool,
                    &mut new_out_neighbors,
                    params_r,
                    data,
                );
                {
                    if update_in_graph {
                        let mut out_nbrs: Vec<usize> = Vec::new();
                        {
                            let node_neighbors_f = self.final_graph[*des].read();
                            out_nbrs = node_neighbors_f.clone();
                        }
                        for out_nbr in out_nbrs.iter() {
                            let mut out_neighbors_i = self.in_graph[*out_nbr].write();
                            out_neighbors_i.retain(|&x| x != *des);
                        }
                    }
                    // write out new graph structure for node: des
                    let mut node_neighbors_f = self.final_graph[*des].write();
                    node_neighbors_f.clear();
                    for new_nbr in new_out_neighbors.iter() {
                        node_neighbors_f.push(*new_nbr);
                    }
                    if update_in_graph {
                        // write out new graph structure for things hitting des
                        for new_nbr in new_out_neighbors.iter() {
                            let mut node_neighbors_i = self.in_graph[*new_nbr].write();
                            node_neighbors_i.push(*des);
                        }
                    }
                }
            }
        }
    }

    fn search_for_point_and_add_links(
        &self,
        vid: usize,
        params_r: &DiskANNParamsInternal,
        pool: &mut Vec<ann::INode>,
        visited: &mut HashSet<usize>,
        des: &mut Vec<usize>,
        best_l_nodes: &mut Vec<ann::INode>,
        inserted_into_pool_hs: &mut HashSet<usize>,
        inserted_into_pool_rb: &mut RoaringTreemap,
    ) {
        let mut init_ids: Vec<usize> = Vec::new();
        self.get_expanded_nodes(
            vid,
            params_r,
            pool,
            visited,
            des,
            best_l_nodes,
            inserted_into_pool_hs,
            inserted_into_pool_rb,
            &mut init_ids,
        );

        {
            let delete_set_r = self.delete_set.read();
            let mut idx: usize = 0;
            while idx < pool.len() {
                if pool[idx].vid == vid {
                    pool.remove(idx);
                    visited.remove(&vid);
                    idx = if idx == 0 { 0 } else { idx - 1 }
                } else if delete_set_r.get(&pool[idx].vid) != None {
                    pool.remove(idx);
                    visited.remove(&vid);
                    idx = if idx == 0 { 0 } else { idx - 1 }
                } else {
                    idx += 1;
                }
            }
        }
        let data = self.data.read();
        let mut pruned_list: Vec<usize> = Vec::new();
        self.prune_neighbors(vid, pool, &mut pruned_list, params_r, &data);
        {
            // necessary to support the eager delete operation
            let indices = self.final_graph[vid].write().clone();
            for idx in indices.iter() {
                let mut node_neighbors_i = self.in_graph[*idx].write();
                node_neighbors_i.retain(|&x| x != vid);
            }
        }
        {
            let mut node_neighbors_f = self.final_graph[vid].write();
            node_neighbors_f.clear();
            node_neighbors_f.shrink_to_fit();
            node_neighbors_f.reserve(
                ((params_r.params_e.indexing_range as f64) * GRAPH_SLACK_FACTOR * 1.05).ceil()
                    as usize,
            );
            for link in pruned_list.iter() {
                node_neighbors_f.push(*link);
                let mut node_neighbors_i = self.in_graph[*link].write();
                node_neighbors_i.push(vid);
            }
        }
        self.inter_insert(vid, &mut pruned_list, params_r, &data)
    }

    fn update_in_graph(&self) {
        for (idx, nbrs) in self.final_graph.iter().enumerate() {
            let fg_nbrs = nbrs.write();
            let mut ig_nbrs = self.in_graph[idx].write();
            ig_nbrs.clear();
            ig_nbrs.resize(fg_nbrs.len(), 0);
            ig_nbrs.splice(..fg_nbrs.len(), fg_nbrs.iter().cloned());
        }
    }

    fn build(&self) {
        {
            let mut paramsw = self.params.write();
            if paramsw.num_frozen_pts > 0 {
                paramsw.start = paramsw.params_e.max_points;
            } else {
                // let mut dataw = self.data.write().unwrap();
                // paramsw.start = DiskANNIndex::calculate_entry_point(&paramsw, &mut dataw);
            }
        }
        // let paramsr = self.params.read();
        self.generate_frozen_point();
        self.link();
        self.update_in_graph();
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
        let shared_in: Vec<_> =
            std::iter::repeat_with(|| (RwLock::new(Vec::with_capacity(reserve_size))))
                .take(total_internal_points)
                .collect();

        let in_graph = Arc::new(shared_in);
        let empty_slots: Arc<RwLock<HashSet<usize>>> = Arc::new(RwLock::new(HashSet::new()));
        let delete_set: Arc<RwLock<HashSet<usize>>> = Arc::new(RwLock::new(HashSet::new()));

        let location_to_tag: Arc<RwLock<HashMap<usize, EId>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let tag_to_location: Arc<RwLock<HashMap<EId, usize>>> =
            Arc::new(RwLock::new(HashMap::new()));

        let mut data: Arc<RwLock<AlignedDataStore>> =
            Arc::new(RwLock::new(AlignedDataStore::new(0, 0)));

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
        let id_increment = Arc::new(AtomicUsize::new(0));
        let obj: DiskANNV1Index<TMetric> = DiskANNV1Index::<TMetric> {
            params: paramsi,
            data,
            final_graph,
            in_graph,
            location_to_tag,
            tag_to_location,
            id_increment,
            delete_set,
            empty_slots,
            metric: PhantomData,
        };
        Ok(obj)
    }
}
