use crate::ann;
use crate::ann::EId;
use crate::ann::Node;
use crate::av_store;
use crate::av_store::AlignedDataStore;
use crate::errors;
use crate::metric;

use parking_lot::RwLock;
use serde::de::Expected;
use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;

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
    fn batch_insert(&self, eids: &[EId], data: &[f32]) -> Result<bool, Box<dyn std::error::Error>> {
        unimplemented!()
    }
    fn insert(&self, eid: EId, data: &[f32]) -> Result<bool, Box<dyn std::error::Error>> {
        unimplemented!()
    }
    fn search(&self, q: &[f32], k: usize) -> Result<Vec<Node>, Box<dyn std::error::Error>> {
        unimplemented!()
    }
    fn save(&self) -> Result<bool, Box<dyn std::error::Error>> {
        unimplemented!()
    }
}

impl<TMetric> DiskANNV1Index<TMetric>
where
    TMetric: metric::Metric,
{
    fn batch_insert(&self, eids: &[EId], data: &[f32]) -> Result<bool, Box<dyn std::error::Error>> {
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
            params_w.nd = eids.len();
            let mut data_w = self.data.write();
            // data_w.iter().enumerate().for_each(|(i, x)| {
            //     data_w.data[i] = *x;
            // });
            // println!("alignment: {}", data_w.data.as_ptr().align_offset(32));
            // println!("data size: {}", data_w.data.len());
        }
        Ok(true)
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
                num_frozen_pts: num_frozen_pts,
                total_internal_points: total_internal_points,
                start: params.max_points,
            }));

        let shared: Vec<_> = std::iter::repeat_with(|| (RwLock::new(Vec::new())))
            .take(total_internal_points)
            .collect();
        let final_graph = Arc::new(shared);
        let shared_in: Vec<_> = std::iter::repeat_with(|| (RwLock::new(Vec::new())))
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
                params.aligned_dim,
                (params.params_e.max_points + 1) * params.aligned_dim,
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
