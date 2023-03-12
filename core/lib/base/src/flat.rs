use log::warn;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::marker::PhantomData;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;

use parking_lot::RwLock;

use super::av_store;
use super::errors;
use crate::ann;
use crate::ann::EId;
use crate::metric;

#[derive(Debug, Clone, Copy)]
pub struct FlatParams {
    pub dim: usize,
    pub segment_size_kb: usize,
}

#[derive(Debug)]
#[allow(dead_code)]
pub struct FlatIndex<TMetric: metric::Metric> {
    pub metric: PhantomData<TMetric>,
    pub(crate) params: Arc<FlatParams>,
    datastore: Arc<RwLock<HashMap<usize, RwLock<av_store::AlignedDataStore>>>>,
    id_increment: Arc<AtomicUsize>,
    delete_set: Arc<RwLock<HashSet<usize>>>,
    eid_to_vid: Arc<RwLock<HashMap<EId, usize>>>,
    vid_to_eid: Arc<RwLock<HashMap<usize, EId>>>,
    v_per_segment: usize,
    aligned_dim: usize,
}

impl<TMetric> ann::ANNIndex for FlatIndex<TMetric>
where
    TMetric: metric::Metric,
{
    fn new(params: &ann::ANNParams) -> Result<FlatIndex<TMetric>, Box<dyn std::error::Error>> {
        let flat_params: &FlatParams = match params {
            ann::ANNParams::Flat { params } => params,
            _ => {
                unreachable!("incorrect params passed for construction")
            }
        };
        FlatIndex::new(flat_params)
    }
    fn batch_insert(&self, eids: &[EId], data: &[f32]) -> Result<bool, Box<dyn std::error::Error>> {
        if (data.len() % eids.len()) != 0 {
            return Err(Box::new(errors::ANNError::GenericError {
                message: format!(
                    "data is not an exact multiple of eids - data_len: {} | eids_len: {}",
                    eids.len(),
                    data.len(),
                )
                .into(),
            }));
        }
        let data_len = data.len() / eids.len();
        for (idx, eid) in eids.iter().enumerate() {
            match self.insert(*eid, &data[idx * data_len..idx * data_len + data_len]) {
                Ok(_) => {}
                Err(err) => {
                    return Err(err);
                }
            }
        }
        Ok(true)
    }

    fn insert(&self, eid: EId, data: &[f32]) -> Result<bool, Box<dyn std::error::Error>> {
        self.insert(eid, data)
    }

    fn search(&self, q: &[f32], k: usize) -> Result<Vec<ann::Node>, Box<dyn std::error::Error>> {
        self.search(q, k)
    }

    fn save(&self) -> Result<bool, Box<dyn std::error::Error>> {
        Ok(true)
    }
}

impl<TMetric> FlatIndex<TMetric>
where
    TMetric: metric::Metric,
{
    pub fn new(params: &FlatParams) -> Result<FlatIndex<TMetric>, Box<dyn std::error::Error>> {
        let aligned_dim = ann::round_up(params.dim as u32) as usize;
        let mut v_per_segment: usize = (params.segment_size_kb * 1000) / (aligned_dim as usize * 4);
        if v_per_segment < 1000 {
            v_per_segment = 1000
        }
        let id_increment = Arc::new(AtomicUsize::new(0));
        let delete_set: Arc<RwLock<HashSet<usize>>> = Arc::new(RwLock::new(HashSet::new()));
        let segement_0 = RwLock::new(av_store::AlignedDataStore::new(
            aligned_dim.try_into().unwrap(),
            v_per_segment,
        ));
        let datastore = Arc::new(RwLock::new(HashMap::new()));
        {
            datastore.write().insert(0, segement_0);
        }
        let location_to_tag: Arc<RwLock<HashMap<usize, ann::EId>>> =
            Arc::new(RwLock::new(HashMap::with_capacity(v_per_segment * 2)));
        let tag_to_location: Arc<RwLock<HashMap<ann::EId, usize>>> =
            Arc::new(RwLock::new(HashMap::with_capacity(v_per_segment * 2)));
        Ok(FlatIndex {
            metric: PhantomData,
            params: Arc::new(params.clone()),
            datastore: datastore,
            id_increment: id_increment,
            delete_set: delete_set,
            eid_to_vid: tag_to_location,
            vid_to_eid: location_to_tag,

            v_per_segment: v_per_segment,
            aligned_dim: aligned_dim,
        })
    }
    pub fn insert(&self, eid: ann::EId, point: &[f32]) -> Result<bool, Box<dyn std::error::Error>> {
        if point.len() > self.aligned_dim {
            return Err(Box::new(errors::ANNError::GenericError {
                message: format!(
                    "point dim: {} > aligned_dim: {}",
                    point.len(),
                    self.aligned_dim,
                )
                .into(),
            }));
        }
        let mut padded_point: &[f32] = point;
        let mut data: Vec<f32>;
        if point.len() < self.aligned_dim {
            data = vec![0.0; self.aligned_dim];
            data[0..point.len()].copy_from_slice(point);
            padded_point = &data[..]
        }
        let vid: usize;
        {
            match self.eid_to_vid.read().get(&eid) {
                Some(vid_existing) => vid = *vid_existing,
                None => {
                    vid = self
                        .id_increment
                        .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                }
            }
        }
        let segment_id = (vid / self.v_per_segment) as usize;
        let need_new_segment: bool;
        {
            match self.datastore.read().get(&segment_id) {
                Some(_segment) => need_new_segment = false,
                None => need_new_segment = true,
            }
        }
        if need_new_segment {
            let new_segment = RwLock::new(av_store::AlignedDataStore::new(
                self.aligned_dim.try_into().unwrap(),
                self.v_per_segment.try_into().unwrap(),
            ));
            self.datastore.write().insert(segment_id, new_segment);
        }
        match self.datastore.read().get(&segment_id) {
            None => {
                return Err(Box::new(errors::ANNError::GenericError {
                    message: "unexpectedly, the segement is missing when it was previously inserted - bailing".to_string(),
                }))
            }
            Some(segment) => segment
                .write()
                .aligned_insert((vid % self.v_per_segment).try_into().unwrap(), padded_point),
        }
        self.eid_to_vid.write().insert(eid, vid);
        self.vid_to_eid.write().insert(vid, eid);

        Ok(true)
    }
    pub fn delete(&self, eid: ann::EId) -> Result<bool, Box<dyn std::error::Error>> {
        let vid: usize;
        {
            match self.eid_to_vid.read().get(&eid) {
                Some(vid_val) => vid = *vid_val,
                None => {
                    return Ok(false);
                }
            }
        }
        // key is in our mapping so do the delete set dance
        self.eid_to_vid.write().remove(&eid);
        self.vid_to_eid.write().remove(&vid);
        self.delete_set.write().insert(vid);
        Ok(true)
    }
    pub fn search(
        &self,
        q: &[f32],
        k: usize,
    ) -> Result<Vec<ann::Node>, Box<dyn std::error::Error>> {
        if q.len() > self.aligned_dim {
            return Err(Box::new(errors::ANNError::GenericError {
                message: format!("query dim: {} > aligned_dim: {}", q.len(), self.aligned_dim)
                    .into(),
            }));
        }
        // maybe we want to keep around a bunch of these in a pool we can pull from?
        let mut res_heap: BinaryHeap<ann::Node> = BinaryHeap::with_capacity(k + 1);
        let mut q_aligned = av_store::AlignedDataStore::new(self.aligned_dim, 1);
        q_aligned.data[..q.len()].copy_from_slice(&q[..]);
        // we should probably use rayon over segments and have multiple vectors
        // in a given segment
        self.datastore
            .read()
            .iter()
            .for_each(|(segment_id, vec_store)| {
                // we are now in a single segment!
                let data = vec_store.read();
                for i in 0..data.num_vectors {
                    let eid: ann::EId;
                    let vid = *segment_id * (self.v_per_segment as usize) + i;
                    match self.vid_to_eid.read().get(&vid) {
                        Some(val) => eid = val.clone(),
                        None => {
                            continue;
                        }
                    }

                    let arr_a: &[f32] = &q_aligned.data[..];
                    let arr_b: &[f32] =
                        &data.data[i * self.aligned_dim..(i * self.aligned_dim) + self.aligned_dim];
                    let dist = TMetric::compare(arr_a, arr_b, arr_a.len());

                    res_heap.push(ann::Node {
                        vid: vid,
                        eid: eid,
                        distance: dist,
                    });
                    if res_heap.len() > k {
                        res_heap.pop().unwrap();
                    }
                }
            });
        let mut res_vec = Vec::with_capacity(res_heap.len());
        while !res_heap.is_empty() {
            let neighbor_rev = res_heap.pop().unwrap();
            res_vec.push(neighbor_rev);
        }
        res_vec.reverse();
        Ok(res_vec)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn insert_large_wpadding() {
        let params = FlatParams {
            dim: 128,
            segment_size_kb: 512,
        };
        let index = FlatIndex::<metric::MetricL2>::new(&params).unwrap();
        // insert the first 1000 vectors into the index (nb: 1000 per segment)
        for i in 0..1000 {
            let mut id = [0u8; 16];
            let id_str = i.clone().to_string();
            let id_bytes = id_str.as_bytes();
            id[0..id_bytes.len()].copy_from_slice(&id_bytes[..]);
            let point = vec![1.2 * (i as f32); 128];
            match index.insert(id, &point[..]) {
                Ok(res) => {
                    assert_eq!(true, res);
                }
                Err(_) => {
                    panic!("error should not be thrown on insert");
                }
            }
        }
        // then insert the 1001 vector - this will cause a new segement to be created
        let mut id = [0u8; 16];
        let id_str = "1000".to_string();
        let id_bytes = id_str.as_bytes();
        id[0..id_bytes.len()].copy_from_slice(&id_bytes[..]);
        let point = vec![1.2 * (1000 as f32); 128];
        match index.insert(id, &point[..]) {
            Ok(res) => {
                assert_eq!(true, res);
            }
            Err(_) => {
                panic!("error should not be thrown on insert");
            }
        }
        // now craft a search that includes the vector in the external segment!
        let point_search = vec![1.2 * (10001 as f32); 128];
        match index.search(&point_search, 1) {
            Ok(res) => {
                let result: Vec<ann::EId> = res.iter().map(|x| (x.eid)).collect();
                assert_eq!(vec![id], result);
            }
            Err(_) => {
                panic!("error should not be thrown on search");
            }
        }
    }
    #[test]
    fn insert_small_delete() {
        let params = FlatParams {
            dim: 31,
            segment_size_kb: 512,
        };
        let index = FlatIndex::<metric::MetricL2>::new(&params).unwrap();
        for i in 0..10 {
            let mut id = [0u8; 16];
            id[0] = i;
            let point = vec![1.2 * (i as f32); 31];
            match index.insert(id, &point[..]) {
                Ok(res) => {
                    assert_eq!(true, res);
                }
                Err(_) => {
                    panic!("error should not be thrown on insert");
                }
            }
        }
        let point_search = vec![0.4; 31];
        match index.search(&point_search, 1) {
            Ok(res) => {
                let result: Vec<ann::EId> = res.iter().map(|x| (x.eid)).collect();
                assert_eq!(vec![[0u8; 16]], result);
            }
            Err(_) => {
                panic!("error should not be thrown on search");
            }
        }
        // now delete the one we care about
        match index.delete([0u8; 16]) {
            Ok(_) => {}
            Err(_) => {
                panic!("no err should throw on deletion of existing item");
            }
        }
        // then issue the search and ensure the old mapping does
        // not come up
        match index.search(&point_search, 1) {
            Ok(res) => {
                let result: Vec<ann::EId> = res.iter().map(|x| (x.eid)).collect();
                let mut expected_id = [0u8; 16];
                expected_id[0] = 1;
                assert_eq!(vec![expected_id], result);
            }
            Err(_) => {
                panic!("error should not be thrown on search");
            }
        }
    }
    #[test]
    fn insert_small_wpadding() {
        let params = FlatParams {
            dim: 31,
            segment_size_kb: 512,
        };
        let index = FlatIndex::<metric::MetricL2>::new(&params).unwrap();
        for i in 0..10 {
            let mut id = [0u8; 16];
            id[0] = i;
            let point = vec![1.2 * (i as f32); 31];
            match index.insert(id, &point[..]) {
                Ok(res) => {
                    assert_eq!(true, res);
                }
                Err(_) => {
                    panic!("error should not be thrown on insert");
                }
            }
        }
        let point_search = vec![0.4; 31];
        match index.search(&point_search, 1) {
            Ok(res) => {
                let result: Vec<ann::EId> = res.iter().map(|x| (x.eid)).collect();
                assert_eq!(vec![[0u8; 16]], result);
            }
            Err(_) => {
                panic!("error should not be thrown on search");
            }
        }
    }
    #[test]
    fn insert_small_npadding() {
        let params = FlatParams {
            dim: 128,
            segment_size_kb: 512,
        };
        let index = FlatIndex::<metric::MetricL2>::new(&params).unwrap();
        for i in 0..10 {
            let mut id = [0u8; 16];
            id[0] = i;
            let point = vec![100.0 * (i as f32); 128];
            match index.insert(id, &point[..]) {
                Ok(res) => {
                    assert_eq!(true, res);
                }
                Err(_) => {
                    panic!("error should not be thrown on insert");
                }
            }
        }
        let point_search = vec![0.1; 128];
        match index.search(&point_search, 1) {
            Ok(res) => {
                let result: Vec<ann::EId> = res.iter().map(|x| (x.eid)).collect();
                assert_eq!(vec![[0u8; 16]], result);
            }
            Err(_) => {
                panic!("error should not be thrown on search");
            }
        }
    }
}
