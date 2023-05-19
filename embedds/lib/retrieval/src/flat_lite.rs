use anyhow::bail;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::io::{Read, Write};
use std::marker::PhantomData;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;

extern crate rmp_serde as rmps;
use rmps::Serializer;

use super::av_store;
use crate::ann;
use crate::ann::EId;
use crate::metric;
use crate::scalar_quantizer;

#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
pub struct FlatLiteParams {
    pub dim: usize,
    pub segment_size_kb: usize,
}

#[derive(Debug)]
#[allow(dead_code)]
pub struct FlatIndex<TMetric, TVal: for<'de> ann::ElementVal<'de>> {
    pub metric: PhantomData<TMetric>,
    pub(crate) params: Arc<FlatLiteParams>,
    datastore: Arc<RwLock<HashMap<usize, RwLock<av_store::AlignedDataStore<TVal>>>>>,
    id_increment: Arc<AtomicUsize>,
    delete_set: Arc<RwLock<HashSet<usize>>>,
    eid_to_vid: Arc<RwLock<HashMap<EId, usize>>>,
    vid_to_eid: Arc<RwLock<HashMap<usize, EId>>>,
    v_per_segment: usize,
    aligned_dim: usize,
    quantizer: Arc<scalar_quantizer::ScalarQuantizer>,
}

// this is used for serialization + deserialization purposes
#[derive(Debug, Serialize, Deserialize)]
struct FlatIndexSerialized {
    metric: Vec<u8>,
    params: Vec<u8>,
    datastore: HashMap<usize, Vec<u8>>,
    id_increment: Vec<u8>,
    delete_set: Vec<u8>,
    eid_to_vid: Vec<u8>,
    vid_to_eid: Vec<u8>,
    v_per_segment: Vec<u8>,
    aligned_dim: Vec<u8>,
    quantizer: Vec<u8>,
}

impl<TMetric, TVal> ann::ANNIndex for FlatIndex<TMetric, TVal>
where
    TVal: for<'a> ann::ElementVal<'a>,
    TMetric: metric::Metric<TVal>,
{
    type Val = TVal;
    fn new(params: &ann::ANNParams) -> anyhow::Result<FlatIndex<TMetric, TVal>> {
        let flat_params: &FlatLiteParams = match params {
            ann::ANNParams::FlatLite { params } => params,
            _ => {
                unreachable!("incorrect params passed for construction")
            }
        };
        FlatIndex::new_core(flat_params)
    }

    fn insert(&self, eids: &[EId], points: ann::Points<TVal>) -> anyhow::Result<()> {
        self.insert(eids, points)
    }
    fn delete(&self, eids: &[EId]) -> anyhow::Result<()> {
        self.delete(eids)
    }

    fn search(&self, q: ann::Points<TVal>, k: usize) -> anyhow::Result<Vec<ann::Node>> {
        self.search(q, k)
    }

    fn save(&self) -> anyhow::Result<()> {
        Ok(())
    }
    fn delete_index(&self) -> anyhow::Result<()> {
        Ok(())
    }
}

impl<TMetric, TVal> FlatIndex<TMetric, TVal>
where
    TVal: for<'de> ann::ElementVal<'de>,
    TMetric: metric::Metric<TVal>,
{
    pub fn deserialize_(reader: impl Read) -> anyhow::Result<FlatIndex<TMetric, TVal>> {
        let obj: FlatIndexSerialized = rmp_serde::decode::from_read(reader)?;
        let metric: String = rmp_serde::from_slice(&obj.metric)?;
        let params: Arc<FlatLiteParams> = rmp_serde::from_slice(&obj.params)?;
        if metric != TMetric::as_string() {
            bail!(
                "incorrect deserialization: provided metric: {:?} expected metric: {:?}",
                TMetric::as_string(),
                obj.metric,
            )
        }
        // rebuild the datastore
        let datastore: Arc<RwLock<HashMap<usize, RwLock<av_store::AlignedDataStore<TVal>>>>> =
            Arc::new(RwLock::new(HashMap::new()));
        {
            let mut datastore_w = datastore.write();
            let datastore_d: &HashMap<usize, Vec<u8>> = &obj.datastore;
            for (seg_id, data_pl) in datastore_d.iter() {
                let data_b: RwLock<av_store::AlignedDataStore<TVal>> =
                    rmp_serde::from_slice(&(data_pl))?;
                datastore_w.insert(*seg_id, data_b);
            }
        }
        // additional items that are necessary to build the struct
        let id_increment: Arc<AtomicUsize> = rmp_serde::from_slice(&obj.id_increment)?;
        let delete_set: Arc<RwLock<HashSet<usize>>> = rmp_serde::from_slice(&obj.delete_set)?;
        let eid_to_vid: Arc<RwLock<HashMap<EId, usize>>> = rmp_serde::from_slice(&obj.eid_to_vid)?;
        let vid_to_eid: Arc<RwLock<HashMap<usize, EId>>> = rmp_serde::from_slice(&obj.vid_to_eid)?;
        let aligned_dim: usize = rmp_serde::from_slice(&obj.aligned_dim)?;
        let quantizer: Arc<scalar_quantizer::ScalarQuantizer> =
            rmp_serde::from_slice(&obj.quantizer)?;
        let v_per_segment: usize = rmp_serde::from_slice(&obj.v_per_segment)?;
        Ok(FlatIndex {
            metric: PhantomData,
            params: params,
            datastore: datastore,
            id_increment: id_increment,
            delete_set: delete_set,
            eid_to_vid: eid_to_vid,
            vid_to_eid: vid_to_eid,
            v_per_segment: v_per_segment,
            aligned_dim: aligned_dim,
            quantizer: quantizer,
        })
    }

    // allows for the serialization | derserilization of a type from the rust
    pub fn serialize_(self, mut writer: impl Write) -> anyhow::Result<()> {
        let mut serializer = Serializer::new(&mut writer);
        let metric_b = rmp_serde::to_vec(&TMetric::as_string())?;
        let params_b = rmp_serde::to_vec(&self.params)?;
        let mut datastore_b: HashMap<usize, Vec<u8>> = HashMap::new();
        // now get the serialization for each particular segment
        let data_write = self.datastore.write();
        for (seg_id, data_pl) in data_write.iter() {
            let data_b = rmp_serde::to_vec(&(data_pl))?;
            datastore_b.insert(*seg_id, data_b);
        }
        // this is a slimmed down representation of the FlatIndex itself
        // it also encodes the <TMetric, TVal> so that we can ensure that
        // any attempts to deserialize are sane.
        let obj = FlatIndexSerialized {
            metric: metric_b,
            params: params_b,
            datastore: datastore_b,
            id_increment: rmp_serde::to_vec(&self.id_increment)?,
            delete_set: rmp_serde::to_vec(&self.delete_set)?,
            eid_to_vid: rmp_serde::to_vec(&self.eid_to_vid)?,
            vid_to_eid: rmp_serde::to_vec(&self.vid_to_eid)?,
            aligned_dim: rmp_serde::to_vec(&self.aligned_dim)?,
            v_per_segment: rmp_serde::to_vec(&self.v_per_segment)?,
            quantizer: rmp_serde::to_vec(&self.quantizer)?,
        };
        obj.serialize(&mut serializer)?;
        writer.flush()?;
        Ok(())
    }

    pub fn new_core(params: &FlatLiteParams) -> anyhow::Result<FlatIndex<TMetric, TVal>> {
        let aligned_dim = ann::round_up(params.dim as u32) as usize;
        let mut v_per_segment: usize = (params.segment_size_kb * 1000) / (aligned_dim as usize * 4);
        if v_per_segment < 1000 {
            v_per_segment = 1000
        }
        let id_increment = Arc::new(AtomicUsize::new(0));
        let delete_set: Arc<RwLock<HashSet<usize>>> = Arc::new(RwLock::new(HashSet::new()));
        let segement_0 = RwLock::new(av_store::AlignedDataStore::new(
            v_per_segment,
            aligned_dim.try_into().unwrap(),
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

            quantizer: Arc::new(scalar_quantizer::ScalarQuantizer::new(0.99)?),
        })
    }
    pub fn insert(&self, eids: &[ann::EId], points: ann::Points<TVal>) -> anyhow::Result<()> {
        let mut idx_by_vid: HashMap<usize, usize> = HashMap::new();
        let mut vids: Vec<usize> = Vec::with_capacity(eids.len());
        {
            let eid_to_vid = self.eid_to_vid.read();
            eids.iter().for_each(|eid| match eid_to_vid.get(eid) {
                Some(vid_existing) => vids.push(*vid_existing),
                None => {
                    vids.push(
                        self.id_increment
                            .fetch_add(1, std::sync::atomic::Ordering::SeqCst),
                    );
                }
            });
        }
        vids.iter().enumerate().for_each(|(idx, vid)| {
            idx_by_vid.insert(*vid, idx);
        });

        let data: &[TVal];
        let quantize_result: Vec<TVal>;
        match points {
            ann::Points::QuantizerIn { vals } => {
                quantize_result = self
                    .quantizer
                    .quantize(&vids, vals, None)
                    .iter()
                    .map(|x| TVal::from_u8(*x).expect("unable to coerce to u8"))
                    .collect();
                data = &quantize_result[..]
            }
            ann::Points::Values { vals } => data = vals,
        }

        let per_vector_dim = data.len() / eids.len() as usize;
        if data.len() % eids.len() != 0 {
            bail!(
                "point dim: {}  aligned_dim: {} not divisble.",
                data.len(),
                eids.len(),
            );
        }
        if per_vector_dim > self.aligned_dim {
            bail!(
                "point dim: {} > aligned dim: {}",
                per_vector_dim,
                self.aligned_dim,
            )
        }

        let padded_vector: Vec<TVal>;
        let padded_points: &[TVal];
        match ann::pad_and_preprocess::<TVal, TMetric>(data, per_vector_dim, self.aligned_dim) {
            Some(vec) => {
                padded_vector = vec;
                padded_points = &padded_vector[..]
            }
            None => padded_points = data,
        }

        let mut vid_by_segment_id: HashMap<usize, Vec<usize>> = HashMap::new();
        vids.iter().for_each(|vid| {
            let segment_id = (vid / self.v_per_segment) as usize;
            let mut need_new_list = false;
            match vid_by_segment_id.get_mut(&segment_id) {
                Some(vids_list) => vids_list.push(*vid),
                None => need_new_list = true,
            }
            if need_new_list {
                let new_list = vec![*vid];
                vid_by_segment_id.insert(segment_id, new_list);
            }
        });
        let mut new_segment_ids: Vec<usize> = Vec::with_capacity(2);
        vid_by_segment_id.iter().for_each(|(segment_id, _)| {
            match self.datastore.read().get(&segment_id) {
                Some(_segment) => {}
                None => new_segment_ids.push(*segment_id),
            }
        });
        new_segment_ids.iter().for_each(|segment_id| {
            let new_segment = RwLock::new(av_store::AlignedDataStore::new(
                self.v_per_segment.try_into().unwrap(),
                self.aligned_dim.try_into().unwrap(),
            ));
            self.datastore.write().insert(*segment_id, new_segment);
        });
        let datastore = self.datastore.read();
        for (segment_id, vids) in vid_by_segment_id {
            match datastore.get(&segment_id) {
                None => {
                    bail!("unexpectedly, the segment: {segment_id} is missing - bailing")
                }
                Some(segment) => {
                    let mut segment_w = segment.write();
                    for vid in vids {
                        let idx: usize;
                        match idx_by_vid.get(&vid) {
                            Some(index) => idx = *index,
                            None => {
                                bail!("every vid should have an associated index - vid: {vid} is missing one")
                            }
                        }
                        segment_w.aligned_insert(
                            (vid % self.v_per_segment).try_into().unwrap(),
                            &padded_points
                                [idx * self.aligned_dim..idx * self.aligned_dim + self.aligned_dim],
                        )
                    }
                }
            }
        }
        let mut eid_to_vid = self.eid_to_vid.write();
        let mut vid_to_eid = self.vid_to_eid.write();
        for (idx, vid) in vids.iter().enumerate() {
            eid_to_vid.insert(eids[idx], *vid);
            vid_to_eid.insert(*vid, eids[idx]);
        }
        Ok(())
    }

    pub fn delete(&self, eids: &[ann::EId]) -> anyhow::Result<()> {
        eids.iter().for_each(|eid| {
            let vid: usize;
            let vid_found: bool;
            {
                match self.eid_to_vid.read().get(eid) {
                    Some(vid_val) => {
                        vid = *vid_val;
                        vid_found = true;
                    }
                    None => {
                        vid = 0;
                        vid_found = false;
                    }
                }
            }
            if vid_found {
                self.eid_to_vid.write().remove(eid);
                self.vid_to_eid.write().remove(&vid);
                self.delete_set.write().insert(vid);
            }
            // key is in our mapping so do the delete set dance
        });

        Ok(())
    }
    pub fn search(&self, q: ann::Points<TVal>, k: usize) -> anyhow::Result<Vec<ann::Node>> {
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
        if data.len() > self.aligned_dim {
            bail!(
                "query dim: {} > aligned_dim: {}",
                data.len(),
                self.aligned_dim
            );
        }

        let padded_vector: Vec<TVal>;
        let padded_points: &[TVal];
        match ann::pad_and_preprocess::<TVal, TMetric>(data, data.len(), self.aligned_dim) {
            Some(vec) => {
                padded_vector = vec;
                padded_points = &padded_vector[..]
            }
            None => padded_points = data,
        }

        // maybe we want to keep around a bunch of these in a pool we can pull from?
        let mut res_heap: BinaryHeap<ann::Node> = BinaryHeap::with_capacity(k + 1);
        let mut q_aligned: av_store::AlignedDataStore<TVal> =
            av_store::AlignedDataStore::<TVal>::new(1, self.aligned_dim);
        q_aligned.data[..padded_points.len()].copy_from_slice(&padded_points[..]);
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

                    let arr_a: &[TVal] = &q_aligned.data[..];
                    let arr_b: &[TVal] =
                        &data.data[i * self.aligned_dim..(i * self.aligned_dim) + self.aligned_dim];
                    let dist = TMetric::compare(arr_a, arr_b);

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
    use std::io::BufWriter;
    use std::io::Cursor;

    #[test]
    fn test_serialization() {
        // setup to ensure that insert + search works as expected
        let dimensions = 128;
        let params = FlatLiteParams {
            dim: dimensions,
            segment_size_kb: 512,
        };
        let index = FlatIndex::<metric::MetricL2, f32>::new_core(&params).unwrap();
        let eids: Vec<ann::EId> = (0..1000)
            .map(|id| {
                let mut eid = EId([0u8; 16]);
                let id_str = id.clone().to_string();
                let id_bytes = id_str.as_bytes();
                eid.0[0..id_bytes.len()].copy_from_slice(&id_bytes[..]);
                eid
            })
            .collect();
        let mut points: Vec<f32> = Vec::with_capacity(dimensions * eids.len());
        (0..1000).for_each(|factor| {
            points.append(&mut vec![1.2 * (factor as f32); dimensions]);
        });
        assert_eq!(
            (),
            index
                .insert(&eids, ann::Points::Values { vals: &points[..] })
                .unwrap()
        );
        let point_search = vec![1.2 * (10000 as f32); dimensions];
        let expect: Vec<ann::EId> = (998..1000)
            .rev()
            .map(|id| {
                let mut eid = EId([0u8; 16]);
                let id_str = id.clone().to_string();
                let id_bytes = id_str.as_bytes();
                eid.0[0..id_bytes.len()].copy_from_slice(&id_bytes[..]);
                eid
            })
            .collect();
        match index.search(
            ann::Points::Values {
                vals: &point_search,
            },
            2,
        ) {
            Ok(res) => {
                let result: Vec<ann::EId> = res.iter().map(|x| (x.eid)).collect();
                assert_eq!(expect, result);
            }
            Err(_) => {
                panic!("error should not be thrown on search");
            }
        }
        // now verify that we can serialize and deserialize the data as expected
        let mut buf = Vec::new();
        index
            .serialize_(BufWriter::new(&mut buf))
            .expect("serialization should not fail");
        let deserialized_index = FlatIndex::<metric::MetricL2, f32>::deserialize_(Cursor::new(buf))
            .expect("deserialization should not fail");
        // now verfiy that the earlier search operation works and returns the same answer
        let expect: Vec<ann::EId> = (998..1000)
            .rev()
            .map(|id| {
                let mut eid = EId([0u8; 16]);
                let id_str = id.clone().to_string();
                let id_bytes = id_str.as_bytes();
                eid.0[0..id_bytes.len()].copy_from_slice(&id_bytes[..]);
                eid
            })
            .collect();
        match deserialized_index.search(
            ann::Points::Values {
                vals: &point_search,
            },
            2,
        ) {
            Ok(res) => {
                let result: Vec<ann::EId> = res.iter().map(|x| (x.eid)).collect();
                assert_eq!(expect, result);
            }
            Err(_) => {
                panic!("error should not be thrown on search");
            }
        }
    }

    #[test]
    fn insert_with_quantization() {
        let dimensions = 128;
        let params = FlatLiteParams {
            dim: dimensions,
            segment_size_kb: 512,
        };
        let index = FlatIndex::<metric::MetricL2, u8>::new_core(&params).unwrap();
        let eids: Vec<ann::EId> = (0..1000)
            .map(|id| {
                let mut eid = EId([0u8; 16]);
                let id_str = id.clone().to_string();
                let id_bytes = id_str.as_bytes();
                eid.0[0..id_bytes.len()].copy_from_slice(&id_bytes[..]);
                eid
            })
            .collect();
        let mut points: Vec<f32> = Vec::with_capacity(dimensions * eids.len());
        (0..1000).for_each(|factor| {
            points.append(&mut vec![1.2 * (factor as f32); dimensions]);
        });
        assert_eq!(
            (),
            index
                .insert(&eids, ann::Points::QuantizerIn { vals: &points[..] })
                .unwrap()
        );
        let expect: Vec<ann::EId> = (998..1000)
            .rev()
            .map(|id| {
                let mut eid = EId([0u8; 16]);
                let id_str = id.clone().to_string();
                let id_bytes = id_str.as_bytes();
                eid.0[0..id_bytes.len()].copy_from_slice(&id_bytes[..]);
                eid
            })
            .collect();
        let point_search = vec![1.2 * (10000 as f32); dimensions];
        match index.search(
            ann::Points::QuantizerIn {
                vals: &point_search,
            },
            expect.len(),
        ) {
            Ok(res) => {
                let result: Vec<ann::EId> = res.iter().map(|x| (x.eid)).collect();
                assert_eq!(expect, result);
            }
            Err(_) => {
                panic!("error should not be thrown on search");
            }
        }
    }

    #[test]
    fn insert_large_wpadding() {
        let dimensions = 126;
        let params = FlatLiteParams {
            dim: dimensions,
            segment_size_kb: 512,
        };
        let index = FlatIndex::<metric::MetricL2, f32>::new_core(&params).unwrap();
        // insert the first 1000 vectors into the index (nb: 1000 per segment)
        let eids: Vec<ann::EId> = (0..1000)
            .map(|id| {
                let mut eid = EId([0u8; 16]);
                let id_str = id.clone().to_string();
                let id_bytes = id_str.as_bytes();
                eid.0[0..id_bytes.len()].copy_from_slice(&id_bytes[..]);
                eid
            })
            .collect();
        let mut points: Vec<f32> = Vec::with_capacity(dimensions * eids.len());
        (0..1000).for_each(|factor| {
            points.append(&mut vec![1.2 * (factor as f32); dimensions]);
        });
        assert_eq!(
            (),
            index
                .insert(&eids, ann::Points::Values { vals: &points[..] })
                .unwrap()
        );
        // then insert the 1001th vector - this will cause a new segement to be created
        let mut id = EId([0u8; 16]);
        let id_str = "1000".to_string();
        let id_bytes = id_str.as_bytes();
        id.0[0..id_bytes.len()].copy_from_slice(&id_bytes[..]);
        let point = vec![1.2 * (1000 as f32); dimensions];
        match index.insert(&vec![id; 1], ann::Points::Values { vals: &point[..] }) {
            Ok(res) => {
                assert_eq!((), res);
            }
            Err(_) => {
                panic!("error should not be thrown on insert");
            }
        }
        // now craft a search that includes the vector in the external segment!
        let point_search = vec![1.2 * (10000 as f32); dimensions];
        match index.search(
            ann::Points::Values {
                vals: &point_search,
            },
            1,
        ) {
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
        let params = FlatLiteParams {
            dim: 32,
            segment_size_kb: 512,
        };
        let index = FlatIndex::<metric::MetricL2, f32>::new_core(&params).unwrap();
        for i in 0..10 {
            let mut id = EId([0u8; 16]);
            id.0[0] = i;
            let point = vec![100.0 * (i as f32); 32];
            match index.insert(&vec![id; 1], ann::Points::Values { vals: &point[..] }) {
                Ok(res) => {
                    assert_eq!((), res);
                }
                Err(_) => {
                    panic!("error should not be thrown on insert");
                }
            }
        }
        let point_search = vec![0.0; 32];
        match index.search(
            ann::Points::Values {
                vals: &point_search,
            },
            1,
        ) {
            Ok(res) => {
                let result: Vec<ann::EId> = res.iter().map(|x| (x.eid)).collect();
                assert_eq!(vec![EId([0u8; 16])], result);
            }
            Err(_) => {
                panic!("error should not be thrown on search");
            }
        }
        // now delete the one we care about
        let delete_set = vec![EId([0u8; 16]); 1];
        index
            .delete(&delete_set)
            .expect("unable to remove the item from the database");
        // then issue the search and ensure the old mapping does
        // not come up
        match index.search(
            ann::Points::Values {
                vals: &point_search,
            },
            1,
        ) {
            Ok(res) => {
                let result: Vec<ann::EId> = res.iter().map(|x| (x.eid)).collect();
                let mut expected_id = [0u8; 16];
                expected_id[0] = 1;
                assert_eq!(vec![EId(expected_id)], result);
            }
            Err(_) => {
                panic!("error should not be thrown on search");
            }
        }
    }
    #[test]
    fn insert_small_wpadding() {
        let params = FlatLiteParams {
            dim: 32,
            segment_size_kb: 512,
        };
        let index = FlatIndex::<metric::MetricL2, f32>::new_core(&params).unwrap();
        for i in 0..10 {
            let mut id = EId([0u8; 16]);
            id.0[0] = i;
            let point = vec![1.2 * (i as f32); 32];
            match index.insert(&vec![id; 1], ann::Points::Values { vals: &point[..] }) {
                Ok(res) => {
                    assert_eq!((), res);
                }
                Err(_) => {
                    panic!("error should not be thrown on insert");
                }
            }
        }
        let point_search = vec![0.4; 32];
        match index.search(
            ann::Points::Values {
                vals: &point_search,
            },
            1,
        ) {
            Ok(res) => {
                let result: Vec<ann::EId> = res.iter().map(|x| (x.eid)).collect();
                assert_eq!(vec![EId([0u8; 16])], result);
            }
            Err(_) => {
                panic!("error should not be thrown on search");
            }
        }
    }
    #[test]
    fn insert_small_npadding() {
        let params = FlatLiteParams {
            dim: 128,
            segment_size_kb: 512,
        };
        let index = FlatIndex::<metric::MetricL2, f32>::new_core(&params).unwrap();
        for i in 0..10 {
            let mut id = EId([0u8; 16]);
            id.0[0] = i;
            let point = vec![100.0 * (i as f32); 128];
            match index.insert(&vec![id; 1], ann::Points::QuantizerIn { vals: &point[..] }) {
                Ok(res) => {
                    assert_eq!((), res);
                }
                Err(_) => {
                    panic!("error should not be thrown on insert");
                }
            }
        }
        let point_search = vec![0.1; 128];
        match index.search(
            ann::Points::Values {
                vals: &point_search,
            },
            1,
        ) {
            Ok(res) => {
                let result: Vec<ann::EId> = res.iter().map(|x| (x.eid)).collect();
                assert_eq!(vec![EId([0u8; 16])], result);
            }
            Err(_) => {
                panic!("error should not be thrown on search");
            }
        }
    }
}
