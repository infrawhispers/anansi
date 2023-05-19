/// provides an implementation of FlatIndex which is backed by RocksDB
/// this allows the index to be created | updated during interuption
/// without explicit serialization calls of the _entire_ structure
///
/// within rocksb, we store the following keys:
/// --------------------------------------------
/// "eid_to_vid::{eid}" -> vid
/// "vid_to_eid::{vid}" -> eid
/// "avail_vid::{vid}" -> vid
/// "segment::{segment_id}" -> a segment
/// "config" -> FlatConfig
/// --------------------------------------------
/// which are then used to reconstruct the HashMaps and HashSets in the process
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::marker::PhantomData;
use std::path::PathBuf;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;

use anyhow::{bail, Context};
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use tracing::{info, warn};
extern crate rmp_serde as rmps;

use rayon::prelude::*;

use super::ann;
use super::ann::EId;
use super::av_store;
use super::flat_lite::FlatLiteParams;
use super::metric;
use super::scalar_quantizer;

#[derive(Serialize, Deserialize)]
struct FlatConfig {
    v_per_segment: usize,
    aligned_dim: usize,
    index_name: String,
    index_cf_name: String,
    dir_path: PathBuf,
}
pub struct FlatIndex<TMetric, TVal: for<'de> ann::ElementVal<'de>> {
    // fields copied from from the FlatIndexLite implementation
    pub metric: PhantomData<TMetric>,
    datastore: Arc<RwLock<HashMap<usize, RwLock<av_store::AlignedDataStore<TVal>>>>>,
    id_increment: Arc<AtomicUsize>,
    delete_set: Arc<RwLock<HashSet<usize>>>,
    eid_to_vid: Arc<RwLock<HashMap<EId, usize>>>,
    vid_to_eid: Arc<RwLock<HashMap<usize, EId>>>,
    config: FlatConfig,
    quantizer: Arc<scalar_quantizer::ScalarQuantizer>,
    rocksdb_instance: Arc<RwLock<rocksdb::DB>>,
}

#[derive(Debug, Clone)]
pub struct FlatFullParams {
    pub params: FlatLiteParams,
    pub index_name: String,
    pub rocksdb_instance: Arc<RwLock<rocksdb::DB>>,
    pub dir_path: PathBuf,
}
impl<TMetric, TVal> ann::ANNIndex for FlatIndex<TMetric, TVal>
where
    TVal: for<'a> ann::ElementVal<'a>,
    TMetric: metric::Metric<TVal>,
{
    type Val = TVal;
    fn new(params: &ann::ANNParams) -> anyhow::Result<FlatIndex<TMetric, TVal>> {
        let flat_params: &FlatFullParams = match params {
            ann::ANNParams::FlatFull { params } => params,
            _ => {
                unreachable!("incorrect params passed for construction")
            }
        };
        FlatIndex::new_from_params(flat_params)
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
        bail!("flat_full::FlatIndex does not have a save function")
    }
    fn delete_index(&self) -> anyhow::Result<()> {
        self.delete_index()
    }
}

impl<TMetric, TVal> FlatIndex<TMetric, TVal>
where
    TVal: for<'de> ann::ElementVal<'de>,
    TMetric: metric::Metric<TVal>,
{
    fn delete_index(&self) -> anyhow::Result<()> {
        let mut rocksdb_instance = self.rocksdb_instance.write();
        rocksdb_instance.drop_cf(&self.config.index_cf_name)?;
        Ok(())
    }
    fn new_empty_index(params: &FlatFullParams) -> anyhow::Result<FlatIndex<TMetric, TVal>> {
        let aligned_dim = ann::round_up(params.params.dim as u32) as usize;
        let mut v_per_segment: usize =
            (params.params.segment_size_kb * 1024) / (aligned_dim as usize * 4);
        if v_per_segment < 1000 {
            info!(
                index_name = params.index_name,
                "v_per_segment requested: {v_per_segment} < 1000, defaulting to 1000"
            );
            v_per_segment = 1000;
        }
        let index_cf_name = format!("flatindxfull::{}", params.index_name);
        let config = FlatConfig {
            v_per_segment: v_per_segment,
            aligned_dim: aligned_dim,
            index_name: params.index_name.clone(),
            index_cf_name: index_cf_name,
            dir_path: params.dir_path.clone(),
        };

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
            datastore: datastore,
            id_increment: id_increment,
            delete_set: delete_set,
            eid_to_vid: tag_to_location,
            vid_to_eid: location_to_tag,
            quantizer: Arc::new(scalar_quantizer::ScalarQuantizer::new(0.99)?),

            config: config,
            rocksdb_instance: params.rocksdb_instance.clone(),
        })
    }

    fn init_fr_rocksdb(&mut self) -> anyhow::Result<()> {
        let mut options = rocksdb::Options::default();
        {
            let mut rocksdb_instance = self.rocksdb_instance.write();
            options.set_error_if_exists(false);
            options.create_if_missing(true);
            options.create_missing_column_families(true);
            let cfs =
                rocksdb::DB::list_cf(&options, self.config.dir_path.clone()).unwrap_or(vec![]);
            let cf_missing = cfs
                .iter()
                .find(|cf| *cf == &self.config.index_cf_name)
                .is_none();
            if cf_missing {
                let options = rocksdb::Options::default();
                rocksdb_instance.create_cf(self.config.index_cf_name.clone(), &options)?;
            }
        }
        let rocksdb_instance = self.rocksdb_instance.read();
        let cf = rocksdb_instance
            .cf_handle(&self.config.index_cf_name)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "unable to fetch colun family: {} - was it created?",
                    self.config.index_cf_name
                )
            })?;
        match rocksdb_instance.get_pinned_cf(cf, "config")? {
            Some(res) => {
                let result: FlatConfig = rmp_serde::from_slice(&res)?;
                self.config = result;
            }
            None => {
                // this is a _new_ index instantiation, so write it out to the database
                // allowing us to persist the config
                rocksdb_instance.put_cf(cf, "config", rmp_serde::to_vec(&self.config)?)?;
            }
        }
        {
            // populate: delete_set
            let mut delete_set = self.delete_set.write();
            let mut options = rocksdb::ReadOptions::default();
            options.set_iterate_range(rocksdb::PrefixRange("avail_vid::".as_bytes()));
            for item in rocksdb_instance.iterator_cf_opt(cf, options, rocksdb::IteratorMode::Start)
            {
                let (_, v) = item?;
                let vid: usize = usize::from_be_bytes(v[0..8].try_into()?);
                delete_set.insert(vid);
            }
        }
        {
            // populate: eid_to_vid
            let k_prefix = "eid_to_vid::";
            let mut eid_to_vid = self.eid_to_vid.write();
            let mut options = rocksdb::ReadOptions::default();
            options.set_iterate_range(rocksdb::PrefixRange(k_prefix.as_bytes()));
            for item in rocksdb_instance.iterator_cf_opt(cf, options, rocksdb::IteratorMode::Start)
            {
                let (k, v) = item?;
                let mut eid = EId([0u8; 16]);
                let vid: usize = usize::from_be_bytes(v[0..8].try_into()?);
                eid.0[..].copy_from_slice(&k[k_prefix.len()..]);
                eid_to_vid.insert(eid, vid);
            }
        }
        {
            // populate: vid_to_eid
            let k_prefix = "vid_to_eid::";
            let mut vid_to_eid = self.vid_to_eid.write();
            let mut options = rocksdb::ReadOptions::default();
            options.set_iterate_range(rocksdb::PrefixRange(k_prefix.as_bytes()));
            for item in rocksdb_instance.iterator_cf_opt(cf, options, rocksdb::IteratorMode::Start)
            {
                let (k, v) = item?;
                let mut eid = EId([0u8; 16]);
                let vid: usize = usize::from_be_bytes(k[k_prefix.len()..].try_into()?);
                eid.0[..].copy_from_slice(&v[..]);
                vid_to_eid.insert(vid, eid);
            }
        }

        match rocksdb_instance.get_pinned_cf(cf, "id_increment")? {
            Some(res) => {
                let result: Arc<AtomicUsize> = rmp_serde::from_slice(&res)?;
                self.id_increment = result;
            }
            None => {}
        }

        {
            // pull all the segments we care about.
            let mut options = rocksdb::ReadOptions::default();
            options.set_iterate_range(rocksdb::PrefixRange("segment::".as_bytes()));
            for item in rocksdb_instance.iterator_cf_opt(cf, options, rocksdb::IteratorMode::Start)
            {
                let (k, v) = item?;
                let key_utf8 = std::str::from_utf8(&k)
                    .with_context(|| "failed to parse utf8 str from the segment key")?;
                let key_parts: Vec<&str> = key_utf8.splitn(2, "::").collect();
                if key_parts.len() != 2 {
                    warn!(
                        "key: \"{:?}\" does not fit the expected format \"index::{{index_name}}\"",
                        key_utf8
                    );
                    continue;
                }
                let segment_id: usize = key_parts[key_parts.len() - 1].parse()?;
                // info!("loaded - segment_id: {segment_id:?}");
                let segment_data_r: RwLock<av_store::AlignedDataStore<TVal>> =
                    rmp_serde::from_slice(&(v))?;
                // this needs to be actually aligned so we _must_ call new
                // in order to avoid segfaults
                let segment_data: RwLock<av_store::AlignedDataStore<TVal>> =
                    RwLock::new(av_store::AlignedDataStore::new(
                        self.config.v_per_segment,
                        self.config.aligned_dim.try_into().unwrap(),
                    ));
                segment_data.write().num_vectors = segment_data_r.read().num_vectors;
                segment_data.write().data[..].copy_from_slice(&segment_data_r.read().data[..]);
                // info!("loaded - segment_data: {segment_data:?}");
                self.datastore.write().insert(segment_id, segment_data);
            }
        }
        Ok(())
    }

    fn new_from_params(params: &FlatFullParams) -> anyhow::Result<FlatIndex<TMetric, TVal>> {
        let mut idx = FlatIndex::new_empty_index(params)?;
        idx.init_fr_rocksdb()?;
        Ok(idx)
    }
    fn insert_segment(
        &self,
        segment_id: usize,
        segment: &parking_lot::RwLock<av_store::AlignedDataStore<TVal>>,
        vids: &Vec<usize>,
        eids: &[ann::EId],
        idx_by_vid: &HashMap<usize, usize>,
        padded_points: &[TVal],
    ) -> anyhow::Result<()> {
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
                (vid % self.config.v_per_segment).try_into().unwrap(),
                &padded_points[idx * self.config.aligned_dim
                    ..idx * self.config.aligned_dim + self.config.aligned_dim],
            )
        }
        let mut eid_to_vid = self.eid_to_vid.write();
        let mut vid_to_eid = self.vid_to_eid.write();
        for (idx, vid) in vids.iter().enumerate() {
            eid_to_vid.insert(eids[idx], *vid);
            vid_to_eid.insert(*vid, eids[idx]);
        }
        // write out all the changes to rocksdb that we need
        let instance = self.rocksdb_instance.read();
        let cf = instance
            .cf_handle(&self.config.index_cf_name)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "unable to fetch colun family: {} - was it created?",
                    self.config.index_cf_name
                )
            })?;
        let mut batch = rocksdb::WriteBatch::default();
        for vid in vids {
            let eid = eids[idx_by_vid[vid]];
            batch.put_cf(
                cf,
                ["eid_to_vid::".as_bytes(), &eid.as_bytes()].concat(),
                vid.to_be_bytes(),
            );
            batch.put_cf(
                cf,
                ["vid_to_eid::".as_bytes(), &vid.to_be_bytes()].concat(),
                &eid.as_bytes(),
            );
            batch.delete_cf(cf, ["avail_vid::".as_bytes(), &vid.to_be_bytes()].concat());
        }
        batch.put_cf(
            cf,
            format!("segment::{segment_id}").as_bytes(),
            rmp_serde::to_vec(&*segment_w)?,
        );
        instance.write(batch)?;
        Ok(())
    }

    fn insert(&self, eids: &[ann::EId], points: ann::Points<TVal>) -> anyhow::Result<()> {
        // pull a bunch of vids from the pool
        let mut idx_by_vid: HashMap<usize, usize> = HashMap::new();
        let mut vids: Vec<usize> = Vec::with_capacity(eids.len());

        {
            let eid_to_vid = self.eid_to_vid.read();
            eids.iter().for_each(|eid| match eid_to_vid.get(eid) {
                Some(vid_existing) => vids.push(*vid_existing),
                None => {
                    // attempt to pop a value from the delete_set
                    let mut delete_set = self.delete_set.write();
                    if let Some(elem) = delete_set.iter().next().cloned() {
                        delete_set.remove(&elem);
                        vids.push(elem);
                    } else {
                        vids.push(
                            self.id_increment
                                .fetch_add(1, std::sync::atomic::Ordering::SeqCst),
                        );
                    }
                }
            });
        }
        vids.iter().enumerate().for_each(|(idx, vid)| {
            idx_by_vid.insert(*vid, idx);
        });

        // run the quantization process (allocates) or
        // point data to the raw points that are provided
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
        // verify that the values are what we expect w.r.t
        // dimensions
        let per_vector_dim = data.len() / eids.len() as usize;
        if data.len() % eids.len() != 0 {
            bail!(
                "point dim: {}  aligned_dim: {} not divisble.",
                data.len(),
                eids.len(),
            );
        }
        if per_vector_dim > self.config.aligned_dim {
            bail!(
                "point dim: {} > aligned dim: {}",
                per_vector_dim,
                self.config.aligned_dim,
            )
        }
        // run any preprocessing that we may need to do for the
        // given metric that is passed in
        let padded_vector: Vec<TVal>;
        let padded_points: &[TVal];
        match ann::pad_and_preprocess::<TVal, TMetric>(
            data,
            per_vector_dim,
            self.config.aligned_dim,
        ) {
            Some(vec) => {
                padded_vector = vec;
                padded_points = &padded_vector[..]
            }
            None => padded_points = data,
        }

        // group items into the segment_ids that we care about
        // for the given items
        let mut vid_by_segment_id: HashMap<usize, Vec<usize>> = HashMap::new();
        vids.iter().for_each(|vid| {
            let segment_id = (vid / self.config.v_per_segment) as usize;
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
        // println!("new_segment_ids: {new_segment_ids:?}");
        // finally write out items on a per-segment basis
        for (segment_id, vids) in vid_by_segment_id {
            if new_segment_ids.contains(&segment_id) {
                let mut datastore = self.datastore.write();
                let segment;
                if datastore.contains_key(&segment_id) {
                    // was already created by another thread!
                    segment = datastore
                        .get(&segment_id)
                        .ok_or_else(|| {
                            anyhow::anyhow!(
                                "unable to get the segment_id {segment_id} - when it should already exist.",
                            )
                        })?;
                } else {
                    // create the new_segment and insert into the datastore
                    let new_segment = RwLock::new(av_store::AlignedDataStore::new(
                        self.config.v_per_segment.try_into().unwrap(),
                        self.config.aligned_dim.try_into().unwrap(),
                    ));
                    datastore.insert(segment_id, new_segment);
                    segment = datastore
                        .get(&segment_id)
                        .ok_or_else(|| {
                            anyhow::anyhow!(
                                "unable to get the segment_id {segment_id} - when it should already exist.",
                            )
                        })?;
                }
                self.insert_segment(segment_id, segment, &vids, eids, &idx_by_vid, padded_points)?;
            } else {
                let datastore = self.datastore.read();
                let segment = datastore.get(&segment_id).ok_or_else(|| {
                    anyhow::anyhow!(
                        "unable to get the segment_id {segment_id} - although we asserted that it should be missing",
                    )
                })?;
                self.insert_segment(segment_id, segment, &vids, eids, &idx_by_vid, padded_points)?;
            }
        }
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
        if data.len() > self.config.aligned_dim {
            bail!(
                "query dim: {} > aligned_dim: {}",
                data.len(),
                self.config.aligned_dim
            );
        }

        let padded_vector: Vec<TVal>;
        let padded_points: &[TVal];
        match ann::pad_and_preprocess::<TVal, TMetric>(data, data.len(), self.config.aligned_dim) {
            Some(vec) => {
                padded_vector = vec;
                padded_points = &padded_vector[..]
            }
            None => padded_points = data,
        }

        // info!("stage: running the search");
        // maybe we want to keep around a bunch of these in a pool we can pull from?
        let res_heap: Mutex<BinaryHeap<ann::Node>> = Mutex::new(BinaryHeap::with_capacity(k + 1));
        let mut q_aligned: av_store::AlignedDataStore<TVal> =
            av_store::AlignedDataStore::<TVal>::new(1, self.config.aligned_dim);
        q_aligned.data[..padded_points.len()].copy_from_slice(&padded_points[..]);
        // info!("stage: generated q_aligned");
        // we should probably use rayon over segments and have multiple vectors
        // in a given segment
        self.datastore
            .read()
            .par_iter()
            .for_each(|(segment_id, vec_store)| {
                // we are now in a single segment
                let data = vec_store.read();
                let mut nodes = Vec::with_capacity(data.num_vectors);
                // info!("stage: running in the segment: {segment_id}");
                for i in 0..data.num_vectors {
                    let eid: ann::EId;
                    let vid = *segment_id * (self.config.v_per_segment as usize) + i;
                    match self.vid_to_eid.read().get(&vid) {
                        Some(val) => eid = val.clone(),
                        None => {
                            continue;
                        }
                    }
                    // info!("stage: fetched the correct vid");
                    let arr_a: &[TVal] = &q_aligned.data[..];
                    let arr_b: &[TVal] = &data.data[i * self.config.aligned_dim
                        ..(i * self.config.aligned_dim) + self.config.aligned_dim];
                    // info!(
                    //     "stage: arr_a len: {} | arr_b len: {}",
                    //     arr_a.len(),
                    //     arr_b.len()
                    // );
                    let dist = TMetric::compare(arr_a, arr_b);
                    // info!("stage: ran the distance comparison");
                    nodes.push(ann::Node {
                        vid: vid,
                        eid: eid,
                        distance: dist,
                    });
                }
                // sort it by distance _ascending_ so that we can stop early
                // allowing us to reuse the lock
                nodes.sort();
                let mut heap = res_heap.lock();
                for nn in nodes.iter() {
                    if heap.len() >= k {
                        // this unwrap will not panic as we have values in the heep
                        // based on the previous line!
                        if nn > heap.peek().unwrap() {
                            break;
                        } else {
                            heap.pop();
                            heap.push(nn.clone());
                        }
                    } else {
                        heap.push(nn.clone())
                    }
                }
            });
        let mut heap = res_heap.lock();
        let mut res_vec = Vec::with_capacity(heap.len());
        while !heap.is_empty() {
            let neighbor_rev = heap.pop().unwrap();
            res_vec.push(neighbor_rev);
        }
        res_vec.reverse();
        // info!("stage: search is complete");
        Ok(res_vec)
    }

    pub fn delete(&self, eids: &[ann::EId]) -> anyhow::Result<()> {
        for eid in eids.iter() {
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
                // write out the changes to rocksdb
                let instance = self.rocksdb_instance.read();
                let cf = instance
                    .cf_handle(&self.config.index_cf_name)
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "unable to fetch colunmn family: {} - was it created?",
                            self.config.index_cf_name
                        )
                    })?;
                let mut batch = rocksdb::WriteBatch::default();
                batch.delete_cf(cf, ["eid_to_vid::".as_bytes(), &eid.as_bytes()].concat());
                batch.delete_cf(
                    cf,
                    ["vid_to_eid::{vid}".as_bytes(), &vid.to_be_bytes()].concat(),
                );
                batch.put_cf(
                    cf,
                    ["avail_vid::".as_bytes(), &vid.to_be_bytes()].concat(),
                    vid.to_be_bytes(),
                );
                instance.write(batch)?;
                // finally update the in-process stores
                self.eid_to_vid.write().remove(eid);
                self.vid_to_eid.write().remove(&vid);
                self.delete_set.write().insert(vid);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    #[test]
    fn flat_rocksdb_normal() {
        let dimensions = 768;
        let dir_path = PathBuf::from(".test/flat_rocksdb_normal");
        let _ = fs::remove_dir_all(dir_path.clone());
        fs::create_dir_all(dir_path.clone()).expect("unable to create the test directory");
        let mut options = rocksdb::Options::default();
        options.set_error_if_exists(false);
        options.create_if_missing(true);
        options.create_missing_column_families(true);
        let cfs = rocksdb::DB::list_cf(&options, dir_path.clone()).unwrap_or(vec![]);
        {
            let rocksdb_instance = rocksdb::DB::open_cf(&options, dir_path.clone(), cfs.clone())
                .expect("unable to create the rocksdb instance");

            let params = FlatFullParams {
                params: FlatLiteParams {
                    dim: dimensions,
                    segment_size_kb: 512,
                },
                index_name: "test-00000".to_string(),
                dir_path: dir_path.clone(),
                rocksdb_instance: Arc::new(RwLock::new(rocksdb_instance)),
            };

            // this index gets dropped at the end of this block
            let index = FlatIndex::<metric::MetricL2, f32>::new_from_params(&params)
                .expect("no error thrown on creation");
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
        }
        let rocksdb_instance = rocksdb::DB::open_cf(&options, dir_path.clone(), cfs)
            .expect("unable to create the rocksdb instance");
        let params = FlatFullParams {
            params: FlatLiteParams {
                dim: 128,
                segment_size_kb: 512,
            },
            index_name: "test-00000".to_string(),
            dir_path: dir_path.clone(),
            rocksdb_instance: Arc::new(RwLock::new(rocksdb_instance)),
        };

        let index = FlatIndex::<metric::MetricL2, f32>::new_from_params(&params)
            .expect("no error on recreation of index");
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
    }
}
