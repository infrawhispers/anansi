use anyhow::{bail, Context};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use crate::ann::Points;
use crate::ann::{ANNIndex, ANNParams, EId, Node};
use crate::diskannv1::DiskANNV1Index;
use crate::flat_lite::FlatIndex as FlatLiteIndex;

#[cfg(feature = "full")]
use crate::flat_full::{FlatFullParams, FlatIndex as FlatFullIndex};

pub struct IndexManager {
    // mapping of index_name -> ANNIndex, all operations on the manager
    // are segmented by the index_name.
    // in the future, we should be able to add ACLs to a particular index
    // to control the _type_ of operation being done.
    indices: RwLock<HashMap<String, Arc<dyn ANNIndex<Val = f32>>>>,
    #[cfg(feature = "full")]
    rocksdb_instance: Arc<RwLock<rocksdb::DB>>,
    #[cfg(feature = "full")]
    dir_path: PathBuf,
}

impl IndexManager {
    #[cfg(feature = "full")]
    pub fn new(dir: &PathBuf) -> Self {
        // let dir_path = PathBuf::from(".test");
        // let _ = fs::remove_dir_all(dir_path.clone());
        // fs::create_dir_all(dir_path.clone()).expect("unable to create the test directory");
        let mut options = rocksdb::Options::default();
        options.set_error_if_exists(false);
        options.create_if_missing(true);
        options.create_missing_column_families(true);
        let cfs = rocksdb::DB::list_cf(&options, dir.clone()).unwrap_or(vec![]);
        let rocksdb_instance = rocksdb::DB::open_cf(&options, dir.clone(), cfs)
            .expect("unable to create the rocksdb instance");

        IndexManager {
            indices: RwLock::new(HashMap::new()),
            rocksdb_instance: Arc::new(RwLock::new(rocksdb_instance)),
            dir_path: dir.clone(),
        }
    }

    #[cfg(not(feature = "full"))]
    pub fn new(_dir: &PathBuf) -> Self {
        IndexManager {
            indices: RwLock::new(HashMap::new()),
        }
    }

    pub fn delete_index(&self, index_name: &str) -> anyhow::Result<()> {
        {
            let indices = self.indices.read();
            match indices.get(index_name) {
                Some(index) => {
                    index.delete_index()?;
                }
                None => return Ok(()),
            }
        }
        {
            let mut indices = self.indices.write();
            indices.remove(index_name);
        }
        Ok(())
    }

    pub fn delete(&self, index_name: String, eids: &[EId]) -> anyhow::Result<()> {
        let indices = self.indices.read();
        let index = indices.get(&index_name).ok_or_else(|| {
            anyhow::anyhow!(
                "index: \"{:?}\" does not exist - was it initialized or created?",
                &index_name
            )
        })?;
        index.delete(eids)
    }

    pub fn search(&self, index_name: &str, q: Points<f32>, k: usize) -> anyhow::Result<Vec<Node>> {
        let indices = self.indices.read();
        let index = indices.get(index_name).ok_or_else(|| {
            anyhow::anyhow!(
                "index: \"{:?}\" does not exist - was it initialized or created?",
                &index_name
            )
        })?;
        index.search(q, k)
    }

    pub fn delete_data(&self, index_name: &str, eids: &[EId]) -> anyhow::Result<()> {
        let indices = self.indices.read();
        let index = indices
            .get(index_name)
            .ok_or_else(|| anyhow::anyhow!("mgr:: index \"{index_name}\" does not exist"))?;
        index.delete(eids)
    }

    pub fn insert(
        &self,
        index_name: String,
        eids: &[EId],
        data: Points<f32>,
    ) -> anyhow::Result<()> {
        let indices = self.indices.read();
        let index = indices
            .get(&index_name)
            .ok_or_else(|| anyhow::anyhow!("mgr:: index \"{index_name}\" does not exist"))?;
        index.insert(eids, data)
    }

    pub fn create_index(
        &self,
        index_name: &str,
        index_type: &str,
        metric_type: &str,
        index_params: &crate::ann::ANNParams,
    ) -> anyhow::Result<Arc<dyn ANNIndex<Val = f32>>> {
        let aidx: Arc<dyn ANNIndex<Val = f32>>;
        match index_type {
            "DiskANNLite" => match metric_type {
                "MetricL2" => {
                    let idx: DiskANNV1Index<crate::metric::MetricL2, f32> =
                        DiskANNV1Index::new(&index_params)
                            .with_context(|| "unable to create the index")?;
                    aidx = Arc::new(idx);
                }
                "MetricL1" => {
                    let idx: DiskANNV1Index<crate::metric::MetricL1, f32> =
                        DiskANNV1Index::new(&index_params)
                            .with_context(|| "unable to create the index")?;
                    aidx = Arc::new(idx);
                }
                "MetricCosine" => {
                    let idx: DiskANNV1Index<crate::metric::MetricL1, f32> =
                        DiskANNV1Index::new(&index_params)
                            .with_context(|| "unable to create the index")?;
                    aidx = Arc::new(idx);
                }
                &_ => bail!("unknown metric type: {metric_type}"),
            },
            "Flat" => {
                #[cfg(feature = "full")]
                {
                    let e_params;
                    match index_params {
                        ANNParams::FlatLite { params } => {
                            e_params = params;
                        }
                        &_ => {
                            bail!("Flat expected FlatLite - recieved: {index_params:?}")
                        }
                    }
                    let params = ANNParams::FlatFull {
                        params: FlatFullParams {
                            params: *e_params,
                            index_name: index_name.to_string(),
                            rocksdb_instance: self.rocksdb_instance.clone(),
                            dir_path: self.dir_path.clone(),
                        },
                    };
                    match metric_type {
                        "MetricL2" => {
                            let idx: FlatFullIndex<crate::metric::MetricL2, f32> =
                                FlatFullIndex::new(&params)
                                    .with_context(|| "unable to create the index")?;
                            aidx = Arc::new(idx);
                        }
                        "MetricL1" => {
                            let idx: FlatFullIndex<crate::metric::MetricL1, f32> =
                                FlatFullIndex::new(&params)
                                    .with_context(|| "unable to create the index")?;
                            aidx = Arc::new(idx);
                        }
                        "MetricCosine" => {
                            let idx: FlatFullIndex<crate::metric::MetricCosine, f32> =
                                FlatFullIndex::new(&params)
                                    .with_context(|| "unable to create the index")?;
                            aidx = Arc::new(idx);
                        }
                        &_ => bail!("unknown metric type: {metric_type}"),
                    }
                }
                #[cfg(not(feature = "full"))]
                {
                    bail!("Flat must be compiled with features: full")
                }
            }
            "FlatLite" => match metric_type {
                "MetricL2" => {
                    let idx: FlatLiteIndex<crate::metric::MetricL2, f32> =
                        FlatLiteIndex::new(index_params)
                            .with_context(|| "unable to create the index")?;
                    aidx = Arc::new(idx);
                }
                "MetricL1" => {
                    let idx: FlatLiteIndex<crate::metric::MetricL1, f32> =
                        FlatLiteIndex::new(index_params)
                            .with_context(|| "unable to create the index")?;
                    aidx = Arc::new(idx);
                }
                "MetricCosine" => {
                    let idx: FlatLiteIndex<crate::metric::MetricCosine, f32> =
                        FlatLiteIndex::new(index_params)
                            .with_context(|| "unable to create the index")?;
                    aidx = Arc::new(idx);
                }
                &_ => bail!("unknown metric_type: {:?}", metric_type),
            },
            &_ => bail!("unknown index_type: {:?}", index_type),
        }
        Ok(aidx)
    }

    pub fn new_index(
        &self,
        index_name: &str,
        index_type: &str,
        metric_type: &str,
        index_params: &crate::ann::ANNParams,
    ) -> anyhow::Result<()> {
        {
            // bail if the index already exists, nothing here to do!
            let indices_r = self.indices.read();
            if indices_r.contains_key(index_name) {
                bail!("index: {} already exists", index_name)
            }
        }
        // create the index and run the operations that we care about
        let index = self.create_index(index_name, index_type, metric_type, index_params)?;
        let mut indices_w = self.indices.write();
        indices_w.insert(index_name.to_string(), index);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_manager() {
        let mgr = IndexManager::new(&PathBuf::from(".test"));
        let index_name = "test-00001";
        let dimensions: usize = 128;
        let params = crate::ann::ANNParams::FlatLite {
            params: crate::flat_lite::FlatLiteParams {
                dim: dimensions,
                segment_size_kb: 1024,
            },
        };
        mgr.new_index(index_name, "Flat", "MetricL1", &params)
            .expect("fresh index creation does not fail");
        // creation of an index with the same name should fail
        assert_eq!(
            mgr.new_index(index_name, "Flat", "MetricL1", &params)
                .is_err(),
            true
        );
        let eids: Vec<EId> = (0..1000)
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

        mgr.insert(
            index_name.to_string(),
            &eids,
            Points::Values { vals: &points[..] },
        )
        .expect("insertion of points to created index should not fail");
        // insertion should fail for index that is not initialized
        assert_eq!(
            mgr.insert(
                "test-00002".to_string(),
                &eids,
                Points::Values { vals: &points[..] },
            )
            .is_err(),
            true
        );
        // finally, querying should be accomplished quite easily
        let point_search = vec![1.2 * (10000 as f32); dimensions];
        let expect: Vec<EId> = (998..1000)
            .rev()
            .map(|id| {
                let mut eid = EId([0u8; 16]);
                let id_str = id.clone().to_string();
                let id_bytes = id_str.as_bytes();
                eid.0[0..id_bytes.len()].copy_from_slice(&id_bytes[..]);
                eid
            })
            .collect();
        match mgr.search(
            index_name,
            Points::Values {
                vals: &point_search,
            },
            2,
        ) {
            Ok(res) => {
                let result: Vec<EId> = res.iter().map(|x| (x.eid)).collect();
                assert_eq!(expect, result);
            }
            Err(_) => {
                panic!("error should not be thrown on search");
            }
        }
    }
}
