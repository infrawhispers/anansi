use std::collections::HashMap;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;

use anyhow::{bail, Context};
use parking_lot::RwLock;

pub struct DocStore {
    // we store a mapping of: [index_name, EId] -> JSON document
    in_mem_store: RwLock<HashMap<retrieval::ann::EId, serde_json::Value>>,
    rocksdb_instance: Arc<RwLock<rocksdb::DB>>,
    cf_name: String,
}
impl DocStore {
    fn parse_key(key: &str) -> anyhow::Result<retrieval::ann::EId> {
        let key_parts: Vec<&str> = key.splitn(3, "::").collect();
        if key_parts.len() != 3 {
            bail!("invalid key: \"{key}\" retrieved from rocksdb");
        }
        Ok(retrieval::ann::EId::from_str(key_parts[2]).with_context(|| "failed to parse EId")?)
    }

    fn init_fr_rocksdb(&self) -> anyhow::Result<()> {
        let instance = self.rocksdb_instance.read();
        let mut store = self.in_mem_store.write();
        let cf = instance.cf_handle(&self.cf_name).ok_or_else(|| {
            anyhow::anyhow!("unable to fetch the column family: {:?}", self.cf_name)
        })?;
        let mut options = rocksdb::ReadOptions::default();
        options.set_iterate_range(rocksdb::PrefixRange("docstore::eid".as_bytes()));
        for item in instance.iterator_cf_opt(cf, options, rocksdb::IteratorMode::Start) {
            let (key, val) = item?;
            let key_utf8 = std::str::from_utf8(&key)
                .with_context(|| "failed to parse utf8 str from the key")?;
            let json_val = serde_json::from_slice(&val)
                .with_context(|| "failed to deserialize value for key: {key_utf8}")?;
            let eid = DocStore::parse_key(key_utf8)?;
            store.insert(eid, json_val);
        }
        Ok(())
    }

    pub fn get(
        &self,
        eids: &[retrieval::ann::EId],
    ) -> anyhow::Result<HashMap<retrieval::ann::EId, serde_json::Value>> {
        let mut res: HashMap<retrieval::ann::EId, serde_json::Value> = HashMap::new();
        let store = self.in_mem_store.read();
        for eid in eids.iter() {
            let val = store.get(eid).ok_or_else(|| {
                anyhow::anyhow!(
                    "no document associated with eid: {}",
                    retrieval::ann::EId::to_string(eid),
                )
            })?;
            res.insert(*eid, val.clone());
        }
        Ok(res)
    }
    pub fn insert(
        &self,
        src_by_id: HashMap<retrieval::ann::EId, serde_json::Value>,
    ) -> anyhow::Result<()> {
        let instance = self.rocksdb_instance.read();
        let cf = instance
            .cf_handle(&self.cf_name)
            .ok_or_else(|| anyhow::anyhow!("unable to fetch column family: {}", self.cf_name))?;
        let mut batch = rocksdb::WriteBatch::default();
        for (k, v) in src_by_id.iter() {
            let k_bytes: Vec<u8> = ["docstore::eid::".as_bytes(), &k.as_bytes()].concat();
            let v_bytes: Vec<u8> = serde_json::to_vec(v)?;
            batch.put_cf(cf, k_bytes, v_bytes);
        }
        instance.write(batch)?;
        let mut store = self.in_mem_store.write();
        for (k, v) in src_by_id.iter() {
            store.insert(*k, v.clone());
        }
        Ok(())
    }

    pub fn new(
        dir_path: &PathBuf,
        rocksdb_instance: Arc<RwLock<rocksdb::DB>>,
        cf_name: &str,
    ) -> anyhow::Result<DocStore> {
        let obj = DocStore {
            in_mem_store: RwLock::new(HashMap::new()),
            rocksdb_instance: rocksdb_instance,
            cf_name: cf_name.to_string(),
        };
        obj.init_fr_rocksdb()?;
        Ok(obj)
    }
}
