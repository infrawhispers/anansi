use std::collections::HashMap;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;

use anyhow::{bail, Context};
use parking_lot::RwLock;

pub struct DocStore {
    // we store a mapping of: [index_name, EId] -> JSON document
    in_mem_store: RwLock<HashMap<String, RwLock<HashMap<retrieval::ann::EId, serde_json::Value>>>>,
    rocksdb_instance: Arc<RwLock<rocksdb::DB>>,
    cf_name: String,
}
impl DocStore {
    fn parse_key(key: &str) -> anyhow::Result<(String, retrieval::ann::EId)> {
        let key_parts: Vec<&str> = key.splitn(3, "::").collect();
        if key_parts.len() != 3 {
            bail!("invalid key: \"{key}\" retrieved from rocksdb");
        }
        Ok((
            key_parts[1].to_string(),
            retrieval::ann::EId::from_str(key_parts[2]).with_context(|| "failed to parse EId")?,
        ))
    }
    fn to_key(index_name: &str, eid: &retrieval::ann::EId) -> Vec<u8> {
        [
            "index_docs::".as_bytes(),
            index_name.as_bytes(),
            "::".as_bytes(),
            &eid.as_bytes(),
        ]
        .concat()
    }

    fn init_fr_rocksdb(&self) -> anyhow::Result<()> {
        let instance = self.rocksdb_instance.read();
        let mut store = self.in_mem_store.write();
        let cf = instance.cf_handle(&self.cf_name).ok_or_else(|| {
            anyhow::anyhow!("unable to fetch the column family: {:?}", self.cf_name)
        })?;
        let mut options = rocksdb::ReadOptions::default();
        options.set_iterate_range(rocksdb::PrefixRange("index_docs::".as_bytes()));
        for item in instance.iterator_cf_opt(cf, options, rocksdb::IteratorMode::Start) {
            let (key, value) = item?;
            let key_utf8 = std::str::from_utf8(&key)
                .with_context(|| "failed to parse utf8 str from the key")?;
            let json_val = serde_json::to_value(value)
                .with_context(|| "failed to deserialize value for key: {key_utf8}")?;
            let (index_name, eid) = DocStore::parse_key(key_utf8)?;
            if !store.contains_key(&index_name) {
                store.insert(index_name.to_string(), RwLock::new(HashMap::new()));
            }
            let index_doc_store = store
                .get(&index_name)
                .ok_or_else(|| anyhow::anyhow!("unexpectedly the index store is missing"))?;
            index_doc_store.write().insert(eid, json_val);
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
