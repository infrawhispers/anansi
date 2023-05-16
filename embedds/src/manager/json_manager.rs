use std::cmp::Ordering;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;

use anyhow::{bail, Context};
use nanoid::nanoid;
use parking_lot::RwLock;
use tracing::{info, warn};

use retrieval::manager::index_manager::IndexManager;
use serde::{Deserialize, Serialize};
use serde_json::{Result, Value};

extern crate rmp_serde as rmps;

#[derive(Debug)]
pub struct IndexSearch<'a> {
    pub embedding: &'a [f32],
    pub attributes: &'a [&'a str],
    pub weighting: HashMap<String, f32>,
    pub limit: usize,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct IndexSettings {
    // configuration for the actual index itself
    pub metric_type: String,
    pub index_type: String,
    pub index_params: retrieval::ann::ANNParams,
    // configuration needed to auto-generate embeddings
    pub embedding_model_name: String,
    pub embedding_model_class: crate::api::ModelClass,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct IndexDetails {
    // settings that will be used to create a subsequent list of
    // indexing fields
    settings: IndexSettings,
    // json field name -> actual index name
    sub_index_by_name: RwLock<HashMap<String, String>>,
}

pub struct JSONIndexManager {
    index_mgr: Arc<IndexManager>,
    cf_name: String,
    index_details: RwLock<HashMap<String, Arc<IndexDetails>>>,
    // within rocksdb we store the following:
    // b"{index_name}" -> b"{index_details}
    rocksdb_instance: Arc<rocksdb::DB>,
}

#[derive(Debug)]
struct Node<'a> {
    ann_node: retrieval::ann::Node,
    field: &'a str,
}

#[derive(Debug, Serialize)]
pub struct NodeHit {
    pub id: String,
    pub distance: f32,
    pub document: Option<String>,
}

impl Ord for Node<'_> {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.ann_node.distance == other.ann_node.distance {
            return std::cmp::Ordering::Equal;
        }
        if self.ann_node.distance < other.ann_node.distance {
            return std::cmp::Ordering::Less;
        }
        std::cmp::Ordering::Greater
    }
}

impl PartialOrd for Node<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Node<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Node<'_> {}

impl JSONIndexManager {
    pub fn new(directory: &Path) -> anyhow::Result<Self> {
        let cf_name = "json_index_manager";
        // create the directory if it does not exist
        fs::create_dir_all(directory)?;
        let mut dir_path = PathBuf::from(directory);
        // create the IndicesMgr directory
        let mut indices_path = dir_path.clone();
        indices_path.push("indices");
        fs::create_dir_all(&indices_path)?;

        // create the JSONIndexMgr directory
        dir_path.push("index_mgr");
        fs::create_dir_all(&dir_path)?;

        // finally open a rocksdb entry to the database
        let mut options = rocksdb::Options::default();
        options.set_error_if_exists(false);
        options.create_if_missing(true);
        options.create_missing_column_families(true);
        let cfs = rocksdb::DB::list_cf(&options, dir_path.clone()).unwrap_or(vec![]);
        let cf_missing = cfs.iter().find(|cf| cf == &cf_name).is_none();
        let mut instance = rocksdb::DB::open_cf(&options, dir_path.clone(), cfs)?;

        if cf_missing {
            // create a the column family if it is missing
            let options = rocksdb::Options::default();
            instance.create_cf(cf_name, &options)?;
        }

        let obj = JSONIndexManager {
            cf_name: cf_name.to_string(),
            index_mgr: Arc::new(IndexManager::new(&indices_path)),
            index_details: RwLock::new(HashMap::new()),
            rocksdb_instance: Arc::new(instance),
        };
        let indices = obj.indices_fr_rocksdb()?;
        indices.iter().for_each(|(index, details)| {
            info!("loading index: {index}");
            details
                .sub_index_by_name
                .read()
                .iter()
                .for_each(|(field, fqn)| {
                    match obj.index_mgr.new_index(
                        fqn,
                        &details.settings.index_type,
                        &details.settings.metric_type,
                        &details.settings.index_params,
                    ) {
                        Ok(()) => {
                            info!("loaded sub_index: {fqn}")
                        }
                        Err(err) => {
                            warn!("unable to load sub_index: {fqn} | err: {err}")
                        }
                    }
                });
        });
        obj.index_details.write().extend(indices);
        Ok(obj)
    }

    fn delete_index_fr_rocksdb(&self, index_name: &str) -> anyhow::Result<()> {
        let cf = self
            .rocksdb_instance
            .cf_handle(&self.cf_name)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "unable to fetch column family: {:?} was it created?",
                    self.cf_name
                )
            })?;
        // let details_b = rmp_serde::to_vec(index_details)?;
        let index_name_full = format!("index::{}", index_name);
        self.rocksdb_instance
            .delete_cf(cf, index_name_full.as_bytes())?;
        Ok(())
    }
    fn index_to_rocksdb(
        &self,
        index_name: &str,
        index_details: Vec<u8>,
        // index_details: &IndexDetails,
    ) -> anyhow::Result<()> {
        let cf = self
            .rocksdb_instance
            .cf_handle(&self.cf_name)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "unable to fetch column family: {:?} was it created?",
                    self.cf_name
                )
            })?;
        let index_name_full = format!("index::{}", index_name);
        self.rocksdb_instance
            .put_cf(cf, index_name_full.as_bytes(), index_details)?;
        Ok(())
    }
    fn indices_fr_rocksdb(&self) -> anyhow::Result<HashMap<String, Arc<IndexDetails>>> {
        let mut res: HashMap<String, Arc<IndexDetails>> = HashMap::new();
        let cf = self
            .rocksdb_instance
            .cf_handle(&self.cf_name)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "unable to fetch column family: {:?} was it created?",
                    self.cf_name
                )
            })?;
        let mut options = rocksdb::ReadOptions::default();
        options.set_iterate_range(rocksdb::PrefixRange("index::".as_bytes()));
        for item in self
            .rocksdb_instance
            .iterator_cf_opt(cf, options, rocksdb::IteratorMode::Start)
        {
            let (key, value) = item?;
            let key_utf8 = std::str::from_utf8(&key)
                .with_context(|| "failed to parse utf8 str from the key")?;

            let key_parts: Vec<&str> = key_utf8.splitn(2, "::").collect();
            if key_parts.len() != 2 {
                warn!(
                    "key: \"{:?}\" does not fit the expected format \"index::{{index_name}}\"",
                    key_utf8
                );
                continue;
            }
            let index_name = key_parts[key_parts.len() - 1];
            info!("index_name: {index_name}");
            let key_settings: IndexDetails = rmp_serde::from_slice(&value)?;
            res.insert(index_name.to_string(), Arc::new(key_settings));
        }
        info!("num ann-index metadata: {}", res.len());
        Ok(res)
    }

    pub fn delete_index(&self, index_name: &str) -> anyhow::Result<()> {
        {
            let settings_r = self.index_details.read();
            let settings;
            match settings_r.get(index_name) {
                Some(res) => settings = res,
                None => {
                    warn!(
                        "delete requested for index: {}, which does not exist",
                        index_name
                    );
                    return Ok(());
                }
            }
            // we take out an exclusive lock on the index structure
            // which blocks the writes on the sub-indices
            let mut sub_indices = settings.sub_index_by_name.write();
            let mut rm_success: Vec<String> = Vec::new();
            let mut rm_errs: Vec<anyhow::Error> = Vec::new();
            // since we are iterating through the items, we cannot update
            // remove the items as we get them
            for (field, sub_index_name) in sub_indices.iter() {
                match self.index_mgr.delete_index(sub_index_name) {
                    Ok(()) => {
                        rm_success.push(sub_index_name.to_string());
                        info!("succesfully removed - field: {field} sub_index: {sub_index_name}");
                    }
                    Err(err) => {
                        rm_errs.push(err);
                    }
                }
            }
            // remove everything that we have from the in-memory index
            // to sub-index mapper
            rm_success.iter().for_each(|sub_index_name| {
                sub_indices.remove(sub_index_name);
            });
            if rm_errs.len() != 0 {
                return Err(anyhow::anyhow!("first err returned: {}", rm_errs[0]));
            }
        }
        // finally remove the metadata and clear everything out
        self.delete_index_fr_rocksdb(index_name)
            .with_context(|| format!("unable to remove metadata for index: {index_name}"))?;
        let mut settings_w = self.index_details.write();
        settings_w.remove(index_name);
        Ok(())
    }

    pub fn search(&self, index_name: &str, req: &IndexSearch) -> anyhow::Result<Vec<NodeHit>> {
        let index = self.index_details.read();
        let details = index
            .get(index_name)
            .ok_or_else(|| anyhow::anyhow!("index: {index_name} does not exist"))?;
        // default to all ANN sub-indices that we have created
        let sub_indices = details.sub_index_by_name.read();
        let active_attributes: Vec<String> = sub_indices
            .keys()
            .map(|k| k.clone())
            .collect::<Vec<String>>();
        let mut attributes: Vec<String> = active_attributes.clone();
        if req.attributes.len() != 0 {
            attributes.clear();
            attributes.extend(
                req.attributes
                    .iter()
                    .map(|x| (*x).to_string())
                    .collect::<Vec<String>>(),
            ); //eq.attributes
        }
        // generate the boost_vals - this is necessary in order to do
        // any boosting
        let re_rank = req.weighting.len() > 0;
        let mut boost_vals: HashMap<String, f32> = HashMap::new();
        let mut boost_denom: f32 = 0.0;
        if re_rank {
            req.weighting.values().for_each(|w| boost_denom += w);
            req.weighting.iter().for_each(|(attr, boost)| {
                boost_vals.insert(attr.to_string(), 1.0 - (boost / boost_denom));
            });
        }

        // walk through all the attributes and generate the nodes
        // that we care about
        let mut max_distance: f32 = f32::MIN;
        let mut all_nodes: Vec<Node> = Vec::new();
        for attribute in attributes.iter() {
            let fqn = sub_indices.get(attribute).ok_or_else(|| {
                anyhow::anyhow!(
                    "unknown attribute: {attribute}, active attributes are: {active_attributes:?}"
                )
            })?;
            let nodes = self.index_mgr.search(
                fqn,
                retrieval::ann::Points::Values {
                    vals: &req.embedding,
                },
                req.limit,
            )?;
            let tagged_nodes: Vec<Node> = nodes
                .into_iter()
                .map(|x| {
                    if x.distance > max_distance {
                        max_distance = x.distance
                    }
                    Node {
                        ann_node: x,
                        field: &attribute,
                    }
                })
                .collect();
            all_nodes.extend(tagged_nodes);
        }
        // normalize by all the distances

        if re_rank {
            all_nodes.iter_mut().for_each(|x| {
                let boost = boost_vals.get(x.field).unwrap_or(&1.0);
                x.ann_node.distance = (x.ann_node.distance / max_distance) * boost;
            });
        }
        // sort all the nodes
        all_nodes.sort();
        // then take the best k
        Ok(all_nodes
            .into_iter()
            .take(req.limit)
            .map(|nn| NodeHit {
                id: nn.ann_node.eid.to_string(),
                distance: nn.ann_node.distance,
                document: None,
            })
            .collect::<Vec<NodeHit>>())
        // info!("all_nodes: {:?}", all_nodes);
        // Ok(())
    }
    pub fn insert(&self, index_name: &str, data: Vec<crate::IndexItems>) -> anyhow::Result<()> {
        for req in data.iter() {
            let sub_index_name = &req.sub_indices[0];
            let mut must_create: bool = false;
            {
                let details = self.index_details.read();
                match details.get(index_name) {
                    None => {
                        bail!("unknown index: {index_name}, was it previously created?")
                    }
                    Some(res) => {
                        let index_details = res.sub_index_by_name.read();
                        if !index_details.contains_key(sub_index_name) {
                            must_create = true;
                        }
                    }
                }
            }
            if must_create {
                self.create_sub_index(index_name, sub_index_name);
            }
            let vals: Vec<f32> = req
                .embedds
                .as_ref()
                .ok_or_else(|| {
                    anyhow::anyhow!("IndexItems should have embedds set for insert(..)")
                })?
                .clone()
                .into_iter()
                .flatten()
                .collect();
            self.index_mgr.insert(
                format!("{}.{}", index_name, req.sub_indices[0]),
                &req.ids,
                retrieval::ann::Points::Values { vals: &vals },
            )?;
        }
        Ok(())
    }

    fn create_sub_index(&self, index_name: &str, field_name: &str) -> anyhow::Result<()> {
        let details_r = self.index_details.read();
        let index_details = details_r.get(index_name).ok_or_else(|| {
            anyhow::anyhow!(
                "index: \"{:?}\" was not previously created - this is a programming error!",
                index_name
            )
        })?;
        {
            // check if the sub-index already exists!
            let sub_index = index_details.sub_index_by_name.read();
            if sub_index.contains_key(field_name) {
                bail!(
                    "sub_index: {:?} for index: {:?} already exists",
                    field_name,
                    index_name
                )
            }
        }
        // sub-index does not exist, so we need to go and create everything
        let sub_index_name = format!("{}.{}", index_name, field_name);
        self.index_mgr.new_index(
            &sub_index_name,
            &index_details.settings.index_type,
            &index_details.settings.metric_type,
            &index_details.settings.index_params,
        )?;
        info!("created sub_index: {sub_index_name} within index: {index_name}");
        // finally add our sub-index to the list of indexes that we care about
        {
            let mut sub_index = index_details.sub_index_by_name.write();
            sub_index.insert(field_name.to_string(), sub_index_name.to_string());
        }
        let details_b = rmp_serde::to_vec(index_details)?;
        // optimistically put it into rocksdb
        match self.index_to_rocksdb(index_name, details_b) {
            Ok(()) => {}
            Err(err) => {
                info!("got an error while trying to put it into rocksdb: {err}");
                // sub_index.remove(field_name);
                return Err(err);
            }
        }

        Ok(())
    }

    pub fn create_index(
        &self,
        index_name: &str,
        field_names: &[String],
        settings: &IndexSettings,
    ) -> anyhow::Result<()> {
        {
            let indices = self.index_details.read();
            if indices.contains_key(index_name) {
                bail!("index: {index_name} already exists",);
            }
        }
        let details = IndexDetails {
            settings: settings.clone(),
            sub_index_by_name: HashMap::new().into(),
        };
        let details_b = rmp_serde::to_vec(&details)?;

        {
            // stick it in persistent storage
            self.index_to_rocksdb(index_name, details_b)?;
            // stick it in the in-memory location that we read from
            let mut settings_w = self.index_details.write();
            settings_w.insert(index_name.to_string(), Arc::new(details));
        }
        // now create the sub-indices that we care about
        for field_name in field_names.iter() {
            self.create_sub_index(index_name, field_name)?;
        }
        Ok(())
    }

    fn index(&self, index_name: &str, data: serde_json::Value) -> anyhow::Result<()> {
        Ok(())
    }

    fn get_obj_id(&self, obj: &Value) -> anyhow::Result<String> {
        // pulls an id from the from the object which is expected to be
        // at the top level under "_id" or creates one
        let id_as_str: String;
        match obj.get("_id") {
            Some(id) => {
                id_as_str = id
                    .as_str()
                    .ok_or_else(|| anyhow::anyhow!("id: is not a string, bailing",))?
                    .to_string();
            }
            None => {
                let alphabet: [char; 36] = [
                    '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'A', 'B', 'C', 'D', 'E', 'F',
                    'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
                    'W', 'X', 'Y', 'Z',
                ];
                id_as_str = nanoid!(16, &alphabet).to_string();
            }
        }
        Ok(id_as_str.to_string())
    }

    /// add content to embedd to the relevant HashMap, we organize our content
    /// based on sub_index and group *across* documents. this is a good tradeoff
    /// for:
    /// 1. reducing the number of embedd() calls
    /// 2. allowing for the mixture of embedding models within a larger index
    /// 3. simplifying the indexing process, allowing for the batching of index reqs
    fn add_to_sub_indices_items(
        &self,
        id: &str,
        content: &str,
        sub_index: &str,
        items: &mut HashMap<String, crate::IndexItems>,
        settings: &IndexSettings,
    ) -> anyhow::Result<()> {
        if items.contains_key(sub_index) {
            let to_encode = items
                .get_mut(sub_index)
                .ok_or_else(|| anyhow::anyhow!("unable to fetch the releveant IndexItems"))?;
            to_encode
                .to_embedd
                .as_mut()
                .ok_or_else(|| anyhow::anyhow!("unexpectedly empty to_embedd obj"))?
                .text
                .push(content.to_string());
            to_encode.ids.push(retrieval::ann::EId::from_str(id)?);
            to_encode.sub_indices.push(sub_index.to_string());
            return Ok(());
        }
        // everything uses the *same* model at the moment, we may need to be
        // smarter if we offer clients the ability to mix and match
        let to_encode = crate::IndexItems {
            embedds: None,
            to_embedd: Some(crate::api::EncodeItem {
                model_name: settings.embedding_model_name.clone(),
                model_class: settings.embedding_model_class.clone().into(),
                text: vec![content.to_string()],
                // these remain unused atm
                image: Vec::new(),
                image_uri: Vec::new(),
                instructions: Vec::new(),
            }),
            ids: vec![retrieval::ann::EId::from_str(id)?],
            sub_indices: vec![sub_index.to_string()],
        };
        items.insert(sub_index.to_string(), to_encode);
        Ok(())
    }

    /// converts a mapping of doc_id -> doc into a de-normalized (i.e flattened) IndexReq
    /// which contains the content_ids and associated sub-indices, as well as either the
    /// data to be embedded or the pre-computed embeddings
    fn docs_to_embedd_req(
        &self,
        settings: &IndexSettings,
        map: &HashMap<String, serde_json::Map<String, Value>>,
    ) -> anyhow::Result<Vec<crate::IndexItems>> {
        // let mut res: Vec<crate::IndexItems> = Vec::new();
        let mut items: HashMap<String, crate::IndexItems> = HashMap::new();
        for (doc_id, mappings) in map.iter() {
            for (k, v) in mappings {
                // skip everything that has the _id - this avoids any useless work on
                // the part of the embedding generators
                if k == "_id" {
                    continue;
                }
                if v.is_string() {
                    let val = v.as_str().ok_or_else(|| {
                        anyhow::anyhow!("unable to convert to str | k: {k} v: {v} doc_id: {doc_id}")
                    })?;
                    self.add_to_sub_indices_items(doc_id, val, k, &mut items, settings)?;
                } else if v.is_array() {
                    let arr = v.as_array().ok_or_else(|| {
                        anyhow::anyhow!("unable to convert to arr | k: {k}, v: {v}")
                    })?;
                    for (idx, arr_item) in arr.iter().enumerate() {
                        if arr_item.is_string() {
                            self.add_to_sub_indices_items(
                                &format!("{}:{idx}", doc_id.to_string()),
                                &arr_item.to_string(),
                                k,
                                &mut items,
                                settings,
                            )?;
                        }
                    }
                }
            }
        }
        let res: Vec<crate::IndexItems> = items.into_values().collect();
        Ok(res)
    }

    pub fn transform_search_req(
        &self,
        index_name: &str,
        queries: &[crate::api::SearchQuery],
    ) -> anyhow::Result<Vec<crate::IndexItems>> {
        // fetch the index as that contains the embedding details
        let details = self.index_details.read();
        let index_settings;
        match details.get(index_name) {
            Some(res) => index_settings = res,
            None => {
                bail!("index: {index_name} does not exist");
            }
        }
        // TODO(infrawhispers) - we can optimize this by grouping the IndexItems
        // this creates a copy of existing embedding search requests
        let mut res: Vec<crate::IndexItems> = Vec::with_capacity(queries.len());
        for q in queries.iter() {
            match &q.query {
                Some(crate::api::search_query::Query::Embedding(e)) => {
                    res.push(crate::IndexItems {
                        ids: vec![retrieval::ann::EId::from_str(&e.id)?],
                        sub_indices: Vec::new(),
                        embedds: Some(vec![e.vals.clone()]),
                        to_embedd: None,
                    })
                }
                Some(crate::api::search_query::Query::Content(c)) => {
                    let text;
                    match &c.data {
                        Some(crate::api::content::Data::Text(v)) => text = v,
                        Some(crate::api::content::Data::Image(_))
                        | Some(crate::api::content::Data::ImageUri(_)) => {
                            bail!("{{image, image_uri}} are not currently supported")
                        }
                        None => {
                            bail!(
                                "one of {{text}} must be specified when providing content queries "
                            )
                        }
                    }
                    res.push(crate::IndexItems {
                        ids: vec![retrieval::ann::EId::from_str(&c.id)?],
                        sub_indices: Vec::new(),
                        embedds: None,
                        to_embedd: Some(crate::api::EncodeItem {
                            model_name: index_settings.settings.embedding_model_name.clone(),
                            model_class: index_settings.settings.embedding_model_class.into(),
                            text: vec![text.to_string()],
                            image: Vec::new(),
                            image_uri: Vec::new(),
                            instructions: Vec::new(),
                        }),
                    });
                }
                None => {
                    bail!("one of {{embedding, content}} must be set")
                }
            }
        }
        Ok(res)
    }

    pub fn extract_documents(
        &self,
        index_name: &str,
        data: &str,
    ) -> anyhow::Result<Vec<crate::IndexItems>> {
        // we need the index to exist in order to select the type of
        // Model to attach to the embedding request
        let details = self.index_details.read();
        let index_settings;
        match details.get(index_name) {
            Some(res) => index_settings = res,
            None => {
                bail!("index: {index_name} does not exist");
            }
        }
        let mut res: HashMap<String, serde_json::Map<String, Value>> = HashMap::new();
        let obj: Value = serde_json::Value::from_str(data)
            .with_context(|| "unable to parse supplied JSON, must be one of: [obj, arr]")?;
        if obj.is_object() {
            // a single object is passed
            let id = self.get_obj_id(&obj)?;
            let val = obj
                .as_object()
                .ok_or_else(|| anyhow::anyhow!("obj must be a valid JSON object"))?;
            res.insert(id.to_string(), super::utils::flatten(val));
        } else if obj.is_array() {
            // a list of objects is passed, we only look at the top level
            // entries
            for (idx, item) in obj
                .as_array()
                .ok_or_else(|| anyhow::anyhow!("obj expected to be an arr"))?
                .iter()
                .enumerate()
            {
                let id = self.get_obj_id(&item)?;
                let val = item.as_object().ok_or_else(|| {
                    anyhow::anyhow!("the obj ({item}) at idx: {idx} must be a valid JSON object")
                })?;
                res.insert(id.to_string(), super::utils::flatten(val));
            }
        } else {
            bail!("unable to split supplied JSON, must be one of: [obj, arr]");
        }
        let req = self.docs_to_embedd_req(&index_settings.settings, &res)?;
        Ok(req)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_json_manager() {
        let dir_path = PathBuf::from(".data/test/json_manager");
        let _ = fs::remove_dir_all(dir_path.clone());
        fs::create_dir_all(dir_path.clone()).expect("unable to create the test directory");

        let mgr = JSONIndexManager::new(&dir_path)
            .expect("creation of a fresh manager from an existing instance");
        let params: retrieval::ann::ANNParams = retrieval::ann::ANNParams::FlatLite {
            params: retrieval::flat_lite::FlatLiteParams {
                dim: 768,
                segment_size_kb: 1024,
            },
        };

        let settings = IndexSettings {
            metric_type: "MetricL2".to_string(),
            index_type: "Flat".to_string(),
            embedding_model_name: "VIT_L_14_336_OPENAI".to_string(),
            embedding_model_class: crate::api::ModelClass::Clip,
            index_params: params,
        };
        let fields = vec!["paragraph".to_string(), "title".to_string()];
        mgr.create_index("test-0000", &fields, &settings)
            .expect("index creation should be a no-op");
        // assert!();
        info!("run the insertions into two seperate indices");
        let index_req = vec![
            crate::IndexItems {
                ids: vec![retrieval::ann::EId::from_str("1").unwrap()],
                sub_indices: vec!["paragraph".to_string()],
                embedds: Some(vec![vec![0.0; 768]]),
                to_embedd: None,
            },
            crate::IndexItems {
                ids: vec![retrieval::ann::EId::from_str("2").unwrap()],
                sub_indices: vec!["title".to_string()],
                embedds: Some(vec![vec![10.0; 768]]),
                to_embedd: None,
            },
        ];
        mgr.insert("test-0000", index_req)
            .expect("insertion should work");
        let search_req = IndexSearch {
            embedding: vec![2.0; 768],
            attributes: vec!["paragraph".to_string(), "title".to_string()],
            weighting: HashMap::from([
                ("paragraph".to_string(), 4 as f32),
                ("title".to_string(), 1 as f32),
            ]),
            limit: 10,
        };
        let search_res = mgr
            .search("test-0000", &search_req)
            .expect("search should work");
        let expected: Vec<NodeHit> = vec![
            NodeHit {
                id: "1".to_string(),
                distance: 0.012499999,
                document: None,
            },
            NodeHit {
                id: "2".to_string(),
                distance: 0.8,
                document: None,
            },
        ];
        assert_eq!(
            serde_json::to_string(&search_res).expect("[search_res] serialization should not err"),
            serde_json::to_string(&expected).expect("[expected] serialization should not err")
        );

        let search_req_bad = IndexSearch {
            embedding: vec![2.0; 768],
            attributes: vec!["non-existent".to_string()],
            weighting: HashMap::new(),
            limit: 10,
        };
        assert!(mgr.search("test-0000", &search_req_bad).is_err());
    }
}
