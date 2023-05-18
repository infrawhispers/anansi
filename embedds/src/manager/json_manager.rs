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
    pub embedding: Vec<f32>,
    pub attributes: &'a [&'a str],
    pub weighting: &'a HashMap<String, f32>,
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

struct CollectionMgrParams<'a> {
    field_names: &'a [String],
    index_settings: IndexSettings,
}

struct CollectionMgr {
    name: String,
    // this houses the actual ANN indices, which does the core insert
    // serarch and filtering operations
    index_mgr: Arc<IndexManager>,
    // raw documents that have been submitted to this collection
    // these are stored in rocksdb and allow for easy access at
    // query time.
    doc_store: Arc<crate::manager::doc_store::DocStore>,
    // settings that will be used to create a subsequent list of
    // indexing fields
    settings: IndexSettings,
    // json field_name -> actual index name
    sub_index_by_name: RwLock<HashMap<String, String>>,
    // we hold onto the rocksdb instance here
    rocksdb_instance: Arc<RwLock<rocksdb::DB>>,
}

impl CollectionMgr {
    fn init_sub_indices(&self) -> anyhow::Result<()> {
        for (field_name, idx_name) in self.sub_index_by_name.write().iter() {
            self.index_mgr.new_index(
                &idx_name,
                &self.settings.index_type,
                &self.settings.metric_type,
                &self.settings.index_params,
            )?;
        }
        Ok(())
    }
    fn create_sub_index(&self, field_name: &str) -> anyhow::Result<()> {
        {
            // check if the sub-index already exists!
            let sub_indices = self.sub_index_by_name.read();
            if sub_indices.contains_key(field_name) {
                bail!(
                    "collection: {}, sub_index: {} already exists ",
                    self.name,
                    field_name
                )
            }
        }
        // sub-index does not exist, so we need to go and create everything
        let sub_index_name = format!("{}.{}", self.name, field_name);
        self.index_mgr.new_index(
            &sub_index_name,
            &self.settings.index_type,
            &self.settings.metric_type,
            &self.settings.index_params,
        )?;
        info!(
            "collection: {}, created sub_index: {}",
            self.name, field_name,
        );
        // write out the sub_index details to rocksdb and update our in-memory listings
        {
            let instance = self.rocksdb_instance.read();
            let cf = instance.cf_handle(&self.name).ok_or_else(|| {
                anyhow::anyhow!(
                    "unable to fetch column family: {} was it created?",
                    self.name
                )
            })?;
            let mut sub_idx = self.sub_index_by_name.write();
            let sub_idx_b = rmp_serde::to_vec(&*sub_idx)?;
            instance.put_cf(cf, b"sub_index_by_name", sub_idx_b);
            sub_idx.insert(field_name.to_string(), sub_index_name.to_string());
        }
        Ok(())
    }

    fn insert_data(&self, data: Vec<crate::IndexItems>) -> anyhow::Result<()> {
        for req in data.iter() {
            let field_name = &req.sub_indices[0];
            let mut must_create: bool = false;
            {
                match self.sub_index_by_name.read().get(field_name) {
                    None => {
                        must_create = true;
                    }
                    Some(res) => {
                        must_create = false;
                    }
                }
            }
            if must_create {
                self.create_sub_index(field_name);
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
                format!("{}.{}", self.name, field_name),
                &req.ids,
                retrieval::ann::Points::Values { vals: &vals },
            )?;
        }
        Ok(())
    }

    fn delete(&self) -> anyhow::Result<()> {
        let mut sub_indices = self.sub_index_by_name.write();
        let mut rm_success: Vec<String> = Vec::new();
        let mut rm_errs: Vec<anyhow::Error> = Vec::new();
        // since we are iterating through the items, we cannot update
        // remove the items as we get them
        for (field, index_name) in sub_indices.iter() {
            match self.index_mgr.delete_index(index_name) {
                Ok(()) => {
                    rm_success.push(index_name.to_string());
                    info!("succesfully removed - field: {field} sub_index: {index_name}");
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
        // blow away the stuff existing in rocksdb
        let instance = self.rocksdb_instance.write();
        let cf = instance.cf_handle(&self.name).ok_or_else(|| {
            anyhow::anyhow!(
                "unable to fetch column family: {:?} was it created?",
                self.name
            )
        })?;
        instance.delete_cf(cf, self.name.as_bytes())?;
        Ok(())
    }

    fn new(
        name: &str,
        // fields map 1:1 with the sub-indices that we create
        dir_path: &PathBuf,
        // rocksdb_instance that will be used for this given collection
        rocksdb_instance: Arc<RwLock<rocksdb::DB>>,
        // actual collection of indices that are used by the database
        index_mgr: Arc<IndexManager>,
        init_params: Option<CollectionMgrParams>,
    ) -> anyhow::Result<Self> {
        info!("cf_name: {name} ");
        let cf_name = name;
        // do any initialization for the columnfamily for this collection
        let mut options = rocksdb::Options::default();
        options.set_error_if_exists(false);
        options.create_if_missing(true);
        options.create_missing_column_families(true);
        let cfs = rocksdb::DB::list_cf(&options, dir_path.clone()).unwrap_or(vec![]);
        let cf_missing = cfs.iter().find(|cf| cf == &cf_name).is_none();
        if cf_missing {
            // create the column family if it is missing
            let options = rocksdb::Options::default();
            rocksdb_instance.write().create_cf(cf_name, &options)?;
        }

        let settings: IndexSettings;
        let sub_index_by_name: RwLock<HashMap<String, String>>;
        let is_init = init_params.is_some();
        {
            let instance = rocksdb_instance.read();
            let cf = instance.cf_handle(name).ok_or_else(|| {
                anyhow::anyhow!("unable to fetch column family: {name} was it created?",)
            })?;

            match instance.get_cf(cf, "index_settings".as_bytes()) {
                Ok(Some(val)) => {
                    settings = rmp_serde::from_slice(&val)?;
                }
                Ok(None) => {
                    settings = init_params
                        .as_ref()
                        .ok_or_else(|| {
                            anyhow::anyhow!(
                                "unexpectedly missing \"index_settings\" for collection: {name}",
                            )
                        })?
                        .index_settings
                        .clone();
                }
                Err(err) => {
                    bail!("unable to fetch: \"index_settings\"")
                }
            }

            match instance.get_cf(cf, "sub_index_by_name".as_bytes()) {
                Ok(Some(val)) => {
                    sub_index_by_name = rmp_serde::from_slice(&val)?;
                }
                Ok(None) => {
                    // let field_names =
                    sub_index_by_name = RwLock::new(HashMap::new());
                    for field_name in init_params
                        .as_ref()
                        .ok_or_else(|| {
                            anyhow::anyhow!(
                                "unexpectedly missing \"sub_index_by_name\" for collection: {name}",
                            )
                        })?
                        .field_names
                        .iter()
                    {
                        sub_index_by_name
                            .write()
                            .insert(field_name.to_string(), format!("{name}.{field_name}"));
                    }
                    // .for_each(|n| {

                    // });
                }
                Err(err) => {
                    bail!("unable to fetch: \"sub_index_by_name\"")
                }
            }
        }

        if is_init {
            // write out the empty values that we have atm
            let instance = rocksdb_instance.write();
            let mut batch = rocksdb::WriteBatch::default();
            let cf = instance.cf_handle(name).ok_or_else(|| {
                anyhow::anyhow!("unable to fetch column family: {name} was it created?",)
            })?;
            let sub_idx_b = rmp_serde::to_vec(&*sub_index_by_name.read())?;
            let index_settings_b = rmp_serde::to_vec(&settings)?;

            batch.put_cf(cf, b"sub_index_by_name", sub_idx_b);
            batch.put_cf(cf, b"index_settings", index_settings_b);
            instance.write(batch)?;
        }

        let obj = CollectionMgr {
            name: name.to_string(),
            doc_store: Arc::new(super::doc_store::DocStore::new(
                &dir_path,
                rocksdb_instance.clone(),
                cf_name,
            )?),
            index_mgr: index_mgr,
            settings: settings,
            sub_index_by_name: RwLock::new(HashMap::new()),
            rocksdb_instance: rocksdb_instance.clone(),
        };
        obj.init_sub_indices()?;

        Ok(obj)
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
            let batch = to_encode
                .to_embedd
                .as_mut()
                .ok_or_else(|| anyhow::anyhow!("unable to fetch the releveant EncodeBatch"))?;
            match batch.content {
                Some(crate::api::encode_batch::Content::Text(ref mut o)) => {
                    o.data.push(crate::api::Content {
                        id: id.to_string(),
                        data: Some(crate::api::content::Data::Value(content.to_string())),
                        instruction: "".to_string(),
                    });
                }
                Some(crate::api::encode_batch::Content::Images(_))
                | Some(crate::api::encode_batch::Content::ImageUris(_))
                | None => {
                    bail!("programming error")
                }
            }
            to_encode.ids.push(retrieval::ann::EId::from_str(id)?);
            to_encode.sub_indices.push(sub_index.to_string());
            return Ok(());
        }
        // everything uses the *same* model at the moment, we may need to be
        // smarter if we offer clients the ability to mix and match
        let to_encode = crate::IndexItems {
            embedds: None,
            to_embedd: Some(crate::api::EncodeBatch {
                model_name: settings.embedding_model_name.clone(),
                model_class: settings.embedding_model_class.clone().into(),
                content: Some(crate::api::encode_batch::Content::Text(
                    crate::api::TextContent {
                        data: vec![crate::api::Content {
                            id: id.to_string(),
                            data: Some(crate::api::content::Data::Value(content.to_string())),
                            instruction: "".to_string(),
                        }],
                    },
                )),
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
        map: &HashMap<String, serde_json::Map<String, Value>>,
    ) -> anyhow::Result<Vec<crate::IndexItems>> {
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
                    self.add_to_sub_indices_items(doc_id, val, k, &mut items, &self.settings)?;
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
                                &self.settings,
                            )?;
                        }
                    }
                }
            }
        }
        let res: Vec<crate::IndexItems> = items.into_values().collect();
        Ok(res)
    }

    fn insert_preprocess(&self, data: &str) -> anyhow::Result<Vec<crate::IndexItems>> {
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
        self.docs_to_embedd_req(&res)
    }

    fn search_preprocess(
        &self,
        queries: &[crate::api::SearchQuery],
    ) -> anyhow::Result<Vec<crate::IndexItems>> {
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
                Some(crate::api::search_query::Query::Text(c)) => res.push(crate::IndexItems {
                    ids: vec![retrieval::ann::EId::from_str(&c.id)?],
                    sub_indices: Vec::new(),
                    embedds: None,
                    to_embedd: Some(crate::api::EncodeBatch {
                        model_name: self.settings.embedding_model_name.clone(),
                        model_class: self.settings.embedding_model_class.into(),
                        content: Some(crate::api::encode_batch::Content::Text(
                            crate::api::TextContent {
                                data: vec![c.clone()],
                            },
                        )),
                    }),
                }),
                &Some(crate::api::search_query::Query::ImageUri(_))
                | &Some(crate::api::search_query::Query::ImageBytes(_)) => todo!(),
                None => {
                    bail!("one of [embedding, content] must be set")
                }
            }
        }
        Ok(res)
    }

    fn search(&self, req: &IndexSearch) -> anyhow::Result<Vec<NodeHit>> {
        let sub_indices = self.sub_index_by_name.read();
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
                "unknown attribute: \"{attribute}\", active attributes are: {active_attributes:?}"
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
        // sort all the nodes then take the best k
        all_nodes.sort();
        Ok(all_nodes
            .into_iter()
            .take(req.limit)
            .map(|nn| NodeHit {
                id: nn.ann_node.eid.to_string(),
                distance: nn.ann_node.distance,
                document: None,
            })
            .collect::<Vec<NodeHit>>())
    }
}

pub struct JSONIndexManager {
    index_mgr: Arc<IndexManager>,
    cf_name: String,
    dir_path: PathBuf,
    index_details: RwLock<HashMap<String, Arc<CollectionMgr>>>,
    // within rocksdb we store the following:
    // b"{index_name}" -> b"{index_details}
    rocksdb_instance: Arc<RwLock<rocksdb::DB>>,
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
        let rocksdb_instance = Arc::new(RwLock::new(rocksdb::DB::open_cf(
            &options,
            dir_path.clone(),
            cfs,
        )?));

        if cf_missing {
            // create a the column family if it is missing
            let options = rocksdb::Options::default();
            rocksdb_instance.write().create_cf(cf_name, &options)?;
        }
        let index_mgr = Arc::new(IndexManager::new(&indices_path));

        let obj = JSONIndexManager {
            cf_name: cf_name.to_string(),
            dir_path: dir_path,
            index_mgr: index_mgr,
            index_details: RwLock::new(HashMap::new()),
            rocksdb_instance: rocksdb_instance,
        };
        let indices = obj.mgrs_fr_rocksdb()?;
        obj.index_details.write().extend(indices);
        Ok(obj)
    }

    fn get_collections(&self) -> anyhow::Result<Vec<String>> {
        let instance = self.rocksdb_instance.read();
        let cf = instance.cf_handle(&self.cf_name).ok_or_else(|| {
            anyhow::anyhow!(
                "unable to fetch column family: {:?} was it created?",
                self.cf_name
            )
        })?;
        let mut collections: Vec<String> = Vec::new();
        let mut options = rocksdb::ReadOptions::default();
        options.set_iterate_range(rocksdb::PrefixRange("collection::".as_bytes()));
        for item in instance.iterator_cf_opt(cf, options, rocksdb::IteratorMode::Start) {
            let (key, val) = item?;
            let key_utf8 = std::str::from_utf8(&key)
                .with_context(|| "failed to parse utf8 str from the key")?;
            let val_utf8 = std::str::from_utf8(&val)
                .with_context(|| "failed to parse utf8 str from the col")?;
            let key_parts: Vec<&str> = key_utf8.splitn(2, "::").collect();
            if key_parts.len() != 2 {
                warn!(
                    "key: \"{:?}\" does not fit the expected format \"index::{{index_name}}\"",
                    key_utf8
                );
                continue;
            }
            collections.push(val_utf8.to_string());
        }
        Ok(collections)
    }

    fn mgrs_fr_rocksdb(&self) -> anyhow::Result<HashMap<String, Arc<CollectionMgr>>> {
        let mut res: HashMap<String, Arc<CollectionMgr>> = HashMap::new();
        let collections = self.get_collections()?;
        for collection in collections.iter() {
            info!("initating collection: {collection}");
            let obj = CollectionMgr::new(
                collection,
                &self.dir_path,
                self.rocksdb_instance.clone(),
                self.index_mgr.clone(),
                None,
            )?;
            res.insert(collection.to_string(), Arc::new(obj));
        }
        Ok(res)
    }

    pub fn search_preprocess(
        &self,
        index_name: &str,
        queries: &[crate::api::SearchQuery],
    ) -> anyhow::Result<Vec<crate::IndexItems>> {
        let mgrs = self.index_details.read();
        let mgr = mgrs
            .get(index_name)
            .ok_or_else(|| anyhow::anyhow!("index: {index_name} does not exist"))?;
        mgr.search_preprocess(queries)
    }

    pub fn search(&self, index_name: &str, req: &IndexSearch) -> anyhow::Result<Vec<NodeHit>> {
        let mgrs = self.index_details.read();
        let mgr = mgrs
            .get(index_name)
            .ok_or_else(|| anyhow::anyhow!("index: {index_name} does not exist"))?;
        mgr.search(req)
    }

    pub fn insert_preprocess(
        &self,
        index_name: &str,
        data: &str,
    ) -> anyhow::Result<Vec<crate::IndexItems>> {
        let mgrs = self.index_details.read();
        let mgr = mgrs
            .get(index_name)
            .ok_or_else(|| anyhow::anyhow!("index: {index_name} does not exist"))?;
        mgr.insert_preprocess(data)
    }

    pub fn insert_data(
        &self,
        index_name: &str,
        data: Vec<crate::IndexItems>,
    ) -> anyhow::Result<()> {
        let mgrs = self.index_details.read();
        let mgr = mgrs
            .get(index_name)
            .ok_or_else(|| anyhow::anyhow!("index: {index_name} does not exist"))?;
        mgr.insert_data(data)
    }

    pub fn delete_index(&self, index_name: &str) -> anyhow::Result<()> {
        {
            let indices = self.index_details.read();
            let mgr;
            match indices.get(index_name) {
                Some(res) => mgr = res,
                None => return Ok(()),
            }
            mgr.delete()?;
        }
        let instance = self.rocksdb_instance.read();
        let cf = instance.cf_handle(&self.cf_name).ok_or_else(|| {
            anyhow::anyhow!(
                "unable to fetch column family: {:?} was it created?",
                self.cf_name
            )
        })?;
        instance.delete_cf(cf, "collection::{index_name}")?;
        self.index_details.write().remove(index_name);
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
        // create a new manager for the given collection that
        // we are working with
        let mgr = CollectionMgr::new(
            index_name,
            &self.dir_path,
            self.rocksdb_instance.clone(),
            self.index_mgr.clone(),
            Some(CollectionMgrParams {
                field_names: field_names,
                index_settings: settings.clone(),
            }),
        )?;

        let instance = self.rocksdb_instance.read();
        let cf = instance.cf_handle(&self.cf_name).ok_or_else(|| {
            anyhow::anyhow!(
                "unable to fetch column family: {:?} was it created?",
                self.cf_name
            )
        })?;
        instance.put_cf(cf, "collection::{index_name}", format!("{index_name}"))?;
        self.index_details
            .write()
            .insert(index_name.to_string(), Arc::new(mgr));

        Ok(())
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
        mgr.insert_data("test-0000", index_req)
            .expect("insertion should work");
        let search_req = IndexSearch {
            embedding: vec![2.0; 768],
            attributes: &vec!["paragraph", "title"],
            weighting: &HashMap::from([
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
            attributes: &vec!["non-existent"],
            weighting: &HashMap::new(),
            limit: 10,
        };
        assert!(mgr.search("test-0000", &search_req_bad).is_err());
    }
}
