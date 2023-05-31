#![allow(dead_code)]

use std::marker::PhantomData;
use std::path::Path;
use std::time::Instant;

use anyhow::Context;
use byteorder::ByteOrder;
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::fs::OpenOptions;
use std::io::Read;
use std::io::Write;
use std::path::PathBuf;
use tracing::info;

use crate::ann::{ANNIndex, EId};
use crate::vamana::pq::PQ;

mod math_utils;
mod pq;
mod pq_chunk_table;
mod pq_flash_index;
mod reader;
mod scratch;
mod utils;

const SECTOR_LEN: usize = 4096;
const MAX_N_CMPS: usize = 16384;
const MAX_GRAPH_DEGREE: usize = 512;
const MAX_N_SECTOR_READS: usize = 128;

#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
pub struct Params {
    pub indexing_range: usize,      // R
    pub indexing_queue_size: usize, // L
    pub indexing_mem_limit: f32,
    pub indexing_num_threads: usize,
    pub index_ram_limit: usize,

    pub dims: usize,
    pub num_points: usize,
    // pub dim: usize,
    // pub max_points: usize,
    // pub indexing_threads: Option<usize>,
    // pub indexing_range: usize,
    // pub indexing_queue_size: usize,
    // pub indexing_maxc: usize,
    // pub indexing_alpha: f32,
    // pub maintenance_period_millis: u64,
}

struct Index<TMetric> {
    metric: PhantomData<TMetric>,
    pq: PQ<TMetric>,
}

// block size for reading/processing large files and matrices in blocks
#[allow(dead_code)]
const BLOCK_SIZE: usize = 5000000;
#[allow(dead_code)]
const SPACE_FOR_CACHED_NODES_IN_GB: f32 = 0.25;
#[allow(dead_code)]
const THRESHOLD_FOR_CACHING_IN_GB: f32 = 0.25;
#[allow(dead_code)]
const MAX_PQ_TRAINING_SET_SIZE: f32 = 256000.0;
#[allow(dead_code)]
const MAX_PQ_CHUNKS: usize = 512;

impl<TMetric> Index<TMetric>
where
    TMetric: crate::metric::Metric<f32>,
{
    #[allow(dead_code)]
    fn new() -> Self {
        Index {
            metric: PhantomData,
            pq: PQ::new(),
        }
    }
    #[allow(dead_code)]
    fn get_memory_budget(search_budget: f32) -> f32 {
        let mut final_index_ram_limit: f32 = search_budget;
        if search_budget - SPACE_FOR_CACHED_NODES_IN_GB > THRESHOLD_FOR_CACHING_IN_GB {
            final_index_ram_limit = search_budget - SPACE_FOR_CACHED_NODES_IN_GB;
        }
        return final_index_ram_limit * 1024.0 * 1024.0 * 1024.0;
    }

    /// writes out the vamana index to disk
    fn create_disk_layout(&self, path: &std::path::Path) -> anyhow::Result<()> {
        let mut graph_filepath = PathBuf::from(path);
        graph_filepath.push("index-in-mem.graph");
        let mut f = File::open(graph_filepath)?;

        let mut data_filepath = PathBuf::from(path);
        data_filepath.push("index-in-mem.data");
        let mut d = File::open(data_filepath)?;

        let num_points = d.read_u64::<LittleEndian>()? as usize;
        let dims = d.read_u64::<LittleEndian>()? as usize;

        info!("[create_disk_layout] generating with num_points: {num_points} dims: {dims}");
        let mut metadata_filpath = PathBuf::from(path);
        metadata_filpath.push("disk-metadata.bin");
        if metadata_filpath.exists() {
            std::fs::remove_file(&metadata_filpath)?;
        }
        let mut m = OpenOptions::new()
            .create_new(true)
            .write(true)
            .append(false)
            .open(metadata_filpath)
            .with_context(|| "unable to create the output.meta file")?;

        let output_file = path.join("disk.bin");
        if output_file.exists() {
            std::fs::remove_file(&output_file)?;
        }
        let mut o = OpenOptions::new()
            .create_new(true)
            .write(true)
            .append(false)
            .open(output_file)
            .with_context(|| "unable to open the output file")?;

        let index_size = f.read_u64::<LittleEndian>()?;
        let max_degree = f.read_u64::<LittleEndian>()?;
        let start = f.read_u64::<LittleEndian>()?;
        let num_frozen_pts = f.read_u64::<LittleEndian>()?;

        info!("[create_disk_layout] source graph metadata: graph_size: {index_size} | max_degree: {max_degree} | start: {start} | num_frozen_pts: {num_frozen_pts}");
        let vamana_frozen_loc = start;
        let max_node_len: u64 = ((max_degree + 1) * (std::mem::size_of::<u64>() as u64))
            + (dims as u64 * (std::mem::size_of::<f32>() as u64));
        let nnodes_per_sector = SECTOR_LEN / (max_node_len as usize);
        let num_sectors = crate::vamana::utils::round_up(num_points.try_into()?, nnodes_per_sector)
            / nnodes_per_sector;

        info!("[create_disk_layout] layout constraints max_node_len: {max_node_len} bytes | nnodes_per_sector: {nnodes_per_sector} | num_sectors: {num_sectors}");

        let mut output_file_meta = vec![];
        output_file_meta.write_u64::<LittleEndian>(num_points.try_into()?)?;
        output_file_meta.write_u64::<LittleEndian>(dims.try_into()?)?;
        output_file_meta.write_u64::<LittleEndian>(start)?;
        output_file_meta.write_u64::<LittleEndian>(max_node_len)?;
        output_file_meta.write_u64::<LittleEndian>(nnodes_per_sector.try_into()?)?;
        output_file_meta.write_u64::<LittleEndian>(1)?;
        output_file_meta.write_u64::<LittleEndian>(vamana_frozen_loc)?;
        output_file_meta.write_u64::<LittleEndian>(index_size)?;

        let mut curr_node_id = 0;
        let mut curr_node_coords = vec![0; std::mem::size_of::<f32>() * dims];

        let mut sector_buf: Vec<u8> = vec![0u8; SECTOR_LEN];
        let mut node_buf: Vec<u8> = vec![0u8; max_node_len.try_into()?];
        // node_buf is organized in the following manner:
        // [0..dims]----[num_nbrs]---[nhood] which translates to:
        // [the f32 or u8 represenation of the point][number of nearest neighbors][neighborhood]
        o.write(&sector_buf)?;
        let num_nbrs_idx_start = dims * std::mem::size_of::<f32>();
        let num_nbrs_idx_end = num_nbrs_idx_start + std::mem::size_of::<u64>();
        let nnbrs_start = num_nbrs_idx_end;

        for sector in 0..num_sectors {
            if sector % 100000 == 0 {
                info!("[create_disk_layout] working on sector number: {sector}");
            }
            sector_buf.fill(0u8);
            let mut sector_node_id = 0;
            while sector_node_id < nnodes_per_sector && curr_node_id < num_points {
                for v in &mut node_buf {
                    *v = 0u8;
                }
                f.read_exact(&mut node_buf[num_nbrs_idx_start..num_nbrs_idx_end])?;
                let num_nnbrs =
                    LittleEndian::read_u64(&node_buf[num_nbrs_idx_start..num_nbrs_idx_end]);
                // if nnbrs > max_degree || nnbrs == 0 {
                //     info!("unexpectedly ")
                // }
                f.read_exact(
                    &mut node_buf[nnbrs_start
                        ..nnbrs_start + num_nnbrs as usize * std::mem::size_of::<u64>()],
                )?;
                d.read_exact(&mut curr_node_coords[..])?;
                node_buf[..curr_node_coords.len()].copy_from_slice(&curr_node_coords);
                let sector_start = sector_node_id * (max_node_len as usize);
                sector_buf[sector_start..sector_start + node_buf.len()].copy_from_slice(&node_buf);
                sector_node_id += 1;
                curr_node_id += 1;
            }
            o.write(&sector_buf)?;
        }
        o.sync_all()?;

        m.write(&output_file_meta)?;
        Ok(m.sync_all()?)
    }

    fn build_merged_vamana_index(
        &self,
        params: &Params,
        eids: &[EId],
        data: &[f32],
    ) -> anyhow::Result<()> {
        let ram_req = utils::estimate_ram_usage(
            params.num_points,
            params.dims,
            std::mem::size_of::<f32>(),
            params.indexing_range,
        );
        if ram_req < (params.index_ram_limit as f32) * 1024.0 * 1024.0 * 1024.0 {
            info!(
                "building in one pass - ram_req: {} GB fits within allocated budget of {} GB",
                ram_req / (1024.0 * 1024.0 * 1024.0),
                params.index_ram_limit,
            );
            let params_mem = crate::ann::ANNParams::DiskANN {
                params: crate::diskannv1::DiskANNParams {
                    dim: params.dims,
                    max_points: params.num_points,
                    indexing_threads: None,
                    indexing_range: params.indexing_range, // R
                    indexing_queue_size: params.indexing_queue_size, // L
                    indexing_maxc: 750,                    // C
                    indexing_alpha: 1.2,                   // alpha
                    maintenance_period_millis: 500,
                },
            };
            let idx: crate::diskannv1::DiskANNV1Index<crate::metric::MetricL2, f32> =
                crate::diskannv1::DiskANNV1Index::new(&params_mem)
                    .with_context(|| "unable to create the in-memory index")?;
            idx.insert(eids, crate::ann::Points::Values { vals: data })
                .with_context(|| "unable to insert points into the in-memory index")?;
            idx.save_ref_bin(Path::new(".disk"))
                .with_context(|| "unable to save in-memory index to disk")?;
        } else {
            info!(
                "building in multiple pass - ram_req: {} GB > allocated budget of {} GB",
                ram_req / (1024.0 * 1024.0 * 1024.0),
                params.index_ram_limit,
            );
            unimplemented!();
        }
        Ok(())
    }

    pub fn build_disk_index(
        &self,
        params: &Params,
        eids: &[crate::ann::EId],
        base_vectors: &[f32],
    ) -> anyhow::Result<()> {
        let final_index_ram_limit =
            Index::<TMetric>::get_memory_budget(params.indexing_mem_limit as f32);
        info!(
            "building: R={}, L={}, Query RAM Budget: {}, Index Ram Budget: {}",
            params.indexing_range,
            params.indexing_queue_size,
            final_index_ram_limit,
            params.index_ram_limit
        );

        // let p_val: f32 = (MAX_PQ_TRAINING_SET_SIZE) / (params.num_points as f32);
        let mut num_pq_chunks: usize =
            (final_index_ram_limit / (params.num_points as f32)) as usize;
        num_pq_chunks = if num_pq_chunks <= 0 { 1 } else { num_pq_chunks };
        num_pq_chunks = if num_pq_chunks > params.dims {
            params.dims
        } else {
            num_pq_chunks
        };
        num_pq_chunks = if num_pq_chunks > MAX_PQ_CHUNKS {
            MAX_PQ_CHUNKS
        } else {
            num_pq_chunks
        };
        info!(
            "compressing {} dimensional data into {} bytes per vector",
            params.dims, num_pq_chunks
        );
        let quantize_timer = Instant::now();
        self.pq
            .generate_quantized_data(&Path::new(".disk"), &base_vectors, num_pq_chunks, false)
            .with_context(|| "unable to generate the quantized data")?;
        info!("time for quantized_data {:?}", quantize_timer.elapsed());

        let merged_vamana = Instant::now();
        self.build_merged_vamana_index(params, eids, base_vectors)
            .with_context(|| "unable to build the merged vamana index")?;
        info!("time for merged_vamana: {:?}", merged_vamana.elapsed());

        let disk_layout = Instant::now();
        self.create_disk_layout(Path::new(".disk"))?;
        info!("time for disk_layout: {:?}", disk_layout.elapsed());
        Ok(())
    }

    pub fn search_disk_index(&self, params: &Params, vectors: &[f32]) -> anyhow::Result<()> {
        let beamwidth: usize = 2;
        let num_threads: usize = 2;
        let search_io_limit: usize = usize::MAX;

        let reader_runtime =
            crate::vamana::reader::Reader::get_runtime().expect("unable to create the runtime");
        let p_flash_index =
            pq_flash_index::PQFlashIndex::new(6, Path::new(".disk"), reader_runtime.clone())?;
        let num_nodes_to_cache: usize = 5000;
        let node_list = p_flash_index.generate_cache_list_from_sample_queries(
            &Path::new("../../../../public/DiskANN/build/data/siftsmall/disk_index_sift_learn_R32_L50_A1.2_sample_data.bin"),
            15,
            6,
            num_nodes_to_cache,
        )?;
        p_flash_index.load_cache_list(node_list)?;
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use byteorder::{BigEndian, ByteOrder};
    use std::fs;
    use std::path::Path;
    use tracing::Level;

    #[test]
    fn disk_index_build() {
        let subscriber = tracing_subscriber::fmt()
            .with_max_level(Level::INFO)
            .finish();
        match tracing::subscriber::set_global_default(subscriber) {
            Ok(()) => {}
            Err(err) => {
                panic!("unable to create the tracing subscriber: {}", err)
            }
        }
        fs::create_dir_all(".disk").expect("creation of disk_index folder is successfull");

        let path = format!("../../../../eval/data/siftsmall/");
        let directory = Path::new(&path);
        let dims: usize = 128;
        let loader = crate::utils::sift::SIFT {
            directory: directory,
            dims: dims,
        };
        let learn_vectors = loader
            .fetch_vectors("sift_learn.fvecs")
            .expect("unable to fetch learn vectors from disk");
        let query_vectors = loader
            .fetch_vectors("sift_query.fvecs")
            .expect("unable to fetch query vectors from disk");

        let mut eids: Vec<crate::ann::EId> =
            vec![crate::ann::EId([0u8; 16]); learn_vectors.len() / 128];
        for i in 0..learn_vectors.len() / 128 {
            let mut eid: crate::ann::EId = crate::ann::EId([0u8; 16]);
            BigEndian::write_uint(
                &mut eid.0,
                i.try_into().unwrap(),
                std::mem::size_of::<usize>(),
            );
            eids[i] = eid;
        }

        let num_points: usize = learn_vectors.len() / dims;
        info!("num_learn: {num_points}");
        // let sift_learn =
        // let (base_vectors, search_vectors, truth_vectors) =
        //     crate::utils::sift::get_sift_vectors("siftsmall");
        // let dims: usize = 128;
        // let num_points: usize = base_vectors.len() / dims;

        let idx = Index::<crate::metric::MetricL2>::new();
        let params = &Params {
            indexing_range: 32,
            indexing_queue_size: 50,
            indexing_mem_limit: 0.003,
            indexing_num_threads: 12,
            index_ram_limit: 1,

            dims: dims,
            num_points: num_points,
        };
        idx.build_disk_index(params, &eids, &learn_vectors)
            .expect("unable to completely build the disk-index");
        idx.search_disk_index(params, &query_vectors)
            .expect("unable to search the index")
    }
}
