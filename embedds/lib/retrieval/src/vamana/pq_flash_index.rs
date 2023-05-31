use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use anyhow::{bail, Context};
use crossbeam_channel::{bounded, Receiver, Sender};
use parking_lot::RwLock;
use tracing::{info, warn};

use crate::av_store::AlignedDataStore;
use crate::metric::Metric;
use crate::vamana::pq::PQResult;
use crate::vamana::pq_chunk_table::FixedChunkPQTable;
use crate::vamana::reader::AlignedRead;
use crate::vamana::reader::Reader;
use crate::vamana::scratch::SSDThreadData;
use crate::vamana::utils::diskann_load_bin_generic;
use crate::vamana::utils::index_metadata_fr_file;
use crate::vamana::utils::ref_load_aligned;
use crate::vamana::utils::IndexMetadata;

fn node_sector_no(node_id: usize, nnodes_per_sector: usize) -> usize {
    return node_id / nnodes_per_sector + 1;
}
fn offset_to_node(node_id: usize, nnodes_per_sector: usize, max_node_len: usize) -> usize {
    (node_id % nnodes_per_sector) * max_node_len
}
fn offset_to_node_nhood(idx_start: usize, disk_bytes_per_point: usize) -> usize {
    return idx_start + disk_bytes_per_point;
}

// full representaion of a node
pub struct Node {
    vid: usize,                    // id that represents this obj uniquely in the graph.
    nnbrs: Vec<usize>, // nearest neighbors to this node, based on the specified metric calculation
    coords: AlignedDataStore<f32>, // full representation of the node in n-dimensional space
}

pub struct PQFlashIndex {
    data_dim: usize,
    disk_data_dim: usize,
    disk_bytes_per_point: usize,
    aligned_dim: usize,
    num_points: usize,
    n_chunks: usize,

    // houses the compressed pq representations of the points in the index
    data: Vec<u8>,
    pq_table: FixedChunkPQTable,
    index_metadata: IndexMetadata,
    // pulls data from the bound file descriptor in an async manner
    // which atm is powered by io_uring and *could* be powered by linuxaio
    // in the futture if we have the time!
    reader: Reader,
    // defaults to 1
    num_medoids: usize,
    // graph has one entry point by default,
    // we can optionally have multiple starting points
    medoids: Vec<usize>,
    thread_data: Arc<RwLock<SSDThreadData>>,

    // threaded_scratch:
    s_scratch: Sender<SSDThreadData>,
    r_scratch: Receiver<SSDThreadData>,

    // If there are multiple centroids, we pick the medoid corresponding
    // to the closest centroid as the starting point of search
    centroids: AlignedDataStore<f32>,
    // vid -> Arc<(Node{coords: [..], nnbrs: [..], vid: vid0})>
    nhood_cache: Arc<RwLock<HashMap<usize, Arc<Node>>>>,
}

const METADATA_SIZE: usize = 4096;
const VISITED_RESERVE_DEFAULT: usize = 4096;
const MAX_N_SECTOR_READS: usize = 128;
impl PQFlashIndex {
    pub fn new(
        num_threads: usize,
        folder: &Path,
        rdr_runtime: Arc<tokio_uring::Runtime>,
    ) -> anyhow::Result<Self> {
        let pq_table_filepath = Path::new("../../../../public/DiskANN/build/data/siftsmall/disk_index_sift_learn_R32_L50_A1.2_pq_pivots.bin");
        let res: PQResult = PQResult::from_ref_bin_format(pq_table_filepath)?;
        let pq_compressed_filepath = Path::new("../../../../public/DiskANN/build/data/siftsmall/disk_index_sift_learn_R32_L50_A1.2_pq_compressed.bin");

        let (pq_file_num_centroids, pq_file_dim) =
            crate::vamana::utils::get_metadata_ref(pq_table_filepath, METADATA_SIZE)?;
        info!("pq_file_dim: {pq_file_dim}, pq_file_num_centroids: {pq_file_num_centroids}");
        if pq_file_num_centroids != 256 {
            bail!("pq_file_num_centroids: {pq_file_num_centroids} expected 256");
        }

        let data_dim: usize = pq_file_dim;
        let disk_data_dim: usize = pq_file_dim;
        let disk_bytes_per_point: usize = pq_file_dim * std::mem::size_of::<f32>();

        let aligned_dim = crate::vamana::utils::round_up(pq_file_dim, 8);
        let (num_points, n_chunks, data) = diskann_load_bin_generic(pq_compressed_filepath, 0)?;
        let num_points = num_points;
        let n_chunks = n_chunks;
        let data = data;

        info!(
            "[PQFlashIndex] num_points: {num_points}, n_chunks: {n_chunks} | data_len: {}",
            data.len()
        );
        let table = FixedChunkPQTable::load_pq_centroid_bin(pq_table_filepath, n_chunks)?;
        info!(
            "loaded PQ centroids and in-memory compressed vectors num_points: {} \
            | data_dim: {} | aligned_dim: {} | chunks: {}",
            num_points, data_dim, aligned_dim, n_chunks,
        );
        let mut index_metadata = index_metadata_fr_file(&folder.join("disk-metadata.bin"))?;

        // TODO(infrawhispers) - remove this
        // index_metadata.start = 22877;
        // index_metadata.nnodes_per_sector = 6;
        // index_metadata.max_node_len = 644;

        // TODO(infrawhispers) return std::mem::size_of::<u32> -> std::mem::size_of::<u64>
        let max_degree =
            ((index_metadata.max_node_len - disk_bytes_per_point) / std::mem::size_of::<u64>()) - 1;
        info!(
            "diskindex metadata: nnodes_per_sector: {} | max_nodes_len: {}B | max_degree: {}",
            index_metadata.nnodes_per_sector, index_metadata.max_node_len, max_degree
        );

        info!("setting up thread data for {} threads", 1);
        let thread_data = SSDThreadData::new(aligned_dim, VISITED_RESERVE_DEFAULT)
            .with_context(|| "unable to create SSDThreadData")?;

        let reader = Reader::new(&folder.join("disk.bin"), rdr_runtime)?;
        // TODO(lneath) - remove this!
        // let reader = Reader::new(Path::new("../../../../public/DiskANN/build/data/siftsmall/disk_index_sift_learn_R32_L50_A1.2_disk.index"), rdr_runtime)?;

        let num_medoids = 1;
        let medoids = vec![index_metadata.start];
        let centroids = PQFlashIndex::use_medoids_data_as_centroids(
            &medoids,
            data_dim,
            aligned_dim,
            disk_bytes_per_point,
            index_metadata.nnodes_per_sector,
            index_metadata.max_node_len,
            &reader,
        )?;

        let (s, r) = bounded(num_threads);
        for _ in 0..num_threads {
            let scratch = SSDThreadData::new(aligned_dim, VISITED_RESERVE_DEFAULT)
                .with_context(|| "unable to create SSDThreadData")?;
            s.send(scratch).unwrap();
        }

        Ok(PQFlashIndex {
            data_dim: data_dim,
            disk_data_dim: disk_data_dim,
            disk_bytes_per_point: disk_bytes_per_point,
            aligned_dim: aligned_dim,
            num_points: num_points,
            n_chunks: n_chunks,
            data: data,
            pq_table: table,
            index_metadata: index_metadata,
            reader: reader,
            num_medoids: 1,
            medoids: medoids,
            centroids: centroids,
            thread_data: Arc::new(RwLock::new(thread_data)),
            s_scratch: s,
            r_scratch: r,

            nhood_cache: Arc::new(RwLock::new(HashMap::with_capacity(5000))),
        })
    }

    fn load_from_sector(
        &self,
        vid: usize,
        sector_idx: usize,
        point: &mut [f32],
        nbrs: &mut [usize],
        sector_scratch: &AlignedDataStore<u8>,
    ) -> anyhow::Result<usize> {
        let idx_node = offset_to_node(
            vid,
            self.index_metadata.nnodes_per_sector,
            self.index_metadata.max_node_len,
        );
        let coords_start = sector_idx * crate::vamana::SECTOR_LEN + idx_node;
        let coords_end = coords_start + self.disk_bytes_per_point;
        let coords: &[f32] = unsafe {
            std::slice::from_raw_parts(
                sector_scratch.data[coords_start..coords_end].as_ptr() as *const f32,
                (coords_end - coords_start) / 4,
            )
        };
        // copy over the coordinates that matter here
        point.copy_from_slice(coords);

        let nnbr_count: &[u64] = unsafe {
            std::slice::from_raw_parts(
                sector_scratch.data[coords_end..coords_end + std::mem::size_of::<u64>()].as_ptr()
                    as *const u64,
                1,
            )
        };
        if nnbr_count.len() != 1 || nnbr_count[0] == 0 {
            warn!(
                "found vid: {vid} with an unexpected number of neighbors: {nnbr_count:?}, skipping",
            );
            bail!("unexpected nnbr_count: {nnbr_count:?}");
        }
        let num_nnbrs: usize = nnbr_count[0].try_into()?;
        // TODO(infrawhispers) - u32 -> u64
        let nnbrs_u32: &[u64] = unsafe {
            std::slice::from_raw_parts(
                sector_scratch.data[coords_end + std::mem::size_of::<u64>()
                    ..coords_end
                        + std::mem::size_of::<u64>()
                        + std::mem::size_of::<u64>() * num_nnbrs]
                    .as_ptr() as *const u64,
                num_nnbrs,
            )
        };
        // let mut nnbrs: Vec<usize> = Vec::new();
        for (idx, nnbr) in nnbrs_u32.iter().enumerate() {
            nbrs[idx] = *nnbr as usize
        }
        Ok(num_nnbrs)
    }

    pub fn load_cache_list(&self, nodes: Vec<usize>) -> anyhow::Result<()> {
        info!("loading cache_list of size: {}", nodes.len());
        let mut scratch_w = self.thread_data.write();
        let num_chunks = 128;
        let block_size = 8;
        let num_blocks = crate::vamana::utils::div_round_up(nodes.len(), block_size);

        let scratch = &mut scratch_w.scratch;
        let sector_scratch = &mut scratch.sector_scratch;
        let mut nhood_cache_w = self.nhood_cache.write();
        for block in 0..num_blocks {
            let start_idx: usize = block * block_size;
            let end_idx: usize = std::cmp::min(nodes.len(), (block + 1) * block_size);

            let mut read_tracking: Vec<(usize, usize)> = Vec::with_capacity(block_size);
            let mut read_reqs: Vec<AlignedRead> = Vec::with_capacity(block_size);
            let mut res = Vec::new();
            for chunk in sector_scratch.data.chunks_mut(crate::vamana::SECTOR_LEN) {
                res.push(chunk);
            }
            for (i, idx) in (start_idx..end_idx).enumerate() {
                read_tracking.push((nodes[idx], num_chunks - 1 - i));
                let buf = res.pop().unwrap();
                read_reqs.push(AlignedRead {
                    len: crate::vamana::SECTOR_LEN,
                    buf: buf,
                    offset: (node_sector_no(nodes[idx], self.index_metadata.nnodes_per_sector)
                        * crate::vamana::SECTOR_LEN) as u64,
                });
            }
            // finally issue the read_reqs
            self.reader
                .read(&mut read_reqs)
                .with_context(|| "unable to read data from file")?;
            for (vid, section_idx) in read_tracking {
                let mut aligned_point: AlignedDataStore<f32> =
                    AlignedDataStore::<f32>::new(1, self.aligned_dim);
                let mut nbrs = vec![0; 128];
                let nnbrs = self.load_from_sector(
                    vid,
                    section_idx,
                    &mut aligned_point.data,
                    &mut nbrs[..],
                    sector_scratch,
                )?;
                nbrs.resize(nnbrs, 0);
                nhood_cache_w.insert(
                    vid,
                    Arc::new(Node {
                        coords: aligned_point,
                        nnbrs: nbrs,
                        vid: vid,
                    }),
                );
            }
        }

        Ok(())
    }

    pub fn generate_cache_list_from_sample_queries(
        &self,
        filepath: &std::path::Path,
        l_search: usize,
        beam_width: usize,
        num_nodes_to_cache: usize,
    ) -> anyhow::Result<Vec<usize>> {
        let samples = ref_load_aligned(filepath).with_context(|| "unable to load sample points")?;
        let mut visited_nodes_cnt: Arc<RwLock<HashMap<usize, AtomicU64>>> =
            Arc::new(RwLock::new(HashMap::new()));
        // run cached_beam_search across the values that are provided while
        // keeping track of the values that we have
        for i in 0..10 {
            let sample = &samples.arr.data
                [i * samples.aligned_dim..i * samples.aligned_dim + samples.aligned_dim];
            let res = self
                .cached_beam_search(
                    sample,
                    1,
                    l_search,
                    beam_width,
                    Some(visited_nodes_cnt.clone()),
                )
                .with_context(|| "could not complete beamsearch for sample: {i}")?;
            info!("[generate_cache_list_from_sample_queries] res: {res:?}");
        }
        // get all the visited nodes during the warmup period
        // and then sort the values by the number of visits
        let cntr = visited_nodes_cnt.write();
        let mut res: Vec<(usize, usize)> = Vec::with_capacity(cntr.len());
        for (vid, cnt) in cntr.iter() {
            res.push((*vid, cnt.load(Ordering::Relaxed).try_into()?));
        }
        res.sort_by_key(|k| k.1);
        res.reverse();
        let node_list_len: usize = std::cmp::min(num_nodes_to_cache, res.len());
        let mut node_list: Vec<usize> = vec![0; node_list_len];
        for k in 0..node_list_len {
            node_list[k] = res[k].0;
        }
        Ok(node_list)
    }

    pub fn cached_beam_search(
        &self,
        arr: &[f32],
        k: usize,
        l_search: usize,
        beam_width: usize,
        visited_nodes_cnter: Option<Arc<RwLock<HashMap<usize, AtomicU64>>>>,
    ) -> anyhow::Result<Vec<crate::ann::Node>> {
        // info!("disk_bytes_per_point: {}", self.disk_bytes_per_point);
        // info!("search_arr: {:?}", arr);
        if beam_width > MAX_N_SECTOR_READS {
            bail!("beam_width cannot be larger than MAX_N_SECTOR_READS");
        }
        let mut scratch_w = self.thread_data.write();
        // scratch.reset();
        let scratch = &mut scratch_w.scratch;
        let pq_scratch = &mut scratch.pq_scratch;
        // copy query to thread specific aligned and allocated memory (for distance
        // calculations we need aligned data)
        // TODO(infrawhispers) there is some other stuff we would need to handle
        // if we are supporting MIPS
        pq_scratch.query.data[..arr.len()].copy_from_slice(arr);
        pq_scratch.rotated_query.data[..arr.len()].copy_from_slice(arr);

        // let data_buf = &mut scratch.scratch.coord_scratch;
        let sector_scratch = &mut scratch.sector_scratch;
        // center the query and rotate if we have a rotation matrix
        self.pq_table
            .preprocess_query(&mut pq_scratch.rotated_query.data[..]);
        self.pq_table.populate_chunk_distances(
            &pq_scratch.rotated_query.data[..],
            &mut pq_scratch.pqtable_dist_scratch.data[..],
        );

        let dist_scratch = &mut pq_scratch.dist_scratch.data[..];
        let pq_coord_scratch = &mut pq_scratch.pq_coord_scratch.data[..];
        let pq_dists = &mut pq_scratch.pqtable_dist_scratch.data[..];
        let mut compute_dists = |ids: &[usize], dists_out: &mut [f32]| {
            crate::vamana::pq::aggregate_coords(ids, &self.data, self.n_chunks, pq_coord_scratch);
            crate::vamana::pq::pq_dist_lookup(
                pq_coord_scratch,
                ids.len(),
                self.n_chunks,
                pq_dists,
                dists_out,
            );
        };

        let visited = &mut scratch.visited;
        visited.clear();

        let retset = &mut scratch.retset;
        retset.reserve(l_search);
        retset.clear();

        let full_retset = &mut scratch.full_retset;
        full_retset.clear();

        let best_medoid: usize = self.medoids[0];
        let best_dist: f32 = std::f32::MAX;
        if self.medoids.len() != 1 {
            bail!("we do not support multiple medoids at the moment")
        }

        compute_dists(&[best_medoid], dist_scratch);
        retset.insert(crate::ann::INode {
            vid: best_medoid,
            distance: dist_scratch[0],
            flag: false,
        });
        visited.insert(best_medoid);

        let mut n_hops: u64 = 0;
        let mut n_cmps: u64 = 0;
        let mut cmps: u64 = 0;
        let mut hops: u32 = 0;
        let mut num_ios: u64 = 0;
        let mut num_4kb: u64 = 0;
        let mut n_cache_hits: u64 = 0;

        // cleared every iteration
        let mut frontiers: Vec<usize> = Vec::with_capacity(2 * beam_width);
        let mut frontier_nhoods: Vec<(usize, usize)> = Vec::with_capacity(2 * beam_width);
        let mut frontier_read_reqs: Vec<AlignedRead> = Vec::with_capacity(2 * beam_width);
        let mut cached_nhoods: Vec<(usize, Arc<Node>)> = Vec::with_capacity(2 * beam_width);
        // this is aligned to f32 to ensure that we can run SIMD operations
        // without panacking
        let mut aligned_compute: AlignedDataStore<f32> =
            AlignedDataStore::<f32>::new(1, self.aligned_dim);
        let io_limit: u64 = 50000;

        // maybe we should raise this?
        let nhood_cache = self.nhood_cache.read();
        while retset.has_unexpanded_node() && num_ios < io_limit {
            frontiers.clear();
            frontier_nhoods.clear();
            frontier_read_reqs.clear();
            cached_nhoods.clear();
            scratch.sector_idx = 0;
            let mut num_seen: u32 = 0;

            while retset.has_unexpanded_node()
                && frontiers.len() < beam_width.try_into()?
                && num_seen < beam_width.try_into()?
            {
                let nbr = retset.closest_unexpanded();
                num_seen += 1;
                match nhood_cache.get(&nbr.vid) {
                    Some(res) => {
                        cached_nhoods.push((nbr.vid, res.clone()));
                        n_cache_hits += 1;
                    }
                    None => {
                        frontiers.push(nbr.vid);
                    }
                }
                // increment the visited_nodes_cnter if we are going to
                // generate a cached object
                match visited_nodes_cnter.as_ref() {
                    Some(cntr) => {
                        let mut create: bool = false;
                        {
                            let cnt_r = cntr.read();
                            match cnt_r.get(&nbr.vid) {
                                Some(v) => {
                                    v.fetch_add(1, Ordering::SeqCst);
                                }
                                None => {
                                    create = true;
                                }
                            }
                        }
                        if create {
                            let mut cnt_w = cntr.write();
                            cnt_w.insert(nbr.vid, AtomicU64::new(0));
                        }
                    }
                    None => {}
                }
                // TODO(infrawhispers) - add visited_nodes count
            }
            // read nhoods of frontier ids
            if !frontiers.is_empty() {
                n_hops += 1;
                let num_chunks = 128;
                // info!("data_len: {}, num_chunks: {num_chunks:?}");
                let mut frontier_read_reqs_1: Vec<AlignedRead> = Vec::with_capacity(2 * beam_width);
                let mut res = Vec::new();
                for chunk in sector_scratch.data.chunks_mut(crate::vamana::SECTOR_LEN) {
                    res.push(chunk);
                }
                for (idx, id) in frontiers.iter().enumerate() {
                    frontier_nhoods.push((*id, num_chunks - 1 - idx));
                    let buf = res.pop().unwrap();
                    frontier_read_reqs_1.push(AlignedRead {
                        len: crate::vamana::SECTOR_LEN,
                        buf: buf,
                        offset: (node_sector_no(*id, self.index_metadata.nnodes_per_sector)
                            * crate::vamana::SECTOR_LEN) as u64,
                    });
                    num_ios += 1;
                    num_4kb += 1;
                }
                self.reader
                    .read(&mut frontier_read_reqs_1)
                    .with_context(|| "unable to read data from file")?;
            }
            // process cached nhoods
            for (id, nbr) in &cached_nhoods {
                let cur_expanded_dist = crate::metric::MetricL2::compare(&nbr.coords.data[..], arr);
                full_retset.push(crate::ann::Node {
                    eid: crate::ann::EId([0u8; 16]),
                    vid: nbr.vid,
                    distance: cur_expanded_dist,
                });
                // TODO(infrawhispers) handle the case where we are using disk_index_pq
                // ---
                // compute node_nbrs <-> query dists in PQ space
                // info!("the neighbors: {:?}", nbr.nnbrs);
                compute_dists(&nbr.nnbrs, dist_scratch);
                n_cmps += nbr.nnbrs.len() as u64;
                for (idx, nnbr) in nbr.nnbrs.iter().enumerate() {
                    if visited.insert(*nnbr) {
                        // TODO(infrawhispers) handle the point being in our set of _dummy_points (i.e the frozen location)
                        // TOOD(infrawhispers) handle the point not having one of the labels we are looking for
                        cmps += 1;
                        retset.insert(crate::ann::INode {
                            vid: *nnbr,
                            distance: dist_scratch[idx],
                            flag: false,
                        })
                    }
                }
            }
            // handle the frontier neighbors
            for frontier_nhood in &frontier_nhoods {
                let mut nnbrs = vec![0; 128];
                let num_nbrs = self.load_from_sector(
                    frontier_nhood.0,
                    frontier_nhood.1,
                    &mut aligned_compute.data,
                    &mut nnbrs[..],
                    sector_scratch,
                )?;

                nnbrs.resize(num_nbrs, 0);
                // if frontier_nhood.0 == best_medoid
                //     || frontier_nhood.0 == 15322
                //     || frontier_nhood.0 == 17
                // {
                //     info!(
                //         "id: {}, cnt: {}, nnbrs: {:?} point: {:?}",
                //         frontier_nhood.0,
                //         nnbrs.len(),
                //         nnbrs,
                //         aligned_compute.data,
                //     );
                // }
                let dist = crate::metric::MetricL2::compare(&aligned_compute.data, arr);
                // info!("[frontier]: {}, dist: {}", frontier_nhood.0, dist);
                full_retset.push(crate::ann::Node {
                    eid: crate::ann::EId([0u8; 16]),
                    vid: frontier_nhood.0,
                    distance: dist,
                });
                compute_dists(&nnbrs, dist_scratch);
                // process prefetch-ed nhood
                for (idx, nnbr) in nnbrs.iter().enumerate() {
                    if visited.insert(*nnbr) {
                        // info!("adding to retset: {}",*nnbr);
                        // TODO(infrawhispers) handle the point being in our set of _dummy_points (i.e the frozen location)
                        // TOOD(infrawhispers) handle the point not having one of the labels we are looking for
                        cmps += 1;
                        n_cmps += 1;
                        retset.insert(crate::ann::INode {
                            vid: *nnbr,
                            distance: dist_scratch[idx],
                            flag: false,
                        })
                    }
                }
            }
            // print!("curr_hops: {}, ret_set:", hops);
            // retset.print();
            hops += 1;
        }
        // info!("hops: {hops}, cmps: {cmps}, num_ios: {num_ios}, num_4kb: {num_4kb}");
        full_retset.sort();
        // info!("full_retset: {full_retset:?}");
        // handle use_reorder_data
        let mut result: Vec<crate::ann::Node> = Vec::with_capacity(k);
        for i in 0..k {
            if i >= full_retset.len() {
                break;
            }
            if full_retset[i].vid == self.index_metadata.start {
                continue;
            }
            match full_retset.get(i) {
                Some(nbr) => result.push(nbr.clone()),
                None => break,
            }
        }
        Ok(result)
    }

    pub fn use_medoids_data_as_centroids(
        medoids: &[usize],
        data_dim: usize,
        aligned_dim: usize,
        disk_bytes_per_point: usize,
        nnodes_per_sector: usize,
        max_node_len: usize,
        reader: &Reader,
    ) -> anyhow::Result<AlignedDataStore<f32>> {
        let mut centroid_data = AlignedDataStore::<f32>::new(1, medoids.len() * aligned_dim);
        for (idx, medoid) in medoids.iter().enumerate() {
            info!("loading centroid data for idx: {idx} medoid: {medoid} ");
            let mut medoid_buf = AlignedDataStore::<u8>::new(1, crate::vamana::SECTOR_LEN);
            let mut read_reqs: Vec<AlignedRead> = vec![AlignedRead {
                len: crate::vamana::SECTOR_LEN,
                buf: &mut medoid_buf.data[..],
                offset: (node_sector_no(*medoid, nnodes_per_sector) * crate::vamana::SECTOR_LEN)
                    as u64,
            }];
            reader
                .read(&mut read_reqs)
                .with_context(|| "unable to read data from file")?;
            let medoid_node_idx = offset_to_node(*medoid, nnodes_per_sector, max_node_len);
            // info!("medoid_node_idx: {medoid_node_idx}");
            unsafe {
                let ptr = (medoid_buf.data.as_ptr() as *mut f32).add(medoid_node_idx);
                let slice: &[f32] = std::slice::from_raw_parts(ptr, aligned_dim);
                info!(
                    "medoid: {medoid} => [{}, {}, {}...{}, {}, {}]",
                    slice[0],
                    slice[1],
                    slice[2],
                    slice[slice.len() - 3],
                    slice[slice.len() - 2],
                    slice[slice.len() - 1],
                );
                for i in 0..data_dim {
                    centroid_data.data[idx * aligned_dim + i] = slice[i];
                }
            }
        }
        Ok(centroid_data)
    }
}
