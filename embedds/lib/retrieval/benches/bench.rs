use std::fs;
use std::io::Cursor;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use byteorder::{BigEndian, ByteOrder};
use byteorder::{LittleEndian, ReadBytesExt};
use criterion::Throughput;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::Rng;
use rayon::prelude::*;
use retrieval::manager::index_manager::IndexManager;

struct SIFT<'a> {
    directory: &'a Path,
    dims: usize,
}

impl SIFT<'_> {
    fn fetch_vectors(&self, filename: &str, dims: usize) -> anyhow::Result<Vec<f32>> {
        let path = self.directory.join(filename);
        let data_r = fs::read(path)?;
        let data_r_slice = data_r.as_slice();

        let num_vectors = data_r.len() / (4 + dims * 4);
        let mut data_w: Vec<f32> = Vec::with_capacity(num_vectors * dims);
        let mut rdr = Cursor::new(data_r_slice);
        for _i in 0..num_vectors {
            // we *must* make a read for the dimensionality first before reading out
            // the components of the vectors themselves
            let dim = rdr.read_u32::<LittleEndian>().unwrap();
            // if dim != self.dims.try_into()? {
            //     panic!("dim mismatch while reading the source data");
            // }
            for _j in 0..dim {
                data_w.push(rdr.read_f32::<LittleEndian>().unwrap());
            }
        }
        Ok(data_w)
    }
    //     fn fetch_ground_truth(&self, filename: &str, num_closest: usize) -> anyhow::Result<Vec<u32>> {
    //         let path = self.directory.join(filename);
    //         let data_r = fs::read(path)?;
    //         let data_r_slice = data_r.as_slice();
    //         let mut rdr = Cursor::new(data_r_slice);

    //         // we store the dimensionality of the vector here
    //         // and then the components are (unsigned char|float | int)*d
    //         // documentation is avail here: http://corpus-texmex.irisa.fr/
    //         // let dims = 100;
    //         let num_vectors = (data_r.len() as u32 / (4 + (num_closest as u32) * 4)) as usize;
    //         let mut data_w: Vec<u32> = Vec::with_capacity(num_vectors * num_closest as usize);
    //         for _ in 0..num_vectors {
    //             let dim = rdr.read_u32::<LittleEndian>().unwrap();
    //             if dim != num_closest.try_into().unwrap() {
    //                 panic!(
    //                     "dim != num_closest: dim: {} num_closest: {}",
    //                     dim, num_closest
    //                 );
    //             }
    //             for _j in 0..dim {
    //                 data_w.push(rdr.read_u32::<LittleEndian>().unwrap());
    //             }
    //         }
    //         Ok(data_w)
    //     }

    //     fn fetch_ground_truth_by_id(
    //         &self,
    //         filename: &str,
    //         k: usize,
    //         num_closest: usize,
    //     ) -> anyhow::Result<HashMap<retrieval::ann::EId, Vec<retrieval::ann::EId>>> {
    //         let truth_vecs = self.fetch_ground_truth(filename, num_closest)?;
    //         let mut truth_by_id: HashMap<retrieval::ann::EId, Vec<retrieval::ann::EId>> =
    //             HashMap::new();
    //         for id in 0..truth_vecs.len() / num_closest {
    //             let truth_vec = truth_vecs
    //                 [(id * num_closest) as usize..(id * num_closest + num_closest) as usize]
    //                 .to_vec();
    //             let mut truth_eids: Vec<retrieval::ann::EId> = Vec::with_capacity(k);
    //             for i in 0..truth_vec.len() {
    //                 if i == k {
    //                     break;
    //                 }
    //                 let mut eid: retrieval::ann::EId = [0u8; 16];
    //                 BigEndian::write_uint(
    //                     &mut eid,
    //                     truth_vec[i].try_into().unwrap(),
    //                     std::mem::size_of::<usize>(),
    //                 );
    //                 truth_eids.push(eid);
    //             }
    //             let mut ref_eid: retrieval::ann::EId = [0u8; 16];
    //             BigEndian::write_uint(
    //                 &mut ref_eid,
    //                 id.try_into().unwrap(),
    //                 std::mem::size_of::<usize>(),
    //             );
    //             truth_by_id.insert(ref_eid, truth_eids);
    //         }
    //         Ok(truth_by_id)
    //     }
}

fn setup_sift(dir_name: &str) -> (Vec<f32>, Vec<f32>) {
    let path = format!("../../../../eval/data/{dir_name}/");
    let directory = Path::new(&path);
    let dims: usize = 128;
    let loader = SIFT {
        directory: directory,
        dims: dims,
    };
    let k: usize = 10;
    let base_vectors = loader
        .fetch_vectors("sift_base.fvecs", 128)
        .expect("unable to fetch base vectors from disk");
    let mut eids: Vec<retrieval::ann::EId> =
        vec![retrieval::ann::EId([0u8; 16]); base_vectors.len() / 128];
    for i in 0..base_vectors.len() / 128 {
        let mut eid: retrieval::ann::EId = retrieval::ann::EId([0u8; 16]);
        BigEndian::write_uint(
            &mut eid.0,
            i.try_into().unwrap(),
            std::mem::size_of::<usize>(),
        );
        eids[i] = eid;
    }
    let query_vectors = loader
        .fetch_vectors("sift_query.fvecs", 128)
        .expect("unable to fetch the query vectors");

    (base_vectors, query_vectors)
}

fn index_benchmark(c: &mut Criterion) {
    let (base_vectors, search_vectors) = setup_sift("sift1m");
    let dimensions: usize = 128;
    let mut eids: Vec<retrieval::ann::EId> =
        vec![retrieval::ann::EId([0u8; 16]); base_vectors.len() / dimensions];
    for i in 0..base_vectors.len() / dimensions {
        let mut eid: retrieval::ann::EId = retrieval::ann::EId([0u8; 16]);
        BigEndian::write_uint(
            &mut eid.0,
            i.try_into().unwrap(),
            std::mem::size_of::<usize>(),
        );
        eids[i] = eid;
    }
    let _ = fs::remove_dir_all(".test/index_benchmark");
    let mgr = IndexManager::new(&PathBuf::from(".test/index_benchmark"));

    let mut group = c.benchmark_group("ann::index");
    group.sample_size(10);
    group.throughput(Throughput::Elements(eids.len() as u64));

    group.bench_function("flat_lite::index", |b| {
        let index_name = "test-0000";
        let params = retrieval::ann::ANNParams::FlatLite {
            params: retrieval::flat_lite::FlatLiteParams {
                dim: dimensions,
                segment_size_kb: 1024,
            },
        };
        mgr.new_index(index_name, "FlatLite", "MetricL2", &params)
            .expect("fresh index creation does not fail");
        b.iter(|| {
            mgr.insert(
                index_name.to_string(),
                &eids,
                retrieval::ann::Points::Values {
                    vals: &base_vectors,
                },
            )
            .expect("insertion should not fail");
        });
        mgr.delete_index(index_name)
            .expect("deletion should be clean")
    });
    group.bench_function("flat::index", |b| {
        let index_name = "test-0000";
        let params = retrieval::ann::ANNParams::FlatLite {
            params: retrieval::flat_lite::FlatLiteParams {
                dim: dimensions,
                segment_size_kb: 1024,
            },
        };
        mgr.new_index(index_name, "Flat", "MetricL2", &params)
            .expect("fresh index creation does not fail");
        b.iter(|| {
            mgr.insert(
                index_name.to_string(),
                &eids,
                retrieval::ann::Points::Values {
                    vals: &base_vectors,
                },
            )
            .expect("insertion should not fail");
        });
        mgr.delete_index(index_name)
            .expect("deletion should be clean")
    });
}
fn search_benchmark(c: &mut Criterion) {
    let (base_vectors, search_vectors) = setup_sift("sift1m");
    let dimensions: usize = 128;
    let mut eids: Vec<retrieval::ann::EId> =
        vec![retrieval::ann::EId([0u8; 16]); base_vectors.len() / dimensions];
    for i in 0..base_vectors.len() / dimensions {
        let mut eid: retrieval::ann::EId = retrieval::ann::EId([0u8; 16]);
        BigEndian::write_uint(
            &mut eid.0,
            i.try_into().unwrap(),
            std::mem::size_of::<usize>(),
        );
        eids[i] = eid;
    }
    let _ = fs::remove_dir_all(".test");
    let mgr = IndexManager::new(&PathBuf::from(".test"));
    let index_name = "test-0000";
    let params = retrieval::ann::ANNParams::DiskANN {
        params: retrieval::diskannv1::DiskANNParams {
            dim: dimensions,
            max_points: eids.len() + 1,
            indexing_threads: None,
            indexing_range: 64,       // R
            indexing_queue_size: 100, // L
            indexing_maxc: 750,       // C
            indexing_alpha: 1.2,      // alpha
            maintenance_period_millis: 500,
        },
    };
    mgr.new_index(index_name, "DiskANNLite", "MetricL2", &params)
        .expect("fresh index creation does not fail");
    mgr.insert(
        index_name.to_string(),
        &eids,
        retrieval::ann::Points::Values {
            vals: &base_vectors,
        },
    )
    .expect("insertion should not fail");
    let pool: rayon::ThreadPool = rayon::ThreadPoolBuilder::new()
        .num_threads(6)
        .build()
        .expect("unable to create the threadpool");
    let mut group = c.benchmark_group("ann::search");
    group.throughput(Throughput::Elements(10000));
    group.sample_size(10);
    group.bench_function("diskann_lite::search", |b| {
        // b.iter(|| {
        //     search_vectors.chunks(dimensions).for_each(|s| {
        //         mgr.search(&index_name, retrieval::ann::Points::Values { vals: s }, 10)
        //             .expect("mgr.search throws no error");
        //     });
        // })
        b.iter_custom(|iters| {
            let start = Instant::now();
            pool.install(|| {
                search_vectors.par_chunks(128).for_each(|vsearch| {
                    mgr.search(
                        &index_name,
                        retrieval::ann::Points::Values { vals: vsearch },
                        10,
                    )
                    .expect("search should not throw an error");
                });
            });
            start.elapsed()
        });
    });
    group.finish();
}

criterion_group!(benches, search_benchmark, index_benchmark);
criterion_main!(benches);
