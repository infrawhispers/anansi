use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use byteorder::{BigEndian, ByteOrder};
use criterion::Throughput;
use criterion::{criterion_group, criterion_main, Criterion};
use rayon::prelude::*;

use retrieval::manager::index_manager::IndexManager;
use retrieval::utils::sift::get_sift_vectors;

fn index_benchmark(c: &mut Criterion) {
    let (base_vectors, _, _) = get_sift_vectors("siftsmall");
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
    let (base_vectors, search_vectors, _) = get_sift_vectors("siftsmall");
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
        b.iter_custom(|_iters| {
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
