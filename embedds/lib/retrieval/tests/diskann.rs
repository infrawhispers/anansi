// use std::collections::HashSet;
// use std::sync::Arc;

// use byteorder::{BigEndian, ByteOrder};
// // use tracing::info;

// use retrieval::ann::ANNIndex;
// use retrieval::utils::sift::get_sift_vectors;

// #[cfg(test)]
// mod test {
//     use super::*;
//     #[test]
//     fn diskann_small() {
//         // let (base_vectors, _, _) = get_sift_vectors("sift1m");
//         // let dims: usize = 128;
//         // println!("num base_vectors: {:?}", base_vectors.len() / dims);
//         // let mut eids: Vec<retrieval::ann::EId> =
//         //     vec![retrieval::ann::EId([0u8; 16]); base_vectors.len() / dims];
//         // for i in 0..base_vectors.len() / dims {
//         //     let mut eid: retrieval::ann::EId = retrieval::ann::EId([0u8; 16]);
//         //     BigEndian::write_uint(
//         //         &mut eid.0,
//         //         i.try_into().unwrap(),
//         //         std::mem::size_of::<usize>(),
//         //     );
//         //     eids[i] = eid;
//         // }

//         // let params = retrieval::ann::ANNParams::DiskANN {
//         //     params: retrieval::diskannv1::DiskANNParams {
//         //         dim: dims,
//         //         max_points: eids.len(),
//         //         indexing_threads: None,
//         //         indexing_range: 64,       // R
//         //         indexing_queue_size: 100, // L
//         //         indexing_maxc: 750,       // C
//         //         indexing_alpha: 1.2,      // alpha
//         //         maintenance_period_millis: 500,
//         //     },
//         // };
//         // let idx: retrieval::diskannv1::DiskANNV1Index<retrieval::metric::MetricL2, f32> =
//         //     retrieval::diskannv1::DiskANNV1Index::new(&params)
//         //         .expect("index creation should not throw err");
//         // let ann_idx: Arc<dyn retrieval::ann::ANNIndex<Val = f32>> = Arc::new(idx);
//         // ann_idx
//         //     .insert(
//         //         &eids,
//         //         retrieval::ann::Points::Values {
//         //             vals: &base_vectors,
//         //         },
//         //     )
//         //     .expect("batch insert should not throw err");
//         // let num_search_vectors = search_vectors.len() / dims;
//         // let k: usize = 10;
//         // (0..num_search_vectors).for_each(|search_idx| {
//         //     let search_vector = search_vectors
//         //         [(search_idx * dims) as usize..(search_idx * dims + dims) as usize]
//         //         .to_vec();
//         //     let mut search_eid: retrieval::ann::EId = retrieval::ann::EId([0u8; 16]);
//         //     BigEndian::write_uint(
//         //         &mut search_eid.0,
//         //         search_idx.try_into().unwrap(),
//         //         std::mem::size_of::<usize>(),
//         //     );
//         //     let nns = ann_idx
//         //         .search(
//         //             retrieval::ann::Points::Values {
//         //                 vals: &search_vector,
//         //             },
//         //             k,
//         //         )
//         //         .expect("unexpected error fetching the closest vectors");

//         //     let mut nodes_found: HashSet<retrieval::ann::EId> = HashSet::new();
//         //     for nn in nns.iter() {
//         //         nodes_found.insert(nn.eid);
//         //     }
//         //     let mut nodes_gnd_truth: HashSet<retrieval::ann::EId> = HashSet::new();
//         //     let closest_ids = truth_vectors
//         //         .get(&(search_eid))
//         //         .expect("closest ids should exist");
//         //     closest_ids[0..k].iter().for_each(|nn_eid| {
//         //         nodes_gnd_truth.insert(*nn_eid);
//         //     });
//         //     let intersection_count = nodes_gnd_truth.intersection(&nodes_found).count();
//         //     assert!(
//         //         (intersection_count as f32 / (k as f32)) * 100.0 >= 90.0f32,
//         //         "for vid: {search_idx} | found: {intersection_count}/{k}",
//         //     );
//         // });
//     }
// }
