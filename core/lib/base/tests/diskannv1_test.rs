use byteorder::{BigEndian, ByteOrder};
use byteorder::{LittleEndian, ReadBytesExt};
use std::collections::HashMap;
use std::collections::HashSet;
use std::fs;
use std::io::Cursor;
use std::path::Path;
use std::sync::Arc;

use base::ann;
use base::ann::ANNIndex;
use base::diskannv1;
use base::metric;

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
            if dim != self.dims.try_into()? {
                panic!("dim mismatch while reading the source data");
            }
            for _j in 0..dim {
                data_w.push(rdr.read_f32::<LittleEndian>().unwrap());
            }
        }
        Ok(data_w)
    }
    fn fetch_ground_truth(&self, filename: &str, num_closest: usize) -> anyhow::Result<Vec<u32>> {
        let path = self.directory.join(filename);
        let data_r = fs::read(path)?;
        let data_r_slice = data_r.as_slice();
        let mut rdr = Cursor::new(data_r_slice);

        // we store the dimensionality of the vector here
        // and then the components are (unsigned char|float | int)*d
        // documentation is avail here: http://corpus-texmex.irisa.fr/
        // let dims = 100;
        let num_vectors = (data_r.len() as u32 / (4 + (num_closest as u32) * 4)) as usize;
        let mut data_w: Vec<u32> = Vec::with_capacity(num_vectors * num_closest as usize);
        for _ in 0..num_vectors {
            let dim = rdr.read_u32::<LittleEndian>().unwrap();
            if dim != num_closest.try_into().unwrap() {
                panic!(
                    "dim != num_closest: dim: {} num_closest: {}",
                    dim, num_closest
                );
            }
            for _j in 0..dim {
                data_w.push(rdr.read_u32::<LittleEndian>().unwrap());
            }
        }
        Ok(data_w)
    }

    fn fetch_ground_truth_by_id(
        &self,
        filename: &str,
        k: usize,
        num_closest: usize,
    ) -> anyhow::Result<HashMap<base::ann::EId, HashSet<base::ann::EId>>> {
        let truth_vecs = self.fetch_ground_truth(filename, num_closest)?;
        let mut truth_by_id: HashMap<base::ann::EId, HashSet<base::ann::EId>> = HashMap::new();
        for id in 0..truth_vecs.len() / num_closest {
            let truth_vec = truth_vecs
                [(id * num_closest) as usize..(id * num_closest + num_closest) as usize]
                .to_vec();
            let mut eids: HashSet<base::ann::EId> = HashSet::with_capacity(k);
            // vec![[0u8; 16]; truth_vec.len()];
            // if id == 10 {
            //     println!("{:?}", truth_vec);
            // }
            for i in 0..truth_vec.len() {
                if i == k {
                    break;
                }
                let mut eid: base::ann::EId = [0u8; 16];
                BigEndian::write_uint(
                    &mut eid,
                    truth_vec[i].try_into().unwrap(),
                    std::mem::size_of::<usize>(),
                );
                eids.insert(eid);
            }
            // truth_vec.into_iter().enumerate().for_each(|(i, x)| {
            //     eids[i] = eid;
            // });
            let mut ref_eid: base::ann::EId = [0u8; 16];
            BigEndian::write_uint(
                &mut ref_eid,
                id.try_into().unwrap(),
                std::mem::size_of::<usize>(),
            );
            truth_by_id.insert(ref_eid, eids);
        }
        Ok(truth_by_id)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn sift_small_deletes() {}
    #[test]
    fn sift_small_ann() {
        // let directory = Path::new("../../../../eval/data/siftsmall/");'
        let directory = Path::new("../../data/siftsmall/");
        let dims: usize = 128;
        let loader = SIFT {
            directory: directory,
            dims: dims,
        };
        let k: usize = 10;

        let base_vectors = loader
            .fetch_vectors("sift_base.fvecs", 128)
            .expect("unable to fetch the base vectors");
        let mut eids: Vec<base::ann::EId> = vec![[0u8; 16]; base_vectors.len() / 128];
        for i in 0..base_vectors.len() / 128 {
            let mut eid: base::ann::EId = [0u8; 16];
            BigEndian::write_uint(
                &mut eid,
                i.try_into().unwrap(),
                std::mem::size_of::<usize>(),
            );
            eids[i] = eid;
        }
        let query_vectors = loader
            .fetch_vectors("sift_query.fvecs", 128)
            .expect("unable to fetch the query vectors");

        let _truth_vectors = loader
            .fetch_ground_truth_by_id("sift_groundtruth.ivecs", k, 100)
            .expect("fetching the items");
        let _num_truth_vectors = _truth_vectors.len();
        let params = ann::ANNParams::DiskANN {
            params: diskannv1::DiskANNParams {
                dim: 128,
                max_points: eids.len(),
                indexing_threads: None,
                indexing_range: 64,       // R
                indexing_queue_size: 100, // L
                indexing_maxc: 140,       // C
                indexing_alpha: 1.2,      // alph
                maintenance_period_millis: 500,
            },
        };
        let ann_idx: Arc<diskannv1::DiskANNV1Index<metric::MetricL2>> =
            Arc::new(ann::ANNIndex::new(&params).expect("error creating diskannv1 index"));
        let _t_ann_idx = ann_idx.clone();
        let _t = std::thread::spawn(move || {
            _t_ann_idx.maintain();
        });

        assert!(
            ann_idx.batch_insert(&eids, &base_vectors).is_ok(),
            "unexpexted err on batch_insert to the vector store"
        );
        let mut total_intersection_count: usize = 0;
        for i in 0.._num_truth_vectors {
            let query_vec = query_vectors[(i * dims) as usize..(i * dims + dims) as usize].to_vec();
            let mut query_eid: ann::EId = [0u8; 16];
            BigEndian::write_uint(
                &mut query_eid,
                i.try_into().unwrap(),
                std::mem::size_of::<usize>(),
            );
            let nns = ann_idx
                .search(&query_vec, k)
                .expect("unexpected error fetching the closest vectors");
            let mut nodes_found: HashSet<ann::EId> = HashSet::new();
            for nn in nns.iter() {
                let mut eid: base::ann::EId = [0u8; 16];
                BigEndian::write_uint(
                    &mut eid,
                    nn.vid.try_into().unwrap(),
                    std::mem::size_of::<usize>(),
                );
                nodes_found.insert(eid);
            }
            let nodes_gnd_truth = _truth_vectors.get(&(query_eid)).unwrap();
            let intersection_count = nodes_gnd_truth.intersection(&nodes_found).count();
            total_intersection_count += intersection_count;
            assert!(
                (intersection_count as f32 / (k as f32)) * 100.0 >= 90.0f32,
                "for vid: {} | found: {}/{} ",
                i,
                intersection_count,
                k
            );
        }
        assert!(
            (total_intersection_count as f32 / (k * _num_truth_vectors) as f32) > 0.95f32,
            "unexpectedly lowered true neighbours found"
        );
    }
}
