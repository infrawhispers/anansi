use byteorder::{BigEndian, ByteOrder};
use byteorder::{LittleEndian, ReadBytesExt};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::Cursor;
use std::path::Path;

use base::ann;
use base::ann::EId;
use base::flat;
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
        let dims = rdr.read_u32::<LittleEndian>().unwrap();
        let num_vectors = (data_r.len() as u32 / (4 + dims * 4)) as usize;
        let mut data_w: Vec<u32> = Vec::with_capacity(num_vectors * num_closest as usize);
        for _ in 0..num_vectors {
            for _j in 0..dims {
                data_w.push(rdr.read_u32::<LittleEndian>().unwrap());
            }
        }
        Ok(data_w)
    }

    fn fetch_ground_truth_by_id(
        &self,
        filename: &str,
        num_closest: usize,
    ) -> anyhow::Result<HashMap<base::ann::EId, Vec<base::ann::EId>>> {
        let truth_vecs = self.fetch_ground_truth(filename, num_closest)?;
        let mut truth_by_id: HashMap<base::ann::EId, Vec<base::ann::EId>> = HashMap::new();
        for id in 0..truth_vecs.len() / num_closest {
            let truth_vec = truth_vecs
                [(id * num_closest) as usize..(id * num_closest + num_closest) as usize]
                .to_vec();
            let mut eids: Vec<base::ann::EId> = vec![[0u8; 16]; truth_vec.len()];
            truth_vec.into_iter().enumerate().for_each(|(i, x)| {
                let mut eid: base::ann::EId = [0u8; 16];
                BigEndian::write_uint(
                    &mut eid,
                    x.try_into().unwrap(),
                    std::mem::size_of::<usize>(),
                );
                eids[i] = eid;
            });
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
pub fn add(left: usize, right: usize) -> usize {
    left + right
}

#[cfg(test)]
mod tests {
    use base::ann::ANNIndex;

    use super::*;

    #[test]
    fn sift_small_exact() {
        let directory = Path::new("../../data/siftsmall/");
        let loader = SIFT {
            directory: directory,
            dims: 128,
        };
        let dims: usize = 128;
        let k: usize = 100;

        let base_vectors = loader
            .fetch_vectors("siftsmall_base.fvecs", 128)
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
            .fetch_vectors("siftsmall_query.fvecs", 128)
            .expect("unable to fetch the query vectors");

        let truth_vectors = loader
            .fetch_ground_truth_by_id("siftsmall_groundtruth.ivecs", k)
            .expect("fetching the items");
        let num_truth_vectors = truth_vectors.len();

        let params = ann::ANNParams::Flat {
            params: flat::FlatParams {
                dim: 128,
                segment_size_kb: 50,
            },
        };
        let index: flat::FlatIndex<metric::MetricL2> =
            ann::ANNIndex::new(&params).expect("error creating flat index");
        index
            .batch_insert(&eids, &base_vectors)
            .expect("unable to batch insert");

        let mut res_vectors: HashMap<base::ann::EId, Vec<base::ann::EId>> =
            HashMap::with_capacity(truth_vectors.len());

        for idx in 0..num_truth_vectors {
            let mut eid: base::ann::EId = [0u8; 16];
            BigEndian::write_uint(
                &mut eid,
                idx.try_into().unwrap(),
                std::mem::size_of::<usize>(),
            );

            let q = query_vectors[(idx * dims) as usize..(idx * dims + dims) as usize].to_vec();
            let res = index
                .search(&q, k)
                .expect("error retrieving nearest k vectors");
            let mut nodes: Vec<base::ann::EId> = Vec::with_capacity(k);
            res.iter().for_each(|node| {
                nodes.push(node.eid);
            });
            res_vectors.insert(eid, nodes);
        }
    }
}
