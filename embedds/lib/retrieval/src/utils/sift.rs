use std::collections::HashMap;
use std::fs;
use std::io::Cursor;
use std::path::Path;

use byteorder::{BigEndian, ByteOrder};
use byteorder::{LittleEndian, ReadBytesExt};

pub struct SIFT<'a> {
    pub directory: &'a Path,
    pub dims: usize,
}

impl SIFT<'_> {
    pub fn fetch_vectors(&self, filename: &str) -> anyhow::Result<Vec<f32>> {
        let path = self.directory.join(filename);
        let data_r = fs::read(path)?;
        let data_r_slice = data_r.as_slice();

        let num_vectors = data_r.len() / (4 + self.dims * 4);
        let mut data_w: Vec<f32> = Vec::with_capacity(num_vectors * self.dims);
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
    fn fetch_ground_truth(&self, filename: &str, num_closest: usize) -> anyhow::Result<Vec<u32>> {
        let path = self.directory.join(filename);
        let data_r = fs::read(path)?;
        let data_r_slice = data_r.as_slice();
        let mut rdr = Cursor::new(data_r_slice);

        // we store the dimensionality of the vector here
        // and then the components are (unsigned char|float | int)*d
        // documentation is avail here: http://corpus-texmex.irisa.fr/
        let num_vectors = (data_r.len() as u32 / (4 + (num_closest as u32) * 4)) as usize;
        let mut data_w: Vec<u32> = Vec::with_capacity(num_vectors * num_closest as usize);
        for _ in 0..num_vectors {
            let dim = rdr.read_u32::<LittleEndian>().unwrap();
            // if dim != num_closest.try_into().unwrap() {
            //     panic!(
            //         "dim != num_closest: dim: {} num_closest: {}",
            //         dim, num_closest
            //     );
            // }
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
    ) -> anyhow::Result<HashMap<crate::ann::EId, Vec<crate::ann::EId>>> {
        let truth_vecs = self.fetch_ground_truth(filename, num_closest)?;
        let mut truth_by_id: HashMap<crate::ann::EId, Vec<crate::ann::EId>> = HashMap::new();
        for id in 0..truth_vecs.len() / num_closest {
            let truth_vec = truth_vecs
                [(id * num_closest) as usize..(id * num_closest + num_closest) as usize]
                .to_vec();
            let mut truth_eids: Vec<crate::ann::EId> = Vec::with_capacity(k);
            for i in 0..truth_vec.len() {
                if i == k {
                    break;
                }
                let mut eid: crate::ann::EId = crate::ann::EId([0u8; 16]);
                BigEndian::write_uint(
                    &mut eid.0,
                    truth_vec[i].try_into().unwrap(),
                    std::mem::size_of::<usize>(),
                );
                truth_eids.push(eid);
            }
            let mut ref_eid: crate::ann::EId = crate::ann::EId([0u8; 16]);
            BigEndian::write_uint(
                &mut ref_eid.0,
                id.try_into().unwrap(),
                std::mem::size_of::<usize>(),
            );
            truth_by_id.insert(ref_eid, truth_eids);
        }
        Ok(truth_by_id)
    }
}

pub fn get_sift_vectors(
    dir_name: &str,
) -> (
    Vec<f32>,
    Vec<f32>,
    HashMap<crate::ann::EId, Vec<crate::ann::EId>>,
) {
    let path = format!("../../../../eval/data/{dir_name}/");
    let directory = Path::new(&path);
    let dims: usize = 128;
    let loader = SIFT {
        directory: directory,
        dims: dims,
    };
    let k: usize = 10;
    let base_vectors = loader
        .fetch_vectors("sift_base.fvecs")
        .expect("unable to fetch base vectors from disk");
    let mut eids: Vec<crate::ann::EId> = vec![crate::ann::EId([0u8; 16]); base_vectors.len() / 128];
    for i in 0..base_vectors.len() / 128 {
        let mut eid: crate::ann::EId = crate::ann::EId([0u8; 16]);
        BigEndian::write_uint(
            &mut eid.0,
            i.try_into().unwrap(),
            std::mem::size_of::<usize>(),
        );
        eids[i] = eid;
    }
    let query_vectors = loader
        .fetch_vectors("sift_query.fvecs")
        .expect("unable to fetch the query vectors");
    let ground_truth = loader
        .fetch_ground_truth_by_id("sift_groundtruth.ivecs", k, 100)
        .expect("unable to fetch the groundtruth");
    (base_vectors, query_vectors, ground_truth)
}
