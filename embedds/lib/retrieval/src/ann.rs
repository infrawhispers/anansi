use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::default::Default;
use std::str::FromStr;

use crate::diskannv1::DiskANNParams;
use crate::flat_lite::FlatLiteParams;
use crate::metric;

#[cfg(feature = "full")]
use crate::flat_full::FlatFullParams;
// use pyo3::prelude::*;

use num::traits::NumAssign;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum ANNParams {
    FlatLite {
        params: FlatLiteParams,
    },
    #[cfg(feature = "full")]
    #[serde(skip)]
    FlatFull {
        params: FlatFullParams,
    },
    DiskANN {
        params: DiskANNParams,
    },
}

// #[pyclass]
#[derive(Serialize, Deserialize, Clone)]
pub enum ANNTypes {
    DiskANN = 1,
    Flat = 2,
}

pub trait IntoCopied {
    fn into_copied<'a, T>(self) -> std::iter::Copied<Self::IntoIter>
    where
        Self: Sized + IntoIterator<Item = &'a T>,
        T: 'a + Copy,
    {
        self.into_iter().copied()
    }
}

// we support f32s and u8s
pub trait ElementVal<'de>:
    num::Num
    + std::marker::Copy
    + std::default::Default
    + std::marker::Sync
    + std::marker::Send
    + NumAssign
    + std::ops::AddAssign
    + num::ToPrimitive
    + num::FromPrimitive
    + Sized
    + std::fmt::Debug
    + Serialize
    + Deserialize<'de>
{
    type Native;
}
impl ElementVal<'_> for f32 {
    type Native = f32;
}
impl ElementVal<'_> for f64 {
    type Native = f64;
}
impl ElementVal<'_> for u8 {
    type Native = u8;
}

pub enum Points<'a, T> {
    QuantizerIn { vals: &'a [f32] },
    Values { vals: &'a [T] },
}

/// primary trait that enables an obj to act as an ANNIndex - this
/// enables us to add different backends in the future as the similarity
/// search space improves
pub trait ANNIndex: Send + Sync {
    type Val;
    fn new(params: &ANNParams) -> anyhow::Result<Self>
    where
        Self: Sized;
    fn insert(&self, eids: &[EId], data: Points<Self::Val>) -> anyhow::Result<()>;
    fn delete(&self, eids: &[EId]) -> anyhow::Result<()>;
    fn search(&self, q: Points<Self::Val>, k: usize) -> anyhow::Result<Vec<Node>>;
    fn save(&self) -> anyhow::Result<()>;
    fn delete_index(&self) -> anyhow::Result<()>;
}

// we use a 16 byte representation for EIds - allowing clients to
// ship us UUIDs as their resource identifiers with no addn work needed
// pub type EId = [u8; 16];
#[derive(Eq, Hash, PartialEq, Default, Clone, Debug, Deserialize, Serialize, Copy)]
pub struct EId(pub [u8; 16]);

impl EId {
    pub fn as_bytes(&self) -> [u8; 16] {
        let mut b = [0u8; 16];
        b[..].copy_from_slice(&self.0[..]);
        return b;
    }
    // pub fn read_from<B: ByteSlice>(bytes: B) -> Option<Self> {
    //     let mut eid = EId([0u8; 16]);
    //     eid.0[0..id_bytes.len()].copy_from_slice(&bytes[..]);
    //     return Some(eid);
    // }
}

#[derive(Debug, PartialEq, Eq)]
pub struct ParseEIdError;

impl std::fmt::Display for ParseEIdError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "unable to parse the EId - these are expected to be at most 16 bytes"
        )
    }
}
impl std::error::Error for ParseEIdError {}

impl FromStr for EId {
    type Err = ParseEIdError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut eid = [0u8; 16];
        let id_bytes = s.as_bytes();
        if id_bytes.len() > 16 {
            return Err(ParseEIdError);
        }
        eid[0..id_bytes.len()].copy_from_slice(&id_bytes[..]);
        return Ok(EId(eid));
    }
}
impl ToString for EId {
    fn to_string(&self) -> String {
        match std::str::from_utf8(&self.0) {
            Ok(res) => return res.trim_end_matches(char::from(0)).to_string(),
            // ÿ -> 255 -> FF in Hex, we should probably log an error here
            // as we expect _all eids_ to be utf8
            Err(_) => return "ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ".to_string(),
        }
    }
}
// impl std::fmt::Display for EId {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         match std::str::from_utf8(&self.0) {
//             Ok(res) => {
//                 write!(f, "{}", res)
//             }
//             Err(err) => return std::fmt::Result::Err(ParseEIdError{})
//         }
//     }
// }

#[derive(Default, Clone, Debug)]
pub struct Node {
    pub vid: usize,
    pub eid: EId,
    pub distance: f32,
}

impl Ord for Node {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.distance == other.distance {
            return std::cmp::Ordering::Equal;
        }
        if self.distance < other.distance {
            return std::cmp::Ordering::Less;
        }
        std::cmp::Ordering::Greater
    }
}

impl PartialOrd for Node {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Node {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Node {}

#[derive(Default, Clone, Debug, Copy)]
pub struct INode {
    pub vid: usize,
    pub distance: f32,
    pub flag: bool,
}
impl Ord for INode {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.distance == other.distance {
            return std::cmp::Ordering::Equal;
        }
        if self.distance < other.distance {
            return std::cmp::Ordering::Less;
        }
        std::cmp::Ordering::Greater
    }
}
impl PartialOrd for INode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl PartialEq for INode {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for INode {}

pub fn round_up(x: u32) -> u32 {
    ((x - 1) | (16 - 1)) + 1
    // let mut v = x;
    // v -= 1;
    // v |= v >> 1;
    // v |= v >> 2;
    // v |= v >> 4;
    // v |= v >> 8;
    // v |= v >> 16;
    // v += 1;
    // return v;
}

pub fn copy_within_a_slice<T: Clone>(v: &mut [T], from: usize, to: usize, len: usize) {
    if from > to {
        let (dst, src) = v.split_at_mut(from);
        dst[to..to + len].clone_from_slice(&src[..len]);
    } else {
        let (src, dst) = v.split_at_mut(to);
        dst[..len].clone_from_slice(&src[from..from + len]);
    }
}

pub fn get_padded_vector<T: for<'de> ElementVal<'de>>(
    data: &[T],
    current_dim: usize,
    aligned_dim: usize,
) -> Vec<T> {
    let mut result: Vec<T> = vec![Default::default(); (data.len() / current_dim) * aligned_dim];
    let mut cnt: usize = 0;
    for idx in (0..data.len()).step_by(current_dim) {
        let into_offset = (aligned_dim - current_dim) * cnt;
        result[idx + into_offset..idx + current_dim + into_offset]
            .copy_from_slice(&data[idx..idx + current_dim]);
        cnt += 1
    }
    result
}

pub fn pad_and_preprocess<T: for<'de> ElementVal<'de>, M: metric::Metric<T>>(
    data: &[T],
    current_dim: usize,
    aligned_dim: usize,
) -> Option<Vec<T>> {
    let mut padded_vector: Vec<T>;
    if !M::uses_preprocessor() && current_dim == aligned_dim {
        return None;
    }
    if current_dim != aligned_dim {
        padded_vector = get_padded_vector::<T>(data, current_dim, aligned_dim);
        // vec = &padded_vector;
    } else {
        padded_vector = vec![Default::default(); (data.len() / current_dim) * aligned_dim];
        padded_vector[..].copy_from_slice(&data[..]);
        // vec = data;
    }
    if M::uses_preprocessor() {
        let num_vectors = padded_vector.len() / aligned_dim;
        for idx in 0..num_vectors {
            let idx_s_fr = idx * aligned_dim;
            let idx_e_fr = idx_s_fr + aligned_dim;
            let vec: &[T] = if current_dim != aligned_dim {
                &padded_vector
            } else {
                data
            };
            let res = M::pre_process(&vec[idx_s_fr..idx_e_fr]);
            match res {
                Some(vec_result) => {
                    padded_vector[idx_s_fr..idx_e_fr].copy_from_slice(&vec_result[..]);
                }
                None => {}
            }
        }
    }
    Some(padded_vector)

    /*
       // STEP 1 - align the vector if we need to!
       // let padded_vector: Vec<TVal>;
       // if per_vector_dim != self.aligned_dim {
       //     padded_vector = ann::get_padded_vector::<TVal>(data, per_vector_dim, self.aligned_dim);
       //     data = &padded_vector;
       // }

       // // STEP 2 - now do any preprocessing that we may need to do!
       // let padded_point: &[TVal];
       // let mut preprocess_scratch: Vec<TVal>;
       // let aligned_dim = self.aligned_dim;
       // if TMetric::uses_preprocessor() {
       //     preprocess_scratch = vec![Default::default(); self.aligned_dim];
       //     for idx in 0..vids.len() {
       //         let idx_s_fr = idx * aligned_dim;
       //         let idx_e_fr = idx_s_fr + aligned_dim;
       //         match TMetric::pre_process(&data[idx_s_fr..idx_e_fr]) {
       //             Some(vec) => {
       //                 preprocess_scratch[idx_s_fr..idx_e_fr].copy_from_slice(&vec[0..vec.len()]);
       //             }
       //             None => {
       //                 preprocess_scratch[idx_s_fr..idx_e_fr]
       //                     .copy_from_slice(&data[idx_s_fr..idx_e_fr]);
       //             }
       //         }
       //     }
       //     padded_point = &preprocess_scratch[..];
       // } else {
       //     padded_point = data;
       // }
    */
}
