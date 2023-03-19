use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

use crate::diskannv1::DiskANNParams;
use crate::flat::FlatParams;

#[derive(Debug)]
pub enum ANNParams {
    Flat { params: FlatParams },
    DiskANN { params: DiskANNParams },
}

#[derive(Serialize, Deserialize)]
pub enum ANNTypes {
    DiskANN = 1,
    Flat = 2,
}

// primary trait that enables an obj to act as an ANNIndex - this
// allows us to use multiple different backends in the future.
pub trait ANNIndex: Send + Sync {
    fn new(params: &ANNParams) -> Result<Self, Box<dyn std::error::Error>>
    where
        Self: Sized;
    fn batch_insert(&self, eids: &[EId], data: &[f32]) -> Result<(), Box<dyn std::error::Error>>;
    fn insert(&self, eids: &[EId], data: &[f32]) -> Result<(), Box<dyn std::error::Error>>;
    fn search(&self, q: &[f32], k: usize) -> Result<Vec<Node>, Box<dyn std::error::Error>>;
    fn save(&self) -> Result<(), Box<dyn std::error::Error>>;
}

// we use a 16 byte representation for EIds - this would allow clients to
// ship us UUIDs as their resource identifiers with no addn work needed
pub type EId = [u8; 16];

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
