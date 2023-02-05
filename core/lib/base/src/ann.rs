use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

use crate::flat::FlatParams;

pub enum ANNParams {
    Flat { params: FlatParams },
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
    fn batch_insert(&self, eids: &[EId], data: &[f32]) -> Result<bool, Box<dyn std::error::Error>>;
    fn insert(&self, eid: EId, data: &[f32]) -> Result<bool, Box<dyn std::error::Error>>;
    fn search(&self, q: &[f32], k: usize) -> Result<Vec<Node>, Box<dyn std::error::Error>>;
    fn save(&self) -> Result<bool, Box<dyn std::error::Error>>;
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

pub fn round_up(x: u32) -> u32 {
    let mut v = x;
    v -= 1;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v += 1;
    return v;
}
