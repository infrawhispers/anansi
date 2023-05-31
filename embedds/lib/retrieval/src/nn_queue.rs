use crate::ann;
use std::ops::Index;

pub struct NNPriorityQueue {
    size: usize,
    capacity: usize,
    curr: usize,
    pub data: Vec<ann::INode>,
}

impl Index<usize> for NNPriorityQueue {
    type Output = ann::INode;
    fn index(&self, i: usize) -> &Self::Output {
        &self.data[i]
    }
}

impl NNPriorityQueue {
    pub fn print(&self) {
        for i in 0..self.size {
            print!(
                "[id: {}, expanded: {}, distance: {}] ",
                self.data[i].vid, self.data[i].flag, self.data[i].distance
            );
        }
        println!("");
    }
    pub fn new(capacity: usize) -> Self {
        NNPriorityQueue {
            size: 0,
            capacity: capacity,
            curr: 0,
            data: vec![
                ann::INode {
                    vid: 0,
                    flag: false,
                    distance: std::f32::INFINITY,
                };
                capacity
            ],
        }
    }
    pub fn has_unexpanded_node(&self) -> bool {
        return self.curr < self.size;
    }
    pub fn closest_unexpanded(&mut self) -> ann::INode {
        self.data[self.curr].flag = true;
        let pre: usize = self.curr;
        while self.curr < self.size && self.data[self.curr].flag {
            self.curr += 1
        }
        return self.data[pre];
    }
    pub fn clear(&mut self) {
        self.size = 0;
        self.curr = 0;
        self.data[..self.capacity].fill(ann::INode {
            vid: 0,
            distance: std::f32::INFINITY,
            flag: false,
        });
    }

    pub fn reserve(&mut self, capacity: usize) {
        if capacity + 1 > self.data.len() {
            self.data.resize(
                capacity + 1,
                ann::INode {
                    vid: 0,
                    distance: std::f32::INFINITY,
                    flag: false,
                },
            );
        }
        self.capacity = capacity;
    }

    pub fn insert(&mut self, nbr: ann::INode) {
        if self.size == self.capacity && self.data[self.size - 1].distance < nbr.distance {
            return;
        }
        let mut lo: usize = 0;
        let mut hi: usize = self.size;

        while lo < hi {
            let mid: usize = (lo + hi) >> 1;
            if nbr.distance < self.data[mid].distance {
                hi = mid;
            } else if self.data[mid].vid == nbr.vid {
                return;
            } else {
                lo = mid + 1;
            }
        }
        // there is no place to insert our items, so let us bail
        if lo >= self.data.len() {
            return;
        }
        if lo < self.capacity {
            let length = self.data.len();
            self.data.copy_within(lo..length - 1, lo + 1);
        }
        // println!("added node!");
        self.data[lo] = ann::INode {
            vid: nbr.vid,
            flag: false,
            distance: nbr.distance,
        };

        if self.size < self.capacity {
            self.size += 1
        }
        if lo < self.curr {
            self.curr = lo;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_nn_queue() {
        let mut queue = NNPriorityQueue::new(3);
        queue.insert(ann::INode {
            vid: 0,
            distance: 0.5,
            flag: false,
        });
        queue.insert(ann::INode {
            vid: 1,
            distance: 0.1,
            flag: false,
        });
        queue.insert(ann::INode {
            vid: 2,
            distance: 0.2,
            flag: false,
        });
        queue.insert(ann::INode {
            vid: 3,
            distance: 0.5,
            flag: false,
        });
        // assert_eq!(queue[0].distance, 0.02);
    }
}
