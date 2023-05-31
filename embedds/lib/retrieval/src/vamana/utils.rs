use anyhow::{bail, Context};
use byteorder::{LittleEndian, ReadBytesExt};
use tracing::info;

use std::io::Cursor;
use std::io::Read;
use std::io::Seek;
use std::io::SeekFrom;
use std::path::Path;

use crate::av_store::AlignedDataStore;

pub const GRAPH_SLACK_FACTOR: f32 = 1.2;
pub const OVERHEAD_FACTOR: f32 = 1.1;

pub fn round_up(num_to_round: usize, multiple: usize) -> usize {
    ((num_to_round + multiple - 1) / multiple) * multiple
}
pub fn div_round_up(x: usize, y: usize) -> usize {
    let mut res = x / y;
    if x % y != 0 {
        res += 1
    }
    res
}

// pub fn get_mut_slices(src: &mut [u8], indexes: Vec<usize>) -> Vec<&mut [u8]> {
// indexes.sort();
// let index_len = indexes.len();
// if index_len == 0 {
//     return Vec::new();
// }
// let max_index = indexes[index_len - 1];
// if max_index > src.len() {
//     panic!("{} index is out of bounds of data", max_index);
// }
// indexes.dedup();
// let uniq_index_len = indexes.len();
// if index_len != uniq_index_len {
//     panic!("cannot return aliased mut refs to overlapping indexes");
// }
// let mut mut_slices_iter = src.iter_mut();
// let mut mut_slices = Vec::with_capacity(index_len);
// let mut last_index = 0;
// for curr_index in indexes {
//     mut_slices.push(
//         mut_slices_iter
//             .nth(curr_index - last_index)
//             .unwrap()
//             .as_mut_slice(),
//     );
//     last_index = curr_index;
// }

// // return results
// mut_slices
// }

pub fn estimate_ram_usage(size: usize, dim: usize, datasize: usize, degree: usize) -> f32 {
    let size_of_data: f32 = (size as f32) * (dim as f32) * (datasize as f32);
    let size_of_graph: f32 = (size as f32)
        * (degree as f32)
        * (std::mem::size_of::<usize>() as f32)
        * GRAPH_SLACK_FACTOR;
    let size_of_locks = size as f32;
    let size_of_outer_vector = (size as f32) * 4.0;
    OVERHEAD_FACTOR * (size_of_data + size_of_graph + size_of_locks + size_of_outer_vector)
}

pub struct BinLoad {
    // pub buf_u32: Option<Vec<u32>>,
    pub buf_usize: Option<Vec<usize>>,
    pub buf_f32: Option<Vec<f32>>,
    pub num_points: usize,
    pub dims: usize,
}

pub fn get_metadata_ref(filepath: &Path, offset: usize) -> anyhow::Result<(usize, usize)> {
    let mut f = std::fs::File::open(filepath)?;
    f.seek(SeekFrom::Start(offset.try_into()?))?;
    let dims: usize = f.read_u32::<LittleEndian>()?.try_into()?;
    let num_points: usize = f.read_u32::<LittleEndian>()?.try_into()?;
    Ok((dims, num_points))
}

/*
let mut output_file_meta = vec![];
        output_file_meta.write_u64::<LittleEndian>(num_points)?;
        output_file_meta.write_u64::<LittleEndian>(dims.try_into()?)?;
        output_file_meta.write_u64::<LittleEndian>(start)?;
        output_file_meta.write_u64::<LittleEndian>(max_node_len)?;
        output_file_meta.write_u64::<LittleEndian>(nnodes_per_sector.try_into()?)?;
        output_file_meta.write_u64::<LittleEndian>(1)?;
        output_file_meta.write_u64::<LittleEndian>(vamana_frozen_loc)?;
        output_file_meta.write_u64::<LittleEndian>(index_size)?;
*/
pub struct IndexMetadata {
    pub num_points: usize,
    pub dims: usize,
    pub start: usize,
    pub max_node_len: usize,
    pub nnodes_per_sector: usize,
    pub num_frozen_pts: usize,
    pub vamana_frozen_loc: usize,
    pub index_size: usize,
}
pub fn index_metadata_fr_file(filepath: &Path) -> anyhow::Result<IndexMetadata> {
    let buf: Vec<u8> = std::fs::read(filepath).with_context(|| "unable to open the file")?;
    let mut cursor = Cursor::new(buf);
    let num_points: usize = cursor.read_u64::<LittleEndian>()?.try_into()?;
    let dims: usize = cursor.read_u64::<LittleEndian>()?.try_into()?;
    let start: usize = cursor.read_u64::<LittleEndian>()?.try_into()?;
    let max_node_len: usize = cursor.read_u64::<LittleEndian>()?.try_into()?;
    let nnodes_per_sector: usize = cursor.read_u64::<LittleEndian>()?.try_into()?;
    let num_frozen_pts: usize = cursor.read_u64::<LittleEndian>()?.try_into()?;
    let vamana_frozen_loc: usize = cursor.read_u64::<LittleEndian>()?.try_into()?;
    let index_size: usize = cursor.read_u64::<LittleEndian>()?.try_into()?;
    Ok(IndexMetadata {
        num_points: num_points,
        dims: dims,
        start: start,
        max_node_len: max_node_len,
        nnodes_per_sector: nnodes_per_sector,
        num_frozen_pts: num_frozen_pts,
        vamana_frozen_loc: vamana_frozen_loc,
        index_size: index_size,
    })
}

pub struct BinAlignedLoad {
    pub arr: AlignedDataStore<f32>,
    pub num_points: usize,
    pub dims: usize,
    pub aligned_dim: usize,
}
pub fn ref_load_aligned(filepath: &Path) -> anyhow::Result<BinAlignedLoad> {
    let buf: Vec<u8> = std::fs::read(filepath).with_context(|| "unable to open the file")?;
    let mut cursor = Cursor::new(buf);

    let num_points: usize = cursor.read_u32::<LittleEndian>()?.try_into()?;
    let dims: usize = cursor.read_u32::<LittleEndian>()?.try_into()?;
    let aligned_dim: usize = crate::vamana::utils::round_up(dims, 8);

    let arr = AlignedDataStore::<f32>::new(num_points, aligned_dim);
    for i in 0..num_points {
        unsafe {
            let ptr =
                (arr.data.as_ptr() as *mut u8).add(i * aligned_dim * std::mem::size_of::<f32>());
            let num_bytes: usize = dims * std::mem::size_of::<f32>();
            let dest: &mut [u8] = std::slice::from_raw_parts_mut(ptr, num_bytes);
            cursor
                .read_exact(dest)
                .with_context(|| "unable to read {num_bytes} u8 fr loaded file")?;
        }
    }
    Ok(BinAlignedLoad {
        arr: arr,
        num_points: num_points,
        dims: dims,
        aligned_dim: aligned_dim,
    })
}

pub fn diskann_load_bin_generic(
    filepath: &Path,
    _offset: usize,
) -> anyhow::Result<(usize, usize, Vec<u8>)> {
    info!("opening bin file: {filepath:?}");
    let buf: Vec<u8> = std::fs::read(filepath).with_context(|| "unable to open the file")?;
    let mut cursor = Cursor::new(buf);
    let num_points = cursor.read_u32::<LittleEndian>()?.try_into()?;
    let dims = cursor.read_u32::<LittleEndian>()?.try_into()?;
    let mut data: Vec<u8> = vec![0u8; num_points * dims];
    for idx in 0..num_points * dims {
        data[idx as usize] = cursor.read_u8()?;
    }
    Ok((num_points, dims, data))
}

pub fn diskann_load_bin(
    filepath: &Path,
    offset: usize,
    data_type: &str,
) -> anyhow::Result<BinLoad> {
    info!("opening bin file: {filepath:?}");
    let buf: Vec<u8> = std::fs::read(filepath).with_context(|| "unable to open the file")?;
    let mut cursor = Cursor::new(buf);
    cursor.seek(SeekFrom::Start(offset.try_into()?))?;
    let num_points = cursor.read_u32::<LittleEndian>()?;
    let dims = cursor.read_u32::<LittleEndian>()?;
    info!("metadata: num_points: {num_points} | dims: {dims}");
    match data_type {
        "f32" => {
            let mut data: Vec<f32> = vec![0.0; (num_points * dims).try_into()?];
            for idx in 0..num_points * dims {
                data[idx as usize] = cursor.read_f32::<LittleEndian>()?;
            }
            return Ok(BinLoad {
                buf_usize: None,
                buf_f32: Some(data),
                num_points: num_points as usize,
                dims: dims as usize,
            });
        }
        "u32" => {
            let mut data: Vec<usize> = vec![0; (num_points * dims).try_into()?];
            for idx in 0..num_points * dims {
                data[idx as usize] = cursor.read_u32::<LittleEndian>()? as usize;
            }
            return Ok(BinLoad {
                buf_usize: Some(data),
                buf_f32: None,
                num_points: num_points as usize,
                dims: dims as usize,
            });
        }
        "u64" => {
            let mut data: Vec<usize> = vec![0; (num_points * dims).try_into()?];
            for idx in 0..num_points * dims {
                data[idx as usize] = cursor.read_u64::<LittleEndian>()? as usize;
            }
            return Ok(BinLoad {
                buf_usize: Some(data),
                buf_f32: None,
                num_points: num_points as usize,
                dims: dims as usize,
            });
        }

        _ => {
            bail!("invalid data_type: {data_type} expected one of [ f32, u32, u64]")
        }
    }
}

#[cfg(test)]
mod test {
    // use super::*;
    // #[test]
    // fn read_pq_pivots() {
    //     // data/siftsmall/disk_index_sift_learn_R32_L50_A1.2_pq_pivots.bin
    //     let filepath = Path::new("../../../../public/DiskANN/build/data/siftsmall/disk_index_sift_learn_R32_L50_A1.2_pq_pivots.bin");
    //     let meta =
    //         diskann_load_bin(filepath, 0, false).expect("loading pq_pivots file is a success");
    //     let file_offsets = meta
    //         .buf_usize
    //         .ok_or_else(|| anyhow::anyhow!("unable to get the fileoffsets",))
    //         .expect("should have fileoffsets");

    //     let full_pivots = diskann_load_bin(filepath, file_offsets[0], true)
    //         .expect("unable to load the full pivots");
    //     // let filepath = Path();
    // }
}
