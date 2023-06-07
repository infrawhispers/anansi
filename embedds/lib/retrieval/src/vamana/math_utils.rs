extern crate blas_src;
extern crate cblas;
use anyhow::{bail, Context};

fn compute_vecs_l2sq(data: &[f32], num_points: usize, dim: usize) -> anyhow::Result<Vec<f32>> {
    let mut norms: Vec<f32> = vec![0.0; num_points];
    for i in 0..num_points {
        let norm: f32;
        unsafe {
            norm = cblas::snrm2(dim.try_into()?, &data[i * dim..i * dim + dim], 1);
        }
        norms[i] = norm * norm;
    }
    Ok(norms)
}

fn compute_closest_centers_in_block(
    data: &[f32],
    num_points: usize,
    dim: usize,
    centers: &[f32],
    num_centers: usize,
    docs_l2sq: &[f32],
    centers_l2sq: &[f32],
    center_index: &mut [usize],
    dist_matrix: &mut [f32],
    k: usize,
) -> anyhow::Result<()> {
    if k > num_centers {
        bail!("k: {k} > num_centers: {num_centers}")
    }
    let ones_a: Vec<f32> = vec![1.0; num_centers];
    let ones_b: Vec<f32> = vec![1.0; num_points];
    unsafe {
        cblas::sgemm(
            cblas::Layout::RowMajor,
            cblas::Transpose::None,
            cblas::Transpose::Ordinary,
            num_points.try_into()?,
            num_centers.try_into()?,
            1,
            1.0,
            docs_l2sq,
            1,
            &ones_a,
            1,
            0.0,
            dist_matrix,
            num_centers.try_into()?,
        );
        cblas::sgemm(
            cblas::Layout::RowMajor,
            cblas::Transpose::None,
            cblas::Transpose::Ordinary,
            num_points.try_into()?,
            num_centers.try_into()?,
            1,
            1.0,
            &ones_b,
            1,
            centers_l2sq,
            1,
            1.0,
            dist_matrix,
            num_centers.try_into()?,
        );
        cblas::sgemm(
            cblas::Layout::RowMajor,
            cblas::Transpose::None,
            cblas::Transpose::Ordinary,
            num_points.try_into()?,
            num_centers.try_into()?,
            dim.try_into()?,
            -2.0,
            data,
            dim.try_into()?,
            centers,
            dim.try_into()?,
            1.0,
            dist_matrix,
            num_centers.try_into()?,
        );
    }
    if k == 1 {
        for i in 0..num_points {
            let mut min = f32::MAX;
            for j in 0..num_centers {
                let current = dist_matrix[i * num_centers + j];
                if current < min {
                    center_index[i] = j;
                    min = current;
                }
            }
        }
    } else {
        unimplemented!()
    }
    Ok(())
}
pub fn compute_closest_centers(
    data: &[f32],
    num_points: usize,
    dim: usize,
    pivot_data: &[f32],
    num_centers: usize,
    k: usize,
    closest_center_ivf: &mut [usize],
) -> anyhow::Result<()> {
    let par_block_size: usize = num_points;
    let pts_norms_squared = compute_vecs_l2sq(data, num_points, dim)?;
    let pivs_norm_squared = compute_vecs_l2sq(pivot_data, num_centers, dim)?;
    let mut closest_centers: Vec<usize> = vec![0; par_block_size * k];
    let mut distance_matrix: Vec<f32> = vec![0.0; num_centers * par_block_size];
    let n_blocks = 1;
    for curr_block in 0..n_blocks {
        let dcurr_start = curr_block * par_block_size * dim;
        let num_pts_blk = std::cmp::min(par_block_size, num_points - curr_block * par_block_size);
        let pnorms_start = curr_block * par_block_size;
        compute_closest_centers_in_block(
            &data[dcurr_start..dcurr_start + num_pts_blk],
            num_pts_blk,
            dim,
            pivot_data,
            num_centers,
            &pts_norms_squared[pnorms_start..],
            &pivs_norm_squared,
            &mut closest_centers,
            &mut distance_matrix,
            k,
        )
        .with_context(|| "unable to compute the closest centers in block: {curr_block}")?;
        for j in curr_block * par_block_size
            ..std::cmp::min(num_points, (curr_block + 1) * par_block_size)
        {
            for l in 0..k {
                let this_center_id: usize =
                    closest_centers[(j - curr_block * par_block_size) * k + l];
                closest_center_ivf[j * k + l] = this_center_id;
            }
        }
    }

    Ok(())
}
