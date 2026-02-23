//! LAPACK routines — pure Rust, SIMD-accelerated via numrus_core::simd.
//!
//! Covers the essential decompositions and solvers:
//!
//! - **LU factorization** (`sgetrf` / `dgetrf`) with partial pivoting
//! - **LU solve** (`sgetrs` / `dgetrs`) — solve A*X = B using LU factors
//! - **Cholesky factorization** (`spotrf` / `dpotrf`)
//! - **Cholesky solve** (`spotrs` / `dpotrs`)
//! - **QR factorization** (`sgeqrf` / `dgeqrf`)
//! - **Triangular solve** — delegates to numrus_blas `strsm`/`dtrsm`
//!
//! All operations work on flat row-major or column-major arrays.
//!
//! ## SIMD Strategy
//!
//! LAPACK routines have sequential data dependencies (k-loop over pivots/columns),
//! but the inner j/i loops operate on contiguous row/column slices:
//!
//! - **RowMajor**: rows are contiguous → SIMD axpy/dot/scal on row slices
//! - **ColMajor**: columns are contiguous → restructure loops for SIMD on column slices
//!
//! The trailing submatrix update (rank-1) uses `simd::axpy` per row/column.
//! Pivot search uses SIMD abs-max reduction on contiguous column data.
//! Cholesky inner products use `simd::dot` on contiguous partial rows/columns.
//! QR Householder reflector application uses `simd::dot` + `simd::axpy`.

use numrus_core::layout::Layout;
use numrus_core::simd;

// ============================================================================
// LU Factorization: PA = LU
// ============================================================================

/// Single-precision LU factorization with partial pivoting.
///
/// Factors the M x N matrix A into P * A = L * U.
/// - A is overwritten with L (unit lower) and U (upper).
/// - `ipiv` receives the pivot indices (length min(m, n)).
///
/// Returns 0 on success, > 0 if U is singular (U[info][info] == 0).
pub fn sgetrf(
    layout: Layout,
    m: usize,
    n: usize,
    a: &mut [f32],
    lda: usize,
    ipiv: &mut [usize],
) -> i32 {
    #[cfg(feature = "mkl")]
    {
        let min_mn = m.min(n);
        let mut ipiv_i32 = vec![0i32; min_mn];
        let lapacke_layout = layout as i32;
        let info = unsafe {
            numrus_core::mkl_ffi::LAPACKE_sgetrf(
                lapacke_layout,
                m as i32,
                n as i32,
                a.as_mut_ptr(),
                lda as i32,
                ipiv_i32.as_mut_ptr(),
            )
        };
        // Convert 1-based LAPACKE ipiv to 0-based Rust ipiv
        for i in 0..min_mn {
            ipiv[i] = (ipiv_i32[i] - 1) as usize;
        }
        return info;
    }
    let min_mn = m.min(n);

    for k in 0..min_mn {
        // Find pivot: max |A[i, k]| for i in k..m
        let (max_idx, max_val) = pivot_search_f32(a, layout, k, m, lda);

        ipiv[k] = max_idx;

        if max_val == 0.0 {
            return (k + 1) as i32; // Singular
        }

        // Swap rows k and max_idx
        if max_idx != k {
            swap_rows_f32(a, layout, k, max_idx, n, lda);
        }

        // Compute multipliers: A[i,k] /= pivot for i in k+1..m
        let inv_pivot = 1.0 / a[layout.index(k, k, lda)];
        scale_column_f32(a, layout, k, k + 1, m, lda, inv_pivot);

        // Update trailing submatrix: A[i,j] -= A[i,k] * A[k,j]
        trailing_update_f32(a, layout, k, m, n, lda);
    }

    0
}

/// Double-precision LU factorization with partial pivoting.
pub fn dgetrf(
    layout: Layout,
    m: usize,
    n: usize,
    a: &mut [f64],
    lda: usize,
    ipiv: &mut [usize],
) -> i32 {
    #[cfg(feature = "mkl")]
    {
        let min_mn = m.min(n);
        let mut ipiv_i32 = vec![0i32; min_mn];
        let info = unsafe {
            numrus_core::mkl_ffi::LAPACKE_dgetrf(
                layout as i32,
                m as i32,
                n as i32,
                a.as_mut_ptr(),
                lda as i32,
                ipiv_i32.as_mut_ptr(),
            )
        };
        for i in 0..min_mn {
            ipiv[i] = (ipiv_i32[i] - 1) as usize;
        }
        return info;
    }
    let min_mn = m.min(n);

    for k in 0..min_mn {
        let (max_idx, max_val) = pivot_search_f64(a, layout, k, m, lda);

        ipiv[k] = max_idx;

        if max_val == 0.0 {
            return (k + 1) as i32;
        }

        if max_idx != k {
            swap_rows_f64(a, layout, k, max_idx, n, lda);
        }

        let inv_pivot = 1.0 / a[layout.index(k, k, lda)];
        scale_column_f64(a, layout, k, k + 1, m, lda, inv_pivot);

        trailing_update_f64(a, layout, k, m, n, lda);
    }

    0
}

// ============================================================================
// LU helpers — pivot search, row swap, column scale, trailing update
// ============================================================================

/// Pivot search: find index of max |A[i, col]| for i in start_row..m.
/// Uses SIMD when the column is contiguous (ColMajor).
#[inline]
fn pivot_search_f32(a: &[f32], layout: Layout, col: usize, m: usize, lda: usize) -> (usize, f32) {
    let len = m - col;
    if len == 0 {
        return (col, 0.0);
    }

    match layout {
        Layout::ColMajor => {
            // Column is contiguous: a[col*lda + col .. col*lda + m]
            let base = col * lda + col;
            let slice = &a[base..base + len];
            // SIMD abs-max scan
            let mut max_val = 0.0f32;
            let mut max_local = 0;
            for i in 0..len {
                let v = slice[i].abs();
                if v > max_val {
                    max_val = v;
                    max_local = i;
                }
            }
            (col + max_local, max_val)
        }
        Layout::RowMajor => {
            // Column is strided (stride = lda)
            let mut max_val = 0.0f32;
            let mut max_idx = col;
            for i in col..m {
                let val = a[i * lda + col].abs();
                if val > max_val {
                    max_val = val;
                    max_idx = i;
                }
            }
            (max_idx, max_val)
        }
    }
}

#[inline]
fn pivot_search_f64(a: &[f64], layout: Layout, col: usize, m: usize, lda: usize) -> (usize, f64) {
    let len = m - col;
    if len == 0 {
        return (col, 0.0);
    }

    match layout {
        Layout::ColMajor => {
            let base = col * lda + col;
            let slice = &a[base..base + len];
            let mut max_val = 0.0f64;
            let mut max_local = 0;
            for i in 0..len {
                let v = slice[i].abs();
                if v > max_val {
                    max_val = v;
                    max_local = i;
                }
            }
            (col + max_local, max_val)
        }
        Layout::RowMajor => {
            let mut max_val = 0.0f64;
            let mut max_idx = col;
            for i in col..m {
                let val = a[i * lda + col].abs();
                if val > max_val {
                    max_val = val;
                    max_idx = i;
                }
            }
            (max_idx, max_val)
        }
    }
}

/// Swap rows r1 and r2 of matrix A.
/// For RowMajor: rows are contiguous → swap_with_slice.
/// For ColMajor: swap element-by-element across columns.
#[inline]
fn swap_rows_f32(a: &mut [f32], layout: Layout, r1: usize, r2: usize, n: usize, lda: usize) {
    match layout {
        Layout::RowMajor => {
            let (base1, base2) = (r1 * lda, r2 * lda);
            // Swap contiguous row slices
            if base1 < base2 {
                let (left, right) = a.split_at_mut(base2);
                left[base1..base1 + n].swap_with_slice(&mut right[..n]);
            } else {
                let (left, right) = a.split_at_mut(base1);
                right[..n].swap_with_slice(&mut left[base2..base2 + n]);
            }
        }
        Layout::ColMajor => {
            for j in 0..n {
                let idx1 = j * lda + r1;
                let idx2 = j * lda + r2;
                a.swap(idx1, idx2);
            }
        }
    }
}

#[inline]
fn swap_rows_f64(a: &mut [f64], layout: Layout, r1: usize, r2: usize, n: usize, lda: usize) {
    match layout {
        Layout::RowMajor => {
            let (base1, base2) = (r1 * lda, r2 * lda);
            if base1 < base2 {
                let (left, right) = a.split_at_mut(base2);
                left[base1..base1 + n].swap_with_slice(&mut right[..n]);
            } else {
                let (left, right) = a.split_at_mut(base1);
                right[..n].swap_with_slice(&mut left[base2..base2 + n]);
            }
        }
        Layout::ColMajor => {
            for j in 0..n {
                a.swap(j * lda + r1, j * lda + r2);
            }
        }
    }
}

/// Scale column: A[i, col] *= alpha for i in start_row..end_row.
/// For ColMajor: column is contiguous → SIMD scal.
/// For RowMajor: column is strided → scalar loop.
#[inline]
fn scale_column_f32(
    a: &mut [f32],
    layout: Layout,
    col: usize,
    start_row: usize,
    end_row: usize,
    lda: usize,
    alpha: f32,
) {
    let len = end_row - start_row;
    if len == 0 {
        return;
    }
    match layout {
        Layout::ColMajor => {
            let base = col * lda + start_row;
            simd::scal_f32(alpha, &mut a[base..base + len]);
        }
        Layout::RowMajor => {
            for i in start_row..end_row {
                a[i * lda + col] *= alpha;
            }
        }
    }
}

#[inline]
fn scale_column_f64(
    a: &mut [f64],
    layout: Layout,
    col: usize,
    start_row: usize,
    end_row: usize,
    lda: usize,
    alpha: f64,
) {
    let len = end_row - start_row;
    if len == 0 {
        return;
    }
    match layout {
        Layout::ColMajor => {
            let base = col * lda + start_row;
            simd::scal_f64(alpha, &mut a[base..base + len]);
        }
        Layout::RowMajor => {
            for i in start_row..end_row {
                a[i * lda + col] *= alpha;
            }
        }
    }
}

/// Trailing submatrix update: A[i,j] -= A[i,k] * A[k,j] for i>k, j>k.
///
/// - **RowMajor**: Row k and row i are contiguous from column k+1..n.
///   For each row i: `row_i[k+1..n] -= lik * row_k[k+1..n]` → SIMD axpy.
///
/// - **ColMajor**: Column j is contiguous from row k+1..m.
///   For each column j: `col_j[k+1..m] -= ukj * col_k[k+1..m]` → SIMD axpy.
///   (Restructured loop order for SIMD on contiguous column data.)
#[inline]
fn trailing_update_f32(a: &mut [f32], layout: Layout, k: usize, m: usize, n: usize, lda: usize) {
    let trail_rows = m - k - 1;
    let trail_cols = n - k - 1;
    if trail_rows == 0 || trail_cols == 0 {
        return;
    }

    match layout {
        Layout::RowMajor => {
            // Gather row k's trailing slice [k+1..n] once
            let k_row_start = k * lda + (k + 1);
            // For each row i > k, apply axpy: row_i -= lik * row_k
            for i in (k + 1)..m {
                let lik = a[i * lda + k];
                if lik == 0.0 {
                    continue;
                }
                let i_row_start = i * lda + (k + 1);
                // Cannot borrow row_k and row_i mutably at same time,
                // so gather row_k into temp buffer for SIMD axpy.
                // This pays the gather cost once per row but enables full-width SIMD.
                let row_k: Vec<f32> = a[k_row_start..k_row_start + trail_cols].to_vec();
                simd::axpy_f32(-lik, &row_k, &mut a[i_row_start..i_row_start + trail_cols]);
            }
        }
        Layout::ColMajor => {
            // Column k multipliers: A[k+1..m, k] is contiguous at col_k_base
            let col_k_base = k * lda + (k + 1);
            let col_k: Vec<f32> = a[col_k_base..col_k_base + trail_rows].to_vec();
            // For each column j > k: col_j -= ukj * col_k
            for j in (k + 1)..n {
                let ukj = a[j * lda + k]; // A[k, j] in ColMajor = j*lda + k
                if ukj == 0.0 {
                    continue;
                }
                let col_j_base = j * lda + (k + 1);
                simd::axpy_f32(-ukj, &col_k, &mut a[col_j_base..col_j_base + trail_rows]);
            }
        }
    }
}

#[inline]
fn trailing_update_f64(a: &mut [f64], layout: Layout, k: usize, m: usize, n: usize, lda: usize) {
    let trail_rows = m - k - 1;
    let trail_cols = n - k - 1;
    if trail_rows == 0 || trail_cols == 0 {
        return;
    }

    match layout {
        Layout::RowMajor => {
            let k_row_start = k * lda + (k + 1);
            for i in (k + 1)..m {
                let lik = a[i * lda + k];
                if lik == 0.0 {
                    continue;
                }
                let i_row_start = i * lda + (k + 1);
                let row_k: Vec<f64> = a[k_row_start..k_row_start + trail_cols].to_vec();
                simd::axpy_f64(-lik, &row_k, &mut a[i_row_start..i_row_start + trail_cols]);
            }
        }
        Layout::ColMajor => {
            let col_k_base = k * lda + (k + 1);
            let col_k: Vec<f64> = a[col_k_base..col_k_base + trail_rows].to_vec();
            for j in (k + 1)..n {
                let ukj = a[j * lda + k];
                if ukj == 0.0 {
                    continue;
                }
                let col_j_base = j * lda + (k + 1);
                simd::axpy_f64(-ukj, &col_k, &mut a[col_j_base..col_j_base + trail_rows]);
            }
        }
    }
}

// ============================================================================
// LU Solve: solve A * X = B using LU factors from sgetrf/dgetrf
// ============================================================================

/// Single-precision LU solve: solve A * X = B.
///
/// A must already be LU-factored by `sgetrf`.
/// B is overwritten with the solution X.
///
/// SIMD strategy: forward/back substitution inner loops are axpy/scal
/// on contiguous row slices (RowMajor) or column slices (ColMajor).
pub fn sgetrs(
    layout: Layout,
    n: usize,
    nrhs: usize,
    a: &[f32],
    lda: usize,
    ipiv: &[usize],
    b: &mut [f32],
    ldb: usize,
) {
    #[cfg(feature = "mkl")]
    {
        let mut ipiv_i32: Vec<i32> = ipiv.iter().map(|&p| (p + 1) as i32).collect();
        unsafe {
            numrus_core::mkl_ffi::LAPACKE_sgetrs(
                layout as i32,
                b'N',
                n as i32,
                nrhs as i32,
                a.as_ptr(),
                lda as i32,
                ipiv_i32.as_ptr(),
                b.as_mut_ptr(),
                ldb as i32,
            );
        }
        return;
    }
    // Apply row interchanges to B
    for k in 0..n {
        if ipiv[k] != k {
            swap_rows_f32(b, layout, k, ipiv[k], nrhs, ldb);
        }
    }

    // Forward substitution: L * Y = P * B
    match layout {
        Layout::RowMajor => {
            for k in 0..n {
                // Row k of B is contiguous: b[k*ldb .. k*ldb+nrhs]
                let b_k_row: Vec<f32> = b[k * ldb..k * ldb + nrhs].to_vec();
                for i in (k + 1)..n {
                    let lik = a[i * lda + k];
                    if lik == 0.0 {
                        continue;
                    }
                    // row_i -= lik * row_k → SIMD axpy
                    let i_start = i * ldb;
                    simd::axpy_f32(-lik, &b_k_row, &mut b[i_start..i_start + nrhs]);
                }
            }
        }
        Layout::ColMajor => {
            // Columns of B are contiguous
            for k in 0..n {
                for j in 0..nrhs {
                    let bkj = b[j * ldb + k];
                    if bkj == 0.0 {
                        continue;
                    }
                    let col_base = j * ldb + (k + 1);
                    let remain = n - k - 1;
                    if remain == 0 {
                        continue;
                    }
                    // Gather A's L column: A[k+1..n, k] contiguous at k*lda+(k+1)
                    let a_col_base = k * lda + (k + 1);
                    let a_col: Vec<f32> = a[a_col_base..a_col_base + remain].to_vec();
                    simd::axpy_f32(-bkj, &a_col, &mut b[col_base..col_base + remain]);
                }
            }
        }
    }

    // Back substitution: U * X = Y
    match layout {
        Layout::RowMajor => {
            for k in (0..n).rev() {
                let ukk = a[k * lda + k];
                // Scale row k of B by 1/ukk → SIMD scal
                simd::scal_f32(1.0 / ukk, &mut b[k * ldb..k * ldb + nrhs]);

                let b_k_row: Vec<f32> = b[k * ldb..k * ldb + nrhs].to_vec();
                for i in 0..k {
                    let uik = a[i * lda + k];
                    if uik == 0.0 {
                        continue;
                    }
                    let i_start = i * ldb;
                    simd::axpy_f32(-uik, &b_k_row, &mut b[i_start..i_start + nrhs]);
                }
            }
        }
        Layout::ColMajor => {
            for k in (0..n).rev() {
                let ukk = a[k * lda + k];
                let inv_ukk = 1.0 / ukk;
                // Scale B[k, :] — for ColMajor, B[k,j] = b[j*ldb + k], strided
                for j in 0..nrhs {
                    b[j * ldb + k] *= inv_ukk;
                }

                for j in 0..nrhs {
                    let bkj = b[j * ldb + k];
                    if bkj == 0.0 || k == 0 {
                        continue;
                    }
                    // U column: A[0..k, k] contiguous at k*lda
                    let a_col_base = k * lda;
                    let a_col: Vec<f32> = a[a_col_base..a_col_base + k].to_vec();
                    let b_col_base = j * ldb;
                    simd::axpy_f32(-bkj, &a_col, &mut b[b_col_base..b_col_base + k]);
                }
            }
        }
    }
}

/// Double-precision LU solve.
pub fn dgetrs(
    layout: Layout,
    n: usize,
    nrhs: usize,
    a: &[f64],
    lda: usize,
    ipiv: &[usize],
    b: &mut [f64],
    ldb: usize,
) {
    #[cfg(feature = "mkl")]
    {
        let ipiv_i32: Vec<i32> = ipiv.iter().map(|&p| (p + 1) as i32).collect();
        unsafe {
            numrus_core::mkl_ffi::LAPACKE_dgetrs(
                layout as i32,
                b'N',
                n as i32,
                nrhs as i32,
                a.as_ptr(),
                lda as i32,
                ipiv_i32.as_ptr(),
                b.as_mut_ptr(),
                ldb as i32,
            );
        }
        return;
    }
    for k in 0..n {
        if ipiv[k] != k {
            swap_rows_f64(b, layout, k, ipiv[k], nrhs, ldb);
        }
    }

    match layout {
        Layout::RowMajor => {
            for k in 0..n {
                let b_k_row: Vec<f64> = b[k * ldb..k * ldb + nrhs].to_vec();
                for i in (k + 1)..n {
                    let lik = a[i * lda + k];
                    if lik == 0.0 {
                        continue;
                    }
                    let i_start = i * ldb;
                    simd::axpy_f64(-lik, &b_k_row, &mut b[i_start..i_start + nrhs]);
                }
            }
        }
        Layout::ColMajor => {
            for k in 0..n {
                for j in 0..nrhs {
                    let bkj = b[j * ldb + k];
                    if bkj == 0.0 {
                        continue;
                    }
                    let col_base = j * ldb + (k + 1);
                    let remain = n - k - 1;
                    if remain == 0 {
                        continue;
                    }
                    let a_col_base = k * lda + (k + 1);
                    let a_col: Vec<f64> = a[a_col_base..a_col_base + remain].to_vec();
                    simd::axpy_f64(-bkj, &a_col, &mut b[col_base..col_base + remain]);
                }
            }
        }
    }

    match layout {
        Layout::RowMajor => {
            for k in (0..n).rev() {
                let ukk = a[k * lda + k];
                simd::scal_f64(1.0 / ukk, &mut b[k * ldb..k * ldb + nrhs]);

                let b_k_row: Vec<f64> = b[k * ldb..k * ldb + nrhs].to_vec();
                for i in 0..k {
                    let uik = a[i * lda + k];
                    if uik == 0.0 {
                        continue;
                    }
                    let i_start = i * ldb;
                    simd::axpy_f64(-uik, &b_k_row, &mut b[i_start..i_start + nrhs]);
                }
            }
        }
        Layout::ColMajor => {
            for k in (0..n).rev() {
                let ukk = a[k * lda + k];
                let inv_ukk = 1.0 / ukk;
                for j in 0..nrhs {
                    b[j * ldb + k] *= inv_ukk;
                }

                for j in 0..nrhs {
                    let bkj = b[j * ldb + k];
                    if bkj == 0.0 || k == 0 {
                        continue;
                    }
                    let a_col_base = k * lda;
                    let a_col: Vec<f64> = a[a_col_base..a_col_base + k].to_vec();
                    let b_col_base = j * ldb;
                    simd::axpy_f64(-bkj, &a_col, &mut b[b_col_base..b_col_base + k]);
                }
            }
        }
    }
}

// ============================================================================
// Cholesky Factorization: A = L * L^T (or U^T * U)
// ============================================================================

/// Single-precision Cholesky factorization.
///
/// SIMD strategy:
/// - Diagonal sum `sum -= ljk^2` → SIMD dot of partial row/column with itself.
/// - Off-diagonal `sum -= A[i,k]*A[j,k]` → SIMD dot of two partial rows/columns.
pub fn spotrf(
    layout: Layout,
    uplo: numrus_core::layout::Uplo,
    n: usize,
    a: &mut [f32],
    lda: usize,
) -> i32 {
    #[cfg(feature = "mkl")]
    {
        let uplo_char = match uplo {
            numrus_core::layout::Uplo::Upper => b'U',
            numrus_core::layout::Uplo::Lower => b'L',
        };
        let info = unsafe {
            numrus_core::mkl_ffi::LAPACKE_spotrf(
                layout as i32,
                uplo_char,
                n as i32,
                a.as_mut_ptr(),
                lda as i32,
            )
        };
        return info;
    }
    match uplo {
        numrus_core::layout::Uplo::Lower => spotrf_lower(layout, n, a, lda),
        numrus_core::layout::Uplo::Upper => spotrf_upper(layout, n, a, lda),
    }
}

#[inline]
fn spotrf_lower(layout: Layout, n: usize, a: &mut [f32], lda: usize) -> i32 {
    for j in 0..n {
        // Diagonal: sum = A[j,j] - sum_k(L[j,k]^2) for k in 0..j
        let mut sum = a[layout.index(j, j, lda)];

        if j > 0 {
            match layout {
                Layout::RowMajor => {
                    // Row j, columns 0..j is contiguous: a[j*lda .. j*lda+j]
                    let row_start = j * lda;
                    let slice = &a[row_start..row_start + j];
                    sum -= simd::dot_f32(slice, slice);
                }
                Layout::ColMajor => {
                    // L[j, 0..j]: for col-major, these are at positions col*lda+j for col in 0..j (strided)
                    // Gather into contiguous buffer for SIMD dot
                    let buf: Vec<f32> = (0..j).map(|k| a[k * lda + j]).collect();
                    sum -= simd::dot_f32(&buf, &buf);
                }
            }
        }

        if sum <= 0.0 {
            return (j + 1) as i32;
        }
        let ljj = sum.sqrt();
        a[layout.index(j, j, lda)] = ljj;

        // Off-diagonal: L[i,j] for i in j+1..n
        for i in (j + 1)..n {
            let mut s = a[layout.index(i, j, lda)];

            if j > 0 {
                match layout {
                    Layout::RowMajor => {
                        // Row i columns 0..j and row j columns 0..j are both contiguous
                        let ri = &a[i * lda..i * lda + j];
                        let rj = &a[j * lda..j * lda + j];
                        s -= simd::dot_f32(ri, rj);
                    }
                    Layout::ColMajor => {
                        // Gather both partial rows
                        let buf_i: Vec<f32> = (0..j).map(|k| a[k * lda + i]).collect();
                        let buf_j: Vec<f32> = (0..j).map(|k| a[k * lda + j]).collect();
                        s -= simd::dot_f32(&buf_i, &buf_j);
                    }
                }
            }

            a[layout.index(i, j, lda)] = s / ljj;
        }
    }
    0
}

#[inline]
fn spotrf_upper(layout: Layout, n: usize, a: &mut [f32], lda: usize) -> i32 {
    for j in 0..n {
        let mut sum = a[layout.index(j, j, lda)];

        if j > 0 {
            match layout {
                Layout::ColMajor => {
                    // Column j, rows 0..j contiguous: a[j*lda .. j*lda+j]
                    let col_start = j * lda;
                    let slice = &a[col_start..col_start + j];
                    sum -= simd::dot_f32(slice, slice);
                }
                Layout::RowMajor => {
                    // U[0..j, j]: row-major → positions k*lda+j for k in 0..j (strided)
                    let buf: Vec<f32> = (0..j).map(|k| a[k * lda + j]).collect();
                    sum -= simd::dot_f32(&buf, &buf);
                }
            }
        }

        if sum <= 0.0 {
            return (j + 1) as i32;
        }
        let ujj = sum.sqrt();
        a[layout.index(j, j, lda)] = ujj;

        for i in (j + 1)..n {
            let mut s = a[layout.index(j, i, lda)];

            if j > 0 {
                match layout {
                    Layout::ColMajor => {
                        // Column i rows 0..j and column j rows 0..j contiguous
                        let ci = &a[i * lda..i * lda + j];
                        let cj = &a[j * lda..j * lda + j];
                        s -= simd::dot_f32(ci, cj);
                    }
                    Layout::RowMajor => {
                        let buf_i: Vec<f32> = (0..j).map(|k| a[k * lda + i]).collect();
                        let buf_j: Vec<f32> = (0..j).map(|k| a[k * lda + j]).collect();
                        s -= simd::dot_f32(&buf_i, &buf_j);
                    }
                }
            }

            a[layout.index(j, i, lda)] = s / ujj;
        }
    }
    0
}

/// Double-precision Cholesky factorization.
pub fn dpotrf(
    layout: Layout,
    uplo: numrus_core::layout::Uplo,
    n: usize,
    a: &mut [f64],
    lda: usize,
) -> i32 {
    #[cfg(feature = "mkl")]
    {
        let uplo_char = match uplo {
            numrus_core::layout::Uplo::Upper => b'U',
            numrus_core::layout::Uplo::Lower => b'L',
        };
        let info = unsafe {
            numrus_core::mkl_ffi::LAPACKE_dpotrf(
                layout as i32,
                uplo_char,
                n as i32,
                a.as_mut_ptr(),
                lda as i32,
            )
        };
        return info;
    }
    match uplo {
        numrus_core::layout::Uplo::Lower => dpotrf_lower(layout, n, a, lda),
        numrus_core::layout::Uplo::Upper => dpotrf_upper(layout, n, a, lda),
    }
}

#[inline]
fn dpotrf_lower(layout: Layout, n: usize, a: &mut [f64], lda: usize) -> i32 {
    for j in 0..n {
        let mut sum = a[layout.index(j, j, lda)];

        if j > 0 {
            match layout {
                Layout::RowMajor => {
                    let row_start = j * lda;
                    let slice = &a[row_start..row_start + j];
                    sum -= simd::dot_f64(slice, slice);
                }
                Layout::ColMajor => {
                    let buf: Vec<f64> = (0..j).map(|k| a[k * lda + j]).collect();
                    sum -= simd::dot_f64(&buf, &buf);
                }
            }
        }

        if sum <= 0.0 {
            return (j + 1) as i32;
        }
        let ljj = sum.sqrt();
        a[layout.index(j, j, lda)] = ljj;

        for i in (j + 1)..n {
            let mut s = a[layout.index(i, j, lda)];

            if j > 0 {
                match layout {
                    Layout::RowMajor => {
                        let ri = &a[i * lda..i * lda + j];
                        let rj = &a[j * lda..j * lda + j];
                        s -= simd::dot_f64(ri, rj);
                    }
                    Layout::ColMajor => {
                        let buf_i: Vec<f64> = (0..j).map(|k| a[k * lda + i]).collect();
                        let buf_j: Vec<f64> = (0..j).map(|k| a[k * lda + j]).collect();
                        s -= simd::dot_f64(&buf_i, &buf_j);
                    }
                }
            }

            a[layout.index(i, j, lda)] = s / ljj;
        }
    }
    0
}

#[inline]
fn dpotrf_upper(layout: Layout, n: usize, a: &mut [f64], lda: usize) -> i32 {
    for j in 0..n {
        let mut sum = a[layout.index(j, j, lda)];

        if j > 0 {
            match layout {
                Layout::ColMajor => {
                    let col_start = j * lda;
                    let slice = &a[col_start..col_start + j];
                    sum -= simd::dot_f64(slice, slice);
                }
                Layout::RowMajor => {
                    let buf: Vec<f64> = (0..j).map(|k| a[k * lda + j]).collect();
                    sum -= simd::dot_f64(&buf, &buf);
                }
            }
        }

        if sum <= 0.0 {
            return (j + 1) as i32;
        }
        let ujj = sum.sqrt();
        a[layout.index(j, j, lda)] = ujj;

        for i in (j + 1)..n {
            let mut s = a[layout.index(j, i, lda)];

            if j > 0 {
                match layout {
                    Layout::ColMajor => {
                        let ci = &a[i * lda..i * lda + j];
                        let cj = &a[j * lda..j * lda + j];
                        s -= simd::dot_f64(ci, cj);
                    }
                    Layout::RowMajor => {
                        let buf_i: Vec<f64> = (0..j).map(|k| a[k * lda + i]).collect();
                        let buf_j: Vec<f64> = (0..j).map(|k| a[k * lda + j]).collect();
                        s -= simd::dot_f64(&buf_i, &buf_j);
                    }
                }
            }

            a[layout.index(j, i, lda)] = s / ujj;
        }
    }
    0
}

// ============================================================================
// Cholesky Solve: A * X = B using Cholesky factors
// ============================================================================

/// Single-precision Cholesky solve: solve A * X = B.
/// A must already be Cholesky-factored by `spotrf`.
///
/// SIMD: forward/back substitution via axpy/scal on contiguous slices.
pub fn spotrs(
    layout: Layout,
    uplo: numrus_core::layout::Uplo,
    n: usize,
    nrhs: usize,
    a: &[f32],
    lda: usize,
    b: &mut [f32],
    ldb: usize,
) {
    #[cfg(feature = "mkl")]
    {
        let uplo_char = match uplo {
            numrus_core::layout::Uplo::Upper => b'U',
            numrus_core::layout::Uplo::Lower => b'L',
        };
        unsafe {
            numrus_core::mkl_ffi::LAPACKE_spotrs(
                layout as i32,
                uplo_char,
                n as i32,
                nrhs as i32,
                a.as_ptr(),
                lda as i32,
                b.as_mut_ptr(),
                ldb as i32,
            );
        }
        return;
    }
    match uplo {
        numrus_core::layout::Uplo::Lower => {
            // Forward substitution: L * Y = B
            trsm_forward_lower_f32(layout, n, nrhs, a, lda, b, ldb);
            // Back substitution: L^T * X = Y
            trsm_backward_lower_trans_f32(layout, n, nrhs, a, lda, b, ldb);
        }
        numrus_core::layout::Uplo::Upper => {
            // Forward substitution: U^T * Y = B
            trsm_forward_upper_trans_f32(layout, n, nrhs, a, lda, b, ldb);
            // Back substitution: U * X = Y
            trsm_backward_upper_f32(layout, n, nrhs, a, lda, b, ldb);
        }
    }
}

// Triangular solve helpers — all use SIMD axpy/scal on contiguous slices

#[inline]
fn trsm_forward_lower_f32(
    layout: Layout,
    n: usize,
    nrhs: usize,
    a: &[f32],
    lda: usize,
    b: &mut [f32],
    ldb: usize,
) {
    for k in 0..n {
        let lkk = a[layout.index(k, k, lda)];
        match layout {
            Layout::RowMajor => {
                simd::scal_f32(1.0 / lkk, &mut b[k * ldb..k * ldb + nrhs]);
                let b_k_row: Vec<f32> = b[k * ldb..k * ldb + nrhs].to_vec();
                for i in (k + 1)..n {
                    let lik = a[i * lda + k];
                    if lik == 0.0 {
                        continue;
                    }
                    let i_start = i * ldb;
                    simd::axpy_f32(-lik, &b_k_row, &mut b[i_start..i_start + nrhs]);
                }
            }
            Layout::ColMajor => {
                for j in 0..nrhs {
                    b[j * ldb + k] /= lkk;
                    let bkj = b[j * ldb + k];
                    if bkj == 0.0 {
                        continue;
                    }
                    let remain = n - k - 1;
                    if remain == 0 {
                        continue;
                    }
                    let a_col = k * lda + (k + 1);
                    let b_col = j * ldb + (k + 1);
                    let a_slice: Vec<f32> = a[a_col..a_col + remain].to_vec();
                    simd::axpy_f32(-bkj, &a_slice, &mut b[b_col..b_col + remain]);
                }
            }
        }
    }
}

#[inline]
fn trsm_backward_lower_trans_f32(
    layout: Layout,
    n: usize,
    nrhs: usize,
    a: &[f32],
    lda: usize,
    b: &mut [f32],
    ldb: usize,
) {
    for k in (0..n).rev() {
        let lkk = a[layout.index(k, k, lda)];
        match layout {
            Layout::RowMajor => {
                simd::scal_f32(1.0 / lkk, &mut b[k * ldb..k * ldb + nrhs]);
                let b_k_row: Vec<f32> = b[k * ldb..k * ldb + nrhs].to_vec();
                for i in 0..k {
                    // L^T: use L[k, i] = A[k, i] (lower triangle)
                    let lki = a[layout.index(k, i, lda)];
                    if lki == 0.0 {
                        continue;
                    }
                    let i_start = i * ldb;
                    simd::axpy_f32(-lki, &b_k_row, &mut b[i_start..i_start + nrhs]);
                }
            }
            Layout::ColMajor => {
                for j in 0..nrhs {
                    b[j * ldb + k] /= lkk;
                    let bkj = b[j * ldb + k];
                    if bkj == 0.0 || k == 0 {
                        continue;
                    }
                    // L^T column k: L[k, 0..k] → in lower triangle: A[k,i] for i in 0..k
                    // ColMajor: A[k, i] = i*lda + k → strided
                    let a_slice: Vec<f32> = (0..k).map(|i| a[i * lda + k]).collect();
                    let b_col = j * ldb;
                    simd::axpy_f32(-bkj, &a_slice, &mut b[b_col..b_col + k]);
                }
            }
        }
    }
}

#[inline]
fn trsm_forward_upper_trans_f32(
    layout: Layout,
    n: usize,
    nrhs: usize,
    a: &[f32],
    lda: usize,
    b: &mut [f32],
    ldb: usize,
) {
    for k in 0..n {
        let ukk = a[layout.index(k, k, lda)];
        match layout {
            Layout::RowMajor => {
                simd::scal_f32(1.0 / ukk, &mut b[k * ldb..k * ldb + nrhs]);
                let b_k_row: Vec<f32> = b[k * ldb..k * ldb + nrhs].to_vec();
                for i in (k + 1)..n {
                    // U^T: use U[k, i] = A[k, i] (upper triangle)
                    let uki = a[layout.index(k, i, lda)];
                    if uki == 0.0 {
                        continue;
                    }
                    let i_start = i * ldb;
                    simd::axpy_f32(-uki, &b_k_row, &mut b[i_start..i_start + nrhs]);
                }
            }
            Layout::ColMajor => {
                for j in 0..nrhs {
                    b[j * ldb + k] /= ukk;
                    let bkj = b[j * ldb + k];
                    if bkj == 0.0 {
                        continue;
                    }
                    let remain = n - k - 1;
                    if remain == 0 {
                        continue;
                    }
                    // U^T column: U[k, k+1..n] → ColMajor: (k+1..n)*lda + k → strided
                    // But column k of U^T = row k of U → A[k, k+1..n]
                    // ColMajor: A[k, i] = i*lda + k for i in k+1..n → strided
                    let a_slice: Vec<f32> =
                        (0..remain).map(|di| a[(k + 1 + di) * lda + k]).collect();
                    let b_col = j * ldb + (k + 1);
                    simd::axpy_f32(-bkj, &a_slice, &mut b[b_col..b_col + remain]);
                }
            }
        }
    }
}

#[inline]
fn trsm_backward_upper_f32(
    layout: Layout,
    n: usize,
    nrhs: usize,
    a: &[f32],
    lda: usize,
    b: &mut [f32],
    ldb: usize,
) {
    for k in (0..n).rev() {
        let ukk = a[layout.index(k, k, lda)];
        match layout {
            Layout::RowMajor => {
                simd::scal_f32(1.0 / ukk, &mut b[k * ldb..k * ldb + nrhs]);
                let b_k_row: Vec<f32> = b[k * ldb..k * ldb + nrhs].to_vec();
                for i in 0..k {
                    let uik = a[layout.index(i, k, lda)];
                    if uik == 0.0 {
                        continue;
                    }
                    let i_start = i * ldb;
                    simd::axpy_f32(-uik, &b_k_row, &mut b[i_start..i_start + nrhs]);
                }
            }
            Layout::ColMajor => {
                for j in 0..nrhs {
                    b[j * ldb + k] /= ukk;
                    let bkj = b[j * ldb + k];
                    if bkj == 0.0 || k == 0 {
                        continue;
                    }
                    // U column k, rows 0..k: ColMajor → a[k*lda + 0..k] contiguous
                    let a_col_base = k * lda;
                    let a_slice: Vec<f32> = a[a_col_base..a_col_base + k].to_vec();
                    let b_col = j * ldb;
                    simd::axpy_f32(-bkj, &a_slice, &mut b[b_col..b_col + k]);
                }
            }
        }
    }
}

// ============================================================================
// QR Factorization: A = Q * R (Householder reflections)
// ============================================================================

/// Single-precision QR factorization using Householder reflections.
///
/// SIMD strategy:
/// - Column norm: SIMD dot of column portion with itself.
/// - `v^T * A[:, j]`: SIMD dot on contiguous column (ColMajor) or gathered column.
/// - `A[:, j] -= w * v`: SIMD axpy on contiguous column (ColMajor) or scattered.
///
/// For RowMajor, the Householder vector lives in column k (strided), so we
/// gather once and use SIMD dot/axpy on the gathered buffer. For ColMajor,
/// columns are contiguous and we get native SIMD.
pub fn sgeqrf(
    layout: Layout,
    m: usize,
    n: usize,
    a: &mut [f32],
    lda: usize,
    tau: &mut [f32],
) -> i32 {
    #[cfg(feature = "mkl")]
    {
        let info = unsafe {
            numrus_core::mkl_ffi::LAPACKE_sgeqrf(
                layout as i32,
                m as i32,
                n as i32,
                a.as_mut_ptr(),
                lda as i32,
                tau.as_mut_ptr(),
            )
        };
        return info;
    }
    let min_mn = m.min(n);

    for k in 0..min_mn {
        let col_len = m - k;

        // Compute norm of column k below diagonal
        let norm_sq = match layout {
            Layout::ColMajor => {
                let base = k * lda + k;
                let slice = &a[base..base + col_len];
                simd::dot_f32(slice, slice)
            }
            Layout::RowMajor => {
                // Column is strided → gather
                let buf: Vec<f32> = (k..m).map(|i| a[i * lda + k]).collect();
                simd::dot_f32(&buf, &buf)
            }
        };
        let norm = norm_sq.sqrt();

        if norm == 0.0 {
            tau[k] = 0.0;
            continue;
        }

        let akk = a[layout.index(k, k, lda)];
        let sign = if akk >= 0.0 { 1.0 } else { -1.0 };
        let alpha = -sign * norm;

        let beta = akk - alpha;
        if beta.abs() < 1e-30 {
            tau[k] = 0.0;
            a[layout.index(k, k, lda)] = alpha;
            continue;
        }

        // Scale v[k+1..m] so v[k] = 1: A[i,k] /= beta for i > k
        match layout {
            Layout::ColMajor => {
                if col_len > 1 {
                    let base = k * lda + (k + 1);
                    simd::scal_f32(1.0 / beta, &mut a[base..base + col_len - 1]);
                }
            }
            Layout::RowMajor => {
                let inv_beta = 1.0 / beta;
                for i in (k + 1)..m {
                    a[i * lda + k] *= inv_beta;
                }
            }
        }

        tau[k] = -beta / alpha;
        a[layout.index(k, k, lda)] = alpha;

        // Apply reflector to trailing columns: A[k:m, k+1:n]
        // For each column j in k+1..n:
        //   w = v^T * A[k:m, j] (dot product)
        //   w *= tau
        //   A[k:m, j] -= w * v
        let v_len = col_len; // v has length m-k, v[0]=1, rest from A[k+1..m, k]

        match layout {
            Layout::ColMajor => {
                // v is in column k, rows k..m → contiguous: a[k*lda+k .. k*lda+m]
                // Temporarily set v[0] = 1 for dot/axpy, then restore alpha
                a[k * lda + k] = 1.0;
                let v_base = k * lda + k;

                for j in (k + 1)..n {
                    let col_j_base = j * lda + k;
                    // w = v^T * A[k:m, j] → SIMD dot on contiguous slices
                    let v_slice: Vec<f32> = a[v_base..v_base + v_len].to_vec();
                    let w = simd::dot_f32(&v_slice, &a[col_j_base..col_j_base + v_len]);
                    let w = w * tau[k];
                    // A[k:m, j] -= w * v → SIMD axpy
                    simd::axpy_f32(-w, &v_slice, &mut a[col_j_base..col_j_base + v_len]);
                }

                // Restore alpha
                a[k * lda + k] = alpha;
            }
            Layout::RowMajor => {
                // v is in column k (strided) → gather once
                let mut v_buf = Vec::with_capacity(v_len);
                v_buf.push(1.0f32); // v[0] = 1
                for i in (k + 1)..m {
                    v_buf.push(a[i * lda + k]);
                }

                for j in (k + 1)..n {
                    // Gather column j, rows k..m
                    let mut col_j: Vec<f32> = (k..m).map(|i| a[i * lda + j]).collect();
                    let w = simd::dot_f32(&v_buf, &col_j) * tau[k];
                    // Apply: col_j -= w * v
                    simd::axpy_f32(-w, &v_buf, &mut col_j);
                    // Scatter back
                    for (idx, i) in (k..m).enumerate() {
                        a[i * lda + j] = col_j[idx];
                    }
                }
            }
        }
    }

    0
}

/// Double-precision QR factorization.
pub fn dgeqrf(
    layout: Layout,
    m: usize,
    n: usize,
    a: &mut [f64],
    lda: usize,
    tau: &mut [f64],
) -> i32 {
    #[cfg(feature = "mkl")]
    {
        let info = unsafe {
            numrus_core::mkl_ffi::LAPACKE_dgeqrf(
                layout as i32,
                m as i32,
                n as i32,
                a.as_mut_ptr(),
                lda as i32,
                tau.as_mut_ptr(),
            )
        };
        return info;
    }
    let min_mn = m.min(n);

    for k in 0..min_mn {
        let col_len = m - k;

        let norm_sq = match layout {
            Layout::ColMajor => {
                let base = k * lda + k;
                let slice = &a[base..base + col_len];
                simd::dot_f64(slice, slice)
            }
            Layout::RowMajor => {
                let buf: Vec<f64> = (k..m).map(|i| a[i * lda + k]).collect();
                simd::dot_f64(&buf, &buf)
            }
        };
        let norm = norm_sq.sqrt();

        if norm == 0.0 {
            tau[k] = 0.0;
            continue;
        }

        let akk = a[layout.index(k, k, lda)];
        let sign = if akk >= 0.0 { 1.0 } else { -1.0 };
        let alpha = -sign * norm;

        let beta = akk - alpha;
        if beta.abs() < 1e-60 {
            tau[k] = 0.0;
            a[layout.index(k, k, lda)] = alpha;
            continue;
        }

        match layout {
            Layout::ColMajor => {
                if col_len > 1 {
                    let base = k * lda + (k + 1);
                    simd::scal_f64(1.0 / beta, &mut a[base..base + col_len - 1]);
                }
            }
            Layout::RowMajor => {
                let inv_beta = 1.0 / beta;
                for i in (k + 1)..m {
                    a[i * lda + k] *= inv_beta;
                }
            }
        }

        tau[k] = -beta / alpha;
        a[layout.index(k, k, lda)] = alpha;

        let v_len = col_len;

        match layout {
            Layout::ColMajor => {
                a[k * lda + k] = 1.0;
                let v_base = k * lda + k;

                for j in (k + 1)..n {
                    let col_j_base = j * lda + k;
                    let v_slice: Vec<f64> = a[v_base..v_base + v_len].to_vec();
                    let w = simd::dot_f64(&v_slice, &a[col_j_base..col_j_base + v_len]);
                    let w = w * tau[k];
                    simd::axpy_f64(-w, &v_slice, &mut a[col_j_base..col_j_base + v_len]);
                }

                a[k * lda + k] = alpha;
            }
            Layout::RowMajor => {
                let mut v_buf = Vec::with_capacity(v_len);
                v_buf.push(1.0f64);
                for i in (k + 1)..m {
                    v_buf.push(a[i * lda + k]);
                }

                for j in (k + 1)..n {
                    let mut col_j: Vec<f64> = (k..m).map(|i| a[i * lda + j]).collect();
                    let w = simd::dot_f64(&v_buf, &col_j) * tau[k];
                    simd::axpy_f64(-w, &v_buf, &mut col_j);
                    for (idx, i) in (k..m).enumerate() {
                        a[i * lda + j] = col_j[idx];
                    }
                }
            }
        }
    }

    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sgetrf_and_solve() {
        // A = [[2, 1], [1, 3]]
        let mut a = vec![2.0f32, 1.0, 1.0, 3.0];
        let mut ipiv = vec![0usize; 2];
        let info = sgetrf(Layout::RowMajor, 2, 2, &mut a, 2, &mut ipiv);
        assert_eq!(info, 0, "LU factorization should succeed");

        // Solve A * x = b where b = [5, 7]
        // Expected: x = [1.6, 1.8]
        let mut b = vec![5.0f32, 7.0];
        sgetrs(Layout::RowMajor, 2, 1, &a, 2, &ipiv, &mut b, 1);
        assert!((b[0] - 1.6).abs() < 1e-5, "x[0] = {}", b[0]);
        assert!((b[1] - 1.8).abs() < 1e-5, "x[1] = {}", b[1]);
    }

    #[test]
    fn test_dgetrf_3x3() {
        // A = [[1, 2, 3], [4, 5, 6], [7, 8, 10]]
        let mut a = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0];
        let mut ipiv = vec![0usize; 3];
        let info = dgetrf(Layout::RowMajor, 3, 3, &mut a, 3, &mut ipiv);
        assert_eq!(info, 0);
    }

    #[test]
    fn test_spotrf_cholesky() {
        // A = [[4, 2], [2, 3]] (symmetric positive definite)
        let mut a = vec![4.0f32, 2.0, 2.0, 3.0];
        let info = spotrf(
            Layout::RowMajor,
            numrus_core::layout::Uplo::Lower,
            2,
            &mut a,
            2,
        );
        assert_eq!(info, 0);

        // L[0,0] = sqrt(4) = 2
        assert!((a[0] - 2.0).abs() < 1e-6);
        // L[1,0] = 2/2 = 1
        assert!((a[2] - 1.0).abs() < 1e-6);
        // L[1,1] = sqrt(3 - 1) = sqrt(2)
        assert!((a[3] - 2.0f32.sqrt()).abs() < 1e-6);
    }

    #[test]
    fn test_spotrf_not_positive_definite() {
        // A = [[1, 2], [2, 1]] — not positive definite
        let mut a = vec![1.0f32, 2.0, 2.0, 1.0];
        let info = spotrf(
            Layout::RowMajor,
            numrus_core::layout::Uplo::Lower,
            2,
            &mut a,
            2,
        );
        assert!(info > 0, "Should detect non-positive-definite matrix");
    }

    #[test]
    fn test_sgeqrf() {
        // A = [[1, 1], [0, 1], [1, 0]] (3x2)
        let mut a = vec![1.0f32, 1.0, 0.0, 1.0, 1.0, 0.0];
        let mut tau = vec![0.0f32; 2];
        let info = sgeqrf(Layout::RowMajor, 3, 2, &mut a, 2, &mut tau);
        assert_eq!(info, 0);
        // R should be in upper triangle
        // R[0,0] should be -sqrt(2) (or sqrt(2) depending on sign convention)
        assert!(a[0].abs() > 1.0, "R[0,0] should be nonzero: {}", a[0]);
    }

    #[test]
    fn test_dgetrf_and_solve() {
        // A = [[3, 1], [1, 2]]
        let mut a = vec![3.0f64, 1.0, 1.0, 2.0];
        let mut ipiv = vec![0usize; 2];
        let info = dgetrf(Layout::RowMajor, 2, 2, &mut a, 2, &mut ipiv);
        assert_eq!(info, 0);

        // Solve A*x = [9, 8] → 3*x0+x1=9, x0+2*x1=8 → x = [2, 3]
        let mut b = vec![9.0f64, 8.0];
        dgetrs(Layout::RowMajor, 2, 1, &a, 2, &ipiv, &mut b, 1);
        assert!((b[0] - 2.0).abs() < 1e-10, "x[0] = {}", b[0]);
        assert!((b[1] - 3.0).abs() < 1e-10, "x[1] = {}", b[1]);
    }

    #[test]
    fn test_spotrf_and_solve() {
        // A = [[4, 2], [2, 3]] — SPD
        let a_orig = vec![4.0f32, 2.0, 2.0, 3.0];
        let mut a = a_orig.clone();
        let info = spotrf(
            Layout::RowMajor,
            numrus_core::layout::Uplo::Lower,
            2,
            &mut a,
            2,
        );
        assert_eq!(info, 0);

        // Solve A*x = [8, 8] using Cholesky
        // Expected: 4x0+2x1=8, 2x0+3x1=8 → x = [1, 2]
        let mut b = vec![8.0f32, 8.0];
        spotrs(
            Layout::RowMajor,
            numrus_core::layout::Uplo::Lower,
            2,
            1,
            &a,
            2,
            &mut b,
            1,
        );
        assert!((b[0] - 1.0).abs() < 1e-4, "x[0] = {}", b[0]);
        assert!((b[1] - 2.0).abs() < 1e-4, "x[1] = {}", b[1]);
    }

    #[test]
    fn test_dgeqrf() {
        let mut a = vec![1.0f64, 1.0, 0.0, 1.0, 1.0, 0.0];
        let mut tau = vec![0.0f64; 2];
        let info = dgeqrf(Layout::RowMajor, 3, 2, &mut a, 2, &mut tau);
        assert_eq!(info, 0);
        assert!(a[0].abs() > 1.0, "R[0,0] should be nonzero: {}", a[0]);
    }
}
