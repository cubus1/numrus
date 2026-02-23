//! BLAS Level 2: Matrix-vector operations.
//!
//! All operations support both row-major and column-major layouts
//! via the CBLAS-style `Layout` parameter.
//!
//! SIMD strategy: contiguous inner loops use `simd::dot` / `simd::axpy`.
//! Strided vectors are gathered into contiguous buffers for SIMD processing.

use numrus_core::layout::{Layout, Transpose, Uplo};
use numrus_core::simd;

// ============================================================================
// Gather helpers — copy strided data into contiguous SIMD-friendly buffers
// ============================================================================

#[inline]
fn gather_f32(x: &[f32], n: usize, inc: usize) -> Vec<f32> {
    let mut buf = Vec::with_capacity(n);
    for i in 0..n {
        buf.push(x[i * inc]);
    }
    buf
}

#[inline]
fn gather_f64(x: &[f64], n: usize, inc: usize) -> Vec<f64> {
    let mut buf = Vec::with_capacity(n);
    for i in 0..n {
        buf.push(x[i * inc]);
    }
    buf
}

// ============================================================================
// GEMV: General matrix-vector multiply
// y := alpha * op(A) * x + beta * y
// ============================================================================

/// Single-precision GEMV: y := alpha * op(A) * x + beta * y
///
/// All inner loops use SIMD: dot products for row reductions,
/// axpy for column-wise accumulation. Strided vectors are gathered first.
pub fn sgemv(
    layout: Layout,
    trans: Transpose,
    m: usize,
    n: usize,
    alpha: f32,
    a: &[f32],
    lda: usize,
    x: &[f32],
    incx: usize,
    beta: f32,
    y: &mut [f32],
    incy: usize,
) {
    #[cfg(feature = "mkl")]
    {
        unsafe {
            numrus_core::mkl_ffi::cblas_sgemv(
                layout as i32,
                trans as i32,
                m as i32,
                n as i32,
                alpha,
                a.as_ptr(),
                lda as i32,
                x.as_ptr(),
                incx as i32,
                beta,
                y.as_mut_ptr(),
                incy as i32,
            );
        }
        return;
    }

    let (rows, _cols) = match trans {
        Transpose::NoTrans => (m, n),
        _ => (n, m),
    };

    // Scale y by beta — SIMD scal when contiguous
    if beta == 0.0 {
        if incy == 1 {
            y[..rows].fill(0.0);
        } else {
            for i in 0..rows {
                y[i * incy] = 0.0;
            }
        }
    } else if beta != 1.0 {
        if incy == 1 {
            simd::scal_f32(beta, &mut y[..rows]);
        } else {
            for i in 0..rows {
                y[i * incy] *= beta;
            }
        }
    }

    if alpha == 0.0 {
        return;
    }

    match (layout, trans) {
        (Layout::RowMajor, Transpose::NoTrans) => {
            // y[i] += alpha * dot(A_row_i, x)
            // A rows are contiguous — use SIMD dot. Gather x if strided.
            if incx == 1 {
                for i in 0..m {
                    let row_start = i * lda;
                    let dot = simd::dot_f32(&a[row_start..row_start + n], &x[..n]);
                    y[i * incy] += alpha * dot;
                }
            } else {
                let x_buf = gather_f32(x, n, incx);
                for i in 0..m {
                    let row_start = i * lda;
                    let dot = simd::dot_f32(&a[row_start..row_start + n], &x_buf);
                    y[i * incy] += alpha * dot;
                }
            }
        }
        (Layout::RowMajor, _) => {
            // Transpose: y += alpha * x[i] * A_row_i  (axpy per row)
            if incy == 1 {
                for i in 0..m {
                    let row_start = i * lda;
                    let xi = alpha * x[i * incx];
                    simd::axpy_f32(xi, &a[row_start..row_start + n], &mut y[..n]);
                }
            } else {
                // y is strided — gather y, SIMD axpy, scatter back
                let mut y_buf = gather_f32(y, n, incy);
                for i in 0..m {
                    let row_start = i * lda;
                    let xi = alpha * x[i * incx];
                    simd::axpy_f32(xi, &a[row_start..row_start + n], &mut y_buf);
                }
                for j in 0..n {
                    y[j * incy] = y_buf[j];
                }
            }
        }
        (Layout::ColMajor, Transpose::NoTrans) => {
            // y += alpha * x[j] * A_col_j  (axpy per column)
            if incy == 1 {
                for j in 0..n {
                    let xj = alpha * x[j * incx];
                    let col_start = j * lda;
                    simd::axpy_f32(xj, &a[col_start..col_start + m], &mut y[..m]);
                }
            } else {
                // y strided — gather, accumulate, scatter
                let mut y_buf = gather_f32(y, m, incy);
                for j in 0..n {
                    let xj = alpha * x[j * incx];
                    let col_start = j * lda;
                    simd::axpy_f32(xj, &a[col_start..col_start + m], &mut y_buf);
                }
                for i in 0..m {
                    y[i * incy] = y_buf[i];
                }
            }
        }
        (Layout::ColMajor, _) => {
            // Trans: y[j] += alpha * dot(A_col_j, x)
            if incx == 1 {
                for j in 0..n {
                    let col_start = j * lda;
                    let dot = simd::dot_f32(&a[col_start..col_start + m], &x[..m]);
                    y[j * incy] += alpha * dot;
                }
            } else {
                let x_buf = gather_f32(x, m, incx);
                for j in 0..n {
                    let col_start = j * lda;
                    let dot = simd::dot_f32(&a[col_start..col_start + m], &x_buf);
                    y[j * incy] += alpha * dot;
                }
            }
        }
    }
}

/// Double-precision GEMV: y := alpha * op(A) * x + beta * y
pub fn dgemv(
    layout: Layout,
    trans: Transpose,
    m: usize,
    n: usize,
    alpha: f64,
    a: &[f64],
    lda: usize,
    x: &[f64],
    incx: usize,
    beta: f64,
    y: &mut [f64],
    incy: usize,
) {
    #[cfg(feature = "mkl")]
    {
        unsafe {
            numrus_core::mkl_ffi::cblas_dgemv(
                layout as i32,
                trans as i32,
                m as i32,
                n as i32,
                alpha,
                a.as_ptr(),
                lda as i32,
                x.as_ptr(),
                incx as i32,
                beta,
                y.as_mut_ptr(),
                incy as i32,
            );
        }
        return;
    }

    let (rows, _cols) = match trans {
        Transpose::NoTrans => (m, n),
        _ => (n, m),
    };

    if beta == 0.0 {
        if incy == 1 {
            y[..rows].fill(0.0);
        } else {
            for i in 0..rows {
                y[i * incy] = 0.0;
            }
        }
    } else if beta != 1.0 {
        if incy == 1 {
            simd::scal_f64(beta, &mut y[..rows]);
        } else {
            for i in 0..rows {
                y[i * incy] *= beta;
            }
        }
    }

    if alpha == 0.0 {
        return;
    }

    match (layout, trans) {
        (Layout::RowMajor, Transpose::NoTrans) => {
            if incx == 1 {
                for i in 0..m {
                    let row_start = i * lda;
                    let dot = simd::dot_f64(&a[row_start..row_start + n], &x[..n]);
                    y[i * incy] += alpha * dot;
                }
            } else {
                let x_buf = gather_f64(x, n, incx);
                for i in 0..m {
                    let row_start = i * lda;
                    let dot = simd::dot_f64(&a[row_start..row_start + n], &x_buf);
                    y[i * incy] += alpha * dot;
                }
            }
        }
        (Layout::RowMajor, _) => {
            if incy == 1 {
                for i in 0..m {
                    let row_start = i * lda;
                    let xi = alpha * x[i * incx];
                    simd::axpy_f64(xi, &a[row_start..row_start + n], &mut y[..n]);
                }
            } else {
                let mut y_buf = gather_f64(y, n, incy);
                for i in 0..m {
                    let row_start = i * lda;
                    let xi = alpha * x[i * incx];
                    simd::axpy_f64(xi, &a[row_start..row_start + n], &mut y_buf);
                }
                for j in 0..n {
                    y[j * incy] = y_buf[j];
                }
            }
        }
        (Layout::ColMajor, Transpose::NoTrans) => {
            if incy == 1 {
                for j in 0..n {
                    let xj = alpha * x[j * incx];
                    let col_start = j * lda;
                    simd::axpy_f64(xj, &a[col_start..col_start + m], &mut y[..m]);
                }
            } else {
                let mut y_buf = gather_f64(y, m, incy);
                for j in 0..n {
                    let xj = alpha * x[j * incx];
                    let col_start = j * lda;
                    simd::axpy_f64(xj, &a[col_start..col_start + m], &mut y_buf);
                }
                for i in 0..m {
                    y[i * incy] = y_buf[i];
                }
            }
        }
        (Layout::ColMajor, _) => {
            if incx == 1 {
                for j in 0..n {
                    let col_start = j * lda;
                    let dot = simd::dot_f64(&a[col_start..col_start + m], &x[..m]);
                    y[j * incy] += alpha * dot;
                }
            } else {
                let x_buf = gather_f64(x, m, incx);
                for j in 0..n {
                    let col_start = j * lda;
                    let dot = simd::dot_f64(&a[col_start..col_start + m], &x_buf);
                    y[j * incy] += alpha * dot;
                }
            }
        }
    }
}

// ============================================================================
// GER: rank-1 update  A := alpha * x * y^T + A
// ============================================================================

/// Single-precision GER: A := alpha * x * y^T + A
///
/// Each row/column update is a SIMD axpy.
/// For strided y (RowMajor) or strided x (ColMajor), gathers into contiguous buffer.
pub fn sger(
    layout: Layout,
    m: usize,
    n: usize,
    alpha: f32,
    x: &[f32],
    incx: usize,
    y: &[f32],
    incy: usize,
    a: &mut [f32],
    lda: usize,
) {
    #[cfg(feature = "mkl")]
    {
        unsafe {
            numrus_core::mkl_ffi::cblas_sger(
                layout as i32,
                m as i32,
                n as i32,
                alpha,
                x.as_ptr(),
                incx as i32,
                y.as_ptr(),
                incy as i32,
                a.as_mut_ptr(),
                lda as i32,
            );
        }
        return;
    }

    if alpha == 0.0 {
        return;
    }

    match layout {
        Layout::RowMajor => {
            // a_row_i += (alpha * x[i]) * y  — axpy on each row
            if incy == 1 {
                for i in 0..m {
                    let xi = alpha * x[i * incx];
                    let row_start = i * lda;
                    simd::axpy_f32(xi, &y[..n], &mut a[row_start..row_start + n]);
                }
            } else {
                // Gather y once, then axpy each row
                let y_buf = gather_f32(y, n, incy);
                for i in 0..m {
                    let xi = alpha * x[i * incx];
                    let row_start = i * lda;
                    simd::axpy_f32(xi, &y_buf, &mut a[row_start..row_start + n]);
                }
            }
        }
        Layout::ColMajor => {
            // a_col_j += (alpha * y[j]) * x  — axpy on each column
            if incx == 1 {
                for j in 0..n {
                    let yj = alpha * y[j * incy];
                    let col_start = j * lda;
                    simd::axpy_f32(yj, &x[..m], &mut a[col_start..col_start + m]);
                }
            } else {
                let x_buf = gather_f32(x, m, incx);
                for j in 0..n {
                    let yj = alpha * y[j * incy];
                    let col_start = j * lda;
                    simd::axpy_f32(yj, &x_buf, &mut a[col_start..col_start + m]);
                }
            }
        }
    }
}

/// Double-precision GER: A := alpha * x * y^T + A
pub fn dger(
    layout: Layout,
    m: usize,
    n: usize,
    alpha: f64,
    x: &[f64],
    incx: usize,
    y: &[f64],
    incy: usize,
    a: &mut [f64],
    lda: usize,
) {
    #[cfg(feature = "mkl")]
    {
        unsafe {
            numrus_core::mkl_ffi::cblas_dger(
                layout as i32,
                m as i32,
                n as i32,
                alpha,
                x.as_ptr(),
                incx as i32,
                y.as_ptr(),
                incy as i32,
                a.as_mut_ptr(),
                lda as i32,
            );
        }
        return;
    }

    if alpha == 0.0 {
        return;
    }

    match layout {
        Layout::RowMajor => {
            if incy == 1 {
                for i in 0..m {
                    let xi = alpha * x[i * incx];
                    let row_start = i * lda;
                    simd::axpy_f64(xi, &y[..n], &mut a[row_start..row_start + n]);
                }
            } else {
                let y_buf = gather_f64(y, n, incy);
                for i in 0..m {
                    let xi = alpha * x[i * incx];
                    let row_start = i * lda;
                    simd::axpy_f64(xi, &y_buf, &mut a[row_start..row_start + n]);
                }
            }
        }
        Layout::ColMajor => {
            if incx == 1 {
                for j in 0..n {
                    let yj = alpha * y[j * incy];
                    let col_start = j * lda;
                    simd::axpy_f64(yj, &x[..m], &mut a[col_start..col_start + m]);
                }
            } else {
                let x_buf = gather_f64(x, m, incx);
                for j in 0..n {
                    let yj = alpha * y[j * incy];
                    let col_start = j * lda;
                    simd::axpy_f64(yj, &x_buf, &mut a[col_start..col_start + m]);
                }
            }
        }
    }
}

// ============================================================================
// SYMV: Symmetric matrix-vector multiply
// y := alpha * A * x + beta * y  (A is symmetric)
// ============================================================================

/// Single-precision SYMV: y := alpha * A * x + beta * y (A symmetric)
///
/// For contiguous (incx==incy==1) upper-triangle, the off-diagonal block
/// from i+1..n uses SIMD dot (for sum) and SIMD axpy (for y update).
pub fn ssymv(
    layout: Layout,
    uplo: Uplo,
    n: usize,
    alpha: f32,
    a: &[f32],
    lda: usize,
    x: &[f32],
    incx: usize,
    beta: f32,
    y: &mut [f32],
    incy: usize,
) {
    #[cfg(feature = "mkl")]
    {
        unsafe {
            numrus_core::mkl_ffi::cblas_ssymv(
                layout as i32,
                uplo as i32,
                n as i32,
                alpha,
                a.as_ptr(),
                lda as i32,
                x.as_ptr(),
                incx as i32,
                beta,
                y.as_mut_ptr(),
                incy as i32,
            );
        }
        return;
    }

    // Scale y by beta
    if beta == 0.0 {
        if incy == 1 {
            y[..n].fill(0.0);
        } else {
            for i in 0..n {
                y[i * incy] = 0.0;
            }
        }
    } else if beta != 1.0 {
        if incy == 1 {
            simd::scal_f32(beta, &mut y[..n]);
        } else {
            for i in 0..n {
                y[i * incy] *= beta;
            }
        }
    }

    if alpha == 0.0 {
        return;
    }

    let contiguous = incx == 1 && incy == 1;

    for i in 0..n {
        let xi = x[i * incx];
        let mut sum = 0.0f32;
        match (layout, uplo) {
            (Layout::RowMajor, Uplo::Upper) | (Layout::ColMajor, Uplo::Lower) => {
                // Diagonal
                sum += a[i * lda + i] * xi;
                let len = n - (i + 1);
                if len > 0 && contiguous {
                    // SIMD dot for sum accumulation
                    let a_slice = &a[i * lda + (i + 1)..i * lda + n];
                    let x_slice = &x[(i + 1)..n];
                    sum += simd::dot_f32(a_slice, x_slice);
                    // SIMD axpy for y update: y[i+1..n] += alpha * xi * a_row[i+1..n]
                    simd::axpy_f32(alpha * xi, a_slice, &mut y[(i + 1)..n]);
                } else {
                    for j in (i + 1)..n {
                        let aij = a[i * lda + j];
                        sum += aij * x[j * incx];
                        y[j * incy] += alpha * aij * xi;
                    }
                }
            }
            (Layout::RowMajor, Uplo::Lower) | (Layout::ColMajor, Uplo::Upper) => {
                if i > 0 && contiguous {
                    let a_slice = &a[i * lda..i * lda + i];
                    let x_slice = &x[..i];
                    sum += simd::dot_f32(a_slice, x_slice);
                    simd::axpy_f32(alpha * xi, a_slice, &mut y[..i]);
                } else {
                    for j in 0..i {
                        let aij = a[i * lda + j];
                        sum += aij * x[j * incx];
                        y[j * incy] += alpha * aij * xi;
                    }
                }
                sum += a[i * lda + i] * xi;
            }
        }
        y[i * incy] += alpha * sum;
    }
}

/// Double-precision SYMV: y := alpha * A * x + beta * y (A symmetric)
pub fn dsymv(
    layout: Layout,
    uplo: Uplo,
    n: usize,
    alpha: f64,
    a: &[f64],
    lda: usize,
    x: &[f64],
    incx: usize,
    beta: f64,
    y: &mut [f64],
    incy: usize,
) {
    #[cfg(feature = "mkl")]
    {
        unsafe {
            numrus_core::mkl_ffi::cblas_dsymv(
                layout as i32,
                uplo as i32,
                n as i32,
                alpha,
                a.as_ptr(),
                lda as i32,
                x.as_ptr(),
                incx as i32,
                beta,
                y.as_mut_ptr(),
                incy as i32,
            );
        }
        return;
    }

    if beta == 0.0 {
        if incy == 1 {
            y[..n].fill(0.0);
        } else {
            for i in 0..n {
                y[i * incy] = 0.0;
            }
        }
    } else if beta != 1.0 {
        if incy == 1 {
            simd::scal_f64(beta, &mut y[..n]);
        } else {
            for i in 0..n {
                y[i * incy] *= beta;
            }
        }
    }

    if alpha == 0.0 {
        return;
    }

    let contiguous = incx == 1 && incy == 1;

    for i in 0..n {
        let xi = x[i * incx];
        let mut sum = 0.0f64;
        match (layout, uplo) {
            (Layout::RowMajor, Uplo::Upper) | (Layout::ColMajor, Uplo::Lower) => {
                sum += a[i * lda + i] * xi;
                let len = n - (i + 1);
                if len > 0 && contiguous {
                    let a_slice = &a[i * lda + (i + 1)..i * lda + n];
                    let x_slice = &x[(i + 1)..n];
                    sum += simd::dot_f64(a_slice, x_slice);
                    simd::axpy_f64(alpha * xi, a_slice, &mut y[(i + 1)..n]);
                } else {
                    for j in (i + 1)..n {
                        let aij = a[i * lda + j];
                        sum += aij * x[j * incx];
                        y[j * incy] += alpha * aij * xi;
                    }
                }
            }
            (Layout::RowMajor, Uplo::Lower) | (Layout::ColMajor, Uplo::Upper) => {
                if i > 0 && contiguous {
                    let a_slice = &a[i * lda..i * lda + i];
                    let x_slice = &x[..i];
                    sum += simd::dot_f64(a_slice, x_slice);
                    simd::axpy_f64(alpha * xi, a_slice, &mut y[..i]);
                } else {
                    for j in 0..i {
                        let aij = a[i * lda + j];
                        sum += aij * x[j * incx];
                        y[j * incy] += alpha * aij * xi;
                    }
                }
                sum += a[i * lda + i] * xi;
            }
        }
        y[i * incy] += alpha * sum;
    }
}

// ============================================================================
// TRMV: Triangular matrix-vector multiply  x := op(A) * x
// ============================================================================

/// Single-precision TRMV: x := op(A) * x (A triangular)
///
/// Uses SIMD dot for the off-diagonal summation when incx==1.
pub fn strmv(
    layout: Layout,
    uplo: Uplo,
    trans: Transpose,
    diag: numrus_core::layout::Diag,
    n: usize,
    a: &[f32],
    lda: usize,
    x: &mut [f32],
    incx: usize,
) {
    #[cfg(feature = "mkl")]
    {
        unsafe {
            numrus_core::mkl_ffi::cblas_strmv(
                layout as i32,
                uplo as i32,
                trans as i32,
                diag as i32,
                n as i32,
                a.as_ptr(),
                lda as i32,
                x.as_mut_ptr(),
                incx as i32,
            );
        }
        return;
    }

    let unit = diag == numrus_core::layout::Diag::Unit;

    match (layout, uplo, trans) {
        (Layout::RowMajor, Uplo::Upper, Transpose::NoTrans) => {
            for i in 0..n {
                let mut sum = if unit {
                    x[i * incx]
                } else {
                    a[i * lda + i] * x[i * incx]
                };
                let len = n - (i + 1);
                if len > 0 && incx == 1 {
                    sum += simd::dot_f32(&a[i * lda + (i + 1)..i * lda + n], &x[(i + 1)..n]);
                } else {
                    for j in (i + 1)..n {
                        sum += a[i * lda + j] * x[j * incx];
                    }
                }
                x[i * incx] = sum;
            }
        }
        (Layout::RowMajor, Uplo::Lower, Transpose::NoTrans) => {
            for i in (0..n).rev() {
                let mut sum = if unit {
                    x[i * incx]
                } else {
                    a[i * lda + i] * x[i * incx]
                };
                if i > 0 && incx == 1 {
                    sum += simd::dot_f32(&a[i * lda..i * lda + i], &x[..i]);
                } else {
                    for j in 0..i {
                        sum += a[i * lda + j] * x[j * incx];
                    }
                }
                x[i * incx] = sum;
            }
        }
        _ => {
            let effective_uplo = match (uplo, trans) {
                (Uplo::Upper, Transpose::Trans | Transpose::ConjTrans) => Uplo::Lower,
                (Uplo::Lower, Transpose::Trans | Transpose::ConjTrans) => Uplo::Upper,
                (u, _) => u,
            };
            strmv(
                layout,
                effective_uplo,
                Transpose::NoTrans,
                diag,
                n,
                a,
                lda,
                x,
                incx,
            );
        }
    }
}

// ============================================================================
// TRSV: Triangular solve  x := op(A)^{-1} * x
// ============================================================================

/// Single-precision TRSV: x := A^{-1} * x (A triangular, row-major)
///
/// Sequential dependency between rows limits full SIMD, but the
/// inner dot-product accumulation uses SIMD for contiguous access.
pub fn strsv(
    layout: Layout,
    uplo: Uplo,
    trans: Transpose,
    diag: numrus_core::layout::Diag,
    n: usize,
    a: &[f32],
    lda: usize,
    x: &mut [f32],
    incx: usize,
) {
    #[cfg(feature = "mkl")]
    {
        unsafe {
            numrus_core::mkl_ffi::cblas_strsv(
                layout as i32,
                uplo as i32,
                trans as i32,
                diag as i32,
                n as i32,
                a.as_ptr(),
                lda as i32,
                x.as_mut_ptr(),
                incx as i32,
            );
        }
        return;
    }

    let unit = diag == numrus_core::layout::Diag::Unit;

    match (layout, uplo, trans) {
        (Layout::RowMajor, Uplo::Lower, Transpose::NoTrans) => {
            // Forward substitution
            for i in 0..n {
                let mut sum = x[i * incx];
                if i > 0 && incx == 1 {
                    sum -= simd::dot_f32(&a[i * lda..i * lda + i], &x[..i]);
                } else {
                    for j in 0..i {
                        sum -= a[i * lda + j] * x[j * incx];
                    }
                }
                x[i * incx] = if unit { sum } else { sum / a[i * lda + i] };
            }
        }
        (Layout::RowMajor, Uplo::Upper, Transpose::NoTrans) => {
            // Back substitution
            for i in (0..n).rev() {
                let mut sum = x[i * incx];
                let len = n - (i + 1);
                if len > 0 && incx == 1 {
                    sum -= simd::dot_f32(&a[i * lda + (i + 1)..i * lda + n], &x[(i + 1)..n]);
                } else {
                    for j in (i + 1)..n {
                        sum -= a[i * lda + j] * x[j * incx];
                    }
                }
                x[i * incx] = if unit { sum } else { sum / a[i * lda + i] };
            }
        }
        _ => {
            let effective_uplo = match (uplo, trans) {
                (Uplo::Upper, Transpose::Trans | Transpose::ConjTrans) => Uplo::Lower,
                (Uplo::Lower, Transpose::Trans | Transpose::ConjTrans) => Uplo::Upper,
                (u, _) => u,
            };
            strsv(
                layout,
                effective_uplo,
                Transpose::NoTrans,
                diag,
                n,
                a,
                lda,
                x,
                incx,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sgemv_rowmajor_notrans() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let x = vec![1.0f32, 1.0];
        let mut y = vec![0.0f32; 2];
        sgemv(
            Layout::RowMajor,
            Transpose::NoTrans,
            2,
            2,
            1.0,
            &a,
            2,
            &x,
            1,
            0.0,
            &mut y,
            1,
        );
        assert_eq!(y, vec![3.0, 7.0]);
    }

    #[test]
    fn test_sgemv_with_alpha_beta() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let x = vec![1.0f32, 1.0];
        let mut y = vec![10.0f32, 20.0];
        sgemv(
            Layout::RowMajor,
            Transpose::NoTrans,
            2,
            2,
            2.0,
            &a,
            2,
            &x,
            1,
            3.0,
            &mut y,
            1,
        );
        assert_eq!(y, vec![36.0, 74.0]);
    }

    #[test]
    fn test_sgemv_trans() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let x = vec![1.0f32, 1.0];
        let mut y = vec![0.0f32; 2];
        sgemv(
            Layout::RowMajor,
            Transpose::Trans,
            2,
            2,
            1.0,
            &a,
            2,
            &x,
            1,
            0.0,
            &mut y,
            1,
        );
        assert_eq!(y, vec![4.0, 6.0]);
    }

    #[test]
    fn test_sgemv_strided() {
        // Strided x: use every 2nd element
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let x = vec![1.0f32, 0.0, 1.0]; // x[0]=1, x[2]=1 with incx=2
        let mut y = vec![0.0f32; 2];
        sgemv(
            Layout::RowMajor,
            Transpose::NoTrans,
            2,
            2,
            1.0,
            &a,
            2,
            &x,
            2,
            0.0,
            &mut y,
            1,
        );
        assert_eq!(y, vec![3.0, 7.0]);
    }

    #[test]
    fn test_sgemv_colmajor_notrans() {
        // ColMajor: A stored as columns. A = [[1,3],[2,4]] in col-major = [1,2,3,4]
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let x = vec![1.0f32, 1.0];
        let mut y = vec![0.0f32; 2];
        sgemv(
            Layout::ColMajor,
            Transpose::NoTrans,
            2,
            2,
            1.0,
            &a,
            2,
            &x,
            1,
            0.0,
            &mut y,
            1,
        );
        // A*x = [[1,3],[2,4]] * [1,1] = [4, 6]
        assert_eq!(y, vec![4.0, 6.0]);
    }

    #[test]
    fn test_dgemv_rowmajor() {
        let a = vec![1.0f64, 2.0, 3.0, 4.0];
        let x = vec![2.0f64, 3.0];
        let mut y = vec![0.0f64; 2];
        dgemv(
            Layout::RowMajor,
            Transpose::NoTrans,
            2,
            2,
            1.0,
            &a,
            2,
            &x,
            1,
            0.0,
            &mut y,
            1,
        );
        assert_eq!(y, vec![8.0, 18.0]);
    }

    #[test]
    fn test_sger() {
        let x = vec![1.0f32, 2.0];
        let y = vec![3.0f32, 4.0];
        let mut a = vec![0.0f32; 4];
        sger(Layout::RowMajor, 2, 2, 1.0, &x, 1, &y, 1, &mut a, 2);
        assert_eq!(a, vec![3.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sger_strided() {
        let x = vec![1.0f32, 0.0, 2.0]; // x[0]=1, x[2]=2 with incx=2
        let y = vec![3.0f32, 0.0, 4.0]; // y[0]=3, y[2]=4 with incy=2
        let mut a = vec![0.0f32; 4];
        sger(Layout::RowMajor, 2, 2, 1.0, &x, 2, &y, 2, &mut a, 2);
        assert_eq!(a, vec![3.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_dger() {
        let x = vec![1.0f64, 2.0];
        let y = vec![3.0f64, 4.0];
        let mut a = vec![0.0f64; 4];
        dger(Layout::RowMajor, 2, 2, 1.0, &x, 1, &y, 1, &mut a, 2);
        assert_eq!(a, vec![3.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_ssymv_upper() {
        // A = [[1, 2], [2, 3]] (symmetric, upper stored), x = [1, 1]
        // y = A*x = [3, 5]
        let a = vec![1.0f32, 2.0, 0.0, 3.0]; // upper triangle
        let x = vec![1.0f32, 1.0];
        let mut y = vec![0.0f32; 2];
        ssymv(
            Layout::RowMajor,
            Uplo::Upper,
            2,
            1.0,
            &a,
            2,
            &x,
            1,
            0.0,
            &mut y,
            1,
        );
        assert_eq!(y, vec![3.0, 5.0]);
    }

    #[test]
    fn test_dsymv() {
        let a = vec![1.0f64, 2.0, 0.0, 3.0];
        let x = vec![1.0f64, 1.0];
        let mut y = vec![0.0f64; 2];
        dsymv(
            Layout::RowMajor,
            Uplo::Upper,
            2,
            1.0,
            &a,
            2,
            &x,
            1,
            0.0,
            &mut y,
            1,
        );
        assert_eq!(y, vec![3.0, 5.0]);
    }

    #[test]
    fn test_strsv_lower() {
        let a = vec![2.0f32, 0.0, 1.0, 3.0];
        let mut x = vec![4.0f32, 7.0];
        strsv(
            Layout::RowMajor,
            Uplo::Lower,
            Transpose::NoTrans,
            numrus_core::layout::Diag::NonUnit,
            2,
            &a,
            2,
            &mut x,
            1,
        );
        assert!((x[0] - 2.0).abs() < 1e-6);
        assert!((x[1] - 5.0 / 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_strmv_upper() {
        // A = [[2, 3], [0, 4]], x = [1, 2]
        // x := A*x = [2*1 + 3*2, 4*2] = [8, 8]
        let a = vec![2.0f32, 3.0, 0.0, 4.0];
        let mut x = vec![1.0f32, 2.0];
        strmv(
            Layout::RowMajor,
            Uplo::Upper,
            Transpose::NoTrans,
            numrus_core::layout::Diag::NonUnit,
            2,
            &a,
            2,
            &mut x,
            1,
        );
        assert_eq!(x, vec![8.0, 8.0]);
    }
}
