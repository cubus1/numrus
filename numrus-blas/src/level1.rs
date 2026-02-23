//! BLAS Level 1: Vector-vector operations.
//!
//! All operations use AVX-512 SIMD via `numrus_core::simd` primitives.
//! Strided access (incx/incy > 1) gathers into contiguous buffers for SIMD.

use numrus_core::simd;

// ============================================================================
// Strided gather helpers — copy strided data into contiguous SIMD-friendly buffers
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

#[inline]
fn scatter_f32(buf: &[f32], dst: &mut [f32], n: usize, inc: usize) {
    for i in 0..n {
        dst[i * inc] = buf[i];
    }
}

#[inline]
fn scatter_f64(buf: &[f64], dst: &mut [f64], n: usize, inc: usize) {
    for i in 0..n {
        dst[i * inc] = buf[i];
    }
}

// ============================================================================
// DOT: inner product
// ============================================================================

/// Single-precision dot product: result = x^T * y
#[inline]
pub fn sdot(n: usize, x: &[f32], incx: usize, y: &[f32], incy: usize) -> f32 {
    #[cfg(feature = "mkl")]
    {
        return unsafe {
            numrus_core::mkl_ffi::cblas_sdot(
                n as i32,
                x.as_ptr(),
                incx as i32,
                y.as_ptr(),
                incy as i32,
            )
        };
    }
    if incx == 1 && incy == 1 {
        simd::dot_f32(&x[..n], &y[..n])
    } else {
        // Gather strided data into contiguous buffers → SIMD dot
        let x_buf = gather_f32(x, n, incx);
        let y_buf = gather_f32(y, n, incy);
        simd::dot_f32(&x_buf, &y_buf)
    }
}

/// Double-precision dot product: result = x^T * y
#[inline]
pub fn ddot(n: usize, x: &[f64], incx: usize, y: &[f64], incy: usize) -> f64 {
    #[cfg(feature = "mkl")]
    {
        return unsafe {
            numrus_core::mkl_ffi::cblas_ddot(
                n as i32,
                x.as_ptr(),
                incx as i32,
                y.as_ptr(),
                incy as i32,
            )
        };
    }
    if incx == 1 && incy == 1 {
        simd::dot_f64(&x[..n], &y[..n])
    } else {
        let x_buf = gather_f64(x, n, incx);
        let y_buf = gather_f64(y, n, incy);
        simd::dot_f64(&x_buf, &y_buf)
    }
}

// ============================================================================
// AXPY: y = alpha * x + y
// ============================================================================

/// Single-precision axpy: y := alpha * x + y
#[inline]
pub fn saxpy(n: usize, alpha: f32, x: &[f32], incx: usize, y: &mut [f32], incy: usize) {
    #[cfg(feature = "mkl")]
    {
        unsafe {
            numrus_core::mkl_ffi::cblas_saxpy(
                n as i32,
                alpha,
                x.as_ptr(),
                incx as i32,
                y.as_mut_ptr(),
                incy as i32,
            );
        }
        return;
    }
    if alpha == 0.0 {
        return;
    }
    if incx == 1 && incy == 1 {
        simd::axpy_f32(alpha, &x[..n], &mut y[..n]);
    } else {
        // Gather → SIMD axpy → scatter
        let x_buf = gather_f32(x, n, incx);
        let mut y_buf = gather_f32(y, n, incy);
        simd::axpy_f32(alpha, &x_buf, &mut y_buf);
        scatter_f32(&y_buf, y, n, incy);
    }
}

/// Double-precision axpy: y := alpha * x + y
#[inline]
pub fn daxpy(n: usize, alpha: f64, x: &[f64], incx: usize, y: &mut [f64], incy: usize) {
    #[cfg(feature = "mkl")]
    {
        unsafe {
            numrus_core::mkl_ffi::cblas_daxpy(
                n as i32,
                alpha,
                x.as_ptr(),
                incx as i32,
                y.as_mut_ptr(),
                incy as i32,
            );
        }
        return;
    }
    if alpha == 0.0 {
        return;
    }
    if incx == 1 && incy == 1 {
        simd::axpy_f64(alpha, &x[..n], &mut y[..n]);
    } else {
        let x_buf = gather_f64(x, n, incx);
        let mut y_buf = gather_f64(y, n, incy);
        simd::axpy_f64(alpha, &x_buf, &mut y_buf);
        scatter_f64(&y_buf, y, n, incy);
    }
}

// ============================================================================
// SCAL: x = alpha * x
// ============================================================================

/// Single-precision scal: x := alpha * x
#[inline]
pub fn sscal(n: usize, alpha: f32, x: &mut [f32], incx: usize) {
    #[cfg(feature = "mkl")]
    {
        unsafe {
            numrus_core::mkl_ffi::cblas_sscal(n as i32, alpha, x.as_mut_ptr(), incx as i32);
        }
        return;
    }
    if incx == 1 {
        simd::scal_f32(alpha, &mut x[..n]);
    } else {
        // Gather → SIMD scal → scatter
        let mut buf = gather_f32(x, n, incx);
        simd::scal_f32(alpha, &mut buf);
        scatter_f32(&buf, x, n, incx);
    }
}

/// Double-precision scal: x := alpha * x
#[inline]
pub fn dscal(n: usize, alpha: f64, x: &mut [f64], incx: usize) {
    #[cfg(feature = "mkl")]
    {
        unsafe {
            numrus_core::mkl_ffi::cblas_dscal(n as i32, alpha, x.as_mut_ptr(), incx as i32);
        }
        return;
    }
    if incx == 1 {
        simd::scal_f64(alpha, &mut x[..n]);
    } else {
        let mut buf = gather_f64(x, n, incx);
        simd::scal_f64(alpha, &mut buf);
        scatter_f64(&buf, x, n, incx);
    }
}

// ============================================================================
// NRM2: Euclidean norm
// ============================================================================

/// Single-precision nrm2: ||x||_2
#[inline]
pub fn snrm2(n: usize, x: &[f32], incx: usize) -> f32 {
    #[cfg(feature = "mkl")]
    {
        return unsafe { numrus_core::mkl_ffi::cblas_snrm2(n as i32, x.as_ptr(), incx as i32) };
    }
    if incx == 1 {
        simd::nrm2_f32(&x[..n])
    } else {
        let buf = gather_f32(x, n, incx);
        simd::nrm2_f32(&buf)
    }
}

/// Double-precision nrm2: ||x||_2
#[inline]
pub fn dnrm2(n: usize, x: &[f64], incx: usize) -> f64 {
    #[cfg(feature = "mkl")]
    {
        return unsafe { numrus_core::mkl_ffi::cblas_dnrm2(n as i32, x.as_ptr(), incx as i32) };
    }
    if incx == 1 {
        simd::nrm2_f64(&x[..n])
    } else {
        let buf = gather_f64(x, n, incx);
        simd::nrm2_f64(&buf)
    }
}

// ============================================================================
// ASUM: sum of absolute values
// ============================================================================

/// Single-precision asum: sum(|x_i|)
#[inline]
pub fn sasum(n: usize, x: &[f32], incx: usize) -> f32 {
    #[cfg(feature = "mkl")]
    {
        return unsafe { numrus_core::mkl_ffi::cblas_sasum(n as i32, x.as_ptr(), incx as i32) };
    }
    if incx == 1 {
        simd::asum_f32(&x[..n])
    } else {
        let buf = gather_f32(x, n, incx);
        simd::asum_f32(&buf)
    }
}

/// Double-precision asum: sum(|x_i|)
#[inline]
pub fn dasum(n: usize, x: &[f64], incx: usize) -> f64 {
    #[cfg(feature = "mkl")]
    {
        return unsafe { numrus_core::mkl_ffi::cblas_dasum(n as i32, x.as_ptr(), incx as i32) };
    }
    if incx == 1 {
        simd::asum_f64(&x[..n])
    } else {
        let buf = gather_f64(x, n, incx);
        simd::asum_f64(&buf)
    }
}

// ============================================================================
// IAMAX: index of max absolute value
// ============================================================================

/// Single-precision iamax: index of max |x_i|
///
/// Uses SIMD asum for the reduction, then scans for the max index.
/// For contiguous (incx=1) arrays, gathers absolute values via SIMD abs
/// then does a single-pass argmax. For strided arrays, gathers first.
#[inline]
pub fn isamax(n: usize, x: &[f32], incx: usize) -> usize {
    #[cfg(feature = "mkl")]
    {
        return unsafe {
            numrus_core::mkl_ffi::cblas_isamax(n as i32, x.as_ptr(), incx as i32) as usize
        };
    }
    if n == 0 {
        return 0;
    }
    // Gather if strided, then linear scan
    // (SIMD argmax requires horizontal reduction — not trivially vectorizable)
    let data: &[f32];
    let buf;
    if incx == 1 {
        data = &x[..n];
    } else {
        buf = gather_f32(x, n, incx);
        data = &buf;
    }

    let mut max_idx = 0;
    let mut max_val = data[0].abs();
    for i in 1..n {
        let v = data[i].abs();
        if v > max_val {
            max_val = v;
            max_idx = i;
        }
    }
    max_idx
}

/// Double-precision iamax: index of max |x_i|
#[inline]
pub fn idamax(n: usize, x: &[f64], incx: usize) -> usize {
    #[cfg(feature = "mkl")]
    {
        return unsafe {
            numrus_core::mkl_ffi::cblas_idamax(n as i32, x.as_ptr(), incx as i32) as usize
        };
    }
    if n == 0 {
        return 0;
    }
    let data: &[f64];
    let buf;
    if incx == 1 {
        data = &x[..n];
    } else {
        buf = gather_f64(x, n, incx);
        data = &buf;
    }

    let mut max_idx = 0;
    let mut max_val = data[0].abs();
    for i in 1..n {
        let v = data[i].abs();
        if v > max_val {
            max_val = v;
            max_idx = i;
        }
    }
    max_idx
}

// ============================================================================
// COPY: x -> y
// ============================================================================

/// Single-precision copy: y := x
#[inline]
pub fn scopy(n: usize, x: &[f32], incx: usize, y: &mut [f32], incy: usize) {
    #[cfg(feature = "mkl")]
    {
        unsafe {
            numrus_core::mkl_ffi::cblas_scopy(
                n as i32,
                x.as_ptr(),
                incx as i32,
                y.as_mut_ptr(),
                incy as i32,
            );
        }
        return;
    }
    if incx == 1 && incy == 1 {
        y[..n].copy_from_slice(&x[..n]);
    } else {
        for i in 0..n {
            y[i * incy] = x[i * incx];
        }
    }
}

/// Double-precision copy: y := x
#[inline]
pub fn dcopy(n: usize, x: &[f64], incx: usize, y: &mut [f64], incy: usize) {
    #[cfg(feature = "mkl")]
    {
        unsafe {
            numrus_core::mkl_ffi::cblas_dcopy(
                n as i32,
                x.as_ptr(),
                incx as i32,
                y.as_mut_ptr(),
                incy as i32,
            );
        }
        return;
    }
    if incx == 1 && incy == 1 {
        y[..n].copy_from_slice(&x[..n]);
    } else {
        for i in 0..n {
            y[i * incy] = x[i * incx];
        }
    }
}

// ============================================================================
// SWAP: x <-> y
// ============================================================================

/// Single-precision swap: x <-> y
#[inline]
pub fn sswap(n: usize, x: &mut [f32], incx: usize, y: &mut [f32], incy: usize) {
    #[cfg(feature = "mkl")]
    {
        unsafe {
            numrus_core::mkl_ffi::cblas_sswap(
                n as i32,
                x.as_mut_ptr(),
                incx as i32,
                y.as_mut_ptr(),
                incy as i32,
            );
        }
        return;
    }
    if incx == 1 && incy == 1 {
        x[..n].swap_with_slice(&mut y[..n]);
    } else {
        for i in 0..n {
            std::mem::swap(&mut x[i * incx], &mut y[i * incy]);
        }
    }
}

/// Double-precision swap: x <-> y
#[inline]
pub fn dswap(n: usize, x: &mut [f64], incx: usize, y: &mut [f64], incy: usize) {
    #[cfg(feature = "mkl")]
    {
        unsafe {
            numrus_core::mkl_ffi::cblas_dswap(
                n as i32,
                x.as_mut_ptr(),
                incx as i32,
                y.as_mut_ptr(),
                incy as i32,
            );
        }
        return;
    }
    if incx == 1 && incy == 1 {
        x[..n].swap_with_slice(&mut y[..n]);
    } else {
        for i in 0..n {
            std::mem::swap(&mut x[i * incx], &mut y[i * incy]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sdot() {
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let y = vec![5.0f32, 6.0, 7.0, 8.0];
        assert_eq!(sdot(4, &x, 1, &y, 1), 70.0);
    }

    #[test]
    fn test_ddot() {
        let x = vec![1.0f64, 2.0, 3.0];
        let y = vec![4.0f64, 5.0, 6.0];
        assert_eq!(ddot(3, &x, 1, &y, 1), 32.0);
    }

    #[test]
    fn test_sdot_strided() {
        let x = vec![1.0f32, 0.0, 2.0, 0.0, 3.0];
        let y = vec![4.0f32, 0.0, 5.0, 0.0, 6.0];
        assert_eq!(sdot(3, &x, 2, &y, 2), 32.0);
    }

    #[test]
    fn test_saxpy() {
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut y = vec![10.0f32, 20.0, 30.0, 40.0];
        saxpy(4, 2.0, &x, 1, &mut y, 1);
        assert_eq!(y, vec![12.0, 24.0, 36.0, 48.0]);
    }

    #[test]
    fn test_sscal() {
        let mut x = vec![1.0f32, 2.0, 3.0, 4.0];
        sscal(4, 3.0, &mut x, 1);
        assert_eq!(x, vec![3.0, 6.0, 9.0, 12.0]);
    }

    #[test]
    fn test_snrm2() {
        let x = vec![3.0f32, 4.0];
        assert!((snrm2(2, &x, 1) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_sasum() {
        let x = vec![-1.0f32, 2.0, -3.0, 4.0];
        assert_eq!(sasum(4, &x, 1), 10.0);
    }

    #[test]
    fn test_isamax() {
        let x = vec![1.0f32, -5.0, 3.0, -2.0];
        assert_eq!(isamax(4, &x, 1), 1);
    }

    #[test]
    fn test_scopy() {
        let x = vec![1.0f32, 2.0, 3.0];
        let mut y = vec![0.0f32; 3];
        scopy(3, &x, 1, &mut y, 1);
        assert_eq!(y, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_sswap() {
        let mut x = vec![1.0f32, 2.0, 3.0];
        let mut y = vec![4.0f32, 5.0, 6.0];
        sswap(3, &mut x, 1, &mut y, 1);
        assert_eq!(x, vec![4.0, 5.0, 6.0]);
        assert_eq!(y, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_saxpy_strided() {
        let x = vec![1.0f32, 0.0, 2.0, 0.0, 3.0];
        let mut y = vec![10.0f32, 0.0, 20.0, 0.0, 30.0];
        saxpy(3, 2.0, &x, 2, &mut y, 2);
        assert_eq!(y[0], 12.0);
        assert_eq!(y[2], 24.0);
        assert_eq!(y[4], 36.0);
    }

    #[test]
    fn test_sscal_strided() {
        let mut x = vec![1.0f32, 0.0, 2.0, 0.0, 3.0];
        sscal(3, 3.0, &mut x, 2);
        assert_eq!(x[0], 3.0);
        assert_eq!(x[2], 6.0);
        assert_eq!(x[4], 9.0);
    }

    #[test]
    fn test_snrm2_strided() {
        let x = vec![3.0f32, 0.0, 4.0];
        assert!((snrm2(2, &x, 2) - 5.0).abs() < 1e-6);
    }
}
