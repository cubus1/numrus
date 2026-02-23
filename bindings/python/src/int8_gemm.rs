// bindings/python/src/int8_gemm.rs
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Quantize f32 data to u8 with scale/zero_point.
/// Returns (quantized_data, scale, zero_point).
#[pyfunction]
pub fn quantize_f32_to_u8(data: Vec<f32>) -> (Vec<u8>, f32, i32) {
    let (q, params) = numrus_blas::int8_gemm::quantize_f32_to_u8(&data);
    (q, params.scale, params.zero_point)
}

/// Quantize f32 data to i8 with scale/zero_point.
/// Returns (quantized_data, scale, zero_point).
#[pyfunction]
pub fn quantize_f32_to_i8(data: Vec<f32>) -> (Vec<i8>, f32, i32) {
    let (q, params) = numrus_blas::int8_gemm::quantize_f32_to_i8(&data);
    (q, params.scale, params.zero_point)
}

/// INT8 GEMM: C[m*n] = A[m*k] x B[k*n] in i32 accumulation.
/// Uses VNNI (vpdpbusd) on capable CPUs — 64 MACs per instruction.
/// A is u8, B is i8, C is i32.
#[pyfunction]
pub fn int8_gemm_i32(a: Vec<u8>, b: Vec<i8>, m: usize, n: usize, k: usize) -> PyResult<Vec<i32>> {
    if a.len() != m * k {
        return Err(PyValueError::new_err(format!(
            "A must be m*k={} elements, got {}",
            m * k,
            a.len()
        )));
    }
    if b.len() != k * n {
        return Err(PyValueError::new_err(format!(
            "B must be k*n={} elements, got {}",
            k * n,
            b.len()
        )));
    }
    let mut c = vec![0i32; m * n];
    numrus_blas::int8_gemm::int8_gemm_i32(&a, &b, &mut c, m, n, k);
    Ok(c)
}

/// INT8 GEMM with f32 dequantization: returns f32 output.
/// Handles quantization parameters for accurate dequantization.
#[pyfunction]
pub fn int8_gemm_f32(
    a: Vec<u8>,
    b: Vec<i8>,
    m: usize,
    n: usize,
    k: usize,
    scale_a: f32,
    zero_point_a: i32,
    scale_b: f32,
) -> PyResult<Vec<f32>> {
    if a.len() != m * k {
        return Err(PyValueError::new_err(format!(
            "A must be m*k={} elements, got {}",
            m * k,
            a.len()
        )));
    }
    if b.len() != k * n {
        return Err(PyValueError::new_err(format!(
            "B must be k*n={} elements, got {}",
            k * n,
            b.len()
        )));
    }
    let mut c = vec![0.0f32; m * n];
    numrus_blas::int8_gemm::int8_gemm_f32(&a, &b, &mut c, m, n, k, scale_a, zero_point_a, scale_b);
    Ok(c)
}
