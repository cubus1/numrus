// bindings/python/src/functions.rs
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyList;
use numrus_rs::{NumArrayF32, NumArrayF64, NumArrayU8};

use crate::array_f32::PyNumArrayF32;
use crate::array_f64::PyNumArrayF64;
use crate::array_u8::PyNumArrayU8;

#[pyfunction]
pub fn zeros_f32(shape: Vec<usize>) -> PyResult<PyNumArrayF32> {
    Python::with_gil(|_py| {
        let result = NumArrayF32::zeros(shape);
        Ok(PyNumArrayF32 { inner: result })
    })
}

#[pyfunction]
pub fn ones_f32(shape: Vec<usize>) -> PyResult<PyNumArrayF32> {
    Python::with_gil(|_py| {
        let result = NumArrayF32::ones(shape);
        Ok(PyNumArrayF32 { inner: result })
    })
}

#[pyfunction]
pub fn matmul_f32(a: &PyNumArrayF32, b: &PyNumArrayF32) -> PyResult<PyNumArrayF32> {
    Python::with_gil(|_py| {
        if a.inner.shape().len() != 2 || b.inner.shape().len() != 2 {
            return Err(PyTypeError::new_err(
                "Both NumArrayF32 instances must be 2D for matrix multiplication.",
            ));
        }
        let result = a.inner.dot(&b.inner);
        Ok(PyNumArrayF32 { inner: result })
    })
}

#[pyfunction]
pub fn dot_f32(a: &PyNumArrayF32, b: &PyNumArrayF32) -> PyResult<PyNumArrayF32> {
    Python::with_gil(|_py| {
        let result = a.inner.dot(&b.inner);
        Ok(PyNumArrayF32 { inner: result })
    })
}

#[pyfunction]
pub fn arange_f32(start: f32, end: f32, step: f32) -> PyResult<PyNumArrayF32> {
    Python::with_gil(|_py| {
        let result = NumArrayF32::arange(start, end, step);
        Ok(PyNumArrayF32 { inner: result })
    })
}

#[pyfunction]
pub fn linspace_f32(start: f32, end: f32, num: usize) -> PyResult<PyNumArrayF32> {
    Python::with_gil(|_py| {
        let result = NumArrayF32::linspace(start, end, num);
        Ok(PyNumArrayF32 { inner: result })
    })
}

#[pyfunction]
pub fn mean_f32(a: &PyNumArrayF32, axis: Option<&PyList>) -> PyResult<PyNumArrayF32> {
    Python::with_gil(|_py| {
        let result = match axis {
            Some(axis_list) => {
                let axis_vec: Vec<usize> = axis_list.extract()?; // Convert PyList to Vec<usize>
                a.inner.mean_axis(Some(&axis_vec))
            }
            None => a.inner.mean_axis(None), // Handle the case where no axis are provided
        };
        Ok(PyNumArrayF32 { inner: result }) // Convert the result data to a Python object
    })
}

#[pyfunction]
pub fn median_f32(a: &PyNumArrayF32, axis: Option<&PyList>) -> PyResult<PyNumArrayF32> {
    Python::with_gil(|_py| {
        let result = match axis {
            Some(axis_list) => {
                let axis_vec: Vec<usize> = axis_list.extract()?; // Convert PyList to Vec<usize>
                a.inner.median_axis(Some(&axis_vec))
            }
            None => a.inner.median_axis(None), // Handle the case where no axis are provided
        };
        Ok(PyNumArrayF32 { inner: result }) // Convert the result data to a Python object
    })
}

#[pyfunction]
pub fn min_f32(a: &PyNumArrayF32) -> PyResult<f32> {
    Ok(a.inner.min())
}

#[pyfunction]
pub fn min_axis_f32(a: &PyNumArrayF32, axis: Option<&PyList>) -> PyResult<PyNumArrayF32> {
    let result = match axis {
        Some(axis_list) => {
            let axis_vec: Vec<usize> = axis_list.extract()?; // Convert PyList to Vec<usize>
            a.inner.min_axis(Some(&axis_vec))
        }
        None => a.inner.min_axis(None),
    };
    Ok(PyNumArrayF32 { inner: result })
}

#[pyfunction]
pub fn max_f32(a: &PyNumArrayF32) -> PyResult<f32> {
    Ok(a.inner.max())
}

#[pyfunction]
pub fn max_axis_f32(a: &PyNumArrayF32, axis: Option<&PyList>) -> PyResult<PyNumArrayF32> {
    let result = match axis {
        Some(axis_list) => {
            let axis_vec: Vec<usize> = axis_list.extract()?; // Convert PyList to Vec<usize>
            a.inner.max_axis(Some(&axis_vec))
        }
        None => a.inner.max_axis(None),
    };
    Ok(PyNumArrayF32 { inner: result })
}

#[pyfunction]
pub fn exp_f32(a: &PyNumArrayF32) -> PyNumArrayF32 {
    PyNumArrayF32 {
        inner: a.inner.exp(),
    }
}

#[pyfunction]
pub fn log_f32(a: &PyNumArrayF32) -> PyNumArrayF32 {
    PyNumArrayF32 {
        inner: a.inner.log(),
    }
}

#[pyfunction]
pub fn sigmoid_f32(a: &PyNumArrayF32) -> PyNumArrayF32 {
    PyNumArrayF32 {
        inner: a.inner.sigmoid(),
    }
}

#[pyfunction]
pub fn concatenate_f32(arrays: Vec<PyNumArrayF32>, axis: usize) -> PyResult<PyNumArrayF32> {
    let rust_arrays: Vec<NumArrayF32> = arrays.iter().map(|array| array.inner.clone()).collect();
    let result = NumArrayF32::concatenate(&rust_arrays, axis);
    Ok(PyNumArrayF32 { inner: result })
}

#[pyfunction]
pub fn zeros_f64(shape: Vec<usize>) -> PyResult<PyNumArrayF64> {
    Python::with_gil(|_py| {
        let result = NumArrayF64::zeros(shape);
        Ok(PyNumArrayF64 { inner: result })
    })
}

#[pyfunction]
pub fn ones_f64(shape: Vec<usize>) -> PyResult<PyNumArrayF64> {
    Python::with_gil(|_py| {
        let result = NumArrayF64::ones(shape);
        Ok(PyNumArrayF64 { inner: result })
    })
}

#[pyfunction]
pub fn matmul_f64(a: &PyNumArrayF64, b: &PyNumArrayF64) -> PyResult<PyNumArrayF64> {
    Python::with_gil(|_py| {
        if a.inner.shape().len() != 2 || b.inner.shape().len() != 2 {
            return Err(PyTypeError::new_err(
                "Both NumArrayF64 instances must be 2D for matrix multiplication.",
            ));
        }
        let result = a.inner.dot(&b.inner);
        Ok(PyNumArrayF64 { inner: result })
    })
}

#[pyfunction]
pub fn dot_f64(a: &PyNumArrayF64, b: &PyNumArrayF64) -> PyResult<PyNumArrayF64> {
    Python::with_gil(|_py| {
        let result = a.inner.dot(&b.inner);
        Ok(PyNumArrayF64 { inner: result })
    })
}

#[pyfunction]
pub fn arange_f64(start: f64, end: f64, step: f64) -> PyResult<PyNumArrayF64> {
    Python::with_gil(|_py| {
        let result = NumArrayF64::arange(start, end, step);
        Ok(PyNumArrayF64 { inner: result })
    })
}

#[pyfunction]
pub fn linspace_f64(start: f64, end: f64, num: usize) -> PyResult<PyNumArrayF64> {
    Python::with_gil(|_py| {
        let result = NumArrayF64::linspace(start, end, num);
        Ok(PyNumArrayF64 { inner: result })
    })
}

#[pyfunction]
pub fn mean_f64(a: &PyNumArrayF64, axis: Option<&PyList>) -> PyResult<PyNumArrayF64> {
    Python::with_gil(|_py| {
        let result = match axis {
            Some(axis_list) => {
                let axis_vec: Vec<usize> = axis_list.extract()?; // Convert PyList to Vec<usize>
                a.inner.mean_axis(Some(&axis_vec))
            }
            None => a.inner.mean_axis(None), // Handle the case where no axis are provided
        };
        Ok(PyNumArrayF64 { inner: result })
    })
}

#[pyfunction]
pub fn median_f64(a: &PyNumArrayF64, axis: Option<&PyList>) -> PyResult<PyNumArrayF64> {
    Python::with_gil(|_py| {
        let result = match axis {
            Some(axis_list) => {
                let axis_vec: Vec<usize> = axis_list.extract()?; // Convert PyList to Vec<usize>
                a.inner.median_axis(Some(&axis_vec))
            }
            None => a.inner.median_axis(None), // Handle the case where no axis are provided
        };
        Ok(PyNumArrayF64 { inner: result })
    })
}

#[pyfunction]
pub fn min_f64(a: &PyNumArrayF64) -> PyResult<f64> {
    Ok(a.inner.min())
}

#[pyfunction]
pub fn min_axis_f64(a: &PyNumArrayF64, axis: Option<&PyList>) -> PyResult<PyNumArrayF64> {
    let result = match axis {
        Some(axis_list) => {
            let axis_vec: Vec<usize> = axis_list.extract()?; // Convert PyList to Vec<usize>
            a.inner.min_axis(Some(&axis_vec))
        }
        None => a.inner.min_axis(None),
    };
    Ok(PyNumArrayF64 { inner: result })
}

#[pyfunction]
pub fn max_f64(a: &PyNumArrayF64) -> PyResult<f64> {
    Ok(a.inner.max())
}

#[pyfunction]
pub fn max_axis_f64(a: &PyNumArrayF64, axis: Option<&PyList>) -> PyResult<PyNumArrayF64> {
    let result = match axis {
        Some(axis_list) => {
            let axis_vec: Vec<usize> = axis_list.extract()?; // Convert PyList to Vec<usize>
            a.inner.max_axis(Some(&axis_vec))
        }
        None => a.inner.max_axis(None),
    };
    Ok(PyNumArrayF64 { inner: result })
}

#[pyfunction]
pub fn exp_f64(a: &PyNumArrayF64) -> PyNumArrayF64 {
    PyNumArrayF64 {
        inner: a.inner.exp(),
    }
}

#[pyfunction]
pub fn log_f64(a: &PyNumArrayF64) -> PyNumArrayF64 {
    PyNumArrayF64 {
        inner: a.inner.log(),
    }
}

#[pyfunction]
pub fn sigmoid_f64(a: &PyNumArrayF64) -> PyNumArrayF64 {
    PyNumArrayF64 {
        inner: a.inner.sigmoid(),
    }
}

#[pyfunction]
pub fn concatenate_f64(arrays: Vec<PyNumArrayF64>, axis: usize) -> PyResult<PyNumArrayF64> {
    let rust_arrays: Vec<NumArrayF64> = arrays.iter().map(|array| array.inner.clone()).collect();
    let result = NumArrayF64::concatenate(&rust_arrays, axis);
    Ok(PyNumArrayF64 { inner: result })
}

#[pyfunction]
pub fn norm_f32(
    a: &PyNumArrayF32,
    p: u32,
    axis: Option<&PyList>,
    keepdims: Option<bool>,
) -> PyResult<PyNumArrayF32> {
    Python::with_gil(|_py| {
        let result = match axis {
            Some(axis_list) => {
                let axis_vec: Vec<usize> = axis_list.extract()?;
                a.inner.norm(p, Some(&axis_vec), keepdims)
            }
            None => a.inner.norm(p, None, keepdims),
        };
        Ok(PyNumArrayF32 { inner: result })
    })
}

#[pyfunction]
pub fn norm_f64(
    a: &PyNumArrayF64,
    p: u32,
    axis: Option<&PyList>,
    keepdims: Option<bool>,
) -> PyResult<PyNumArrayF64> {
    Python::with_gil(|_py| {
        let result = match axis {
            Some(axis_list) => {
                let axis_vec: Vec<usize> = axis_list.extract()?;
                a.inner.norm(p, Some(&axis_vec), keepdims)
            }
            None => a.inner.norm(p, None, keepdims),
        };
        Ok(PyNumArrayF64 { inner: result })
    })
}

// === HDC/VSA Free Functions ===

/// Majority-vote bundle of multiple u8 vectors.
/// This is the VSA superposition operator.
#[pyfunction]
pub fn bundle_u8(vectors: Vec<PyRef<PyNumArrayU8>>) -> PyResult<PyNumArrayU8> {
    let refs: Vec<&NumArrayU8> = vectors.iter().map(|v| &v.inner).collect();
    Ok(PyNumArrayU8 {
        inner: NumArrayU8::bundle(&refs),
    })
}

/// Hamming distance between two byte arrays (VPOPCNTDQ-accelerated).
#[pyfunction]
pub fn hamming_distance(a: Vec<u8>, b: Vec<u8>) -> PyResult<u64> {
    if a.len() != b.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Arrays must be same length",
        ));
    }
    Ok(numrus_core::simd::hamming_distance(&a, &b))
}

/// Batch Hamming: distances from query to each row in database.
/// database is flat, with num_rows rows of row_bytes each.
#[pyfunction]
pub fn hamming_batch(
    query: Vec<u8>,
    database: Vec<u8>,
    num_rows: usize,
    row_bytes: usize,
) -> PyResult<Vec<u64>> {
    if query.len() != row_bytes {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Query must be {} bytes, got {}",
            row_bytes,
            query.len()
        )));
    }
    if database.len() != num_rows * row_bytes {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Database must be {} bytes, got {}",
            num_rows * row_bytes,
            database.len()
        )));
    }
    Ok(numrus_core::simd::hamming_batch(
        &query, &database, num_rows, row_bytes,
    ))
}

/// Top-K nearest by Hamming distance. Returns (indices, distances).
#[pyfunction]
pub fn hamming_top_k(
    query: Vec<u8>,
    database: Vec<u8>,
    num_rows: usize,
    row_bytes: usize,
    k: usize,
) -> PyResult<(Vec<usize>, Vec<u64>)> {
    if query.len() != row_bytes {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Query must be {} bytes, got {}",
            row_bytes,
            query.len()
        )));
    }
    if database.len() != num_rows * row_bytes {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Database must be {} bytes, got {}",
            num_rows * row_bytes,
            database.len()
        )));
    }
    Ok(numrus_core::simd::hamming_top_k(
        &query, &database, num_rows, row_bytes, k,
    ))
}
