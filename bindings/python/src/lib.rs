// bindings/python/src/lib.rs

#![allow(non_local_definitions)]
#![allow(clippy::too_many_arguments)]

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

mod array_f32;
mod array_f64;
mod array_u8;
mod cogrecord;
mod functions;
mod int8_gemm;

use array_f32::PyNumArrayF32;
use array_f64::PyNumArrayF64;
use array_u8::PyNumArrayU8;
use cogrecord::PyCogRecord;

use functions::*;
use int8_gemm::*;

#[pymodule]
fn _numrus(_py: Python, m: &PyModule) -> PyResult<()> {
    // Array classes
    m.add_class::<PyNumArrayF32>()?;
    m.add_class::<PyNumArrayF64>()?;
    m.add_class::<PyNumArrayU8>()?;
    m.add_class::<PyCogRecord>()?;

    // f32 functions (existing)
    m.add_function(wrap_pyfunction!(zeros_f32, m)?)?;
    m.add_function(wrap_pyfunction!(ones_f32, m)?)?;
    m.add_function(wrap_pyfunction!(matmul_f32, m)?)?;
    m.add_function(wrap_pyfunction!(dot_f32, m)?)?;
    m.add_function(wrap_pyfunction!(arange_f32, m)?)?;
    m.add_function(wrap_pyfunction!(linspace_f32, m)?)?;
    m.add_function(wrap_pyfunction!(mean_f32, m)?)?;
    m.add_function(wrap_pyfunction!(median_f32, m)?)?;
    m.add_function(wrap_pyfunction!(min_f32, m)?)?;
    m.add_function(wrap_pyfunction!(min_axis_f32, m)?)?;
    m.add_function(wrap_pyfunction!(max_f32, m)?)?;
    m.add_function(wrap_pyfunction!(max_axis_f32, m)?)?;
    m.add_function(wrap_pyfunction!(exp_f32, m)?)?;
    m.add_function(wrap_pyfunction!(log_f32, m)?)?;
    m.add_function(wrap_pyfunction!(sigmoid_f32, m)?)?;
    m.add_function(wrap_pyfunction!(concatenate_f32, m)?)?;

    // f64 functions (existing)
    m.add_function(wrap_pyfunction!(zeros_f64, m)?)?;
    m.add_function(wrap_pyfunction!(ones_f64, m)?)?;
    m.add_function(wrap_pyfunction!(matmul_f64, m)?)?;
    m.add_function(wrap_pyfunction!(dot_f64, m)?)?;
    m.add_function(wrap_pyfunction!(arange_f64, m)?)?;
    m.add_function(wrap_pyfunction!(linspace_f64, m)?)?;
    m.add_function(wrap_pyfunction!(mean_f64, m)?)?;
    m.add_function(wrap_pyfunction!(median_f64, m)?)?;
    m.add_function(wrap_pyfunction!(min_f64, m)?)?;
    m.add_function(wrap_pyfunction!(min_axis_f64, m)?)?;
    m.add_function(wrap_pyfunction!(max_f64, m)?)?;
    m.add_function(wrap_pyfunction!(max_axis_f64, m)?)?;
    m.add_function(wrap_pyfunction!(exp_f64, m)?)?;
    m.add_function(wrap_pyfunction!(log_f64, m)?)?;
    m.add_function(wrap_pyfunction!(sigmoid_f64, m)?)?;
    m.add_function(wrap_pyfunction!(concatenate_f64, m)?)?;
    m.add_function(wrap_pyfunction!(norm_f32, m)?)?;
    m.add_function(wrap_pyfunction!(norm_f64, m)?)?;

    // HDC/VSA functions (NEW)
    m.add_function(wrap_pyfunction!(bundle_u8, m)?)?;
    m.add_function(wrap_pyfunction!(hamming_distance, m)?)?;
    m.add_function(wrap_pyfunction!(hamming_batch, m)?)?;
    m.add_function(wrap_pyfunction!(hamming_top_k, m)?)?;

    // INT8 GEMM functions (NEW)
    m.add_function(wrap_pyfunction!(quantize_f32_to_u8, m)?)?;
    m.add_function(wrap_pyfunction!(quantize_f32_to_i8, m)?)?;
    m.add_function(wrap_pyfunction!(int8_gemm_i32, m)?)?;
    m.add_function(wrap_pyfunction!(int8_gemm_f32, m)?)?;

    Ok(())
}
