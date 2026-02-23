// bindings/python/src/array_u8.rs
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};
use numrus_rs::NumArrayU8;

#[pyclass(module = "_numrus")]
#[derive(Clone)]
pub struct PyNumArrayU8 {
    pub(crate) inner: NumArrayU8,
}

#[pymethods]
impl PyNumArrayU8 {
    #[new]
    fn new(data: Vec<u8>, shape: Option<Vec<usize>>) -> Self {
        let inner = match shape {
            Some(s) => NumArrayU8::new_with_shape(data, s),
            None => NumArrayU8::new(data),
        };
        PyNumArrayU8 { inner }
    }

    fn add_scalar(&self, scalar: u8) -> PyResult<PyNumArrayU8> {
        Python::with_gil(|_py| {
            let result = &self.inner + scalar; // Leveraging Rust's Add implementation
            Ok(PyNumArrayU8 { inner: result })
        })
    }

    fn add_array(&self, other: PyRef<PyNumArrayU8>) -> PyResult<PyNumArrayU8> {
        Python::with_gil(|_py| {
            let result = &self.inner + &other.inner; // Leveraging Rust's Add implementation
            Ok(PyNumArrayU8 { inner: result })
        })
    }

    fn sub_scalar(&self, scalar: u8) -> PyResult<PyNumArrayU8> {
        Python::with_gil(|_py| {
            let result = &self.inner - scalar; // Leveraging Rust's Add implementation
            Ok(PyNumArrayU8 { inner: result })
        })
    }

    fn sub_array(&self, other: PyRef<PyNumArrayU8>) -> PyResult<PyNumArrayU8> {
        Python::with_gil(|_py| {
            let result = &self.inner - &other.inner; // Leveraging Rust's Add implementation
            Ok(PyNumArrayU8 { inner: result })
        })
    }

    fn mul_scalar(&self, scalar: u8) -> PyResult<PyNumArrayU8> {
        Python::with_gil(|_py| {
            let result = &self.inner * scalar; // Leveraging Rust's Add implementation
            Ok(PyNumArrayU8 { inner: result })
        })
    }

    fn mul_array(&self, other: PyRef<PyNumArrayU8>) -> PyResult<PyNumArrayU8> {
        Python::with_gil(|_py| {
            let result = &self.inner * &other.inner; // Leveraging Rust's Add implementation
            Ok(PyNumArrayU8 { inner: result })
        })
    }

    fn div_scalar(&self, scalar: u8) -> PyResult<PyNumArrayU8> {
        Python::with_gil(|_py| {
            let result = &self.inner / scalar; // Leveraging Rust's Add implementation
            Ok(PyNumArrayU8 { inner: result })
        })
    }

    fn div_array(&self, other: PyRef<PyNumArrayU8>) -> PyResult<PyNumArrayU8> {
        Python::with_gil(|_py| {
            let result = &self.inner / &other.inner; // Leveraging Rust's Add implementation
            Ok(PyNumArrayU8 { inner: result })
        })
    }

    fn tolist(&self, py: Python) -> PyObject {
        let list = PyList::new(py, self.inner.get_data());
        list.into()
    }

    fn slice(&self, axis: usize, start: usize, end: usize) -> PyResult<PyNumArrayU8> {
        Ok(PyNumArrayU8 {
            inner: self.inner.slice(axis, start, end),
        })
    }

    fn shape(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let shape_vec = self.inner.shape();
            Ok(PyTuple::new(py, shape_vec.iter()).to_object(py))
        })
    }

    fn reshape(&self, shape: Vec<usize>) -> PyResult<PyNumArrayU8> {
        Ok(PyNumArrayU8 {
            inner: self.inner.reshape(&shape),
        })
    }

    fn flip_axis(&self, axis: Option<&PyList>) -> PyResult<PyNumArrayU8> {
        Python::with_gil(|_py| {
            let axis_vec: Vec<usize> = match axis {
                Some(list) => list.extract()?,
                None => vec![],
            };
            let result = if axis_vec.is_empty() {
                self.inner.clone()
            } else {
                self.inner.flip_axis(axis_vec)
            };
            Ok(PyNumArrayU8 { inner: result })
        })
    }

    fn __imul__(&mut self, scalar: u8) -> PyResult<()> {
        self.inner = &self.inner * scalar;
        Ok(())
    }

    // === HDC/VSA Operations ===

    /// XOR bind: a ^ b (element-wise). Self-inverse: bind(bind(a,b), b) == a
    fn bind(&self, other: PyRef<PyNumArrayU8>) -> PyResult<PyNumArrayU8> {
        Ok(PyNumArrayU8 {
            inner: self.inner.bind(&other.inner),
        })
    }

    /// Permute: circular bit rotation by k positions.
    fn permute(&self, k: usize) -> PyResult<PyNumArrayU8> {
        Ok(PyNumArrayU8 {
            inner: self.inner.permute(k),
        })
    }

    /// Hamming distance (VPOPCNTDQ-accelerated on AVX-512 CPUs).
    fn hamming_distance(&self, other: PyRef<PyNumArrayU8>) -> PyResult<u64> {
        Ok(self.inner.hamming_distance(&other.inner))
    }

    /// Population count: number of set bits.
    fn popcount(&self) -> PyResult<u64> {
        Ok(self.inner.popcount())
    }

    /// Signed dot product (interprets u8 as i8).
    fn dot_i8(&self, other: PyRef<PyNumArrayU8>) -> PyResult<i64> {
        Ok(self.inner.dot_i8(&other.inner))
    }

    /// Cosine similarity (signed i8 interpretation).
    fn cosine_i8(&self, other: PyRef<PyNumArrayU8>) -> PyResult<f64> {
        Ok(self.inner.cosine_i8(&other.inner))
    }

    /// Adaptive Hamming: returns None if distance exceeds threshold (early exit).
    fn hamming_distance_adaptive(
        &self,
        other: PyRef<PyNumArrayU8>,
        threshold: u64,
    ) -> PyResult<Option<u64>> {
        Ok(self
            .inner
            .hamming_distance_adaptive(&other.inner, threshold))
    }

    /// Adaptive Hamming batch search: returns vec of (index, distance) for matches below threshold.
    fn hamming_search_adaptive(
        &self,
        database: PyRef<PyNumArrayU8>,
        vec_len: usize,
        count: usize,
        threshold: u64,
    ) -> PyResult<Vec<(usize, u64)>> {
        if database.inner.len() != vec_len * count {
            return Err(PyValueError::new_err(format!(
                "database length {} != vec_len * count = {}",
                database.inner.len(),
                vec_len * count
            )));
        }
        if self.inner.len() != vec_len {
            return Err(PyValueError::new_err(format!(
                "query length {} != vec_len = {}",
                self.inner.len(),
                vec_len
            )));
        }
        Ok(self
            .inner
            .hamming_search_adaptive(&database.inner, vec_len, count, threshold))
    }

    /// Adaptive cosine search: returns vec of (index, similarity) for matches above min_similarity.
    fn cosine_search_adaptive(
        &self,
        database: PyRef<PyNumArrayU8>,
        vec_len: usize,
        count: usize,
        min_similarity: f64,
    ) -> PyResult<Vec<(usize, f64)>> {
        Ok(self
            .inner
            .cosine_search_adaptive(&database.inner, vec_len, count, min_similarity))
    }

    /// HDR cascade search: 3-stroke adaptive with cosine precision tier.
    /// Returns list of (index, hamming_distance, cosine_similarity).
    fn hdr_search(
        &self,
        database: PyRef<PyNumArrayU8>,
        vec_len: usize,
        count: usize,
        threshold: u64,
    ) -> PyResult<Vec<(usize, u64, f64)>> {
        if database.inner.len() != vec_len * count {
            return Err(PyValueError::new_err(format!(
                "database length {} != vec_len * count = {}",
                database.inner.len(),
                vec_len * count
            )));
        }
        if self.inner.len() != vec_len {
            return Err(PyValueError::new_err(format!(
                "query length {} != vec_len = {}",
                self.inner.len(),
                vec_len
            )));
        }
        Ok(self
            .inner
            .hdr_search(&database.inner, vec_len, count, threshold))
    }

    /// HDR cascade search with f32 dequantization precision tier.
    /// Dequantizes finalists using scale/zero_point, then computes f32 cosine.
    /// Returns list of (index, hamming_distance, cosine_similarity).
    fn hdr_search_f32(
        &self,
        database: PyRef<PyNumArrayU8>,
        vec_len: usize,
        count: usize,
        threshold: u64,
        scale: f32,
        zero_point: i32,
    ) -> PyResult<Vec<(usize, u64, f64)>> {
        if database.inner.len() != vec_len * count {
            return Err(PyValueError::new_err(format!(
                "database length {} != vec_len * count = {}",
                database.inner.len(),
                vec_len * count
            )));
        }
        if self.inner.len() != vec_len {
            return Err(PyValueError::new_err(format!(
                "query length {} != vec_len = {}",
                self.inner.len(),
                vec_len
            )));
        }
        Ok(self.inner.hdr_search_f32(
            &database.inner,
            vec_len,
            count,
            threshold,
            scale,
            zero_point,
        ))
    }

    /// HDR cascade search with XOR Delta + INT8 residual precision tier.
    /// Blends Hamming and INT8 cosine with delta_weight (0.0=pure Hamming, 1.0=pure INT8).
    /// Returns list of (index, hamming_distance, blended_similarity).
    fn hdr_search_delta(
        &self,
        database: PyRef<PyNumArrayU8>,
        vec_len: usize,
        count: usize,
        threshold: u64,
        delta_weight: f32,
    ) -> PyResult<Vec<(usize, u64, f64)>> {
        if database.inner.len() != vec_len * count {
            return Err(PyValueError::new_err(format!(
                "database length {} != vec_len * count = {}",
                database.inner.len(),
                vec_len * count
            )));
        }
        if self.inner.len() != vec_len {
            return Err(PyValueError::new_err(format!(
                "query length {} != vec_len = {}",
                self.inner.len(),
                vec_len
            )));
        }
        if !(0.0..=1.0).contains(&delta_weight) {
            return Err(PyValueError::new_err(format!(
                "delta_weight {} must be in [0.0, 1.0]",
                delta_weight
            )));
        }
        Ok(self
            .inner
            .hdr_search_delta(&database.inner, vec_len, count, threshold, delta_weight))
    }
}
