// bindings/python/src/cogrecord.rs
use crate::array_u8::PyNumArrayU8;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use numrus_rs::{sweep_cogrecords, CogRecord, SweepResult};

/// A 4-channel holographic container: meta + cam + btree + embed.
/// Each channel is 2048 bytes (16384 bits). Total: 8KB per record.
///
/// Used for content-addressable cognitive storage with cascade
/// rejection search (checks channels in order, early-exits on mismatch).
#[pyclass(module = "_numrus")]
#[derive(Clone)]
pub struct PyCogRecord {
    pub(crate) inner: CogRecord,
}

#[pymethods]
impl PyCogRecord {
    /// Create from 4 u8 arrays (each must be 2048 bytes).
    #[new]
    fn new(
        meta: PyRef<PyNumArrayU8>,
        cam: PyRef<PyNumArrayU8>,
        btree: PyRef<PyNumArrayU8>,
        embed: PyRef<PyNumArrayU8>,
    ) -> Self {
        PyCogRecord {
            inner: CogRecord::new(
                meta.inner.clone(),
                cam.inner.clone(),
                btree.inner.clone(),
                embed.inner.clone(),
            ),
        }
    }

    /// Create a zeroed CogRecord.
    #[staticmethod]
    fn zeros() -> Self {
        PyCogRecord {
            inner: CogRecord::zeros(),
        }
    }

    /// 4-channel Hamming distances: [meta_dist, cam_dist, btree_dist, embed_dist]
    fn hamming_4ch(&self, other: &PyCogRecord) -> [u64; 4] {
        self.inner.hamming_4ch(&other.inner)
    }

    /// Adaptive 4-channel sweep with early exit.
    /// Returns None if any channel exceeds its threshold.
    /// Returns [u64; 4] distances if all channels pass.
    fn sweep_adaptive(&self, other: &PyCogRecord, thresholds: [u64; 4]) -> Option<[u64; 4]> {
        self.inner.sweep_adaptive(&other.inner, thresholds)
    }

    /// Batch sweep: query against a flat database of N CogRecords (8192*N bytes).
    /// Returns list of (index, [u64; 4]) for records passing all thresholds.
    #[staticmethod]
    fn sweep_batch(
        query: &PyCogRecord,
        database: Vec<u8>,
        n: usize,
        thresholds: [u64; 4],
    ) -> PyResult<Vec<(usize, [u64; 4])>> {
        if database.len() != n * 8192 {
            return Err(PyValueError::new_err(format!(
                "Database must be n*8192={} bytes, got {}",
                n * 8192,
                database.len()
            )));
        }
        let results: Vec<SweepResult> = sweep_cogrecords(&query.inner, &database, n, thresholds);
        Ok(results
            .into_iter()
            .map(|r| (r.index, r.distances))
            .collect())
    }

    /// Serialize to bytes (8192 bytes).
    fn to_bytes(&self) -> Vec<u8> {
        self.inner.to_bytes()
    }

    /// Deserialize from bytes.
    #[staticmethod]
    fn from_bytes(data: Vec<u8>) -> PyResult<Self> {
        if data.len() < 8192 {
            return Err(PyValueError::new_err(format!(
                "CogRecord requires 8192 bytes, got {}",
                data.len()
            )));
        }
        Ok(PyCogRecord {
            inner: CogRecord::from_bytes(&data),
        })
    }

    /// Get a single channel as u8 array (0=meta, 1=cam, 2=btree, 3=embed).
    fn channel(&self, idx: usize) -> PyResult<PyNumArrayU8> {
        if idx > 3 {
            return Err(PyValueError::new_err("Channel index must be 0-3"));
        }
        Ok(PyNumArrayU8 {
            inner: self.inner.container(idx).clone(),
        })
    }

    /// HDR sweep: 4-channel compound early exit with VNNI cosine on EMBED.
    /// Returns list of (index, [4 distances], cosine_similarity) sorted by cosine.
    fn hdr_sweep(
        &self,
        database: Vec<u8>,
        n: usize,
        thresholds: [u64; 4],
    ) -> PyResult<Vec<(usize, [u64; 4], f64)>> {
        if database.len() != n * 8192 {
            return Err(PyValueError::new_err(format!(
                "Database must be n*8192={} bytes, got {}",
                n * 8192,
                database.len()
            )));
        }
        Ok(self.inner.hdr_sweep(&database, n, thresholds))
    }
}
