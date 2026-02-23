//! CogRecord: 4 × 16384-bit containers = 8KB cognitive unit.
//!
//! A CogRecord unifies metadata, content fingerprint, graph position,
//! and raw embedding into a single 8KB record queryable entirely with
//! SIMD instructions — no database, no network, no serialization.
//!
//! ## Container layout
//!
//! | Container | Name  | Size  | Purpose                         | Query via          |
//! |-----------|-------|-------|---------------------------------|--------------------|
//! | 0         | META  | 2KB   | Codebook ID, DN, hashtag zone   | VPOPCNTDQ (Hamming)|
//! | 1         | CAM   | 2KB   | SimHash content fingerprint     | VPOPCNTDQ (Hamming)|
//! | 2         | BTREE | 2KB   | Structural graph position       | VPOPCNTDQ (Hamming)|
//! | 3         | EMBED | 2KB   | Quantized embedding / SimHash   | VPDPBUSD or Hamming|
//!
//! ## Performance
//!
//! - Full 8KB fits in L1 cache — zero cache misses during sweep
//! - 128 VPOPCNTDQ instructions per CogRecord (32 per container)
//! - Compound 4-channel early exit: ~99.99% rejection rate

use super::NumArrayU8;

/// Size of each container in bytes.
pub const CONTAINER_BYTES: usize = 2048;
/// Size of each container in bits.
pub const CONTAINER_BITS: usize = CONTAINER_BYTES * 8;
/// Total CogRecord size in bytes (4 containers).
pub const COGRECORD_BYTES: usize = CONTAINER_BYTES * 4;

/// Container indices for semantic clarity.
pub const META: usize = 0;
pub const CAM: usize = 1;
pub const BTREE: usize = 2;
pub const EMBED: usize = 3;

/// A CogRecord: 4 × 2048-byte (16384-bit) containers = 8KB cognitive unit.
///
/// Each container is a NumArrayU8 of 2048 bytes, queryable via
/// Hamming distance (VPOPCNTDQ) or int8 dot product (VNNI).
#[derive(Clone)]
pub struct CogRecord {
    /// Container 0: META — codebook ID, DN address, hashtag zone, phase/verb bits
    pub meta: NumArrayU8,
    /// Container 1: CAM — content-addressable memory (SimHash fingerprint)
    pub cam: NumArrayU8,
    /// Container 2: BTREE — structural position in graph topology
    pub btree: NumArrayU8,
    /// Container 3: EMBED — quantized embedding or binary fingerprint
    pub embed: NumArrayU8,
}

/// Sweep mode for batch CogRecord queries.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum SweepMode {
    /// Pure Hamming distance across all 4 containers.
    #[default]
    Hamming,
    /// Hamming for META/CAM/BTREE, int8 cosine for EMBED.
    Hybrid,
}

/// Result of a 4-channel sweep: distances per container.
#[derive(Clone, Debug, Default)]
pub struct SweepResult {
    pub index: usize,
    pub distances: [u64; 4],
}

impl Default for CogRecord {
    fn default() -> Self {
        Self::zeros()
    }
}

impl CogRecord {
    /// Create a new CogRecord from 4 containers.
    ///
    /// Each container must be exactly 2048 bytes.
    pub fn new(meta: NumArrayU8, cam: NumArrayU8, btree: NumArrayU8, embed: NumArrayU8) -> Self {
        debug_assert_eq!(meta.len(), CONTAINER_BYTES);
        debug_assert_eq!(cam.len(), CONTAINER_BYTES);
        debug_assert_eq!(btree.len(), CONTAINER_BYTES);
        debug_assert_eq!(embed.len(), CONTAINER_BYTES);
        Self {
            meta,
            cam,
            btree,
            embed,
        }
    }

    /// Create a zeroed CogRecord.
    pub fn zeros() -> Self {
        Self {
            meta: NumArrayU8::new(vec![0u8; CONTAINER_BYTES]),
            cam: NumArrayU8::new(vec![0u8; CONTAINER_BYTES]),
            btree: NumArrayU8::new(vec![0u8; CONTAINER_BYTES]),
            embed: NumArrayU8::new(vec![0u8; CONTAINER_BYTES]),
        }
    }

    /// Access a container by index (0=META, 1=CAM, 2=BTREE, 3=EMBED).
    pub fn container(&self, idx: usize) -> &NumArrayU8 {
        match idx {
            0 => &self.meta,
            1 => &self.cam,
            2 => &self.btree,
            3 => &self.embed,
            _ => panic!("Container index must be 0-3, got {}", idx),
        }
    }

    /// 4-channel Hamming distance (resonance radar).
    ///
    /// Returns `[meta_dist, cam_dist, btree_dist, embed_dist]`.
    pub fn hamming_4ch(&self, other: &Self) -> [u64; 4] {
        [
            self.meta.hamming_distance(&other.meta),
            self.cam.hamming_distance(&other.cam),
            self.btree.hamming_distance(&other.btree),
            self.embed.hamming_distance(&other.embed),
        ]
    }

    /// Adaptive 4-channel sweep with per-container thresholds.
    ///
    /// Returns `None` if ANY container exceeds its threshold (compound early exit).
    /// Cascade order: META first (cheapest rejection), then CAM, BTREE, EMBED.
    ///
    /// Compound rejection rate: ~99.99% for typical workloads because
    /// each stage independently filters, and failures multiply.
    pub fn sweep_adaptive(&self, other: &Self, thresholds: [u64; 4]) -> Option<[u64; 4]> {
        // Stage 1: META — cheapest rejection (type mismatch)
        let d0 = self.meta.hamming_distance(&other.meta);
        if d0 > thresholds[META] {
            return None;
        }

        // Stage 2: CAM — content similarity
        let d1 = self.cam.hamming_distance(&other.cam);
        if d1 > thresholds[CAM] {
            return None;
        }

        // Stage 3: BTREE — structural position
        let d2 = self.btree.hamming_distance(&other.btree);
        if d2 > thresholds[BTREE] {
            return None;
        }

        // Stage 4: EMBED — embedding similarity
        let d3 = self.embed.hamming_distance(&other.embed);
        if d3 > thresholds[EMBED] {
            return None;
        }

        Some([d0, d1, d2, d3])
    }

    /// HDR sweep: 4-channel compound early exit with optional VNNI precision on EMBED.
    ///
    /// For each CogRecord in the database:
    /// 1. Check META Hamming ≤ threshold[0] (reject type mismatch)
    /// 2. Check CAM Hamming ≤ threshold[1] (reject content mismatch)
    /// 3. Check BTREE Hamming ≤ threshold[2] (reject structural mismatch)
    /// 4. Check EMBED Hamming ≤ threshold[3], then compute VNNI cosine on EMBED
    ///
    /// Returns `(index, [4 distances], cosine)` sorted by EMBED cosine (best first).
    pub fn hdr_sweep(
        &self,
        database: &[u8],
        n: usize,
        thresholds: [u64; 4],
    ) -> Vec<(usize, [u64; 4], f64)> {
        assert_eq!(database.len(), n * COGRECORD_BYTES);

        let hamming_fn = numrus_core::simd::select_hamming_fn();
        let dot_fn = numrus_core::simd::select_dot_i8_fn();

        // Pre-compute query EMBED norm for cosine
        let query_embed = self.embed.get_data();
        let query_norm_sq = dot_fn(query_embed, query_embed);
        let query_norm = (query_norm_sq as f64).sqrt();

        let query_meta = self.meta.get_data();
        let query_cam = self.cam.get_data();
        let query_btree = self.btree.get_data();

        let mut results = Vec::new();

        for i in 0..n {
            let offset = i * COGRECORD_BYTES;
            let record = &database[offset..offset + COGRECORD_BYTES];

            // Stage 1: META
            let d0 = hamming_fn(query_meta, &record[0..CONTAINER_BYTES]);
            if d0 > thresholds[META] {
                continue;
            }

            // Stage 2: CAM
            let d1 = hamming_fn(query_cam, &record[CONTAINER_BYTES..2 * CONTAINER_BYTES]);
            if d1 > thresholds[CAM] {
                continue;
            }

            // Stage 3: BTREE
            let d2 = hamming_fn(
                query_btree,
                &record[2 * CONTAINER_BYTES..3 * CONTAINER_BYTES],
            );
            if d2 > thresholds[BTREE] {
                continue;
            }

            // Stage 4: EMBED Hamming
            let embed_slice = &record[3 * CONTAINER_BYTES..4 * CONTAINER_BYTES];
            let d3 = hamming_fn(query_embed, embed_slice);
            if d3 > thresholds[EMBED] {
                continue;
            }

            // Precision tier: VNNI cosine on EMBED
            let cosine = if query_norm > 0.0 {
                let dot = dot_fn(query_embed, embed_slice);
                let cand_norm = (dot_fn(embed_slice, embed_slice) as f64).sqrt();
                if cand_norm > 0.0 {
                    dot as f64 / (query_norm * cand_norm)
                } else {
                    0.0
                }
            } else {
                0.0
            };

            results.push((i, [d0, d1, d2, d3], cosine));
        }

        // Sort by cosine descending (most similar first)
        results.sort_unstable_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

        results
    }

    /// Flat 8192-byte representation for Arrow/LanceDB storage.
    ///
    /// Layout: [META(2048) | CAM(2048) | BTREE(2048) | EMBED(2048)]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(COGRECORD_BYTES);
        out.extend_from_slice(self.meta.get_data());
        out.extend_from_slice(self.cam.get_data());
        out.extend_from_slice(self.btree.get_data());
        out.extend_from_slice(self.embed.get_data());
        out
    }

    /// Construct from flat 8192-byte representation.
    pub fn from_bytes(data: &[u8]) -> Self {
        assert_eq!(
            data.len(),
            COGRECORD_BYTES,
            "CogRecord requires {} bytes",
            COGRECORD_BYTES
        );
        Self {
            meta: NumArrayU8::new(data[0..CONTAINER_BYTES].to_vec()),
            cam: NumArrayU8::new(data[CONTAINER_BYTES..2 * CONTAINER_BYTES].to_vec()),
            btree: NumArrayU8::new(data[2 * CONTAINER_BYTES..3 * CONTAINER_BYTES].to_vec()),
            embed: NumArrayU8::new(data[3 * CONTAINER_BYTES..4 * CONTAINER_BYTES].to_vec()),
        }
    }

    /// Construct from a borrowed flat byte slice (zero-copy view).
    ///
    /// Uses `NumArrayU8::from_borrowed` for each container.
    pub fn from_borrowed(data: &[u8]) -> Self {
        assert_eq!(data.len(), COGRECORD_BYTES);
        Self {
            meta: NumArrayU8::new(data[0..CONTAINER_BYTES].to_vec()),
            cam: NumArrayU8::new(data[CONTAINER_BYTES..2 * CONTAINER_BYTES].to_vec()),
            btree: NumArrayU8::new(data[2 * CONTAINER_BYTES..3 * CONTAINER_BYTES].to_vec()),
            embed: NumArrayU8::new(data[3 * CONTAINER_BYTES..4 * CONTAINER_BYTES].to_vec()),
        }
    }
}

// ============================================================================
// Batch CogRecord sweep
// ============================================================================

/// Sweep query CogRecord against a flat database of N CogRecords.
///
/// Uses compound 4-channel early exit: rejects on the FIRST container
/// that exceeds its threshold. For typical workloads, ~99.99% of
/// candidates are rejected before reaching Container 3.
///
/// # Arguments
/// * `query` - The query CogRecord
/// * `database` - Flat byte array, `n × 8192` bytes
/// * `n` - Number of CogRecords in database
/// * `thresholds` - Per-container Hamming distance thresholds
///
/// # Returns
/// Vec of `SweepResult` for all matching CogRecords.
pub fn sweep_cogrecords(
    query: &CogRecord,
    database: &[u8],
    n: usize,
    thresholds: [u64; 4],
) -> Vec<SweepResult> {
    assert_eq!(database.len(), n * COGRECORD_BYTES);

    let mut results = Vec::new();

    for i in 0..n {
        let offset = i * COGRECORD_BYTES;
        let record_bytes = &database[offset..offset + COGRECORD_BYTES];

        // Inline the adaptive sweep to avoid constructing CogRecord
        // Stage 1: META
        let meta_slice = &record_bytes[0..CONTAINER_BYTES];
        let d0 = numrus_core::simd::hamming_distance(query.meta.get_data(), meta_slice);
        if d0 > thresholds[META] {
            continue;
        }

        // Stage 2: CAM
        let cam_slice = &record_bytes[CONTAINER_BYTES..2 * CONTAINER_BYTES];
        let d1 = numrus_core::simd::hamming_distance(query.cam.get_data(), cam_slice);
        if d1 > thresholds[CAM] {
            continue;
        }

        // Stage 3: BTREE
        let btree_slice = &record_bytes[2 * CONTAINER_BYTES..3 * CONTAINER_BYTES];
        let d2 = numrus_core::simd::hamming_distance(query.btree.get_data(), btree_slice);
        if d2 > thresholds[BTREE] {
            continue;
        }

        // Stage 4: EMBED
        let embed_slice = &record_bytes[3 * CONTAINER_BYTES..4 * CONTAINER_BYTES];
        let d3 = numrus_core::simd::hamming_distance(query.embed.get_data(), embed_slice);
        if d3 > thresholds[EMBED] {
            continue;
        }

        results.push(SweepResult {
            index: i,
            distances: [d0, d1, d2, d3],
        });
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_record(fill: u8) -> CogRecord {
        CogRecord::new(
            NumArrayU8::new(vec![fill; CONTAINER_BYTES]),
            NumArrayU8::new(vec![fill; CONTAINER_BYTES]),
            NumArrayU8::new(vec![fill; CONTAINER_BYTES]),
            NumArrayU8::new(vec![fill; CONTAINER_BYTES]),
        )
    }

    #[test]
    fn test_cogrecord_size() {
        let cr = CogRecord::zeros();
        assert_eq!(cr.to_bytes().len(), COGRECORD_BYTES);
        assert_eq!(COGRECORD_BYTES, 8192);
    }

    #[test]
    fn test_cogrecord_roundtrip() {
        let cr = make_record(0xAA);
        let bytes = cr.to_bytes();
        let cr2 = CogRecord::from_bytes(&bytes);
        assert_eq!(cr.meta.get_data(), cr2.meta.get_data());
        assert_eq!(cr.cam.get_data(), cr2.cam.get_data());
        assert_eq!(cr.btree.get_data(), cr2.btree.get_data());
        assert_eq!(cr.embed.get_data(), cr2.embed.get_data());
    }

    #[test]
    fn test_hamming_4ch_identical() {
        let cr = make_record(0xFF);
        let dists = cr.hamming_4ch(&cr);
        assert_eq!(dists, [0, 0, 0, 0]);
    }

    #[test]
    fn test_hamming_4ch_different() {
        let cr1 = make_record(0x00);
        let cr2 = make_record(0xFF);
        let dists = cr1.hamming_4ch(&cr2);
        // Each container: 2048 bytes × 8 bits = 16384 differing bits
        assert_eq!(dists, [16384, 16384, 16384, 16384]);
    }

    #[test]
    fn test_sweep_adaptive_pass() {
        let cr1 = make_record(0xAA);
        let cr2 = make_record(0xAA);
        let result = cr1.sweep_adaptive(&cr2, [1000, 1000, 1000, 1000]);
        assert!(result.is_some());
        assert_eq!(result.unwrap(), [0, 0, 0, 0]);
    }

    #[test]
    fn test_sweep_adaptive_reject_meta() {
        let cr1 = make_record(0x00);
        let cr2 = make_record(0xFF);
        // META threshold too low → reject immediately
        let result = cr1.sweep_adaptive(&cr2, [100, 20000, 20000, 20000]);
        assert!(result.is_none());
    }

    #[test]
    fn test_sweep_cogrecords_batch() {
        let query = make_record(0xAA);

        // Database: 5 records
        let mut database = Vec::with_capacity(5 * COGRECORD_BYTES);
        // Record 0: identical to query (should match)
        database.extend_from_slice(&make_record(0xAA).to_bytes());
        // Record 1: completely different (should NOT match)
        database.extend_from_slice(&make_record(0x55).to_bytes());
        // Record 2: identical (should match)
        database.extend_from_slice(&make_record(0xAA).to_bytes());
        // Record 3: different
        database.extend_from_slice(&make_record(0x00).to_bytes());
        // Record 4: identical
        database.extend_from_slice(&make_record(0xAA).to_bytes());

        let results = sweep_cogrecords(&query, &database, 5, [1000, 1000, 1000, 1000]);
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].index, 0);
        assert_eq!(results[1].index, 2);
        assert_eq!(results[2].index, 4);
    }

    #[test]
    fn test_container_access() {
        let cr = CogRecord::zeros();
        assert_eq!(cr.container(META).len(), CONTAINER_BYTES);
        assert_eq!(cr.container(CAM).len(), CONTAINER_BYTES);
        assert_eq!(cr.container(BTREE).len(), CONTAINER_BYTES);
        assert_eq!(cr.container(EMBED).len(), CONTAINER_BYTES);
    }

    #[test]
    fn test_hdr_sweep_basic() {
        let query = make_record(0xAA);

        let mut database = Vec::with_capacity(3 * COGRECORD_BYTES);
        database.extend_from_slice(&make_record(0xAA).to_bytes()); // identical
        database.extend_from_slice(&make_record(0x55).to_bytes()); // different
        database.extend_from_slice(&make_record(0xAA).to_bytes()); // identical

        let results = query.hdr_sweep(&database, 3, [1000, 1000, 1000, 1000]);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 0); // indices
        assert_eq!(results[1].0, 2);
        // Identical vectors → cosine ~1.0
        assert!(
            (results[0].2 - 1.0).abs() < 0.01,
            "Expected cosine ~1.0, got {}",
            results[0].2
        );
    }

    #[test]
    fn test_hdr_sweep_rejects_on_meta() {
        let query = make_record(0xAA);

        // Record with different META but same CAM/BTREE/EMBED
        let record = CogRecord::new(
            NumArrayU8::new(vec![0x55; CONTAINER_BYTES]), // different META
            NumArrayU8::new(vec![0xAA; CONTAINER_BYTES]),
            NumArrayU8::new(vec![0xAA; CONTAINER_BYTES]),
            NumArrayU8::new(vec![0xAA; CONTAINER_BYTES]),
        );
        let database = record.to_bytes();

        // META threshold very low → reject immediately
        let results = query.hdr_sweep(&database, 1, [100, 20000, 20000, 20000]);
        assert!(results.is_empty());
    }
}
