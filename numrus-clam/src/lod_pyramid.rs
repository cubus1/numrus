//! LOD (Level of Detail) Pyramid for hierarchical bitplane search.
//!
//! Multi-resolution OR-reduce summaries of binary fingerprints enable
//! early rejection during CLAM tree traversal:
//!
//! - Coarsest level: OR of all fingerprints in cluster → superset mask
//! - Finer levels: OR of spatial 2×2 quadrants
//! - Finest level: individual fingerprints
//!
//! ## Rejection rule
//!
//! For query Q and cluster OR-mask M:
//!
//! ```text
//! lower_bound = popcount(Q & ~M)
//! ```
//!
//! Bits set in Q but absent from every cluster member can never match.
//! If `lower_bound > threshold`, the entire cluster is pruned.
//!
//! ## Complexity
//!
//! For a 16K×16K grid of 2048-byte fingerprints:
//! - Level 0: 16384×16384 (268M cells, 549 GB)
//! - Level 1: 8192×8192 (67M cells, 137 GB)
//! - ...
//! - Level 14: 1×1 (single OR summary)
//!
//! Total pyramid overhead: ~1/3 of base data (geometric series 1/4 + 1/16 + ...).

// ─────────────────────────────────────────────────────────────────────
// LodLevel — one resolution layer
// ─────────────────────────────────────────────────────────────────────

/// One level of the LOD pyramid: a row-major grid of OR-reduced fingerprints.
#[derive(Debug, Clone)]
pub struct LodLevel {
    /// OR-reduced fingerprints, row-major: `data[cell_index * vec_bytes..]`.
    pub data: Vec<u8>,
    /// Number of rows at this level.
    pub rows: usize,
    /// Number of columns at this level.
    pub cols: usize,
    /// Bytes per fingerprint.
    pub vec_bytes: usize,
}

impl LodLevel {
    /// Get the fingerprint for the cell at (row, col).
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> &[u8] {
        let idx = row * self.cols + col;
        &self.data[idx * self.vec_bytes..(idx + 1) * self.vec_bytes]
    }

    /// Total number of cells at this level.
    #[inline]
    pub fn num_cells(&self) -> usize {
        self.rows * self.cols
    }
}

/// OR-reduce a 2D grid by factor 2 in each dimension.
///
/// Adjacent 2×2 blocks of fingerprints are OR'd together.
/// If the grid has odd dimensions, the last row/col is carried forward
/// as-is (ceil division).
pub fn or_reduce_2d(level: &LodLevel) -> LodLevel {
    let new_rows = level.rows.div_ceil(2);
    let new_cols = level.cols.div_ceil(2);
    let vb = level.vec_bytes;
    let mut data = vec![0u8; new_rows * new_cols * vb];

    for r in 0..new_rows {
        for c in 0..new_cols {
            let dst_offset = (r * new_cols + c) * vb;
            let dst = &mut data[dst_offset..dst_offset + vb];

            // Top-left (always exists)
            let r0 = r * 2;
            let c0 = c * 2;
            dst.copy_from_slice(level.get(r0, c0));

            // Top-right
            if c0 + 1 < level.cols {
                let src = level.get(r0, c0 + 1);
                for i in 0..vb {
                    dst[i] |= src[i];
                }
            }

            // Bottom-left
            if r0 + 1 < level.rows {
                let src = level.get(r0 + 1, c0);
                for i in 0..vb {
                    dst[i] |= src[i];
                }
            }

            // Bottom-right
            if r0 + 1 < level.rows && c0 + 1 < level.cols {
                let src = level.get(r0 + 1, c0 + 1);
                for i in 0..vb {
                    dst[i] |= src[i];
                }
            }
        }
    }

    LodLevel {
        data,
        rows: new_rows,
        cols: new_cols,
        vec_bytes: vb,
    }
}

// ─────────────────────────────────────────────────────────────────────
// LodPyramid — full multi-resolution stack
// ─────────────────────────────────────────────────────────────────────

/// Full LOD pyramid from finest (level 0) to coarsest (level N).
///
/// Level 0 is the original data grid. Each subsequent level halves both
/// dimensions via `or_reduce_2d`. Construction stops when both dimensions
/// reach 1.
#[derive(Debug, Clone)]
pub struct LodPyramid {
    /// Levels from finest (index 0) to coarsest (last index).
    pub levels: Vec<LodLevel>,
    /// Bytes per fingerprint (same at all levels).
    pub vec_bytes: usize,
}

impl LodPyramid {
    /// Build a full LOD pyramid from a flat row-major grid of fingerprints.
    ///
    /// `data` must have length `rows * cols * vec_bytes`.
    pub fn build(data: &[u8], rows: usize, cols: usize, vec_bytes: usize) -> Self {
        assert_eq!(data.len(), rows * cols * vec_bytes);

        let base = LodLevel {
            data: data.to_vec(),
            rows,
            cols,
            vec_bytes,
        };

        let mut levels = vec![base];

        while levels.last().unwrap().rows > 1 || levels.last().unwrap().cols > 1 {
            let coarser = or_reduce_2d(levels.last().unwrap());
            levels.push(coarser);
        }

        LodPyramid { levels, vec_bytes }
    }

    /// The OR of all fingerprints (coarsest level, single cell).
    pub fn root_or(&self) -> &[u8] {
        self.levels.last().unwrap().get(0, 0)
    }

    /// Number of pyramid levels (including the base).
    pub fn num_levels(&self) -> usize {
        self.levels.len()
    }

    /// Compute a lower bound on the Hamming distance between `query` and
    /// any fingerprint in the cell at `(row, col)` at the given `level`.
    ///
    /// Uses the OR-mask rejection rule: bits set in the query but absent
    /// from the OR-mask cannot contribute to a match.
    #[inline]
    pub fn lower_bound(&self, query: &[u8], level: usize, row: usize, col: usize) -> u64 {
        let mask = self.levels[level].get(row, col);
        or_mask_lower_bound(query, mask)
    }
}

/// Compute popcount(query & ~mask) — the number of query bits not covered
/// by the OR-mask. This is a lower bound on Hamming distance to any
/// fingerprint contributing to the mask.
#[inline]
pub fn or_mask_lower_bound(query: &[u8], or_mask: &[u8]) -> u64 {
    let mut count = 0u64;
    // Process 8 bytes at a time for POPCNT throughput
    let chunks = query.len() / 8;
    for i in 0..chunks {
        let base = i * 8;
        let q = u64::from_ne_bytes(query[base..base + 8].try_into().unwrap());
        let m = u64::from_ne_bytes(or_mask[base..base + 8].try_into().unwrap());
        count += (q & !m).count_ones() as u64;
    }
    for i in chunks * 8..query.len() {
        count += (query[i] & !or_mask[i]).count_ones() as u64;
    }
    count
}

// ─────────────────────────────────────────────────────────────────────
// LodAnnotation — per-cluster metadata for CLAM tree search
// ─────────────────────────────────────────────────────────────────────

/// Per-cluster LOD annotation for CLAM tree integration.
///
/// The OR-mask is the bitwise OR of all fingerprints in the cluster.
/// It serves as a superset filter: if a query bit is not set in the
/// OR-mask, no cluster member has that bit, guaranteeing a minimum
/// Hamming distance contribution.
#[derive(Debug, Clone)]
pub struct LodAnnotation {
    /// Bitwise OR of all fingerprints in this cluster.
    pub or_mask: Vec<u8>,
    /// popcount(or_mask) — useful for density estimation.
    pub or_popcount: u64,
    /// Number of fingerprints summarized.
    pub cardinality: usize,
}

impl LodAnnotation {
    /// Build an annotation from a set of fingerprints.
    pub fn from_fingerprints(
        fingerprints: impl Iterator<Item = impl AsRef<[u8]>>,
        vec_bytes: usize,
    ) -> Self {
        let mut or_mask = vec![0u8; vec_bytes];
        let mut cardinality = 0usize;

        for fp in fingerprints {
            let fp = fp.as_ref();
            for i in 0..vec_bytes {
                or_mask[i] |= fp[i];
            }
            cardinality += 1;
        }

        let or_popcount = or_mask.iter().map(|b| b.count_ones() as u64).sum();

        LodAnnotation {
            or_mask,
            or_popcount,
            cardinality,
        }
    }

    /// Lower bound on Hamming distance from `query` to any fingerprint
    /// in this cluster.
    #[inline]
    pub fn hamming_lower_bound(&self, query: &[u8]) -> u64 {
        or_mask_lower_bound(query, &self.or_mask)
    }
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_or_reduce_2x2() {
        // 2×2 grid, 4 bytes per fingerprint
        let data = vec![
            0b1000_0000,
            0,
            0,
            0, // (0,0)
            0b0100_0000,
            0,
            0,
            0, // (0,1)
            0b0010_0000,
            0,
            0,
            0, // (1,0)
            0b0001_0000,
            0,
            0,
            0, // (1,1)
        ];
        let level = LodLevel {
            data,
            rows: 2,
            cols: 2,
            vec_bytes: 4,
        };
        let reduced = or_reduce_2d(&level);
        assert_eq!(reduced.rows, 1);
        assert_eq!(reduced.cols, 1);
        assert_eq!(reduced.get(0, 0), &[0b1111_0000, 0, 0, 0]);
    }

    #[test]
    fn test_or_reduce_odd_dims() {
        // 3×3 grid, 1 byte per fingerprint
        let data = vec![0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0xFF];
        let level = LodLevel {
            data,
            rows: 3,
            cols: 3,
            vec_bytes: 1,
        };
        let reduced = or_reduce_2d(&level);
        assert_eq!(reduced.rows, 2);
        assert_eq!(reduced.cols, 2);
        // (0,0) = OR(0x01, 0x02, 0x08, 0x10) = 0x1B
        assert_eq!(reduced.get(0, 0), &[0x1B]);
        // (0,1) = OR(0x04, 0x20) = 0x24 (no right neighbor for col 2)
        assert_eq!(reduced.get(0, 1), &[0x24]);
        // (1,0) = OR(0x40, 0x80) = 0xC0 (no bottom neighbor for row 2)
        assert_eq!(reduced.get(1, 0), &[0xC0]);
        // (1,1) = 0xFF (corner, no neighbors)
        assert_eq!(reduced.get(1, 1), &[0xFF]);
    }

    #[test]
    fn test_lod_pyramid_build() {
        // 4×4 grid, 2 bytes per fingerprint
        let mut data = vec![0u8; 4 * 4 * 2];
        // Set some bits
        data[0] = 0xFF; // (0,0) byte 0
        data[6 * 2 + 1] = 0xAA; // (1,2) byte 1

        let pyramid = LodPyramid::build(&data, 4, 4, 2);
        assert_eq!(pyramid.num_levels(), 3); // 4×4 → 2×2 → 1×1
        assert_eq!(pyramid.levels[0].rows, 4);
        assert_eq!(pyramid.levels[1].rows, 2);
        assert_eq!(pyramid.levels[2].rows, 1);
    }

    #[test]
    fn test_or_mask_lower_bound() {
        let query = vec![0xFF, 0xFF]; // all bits set
        let mask = vec![0x0F, 0xF0]; // half the bits
                                     // Lower bound = popcount(0xFF & ~0x0F, 0xFF & ~0xF0)
                                     //             = popcount(0xF0, 0x0F) = 4 + 4 = 8
        assert_eq!(or_mask_lower_bound(&query, &mask), 8);
    }

    #[test]
    fn test_or_mask_lower_bound_exact_match() {
        let query = vec![0xAA, 0x55];
        let mask = vec![0xFF, 0xFF]; // superset
                                     // All query bits covered → lower bound = 0
        assert_eq!(or_mask_lower_bound(&query, &mask), 0);
    }

    #[test]
    fn test_lod_annotation_from_fingerprints() {
        let fps: Vec<Vec<u8>> = vec![
            vec![0b1010_0000, 0x00],
            vec![0b0101_0000, 0x00],
            vec![0b0000_0000, 0xFF],
        ];
        let ann = LodAnnotation::from_fingerprints(fps.iter(), 2);
        assert_eq!(ann.or_mask, vec![0b1111_0000, 0xFF]);
        assert_eq!(ann.cardinality, 3);
        assert_eq!(ann.or_popcount, 4 + 8); // 4 bits in first byte, 8 in second
    }

    #[test]
    fn test_lod_annotation_lower_bound() {
        let fps: Vec<Vec<u8>> = vec![vec![0x0F], vec![0xF0]];
        let ann = LodAnnotation::from_fingerprints(fps.iter(), 1);
        assert_eq!(ann.or_mask, vec![0xFF]);

        // Query fully covered by OR mask → lower bound = 0
        assert_eq!(ann.hamming_lower_bound(&[0xAA]), 0);

        // OR mask is 0xFF, covers everything → bound is always 0
        assert_eq!(ann.hamming_lower_bound(&[0xFF]), 0);
    }

    #[test]
    fn test_lod_annotation_lower_bound_rejects() {
        let fps: Vec<Vec<u8>> = vec![
            vec![0x0F], // only low 4 bits
        ];
        let ann = LodAnnotation::from_fingerprints(fps.iter(), 1);
        assert_eq!(ann.or_mask, vec![0x0F]);

        // Query has high bits set → lower bound = 4
        assert_eq!(ann.hamming_lower_bound(&[0xF0]), 4);
    }

    #[test]
    fn test_pyramid_lower_bound() {
        // 2×2 grid, 1 byte per fingerprint
        let data = vec![0x0F, 0xF0, 0x33, 0xCC];
        let pyramid = LodPyramid::build(&data, 2, 2, 1);

        // Root OR = 0x0F | 0xF0 | 0x33 | 0xCC = 0xFF
        assert_eq!(pyramid.root_or(), &[0xFF]);
        // Root lower bound for any query = 0 (all bits covered)
        assert_eq!(pyramid.lower_bound(&[0xFF], 1, 0, 0), 0);

        // Level 0 cell (0,0) = 0x0F → query 0xF0 has lower bound 4
        assert_eq!(pyramid.lower_bound(&[0xF0], 0, 0, 0), 4);
    }
}
