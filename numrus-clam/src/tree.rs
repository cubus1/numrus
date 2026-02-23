//! CLAM Tree: divisive hierarchical clustering with LFD estimation.
//!
//! Implements Algorithm 1 (Partition) from CAKES (arXiv:2309.05491):
//!
//! ```text
//! 1. seeds ← random sample of ⌈√|C|⌉ points from C
//! 2. c     ← geometric median of seeds
//! 3. l     ← argmax f(c, x)  ∀x ∈ C          (left pole)
//! 4. r     ← argmax f(l, x)  ∀x ∈ C          (right pole)
//! 5. L     ← { x | f(l,x) ≤ f(r,x) }
//! 6. R     ← { x | f(r,x) < f(l,x) }
//! 7. recurse on L, R
//! ```
//!
//! After construction, the dataset is depth-first reordered so each cluster
//! is a contiguous slice `[offset..offset+cardinality]` — O(n) memory
//! instead of O(n log n) from storing index lists (CAKES §2.1.2).
//!
//! ## LFD (Local Fractal Dimension)
//!
//! Per-cluster LFD is computed during construction using Equation 2 from CAKES:
//!
//! ```text
//! LFD(q, r) = log₂( |B(q, r)| / |B(q, r/2)| )
//! ```
//!
//! where B(q, r) is the set of points in the cluster within radius r of center q.
//! LFD quantifies whether the data is locally spread (high LFD → hard to prune)
//! or locally concentrated (low LFD → easy to prune).

// ─────────────────────────────────────────────────────────────────────
// Distance trait
// ─────────────────────────────────────────────────────────────────────

/// Generic distance function. Any type implementing this can be used
/// for CLAM tree construction, search, and compression.
///
/// If `is_metric()` returns true, search is guaranteed exact via
/// the triangle inequality: d(x,z) ≤ d(x,y) + d(y,z).
pub trait Distance {
    /// The point type this distance operates on.
    type Point: ?Sized;

    /// Compute distance between two points. Must be:
    /// - Non-negative: f(x, y) ≥ 0
    /// - Symmetric:    f(x, y) = f(y, x)
    /// - Identity:     f(x, y) = 0 ⟺ x = y
    fn distance(&self, a: &Self::Point, b: &Self::Point) -> u64;

    /// Whether this distance function satisfies the triangle inequality.
    /// When true, all CAKES search algorithms are exact (perfect recall).
    fn is_metric(&self) -> bool;
}

/// Hamming distance on byte slices (fingerprints).
///
/// Uses the same 4× unrolled u64 XOR+POPCNT as numrus's
/// `hamming_chunk_inline` — compiler emits VPOPCNTDQ on AVX-512.
///
/// Hamming distance IS a metric: it satisfies the triangle inequality.
pub struct HammingDistance;

impl Distance for HammingDistance {
    type Point = [u8];

    fn distance(&self, a: &[u8], b: &[u8]) -> u64 {
        assert_eq!(a.len(), b.len(), "Hamming distance requires equal lengths");
        hamming_inline(a, b)
    }

    fn is_metric(&self) -> bool {
        true
    }
}

/// SIMD-accelerated Hamming distance using numrus-core's VPOPCNTDQ path.
///
/// Runtime dispatches to AVX-512 VPOPCNTDQ when available (64 bytes/iter,
/// ~8x scalar throughput), otherwise falls back to scalar POPCNT on u64 chunks.
///
/// This is an ADDITIONAL distance implementation alongside `HammingDistance`.
/// Use `HammingSIMD` when you want guaranteed hardware acceleration via
/// numrus-core's intrinsic path, vs `HammingDistance` which uses the
/// inline scalar path with compiler auto-vectorization.
pub struct HammingSIMD;

impl Distance for HammingSIMD {
    type Point = [u8];

    fn distance(&self, a: &[u8], b: &[u8]) -> u64 {
        assert_eq!(a.len(), b.len(), "Hamming distance requires equal lengths");
        numrus_core::simd::hamming_distance(a, b)
    }

    fn is_metric(&self) -> bool {
        true
    }
}

/// SIMD-accelerated batch Hamming distance using numrus-core.
///
/// Computes distances from `query` to each row in a flat database.
/// 4x unrolled for ILP — processes 4 database rows per outer iteration.
pub fn hamming_batch_simd(
    query: &[u8],
    database: &[u8],
    num_rows: usize,
    row_bytes: usize,
) -> Vec<u64> {
    numrus_core::simd::hamming_batch(query, database, num_rows, row_bytes)
}

/// SIMD-accelerated top-k Hamming search using numrus-core.
///
/// Returns (indices, distances) of the k closest rows in the database.
/// Uses partial sort (select_nth_unstable) for O(n) instead of O(n log n).
pub fn hamming_top_k_simd(
    query: &[u8],
    database: &[u8],
    num_rows: usize,
    row_bytes: usize,
    k: usize,
) -> (Vec<usize>, Vec<u64>) {
    numrus_core::simd::hamming_top_k(query, database, num_rows, row_bytes, k)
}

/// Hamming distance with 3-tier SIMD dispatch via numrus_core.
/// Dispatch: VPOPCNTDQ (AVX-512) → Harley-Seal (AVX2) → scalar POPCNT.
#[inline(always)]
pub(crate) fn hamming_inline(a: &[u8], b: &[u8]) -> u64 {
    numrus_core::simd::hamming_distance(a, b)
}

// ─────────────────────────────────────────────────────────────────────
// LFD (Local Fractal Dimension)
// ─────────────────────────────────────────────────────────────────────

/// Local Fractal Dimension of a cluster.
///
/// LFD = log₂(|B(c, r)| / |B(c, r/2)|)
///
/// - LFD ≈ 1: data is essentially 1D (a line/string)
/// - LFD ≈ 2: data fills a 2D surface
/// - LFD ≪ embedding_dim: manifold hypothesis holds, CLAM wins
/// - LFD ≈ embedding_dim: uniform distribution, CLAM degrades to linear
#[derive(Debug, Clone, Copy)]
pub struct Lfd {
    pub value: f64,
    /// Number of points within radius r of center.
    pub count_r: usize,
    /// Number of points within radius r/2 of center.
    pub count_half_r: usize,
}

impl Lfd {
    /// Compute LFD from ball cardinalities.
    ///
    /// Returns LFD = log₂(count_r / count_half_r).
    /// When count_half_r == 0 or count_r == count_half_r, returns 0.0
    /// (degenerate: all points are at the same distance from center).
    pub fn compute(count_r: usize, count_half_r: usize) -> Self {
        // Guard: div-by-zero when count_half_r == 0. The count_r <= count_half_r
        // case returns 0.0 (degenerate cluster: all points equidistant from center,
        // or too few points in the half-radius ball to estimate LFD).
        let value = if count_half_r == 0 || count_r <= count_half_r {
            0.0
        } else {
            (count_r as f64 / count_half_r as f64).log2()
        };
        Lfd {
            value,
            count_r,
            count_half_r,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Cluster node
// ─────────────────────────────────────────────────────────────────────

/// A node in the CLAM binary tree.
///
/// After depth-first reordering, the cluster's data points are the
/// contiguous slice `data[offset..offset+cardinality]` in the reordered
/// dataset. This means each cluster requires only 2 integers (offset +
/// cardinality) to identify its points — O(n) total, not O(n log n).
#[derive(Debug, Clone)]
pub struct Cluster {
    /// Index of this cluster's center in the reordered dataset.
    pub center_idx: usize,

    /// Maximum distance from center to any point in this cluster.
    pub radius: u64,

    /// Number of points in this cluster.
    pub cardinality: usize,

    /// Start offset in the depth-first reordered dataset.
    pub offset: usize,

    /// Depth in the tree (root = 0).
    pub depth: usize,

    /// Local Fractal Dimension at this cluster's scale.
    pub lfd: Lfd,

    /// Left child index in the tree's node array, or None if leaf.
    pub left: Option<usize>,

    /// Right child index in the tree's node array, or None if leaf.
    pub right: Option<usize>,
}

impl Cluster {
    /// Whether this cluster is a leaf (no children).
    #[inline]
    pub fn is_leaf(&self) -> bool {
        self.left.is_none()
    }

    /// δ⁺ = f(q, center) + radius — the theoretically farthest point.
    /// If δ⁺ ≤ search_radius, the entire cluster is inside the query ball.
    ///
    /// (Figure 1, CAKES arXiv:2309.05491)
    #[inline]
    pub fn delta_plus(&self, dist_to_center: u64) -> u64 {
        dist_to_center.saturating_add(self.radius)
    }

    /// δ⁻ = max(0, f(q, center) − radius) — the theoretically closest point.
    /// If δ⁻ > search_radius, no point in this cluster can be a hit → prune.
    ///
    /// (Figure 1, CAKES arXiv:2309.05491)
    #[inline]
    pub fn delta_minus(&self, dist_to_center: u64) -> u64 {
        dist_to_center.saturating_sub(self.radius)
    }
}

// ─────────────────────────────────────────────────────────────────────
// CLAM Tree
// ─────────────────────────────────────────────────────────────────────

/// Stopping criteria for tree construction.
#[derive(Debug, Clone)]
pub struct BuildConfig {
    /// Stop splitting when cluster has ≤ this many points.
    pub min_cardinality: usize,
    /// Stop splitting at this tree depth.
    pub max_depth: usize,
    /// Stop splitting when cluster radius ≤ this.
    pub min_radius: u64,
}

impl Default for BuildConfig {
    fn default() -> Self {
        BuildConfig {
            min_cardinality: 1,
            max_depth: 256,
            min_radius: 0,
        }
    }
}

/// The CLAM tree: a divisive hierarchical clustering over a dataset.
///
/// ```text
/// ┌──────────────────── root ─────────────────────┐
/// │  center=42, radius=8192, cardinality=1000000  │
/// │  LFD=1.7, offset=0                            │
/// ├────────────┬─────────────────────────────────── │
/// │  left      │  right                            │
/// │  c=17      │  c=893                            │
/// │  r=4096    │  r=5100                           │
/// │  n=600000  │  n=400000                         │
/// │  LFD=1.3   │  LFD=2.1                         │
/// └────────────┴───────────────────────────────────┘
/// ```
///
/// After construction, the dataset indices are reordered depth-first so
/// each cluster is a contiguous range. The original data is NOT moved —
/// only an index permutation is stored.
pub struct ClamTree {
    /// All cluster nodes, stored flat. nodes[0] is the root.
    pub nodes: Vec<Cluster>,

    /// Depth-first permutation: reordered[i] = original dataset index.
    /// After reordering, cluster at offset..offset+cardinality maps to
    /// reordered[offset..offset+cardinality] in the original dataset.
    pub reordered: Vec<usize>,

    /// Number of leaf clusters (= metric entropy 𝒩 at mean leaf radius).
    pub num_leaves: usize,

    /// Mean radius of leaf clusters.
    pub mean_leaf_radius: f64,
}

impl ClamTree {
    /// Build a CLAM tree over a flat database of fixed-length vectors.
    ///
    /// # Arguments
    /// * `data`     — flat byte buffer: `data[i*vec_len..(i+1)*vec_len]` is point i
    /// * `vec_len`  — length of each vector in bytes (e.g., 2048 for 16K-bit fingerprints)
    /// * `count`    — number of vectors in the database
    /// * `config`   — stopping criteria
    ///
    /// # Algorithm (CAKES Algorithm 1)
    ///
    /// ```text
    /// 1. seeds ← random ⌈√n⌉ points
    /// 2. center ← geometric median of seeds (point minimizing Σ distances)
    /// 3. left_pole ← farthest from center
    /// 4. right_pole ← farthest from left_pole
    /// 5. L ← points closer to left_pole
    /// 6. R ← points closer to right_pole
    /// 7. recurse
    /// ```
    pub fn build(data: &[u8], vec_len: usize, count: usize, config: &BuildConfig) -> Self {
        assert_eq!(data.len(), vec_len * count);

        let mut indices: Vec<usize> = (0..count).collect();
        let mut nodes = Vec::with_capacity(2 * count); // upper bound
        let mut rng = numrus_core::SplitMix64::new(0xDEAD_BEEF_CAFE_BABE);

        Self::partition(
            data,
            vec_len,
            &mut indices,
            0,     // start
            count, // end
            0,     // depth
            config,
            &mut nodes,
            &mut rng,
        );

        // Compute summary statistics
        let mut num_leaves = 0usize;
        let mut leaf_radius_sum = 0u64;
        for node in &nodes {
            if node.is_leaf() {
                num_leaves += 1;
                leaf_radius_sum += node.radius;
            }
        }
        let mean_leaf_radius = if num_leaves > 0 {
            leaf_radius_sum as f64 / num_leaves as f64
        } else {
            0.0
        };

        ClamTree {
            nodes,
            reordered: indices,
            num_leaves,
            mean_leaf_radius,
        }
    }

    /// Recursive partition (Algorithm 1 from CAKES).
    ///
    /// Operates on `indices[start..end]` — the subset of dataset indices
    /// belonging to this cluster. Modifies `indices` in place to achieve
    /// depth-first ordering.
    #[allow(clippy::too_many_arguments)]
    fn partition(
        data: &[u8],
        vec_len: usize,
        indices: &mut [usize],
        start: usize,
        end: usize,
        depth: usize,
        config: &BuildConfig,
        nodes: &mut Vec<Cluster>,
        rng: &mut numrus_core::SplitMix64,
    ) -> usize {
        let n = end - start;
        let node_idx = nodes.len();

        // ── Step 1: Find center (geometric median of √n seeds) ──
        let num_seeds = (n as f64).sqrt().ceil() as usize;
        let num_seeds = num_seeds.max(1).min(n);

        // Fisher-Yates partial shuffle for seed selection
        let working = &mut indices[start..end];
        for i in 0..num_seeds.min(working.len()) {
            let j = i + (rng.next_u64() as usize % (working.len() - i));
            working.swap(i, j);
        }

        // Geometric median: point among seeds minimizing sum of distances
        let center_local = if num_seeds <= 1 {
            0
        } else {
            let mut best_idx = 0;
            let mut best_sum = u64::MAX;
            for s in 0..num_seeds {
                let si = working[s];
                let si_data = &data[si * vec_len..(si + 1) * vec_len];
                let mut sum = 0u64;
                for t in 0..num_seeds {
                    if s != t {
                        let ti = working[t];
                        let ti_data = &data[ti * vec_len..(ti + 1) * vec_len];
                        sum += hamming_inline(si_data, ti_data);
                    }
                }
                if sum < best_sum {
                    best_sum = sum;
                    best_idx = s;
                }
            }
            best_idx
        };

        // Move center to front
        working.swap(0, center_local);
        let center_idx = working[0];
        let center_data = &data[center_idx * vec_len..(center_idx + 1) * vec_len];

        // ── Step 2: Compute radius + find left pole (farthest from center) ──
        let mut radius = 0u64;
        let mut left_pole_local = 0;
        let mut left_pole_dist = 0u64;

        // Also compute LFD while scanning all points
        let mut distances: Vec<u64> = Vec::with_capacity(n);
        for i in 0..n {
            let pi = working[i];
            let pi_data = &data[pi * vec_len..(pi + 1) * vec_len];
            let d = hamming_inline(center_data, pi_data);
            distances.push(d);
            if d > radius {
                radius = d;
            }
            if d > left_pole_dist {
                left_pole_dist = d;
                left_pole_local = i;
            }
        }

        // ── Compute LFD: count points within r and r/2 ──
        let half_radius = radius / 2;
        let count_r = distances.iter().filter(|&&d| d <= radius).count();
        let count_half_r = distances.iter().filter(|&&d| d <= half_radius).count();
        let lfd = Lfd::compute(count_r, count_half_r);

        // ── Step 3: Find right pole (farthest from left pole) ──
        let left_pole_idx = working[left_pole_local];
        let left_pole_data = &data[left_pole_idx * vec_len..(left_pole_idx + 1) * vec_len];

        let mut right_pole_local = 0;
        let mut right_pole_dist = 0u64;
        for i in 0..n {
            let pi = working[i];
            let pi_data = &data[pi * vec_len..(pi + 1) * vec_len];
            let d = hamming_inline(left_pole_data, pi_data);
            if d > right_pole_dist {
                right_pole_dist = d;
                right_pole_local = i;
            }
        }
        let right_pole_idx = working[right_pole_local];
        let right_pole_data = &data[right_pole_idx * vec_len..(right_pole_idx + 1) * vec_len];

        // ── Step 4: Partition into L (closer to left) and R (closer to right) ──
        // Use in-place partitioning (like quicksort pivot)
        // Tag each point: dist_to_left ≤ dist_to_right → left, else right
        let mut side: Vec<bool> = Vec::with_capacity(n); // true = left
        for i in 0..n {
            let pi = working[i];
            let pi_data = &data[pi * vec_len..(pi + 1) * vec_len];
            let dl = hamming_inline(left_pole_data, pi_data);
            let dr = hamming_inline(right_pole_data, pi_data);
            side.push(dl <= dr); // ties go left (as per CAKES Algorithm 1)
        }

        // Dutch-flag partition: lefts to front, rights to back
        let mut cursor = 0;
        for i in 0..n {
            if side[i] {
                working.swap(cursor, i);
                side.swap(cursor, i);
                cursor += 1;
            }
        }
        let split = cursor; // working[0..split] = left, working[split..n] = right

        // ── Create this node ──
        nodes.push(Cluster {
            center_idx,
            radius,
            cardinality: n,
            offset: start,
            depth,
            lfd,
            left: None,
            right: None,
        });

        // ── Step 5: Recurse if criteria met ──
        let should_split = n > config.min_cardinality
            && depth < config.max_depth
            && radius > config.min_radius
            && split > 0
            && split < n;

        if should_split {
            // Left child
            let left_idx = Self::partition(
                data,
                vec_len,
                indices,
                start,
                start + split,
                depth + 1,
                config,
                nodes,
                rng,
            );
            nodes[node_idx].left = Some(left_idx);

            // Right child
            let right_idx = Self::partition(
                data,
                vec_len,
                indices,
                start + split,
                end,
                depth + 1,
                config,
                nodes,
                rng,
            );
            nodes[node_idx].right = Some(right_idx);
        }

        node_idx
    }

    /// Get the root cluster.
    pub fn root(&self) -> &Cluster {
        &self.nodes[0]
    }

    /// Get the data slice for a cluster from the original dataset.
    /// Returns an iterator of (original_index, data_slice) pairs.
    pub fn cluster_points<'a>(
        &'a self,
        cluster: &Cluster,
        data: &'a [u8],
        vec_len: usize,
    ) -> impl Iterator<Item = (usize, &'a [u8])> + 'a {
        let start = cluster.offset;
        let end = start + cluster.cardinality;
        self.reordered[start..end].iter().map(move |&orig_idx| {
            (
                orig_idx,
                &data[orig_idx * vec_len..(orig_idx + 1) * vec_len],
            )
        })
    }

    /// Get the center point data.
    pub fn center_data<'a>(&self, cluster: &Cluster, data: &'a [u8], vec_len: usize) -> &'a [u8] {
        &data[cluster.center_idx * vec_len..(cluster.center_idx + 1) * vec_len]
    }

    /// Get LFD statistics across the tree.
    pub fn lfd_percentiles(&self) -> LfdStats {
        let mut lfds: Vec<f64> = self.nodes.iter().map(|c| c.lfd.value).collect();
        lfds.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = lfds.len();
        if n == 0 {
            return LfdStats::default();
        }

        LfdStats {
            min: lfds[0],
            p5: lfds[n * 5 / 100],
            p25: lfds[n * 25 / 100],
            p50: lfds[n / 2],
            p75: lfds[n * 75 / 100],
            p95: lfds[n * 95 / 100],
            max: lfds[n - 1],
            mean: lfds.iter().sum::<f64>() / n as f64,
        }
    }

    /// Get LFD values by depth (for plotting like Figure 2 in CAKES).
    pub fn lfd_by_depth(&self) -> Vec<(usize, Vec<f64>)> {
        let max_depth = self.nodes.iter().map(|c| c.depth).max().unwrap_or(0);
        let mut by_depth: Vec<Vec<f64>> = vec![Vec::new(); max_depth + 1];

        for node in &self.nodes {
            // Weight by cardinality (as per CAKES Figure 2 caption)
            for _ in 0..node.cardinality {
                by_depth[node.depth].push(node.lfd.value);
            }
        }

        by_depth
            .into_iter()
            .enumerate()
            .filter(|(_, v)| !v.is_empty())
            .collect()
    }
}

/// Summary statistics of LFD across the tree.
#[derive(Debug, Clone, Default)]
pub struct LfdStats {
    pub min: f64,
    pub p5: f64,
    pub p25: f64,
    pub p50: f64,
    pub p75: f64,
    pub p95: f64,
    pub max: f64,
    pub mean: f64,
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use numrus_core::SplitMix64;

    /// Create a simple dataset of 100 random 64-byte vectors.
    fn make_test_data(n: usize, vec_len: usize) -> Vec<u8> {
        let mut rng = SplitMix64::new(42);
        let mut data = vec![0u8; n * vec_len];
        for byte in data.iter_mut() {
            *byte = (rng.next_u64() & 0xFF) as u8;
        }
        data
    }

    #[test]
    fn test_hamming_inline_identical() {
        let a = vec![0xAA; 2048];
        let d = hamming_inline(&a, &a);
        assert_eq!(d, 0);
    }

    #[test]
    fn test_hamming_inline_all_different() {
        let a = vec![0xFF; 8];
        let b = vec![0x00; 8];
        let d = hamming_inline(&a, &b);
        assert_eq!(d, 64); // 8 bytes × 8 bits
    }

    #[test]
    fn test_lfd_compute() {
        // 100 points within r, 25 within r/2 → LFD = log₂(4) = 2.0
        let lfd = Lfd::compute(100, 25);
        assert!((lfd.value - 2.0).abs() < 1e-10);

        // All at same distance → LFD = 0
        let lfd = Lfd::compute(100, 100);
        assert_eq!(lfd.value, 0.0);

        // Empty inner ball → LFD = 0 (degenerate)
        let lfd = Lfd::compute(100, 0);
        assert_eq!(lfd.value, 0.0);
    }

    #[test]
    fn test_delta_bounds() {
        let c = Cluster {
            center_idx: 0,
            radius: 100,
            cardinality: 50,
            offset: 0,
            depth: 0,
            lfd: Lfd::compute(50, 25),
            left: None,
            right: None,
        };

        // Query at distance 500 from center
        assert_eq!(c.delta_plus(500), 600); // 500 + 100
        assert_eq!(c.delta_minus(500), 400); // 500 - 100

        // Query at distance 50 from center (inside cluster)
        assert_eq!(c.delta_plus(50), 150); // 50 + 100
        assert_eq!(c.delta_minus(50), 0); // max(0, 50 - 100) = 0
    }

    #[test]
    fn test_build_tree_basic() {
        let vec_len = 64;
        let count = 100;
        let data = make_test_data(count, vec_len);

        let config = BuildConfig {
            min_cardinality: 1,
            max_depth: 50,
            min_radius: 0,
        };

        let tree = ClamTree::build(&data, vec_len, count, &config);

        // Root should contain all points
        assert_eq!(tree.root().cardinality, count);
        assert_eq!(tree.root().depth, 0);
        assert_eq!(tree.reordered.len(), count);

        // Every original index should appear exactly once
        let mut seen = vec![false; count];
        for &idx in &tree.reordered {
            assert!(!seen[idx], "duplicate index {}", idx);
            seen[idx] = true;
        }

        // Should have leaf nodes
        assert!(tree.num_leaves > 0);
        println!(
            "Tree: {} nodes, {} leaves, mean leaf radius: {:.1}",
            tree.nodes.len(),
            tree.num_leaves,
            tree.mean_leaf_radius
        );
    }

    #[test]
    fn test_build_tree_singleton_leaves() {
        let vec_len = 32;
        let count = 16;
        let data = make_test_data(count, vec_len);

        let config = BuildConfig::default(); // min_cardinality = 1

        let tree = ClamTree::build(&data, vec_len, count, &config);

        // With min_cardinality=1, most leaves should be singletons
        let singleton_count = tree
            .nodes
            .iter()
            .filter(|c| c.is_leaf() && c.cardinality == 1)
            .count();
        assert!(singleton_count > 0);
    }

    #[test]
    fn test_lfd_statistics() {
        let vec_len = 64;
        let count = 200;
        let data = make_test_data(count, vec_len);

        let config = BuildConfig {
            min_cardinality: 5,
            max_depth: 20,
            min_radius: 0,
        };

        let tree = ClamTree::build(&data, vec_len, count, &config);
        let stats = tree.lfd_percentiles();

        println!(
            "LFD stats: min={:.2} p25={:.2} p50={:.2} p75={:.2} max={:.2}",
            stats.min, stats.p25, stats.p50, stats.p75, stats.max
        );

        // LFD should be non-negative
        assert!(stats.min >= 0.0);
    }

    #[test]
    fn test_lfd_by_depth() {
        let vec_len = 64;
        let count = 100;
        let data = make_test_data(count, vec_len);

        let config = BuildConfig {
            min_cardinality: 2,
            max_depth: 15,
            min_radius: 0,
        };

        let tree = ClamTree::build(&data, vec_len, count, &config);
        let lfd_depths = tree.lfd_by_depth();

        // Should have LFD data at multiple depths
        assert!(!lfd_depths.is_empty());

        for (depth, lfds) in &lfd_depths {
            println!(
                "Depth {}: {} points, median LFD = {:.2}",
                depth,
                lfds.len(),
                lfds[lfds.len() / 2]
            );
        }
    }

    // ── HammingSIMD tests (require avx512 or avx2 feature) ──

    #[test]
    fn test_hamming_simd_identical() {
        let a = vec![0xAA; 2048];
        let dist = HammingSIMD;
        assert_eq!(dist.distance(&a, &a), 0);
    }

    #[test]
    fn test_hamming_simd_all_different() {
        let a = vec![0xFF; 8];
        let b = vec![0x00; 8];
        let dist = HammingSIMD;
        assert_eq!(dist.distance(&a, &b), 64);
    }

    #[test]
    fn test_hamming_simd_matches_inline() {
        // Verify SIMD path produces identical results to scalar path
        let mut rng = SplitMix64::new(12345);
        let a: Vec<u8> = (0..2048).map(|_| (rng.next_u64() & 0xFF) as u8).collect();
        let b: Vec<u8> = (0..2048).map(|_| (rng.next_u64() & 0xFF) as u8).collect();

        let inline_result = hamming_inline(&a, &b);
        let simd_result = HammingSIMD.distance(&a, &b);
        assert_eq!(
            inline_result, simd_result,
            "SIMD and inline Hamming must agree: inline={} simd={}",
            inline_result, simd_result
        );
    }

    #[test]
    fn test_hamming_simd_various_sizes() {
        let dist = HammingSIMD;
        // Test sizes: 8, 64, 128, 2048, 8192 bytes
        for &size in &[8, 64, 128, 2048, 8192] {
            let a = vec![0xFF; size];
            let b = vec![0x00; size];
            let expected = (size * 8) as u64;
            assert_eq!(
                dist.distance(&a, &b),
                expected,
                "size={}: expected {} got {}",
                size,
                expected,
                dist.distance(&a, &b)
            );
        }
    }

    #[test]
    fn test_hamming_batch_simd() {
        let vec_len = 64;
        let count = 50;
        let data = make_test_data(count, vec_len);
        let query = &data[0..vec_len]; // first vector as query

        let distances = hamming_batch_simd(query, &data, count, vec_len);
        assert_eq!(distances.len(), count);
        assert_eq!(distances[0], 0); // self-distance = 0
                                     // All distances should be non-negative (they're u64, so always true)
                                     // And should match inline computation
        for i in 0..count {
            let row = &data[i * vec_len..(i + 1) * vec_len];
            let expected = hamming_inline(query, row);
            assert_eq!(
                distances[i], expected,
                "batch distance mismatch at row {}: batch={} inline={}",
                i, distances[i], expected
            );
        }
    }

    #[test]
    #[cfg(any(feature = "avx512", feature = "avx2"))]
    fn test_hamming_top_k_simd() {
        let vec_len = 64;
        let count = 50;
        let data = make_test_data(count, vec_len);
        let query = &data[0..vec_len];

        let k = 5;
        let (indices, distances) = hamming_top_k_simd(query, &data, count, vec_len, k);
        assert_eq!(indices.len(), k);
        assert_eq!(distances.len(), k);
        assert_eq!(indices[0], 0); // query itself should be closest
        assert_eq!(distances[0], 0);

        // Distances should be sorted ascending
        for i in 1..k {
            assert!(
                distances[i] >= distances[i - 1],
                "top-k distances not sorted: d[{}]={} < d[{}]={}",
                i,
                distances[i],
                i - 1,
                distances[i - 1]
            );
        }
    }
}
