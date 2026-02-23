//! CLAM-accelerated holographic search with LOD OR-mask pruning.
//!
//! Combines the CLAM tree's hierarchical clustering (triangle inequality
//! pruning) with LOD OR-mask rejection for binary fingerprints.
//!
//! ## Algorithm
//!
//! ```text
//! 1. Build LodIndex: OR-mask for every cluster in the CLAM tree
//! 2. Search: priority queue traversal of tree
//!    a. For each cluster, compute OR-mask lower bound
//!    b. If lower bound > current k-th best, prune entire subtree
//!    c. At leaves, compute exact Hamming distances
//! 3. Return top-k results
//! ```
//!
//! The OR-mask lower bound is tighter than the triangle inequality alone
//! because it exploits bit-level structure: bits set in the query but
//! absent from the OR-mask guarantee a minimum distance contribution.

use std::collections::{BinaryHeap, HashMap};

use crate::lod_pyramid::{or_mask_lower_bound, LodAnnotation};
use crate::tree::ClamTree;

// ─────────────────────────────────────────────────────────────────────
// LodIndex — sidecar annotation for CLAM trees
// ─────────────────────────────────────────────────────────────────────

/// Sidecar index mapping CLAM tree node indices to their LOD annotations.
///
/// The index is separate from the tree to avoid modifying numrus-clam.
/// Build once after tree construction, then reuse for all queries.
#[derive(Debug, Clone)]
pub struct LodIndex {
    annotations: HashMap<usize, LodAnnotation>,
}

impl LodIndex {
    /// Build LOD annotations for every cluster in the tree.
    ///
    /// Walks the tree bottom-up: leaf annotations are computed from their
    /// fingerprints, internal annotations are the OR of their children.
    pub fn build(tree: &ClamTree, data: &[u8], vec_bytes: usize) -> Self {
        let mut annotations = HashMap::with_capacity(tree.nodes.len());
        Self::build_recursive(tree, data, vec_bytes, 0, &mut annotations);
        LodIndex { annotations }
    }

    fn build_recursive(
        tree: &ClamTree,
        data: &[u8],
        vec_bytes: usize,
        node_idx: usize,
        annotations: &mut HashMap<usize, LodAnnotation>,
    ) {
        let node = &tree.nodes[node_idx];

        if node.is_leaf() {
            // Leaf: OR all fingerprints in this cluster
            let fps = tree.cluster_points(node, data, vec_bytes);
            let ann = LodAnnotation::from_fingerprints(fps.map(|(_, fp)| fp), vec_bytes);
            annotations.insert(node_idx, ann);
        } else {
            // Internal: recurse on children, then OR children's masks
            if let Some(left_idx) = node.left {
                Self::build_recursive(tree, data, vec_bytes, left_idx, annotations);
            }
            if let Some(right_idx) = node.right {
                Self::build_recursive(tree, data, vec_bytes, right_idx, annotations);
            }

            // OR children's masks together
            let mut or_mask = vec![0u8; vec_bytes];
            let mut cardinality = 0;

            if let Some(left_idx) = node.left {
                if let Some(left_ann) = annotations.get(&left_idx) {
                    for i in 0..vec_bytes {
                        or_mask[i] |= left_ann.or_mask[i];
                    }
                    cardinality += left_ann.cardinality;
                }
            }
            if let Some(right_idx) = node.right {
                if let Some(right_ann) = annotations.get(&right_idx) {
                    for i in 0..vec_bytes {
                        or_mask[i] |= right_ann.or_mask[i];
                    }
                    cardinality += right_ann.cardinality;
                }
            }

            let or_popcount = or_mask.iter().map(|b| b.count_ones() as u64).sum();
            annotations.insert(
                node_idx,
                LodAnnotation {
                    or_mask,
                    or_popcount,
                    cardinality,
                },
            );
        }
    }

    /// Get annotation for a node.
    pub fn get(&self, node_idx: usize) -> Option<&LodAnnotation> {
        self.annotations.get(&node_idx)
    }

    /// Number of annotated nodes.
    pub fn len(&self) -> usize {
        self.annotations.len()
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.annotations.is_empty()
    }
}

// ─────────────────────────────────────────────────────────────────────
// Search result
// ─────────────────────────────────────────────────────────────────────

/// Result from LOD-accelerated k-NN search.
#[derive(Debug, Clone)]
pub struct LodSearchResult {
    /// Index in the original (reordered) dataset.
    pub index: usize,
    /// Exact Hamming distance to the query.
    pub distance: u64,
}

/// Statistics from a search — useful for tuning and profiling.
#[derive(Debug, Clone, Default)]
pub struct LodSearchStats {
    /// Number of tree nodes visited.
    pub nodes_visited: usize,
    /// Number of nodes pruned by OR-mask lower bound.
    pub nodes_pruned_lod: usize,
    /// Number of nodes pruned by triangle inequality.
    pub nodes_pruned_triangle: usize,
    /// Number of leaf points with exact distance computed.
    pub exact_distances_computed: usize,
}

// ─────────────────────────────────────────────────────────────────────
// K-NN search
// ─────────────────────────────────────────────────────────────────────

/// Priority queue entry for tree traversal.
#[derive(Debug)]
struct SearchEntry {
    node_idx: usize,
    lower_bound: u64,
}

impl PartialEq for SearchEntry {
    fn eq(&self, other: &Self) -> bool {
        self.lower_bound == other.lower_bound
    }
}

impl Eq for SearchEntry {}

impl PartialOrd for SearchEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SearchEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Min-heap: smallest lower_bound first
        other.lower_bound.cmp(&self.lower_bound)
    }
}

/// K-NN search using CLAM tree + LOD OR-mask pruning.
///
/// Dual pruning strategy:
/// 1. **LOD OR-mask**: `popcount(query & ~cluster_or)` lower bound
/// 2. **Triangle inequality**: `d(q, center) - radius` lower bound
///
/// The tighter of the two bounds is used for each node.
pub fn lod_knn_search(
    query: &[u8],
    tree: &ClamTree,
    index: &LodIndex,
    data: &[u8],
    vec_bytes: usize,
    k: usize,
) -> (Vec<LodSearchResult>, LodSearchStats) {
    let hamming_fn = numrus_core::simd::select_hamming_fn();
    let mut stats = LodSearchStats::default();

    // Top-k results, sorted by distance (max-heap for efficient eviction)
    let mut results: BinaryHeap<(u64, usize)> = BinaryHeap::new();
    let mut kth_distance = u64::MAX;

    // Priority queue of nodes to explore (min-heap by lower bound)
    let mut queue: BinaryHeap<SearchEntry> = BinaryHeap::new();

    // Start at root
    queue.push(SearchEntry {
        node_idx: 0,
        lower_bound: 0,
    });

    while let Some(entry) = queue.pop() {
        // Prune: if this node's lower bound exceeds current k-th best, skip
        if results.len() >= k && entry.lower_bound >= kth_distance {
            stats.nodes_pruned_triangle += 1;
            continue;
        }

        stats.nodes_visited += 1;
        let node = &tree.nodes[entry.node_idx];

        if node.is_leaf() {
            // Leaf: compute exact distances for all points
            for (orig_idx, fp) in tree.cluster_points(node, data, vec_bytes) {
                stats.exact_distances_computed += 1;
                let dist = hamming_fn(query, fp);

                if results.len() < k {
                    results.push((dist, orig_idx));
                    if results.len() == k {
                        kth_distance = results.peek().unwrap().0;
                    }
                } else if dist < kth_distance {
                    results.pop();
                    results.push((dist, orig_idx));
                    kth_distance = results.peek().unwrap().0;
                }
            }
        } else {
            // Internal: compute bounds for children and enqueue
            for &child_idx in [node.left, node.right].iter().flatten() {
                let child = &tree.nodes[child_idx];

                // Bound 1: Triangle inequality
                let center_data = tree.center_data(child, data, vec_bytes);
                let dist_to_center = hamming_fn(query, center_data);
                let tri_lower = child.delta_minus(dist_to_center);

                // Bound 2: LOD OR-mask
                let lod_lower = index
                    .get(child_idx)
                    .map(|ann| or_mask_lower_bound(query, &ann.or_mask))
                    .unwrap_or(0);

                let lower_bound = tri_lower.max(lod_lower);

                if results.len() >= k && lower_bound >= kth_distance {
                    if lod_lower >= kth_distance {
                        stats.nodes_pruned_lod += 1;
                    } else {
                        stats.nodes_pruned_triangle += 1;
                    }
                    continue;
                }

                queue.push(SearchEntry {
                    node_idx: child_idx,
                    lower_bound,
                });
            }
        }
    }

    // Drain results into sorted vec (ascending distance)
    let mut sorted: Vec<LodSearchResult> = results
        .into_iter()
        .map(|(distance, index)| LodSearchResult { index, distance })
        .collect();
    sorted.sort_unstable_by_key(|r| r.distance);

    (sorted, stats)
}

/// Convenience wrapper: build tree + index, then search.
///
/// For repeated queries against the same dataset, prefer building the
/// tree and index once and calling `lod_knn_search` directly.
pub fn lod_knn_search_oneshot(
    query: &[u8],
    database: &[u8],
    vec_bytes: usize,
    count: usize,
    k: usize,
) -> Vec<LodSearchResult> {
    use crate::tree::BuildConfig;

    let config = BuildConfig {
        min_cardinality: 16,
        max_depth: 64,
        min_radius: 0,
    };

    let tree = ClamTree::build(database, vec_bytes, count, &config);
    let index = LodIndex::build(&tree, database, vec_bytes);
    let (results, _stats) = lod_knn_search(query, &tree, &index, database, vec_bytes, k);
    results
}

/// Build a LodIndex for an existing ClamTree (alias for `LodIndex::build`).
pub fn annotate_tree_with_lod(tree: &ClamTree, data: &[u8], vec_bytes: usize) -> LodIndex {
    LodIndex::build(tree, data, vec_bytes)
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::BuildConfig;

    fn make_test_data(count: usize, vec_bytes: usize) -> Vec<u8> {
        let mut data = vec![0u8; count * vec_bytes];
        // Each vector has a distinct pattern
        for i in 0..count {
            for j in 0..vec_bytes {
                data[i * vec_bytes + j] = ((i * 7 + j * 13) % 256) as u8;
            }
        }
        data
    }

    #[test]
    fn test_lod_index_build() {
        let vec_bytes = 64;
        let count = 100;
        let data = make_test_data(count, vec_bytes);

        let config = BuildConfig {
            min_cardinality: 4,
            max_depth: 10,
            min_radius: 0,
        };
        let tree = ClamTree::build(&data, vec_bytes, count, &config);
        let index = LodIndex::build(&tree, &data, vec_bytes);

        // Every node should have an annotation
        assert_eq!(index.len(), tree.nodes.len());

        // Root annotation should have cardinality == count
        let root_ann = index.get(0).unwrap();
        assert_eq!(root_ann.cardinality, count);
    }

    #[test]
    fn test_lod_knn_search_finds_exact_match() {
        let vec_bytes = 64;
        let count = 50;
        let data = make_test_data(count, vec_bytes);

        // Query = exact copy of vector 10
        let query = data[10 * vec_bytes..(10 + 1) * vec_bytes].to_vec();

        let config = BuildConfig {
            min_cardinality: 4,
            max_depth: 10,
            min_radius: 0,
        };
        let tree = ClamTree::build(&data, vec_bytes, count, &config);
        let index = LodIndex::build(&tree, &data, vec_bytes);

        let (results, stats) = lod_knn_search(&query, &tree, &index, &data, vec_bytes, 1);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].distance, 0);

        // The result index maps through reordered — the fingerprint should match
        let result_fp = &data[results[0].index * vec_bytes..(results[0].index + 1) * vec_bytes];
        assert_eq!(result_fp, &query[..]);

        // Should have pruned some nodes
        assert!(stats.exact_distances_computed < count);
    }

    #[test]
    fn test_lod_knn_search_top_k() {
        let vec_bytes = 32;
        let count = 200;
        let data = make_test_data(count, vec_bytes);
        let query = data[0..vec_bytes].to_vec();

        let config = BuildConfig {
            min_cardinality: 8,
            max_depth: 10,
            min_radius: 0,
        };
        let tree = ClamTree::build(&data, vec_bytes, count, &config);
        let index = LodIndex::build(&tree, &data, vec_bytes);

        let (results, _stats) = lod_knn_search(&query, &tree, &index, &data, vec_bytes, 5);

        assert_eq!(results.len(), 5);

        // Results should be sorted by distance
        for i in 1..results.len() {
            assert!(results[i].distance >= results[i - 1].distance);
        }

        // First result should be exact match (distance 0)
        assert_eq!(results[0].distance, 0);
    }

    #[test]
    fn test_lod_knn_search_oneshot() {
        let vec_bytes = 64;
        let count = 100;
        let data = make_test_data(count, vec_bytes);
        let query = data[25 * vec_bytes..26 * vec_bytes].to_vec();

        let results = lod_knn_search_oneshot(&query, &data, vec_bytes, count, 3);
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].distance, 0);
    }

    #[test]
    fn test_lod_knn_search_correctness() {
        // Verify LOD search returns the same top-k as brute force
        let vec_bytes = 32;
        let count = 100;
        let data = make_test_data(count, vec_bytes);
        let query = data[50 * vec_bytes..51 * vec_bytes].to_vec();
        let k = 5;

        // Brute force
        let hamming_fn = numrus_core::simd::select_hamming_fn();
        let mut brute: Vec<(u64, usize)> = (0..count)
            .map(|i| {
                let fp = &data[i * vec_bytes..(i + 1) * vec_bytes];
                (hamming_fn(&query, fp), i)
            })
            .collect();
        brute.sort_unstable_by_key(|&(d, _)| d);
        brute.truncate(k);

        // LOD search
        let config = BuildConfig {
            min_cardinality: 4,
            max_depth: 10,
            min_radius: 0,
        };
        let tree = ClamTree::build(&data, vec_bytes, count, &config);
        let index = LodIndex::build(&tree, &data, vec_bytes);
        let (results, _) = lod_knn_search(&query, &tree, &index, &data, vec_bytes, k);

        // Same top-k distances (indices may differ for ties)
        let brute_dists: Vec<u64> = brute.iter().map(|&(d, _)| d).collect();
        let lod_dists: Vec<u64> = results.iter().map(|r| r.distance).collect();
        assert_eq!(brute_dists, lod_dists, "LOD search must match brute force");
    }

    #[test]
    fn test_annotate_tree_with_lod_alias() {
        let vec_bytes = 64;
        let count = 50;
        let data = make_test_data(count, vec_bytes);

        let config = BuildConfig {
            min_cardinality: 4,
            max_depth: 10,
            min_radius: 0,
        };
        let tree = ClamTree::build(&data, vec_bytes, count, &config);
        let index = annotate_tree_with_lod(&tree, &data, vec_bytes);

        assert_eq!(index.len(), tree.nodes.len());
    }
}
