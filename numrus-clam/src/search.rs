//! Triangle-inequality search: exact k-NN and ρ-NN on the CLAM tree.
//!
//! Implements three algorithms from CAKES (arXiv:2309.05491):
//!
//! 1. **ρ-NN search** (Algorithms 2+3): tree-search to find overlapping
//!    clusters, then leaf-search to find exact hits within radius ρ.
//!
//! 2. **Repeated ρ-NN** (Algorithm 4): starts with a small radius and
//!    increases it guided by LFD until ≥k points are found.
//!
//! 3. **Depth-First Sieve** (Algorithm 6): priority-queue traversal
//!    using δ⁻ ordering — the fastest CAKES algorithm empirically.
//!
//! ## Key improvement over numrus's adaptive cascade
//!
//! The current `hamming_search_adaptive()` uses fixed 3σ/2σ statistical
//! thresholds to prune candidates. This works well statistically but:
//! - Can produce false negatives (rejects true hits at the σ boundary)
//! - Thresholds are fixed, not adaptive to local data density
//! - No formal correctness guarantee
//!
//! CLAM search uses the triangle inequality: d_min = max(0, d(q,c) - r).
//! When the distance function is a metric (Hamming IS a metric):
//! - **Zero false negatives** — mathematically exact pruning
//! - **Adaptive** — pruning power depends on actual cluster radius, not σ
//! - **Proven complexity** — O(k · 2^LFD · log 𝒩) for DFS Sieve

use crate::tree::{hamming_inline, ClamTree};
use std::cmp::Reverse;
use std::collections::BinaryHeap;

// ─────────────────────────────────────────────────────────────────────
// Result types
// ─────────────────────────────────────────────────────────────────────

/// Result of ρ-NN search: all points within radius ρ of the query.
#[derive(Debug, Clone)]
pub struct RhoNnResult {
    /// (original_index, distance) pairs, sorted by distance ascending.
    pub hits: Vec<(usize, u64)>,
    /// Number of distance computations performed.
    pub distance_calls: usize,
    /// Number of clusters pruned by triangle inequality.
    pub clusters_pruned: usize,
}

/// Result of k-NN search: the k nearest neighbors to the query.
#[derive(Debug, Clone)]
pub struct KnnResult {
    /// (original_index, distance) pairs, sorted by distance ascending.
    /// Length is min(k, dataset_size).
    pub hits: Vec<(usize, u64)>,
    /// Number of distance computations performed.
    pub distance_calls: usize,
    /// Number of clusters pruned.
    pub clusters_pruned: usize,
}

/// Configuration for search.
#[derive(Debug, Clone, Default)]
pub struct SearchConfig {
    /// For Repeated ρ-NN: initial radius as fraction of root radius.
    /// Default: 1/n where n = dataset cardinality.
    pub initial_radius_fraction: Option<f64>,
}

// ─────────────────────────────────────────────────────────────────────
// ρ-NN search (Algorithms 2 + 3 from CHESS/CAKES)
// ─────────────────────────────────────────────────────────────────────

/// ρ-nearest neighbor search: find all points within radius ρ of query.
///
/// Two phases:
/// 1. **Tree search** (Algorithm 2): walk the tree, pruning clusters
///    where δ⁻ > ρ (no point can be within ρ).
/// 2. **Leaf search** (Algorithm 3): linearly scan surviving clusters.
///
/// ## Exactness guarantee
///
/// When the distance function is a metric (Hamming is a metric),
/// this search has **perfect recall** — zero false negatives.
/// The triangle inequality guarantees:
///
/// ```text
/// d(q, p) ≥ |d(q, center) - d(center, p)| ≥ d(q, center) - radius
/// ```
///
/// So if `d(q, center) - radius > ρ`, no point p in the cluster can
/// satisfy `d(q, p) ≤ ρ`.
pub fn rho_nn(tree: &ClamTree, data: &[u8], vec_len: usize, query: &[u8], rho: u64) -> RhoNnResult {
    let mut hits = Vec::new();
    let mut distance_calls = 0usize;
    let mut clusters_pruned = 0usize;

    // Phase 1: Tree search — find leaf clusters that overlap query ball
    let mut candidate_leaves = Vec::new();
    let mut stack = vec![0usize]; // start at root

    while let Some(node_idx) = stack.pop() {
        let cluster = &tree.nodes[node_idx];
        let center = tree.center_data(cluster, data, vec_len);
        let delta = hamming_inline(query, center);
        distance_calls += 1;

        let d_minus = cluster.delta_minus(delta);
        let d_plus = cluster.delta_plus(delta);

        if d_minus > rho {
            // Triangle inequality prune: closest possible point is beyond ρ
            clusters_pruned += 1;
            continue;
        }

        if cluster.is_leaf() {
            if d_plus <= rho {
                // Entire cluster is within ρ — add all points without checking
                candidate_leaves.push((node_idx, true, delta));
            } else {
                // Cluster overlaps query ball — need to check individual points
                candidate_leaves.push((node_idx, false, delta));
            }
        } else {
            // Internal node: recurse into children
            if let Some(left) = cluster.left {
                stack.push(left);
            }
            if let Some(right) = cluster.right {
                stack.push(right);
            }
        }
    }

    // Phase 2: Leaf search — scan surviving clusters
    for (node_idx, all_inside, _) in &candidate_leaves {
        let cluster = &tree.nodes[*node_idx];

        if *all_inside {
            // δ⁺ ≤ ρ: every point in this cluster is a hit
            for (orig_idx, point_data) in tree.cluster_points(cluster, data, vec_len) {
                let d = hamming_inline(query, point_data);
                distance_calls += 1;
                hits.push((orig_idx, d));
            }
        } else {
            // Need to check each point
            for (orig_idx, point_data) in tree.cluster_points(cluster, data, vec_len) {
                let d = hamming_inline(query, point_data);
                distance_calls += 1;
                if d <= rho {
                    hits.push((orig_idx, d));
                }
            }
        }
    }

    hits.sort_by_key(|&(_, d)| d);

    RhoNnResult {
        hits,
        distance_calls,
        clusters_pruned,
    }
}

// ─────────────────────────────────────────────────────────────────────
// Repeated ρ-NN for k-NN (Algorithm 4 from CAKES)
// ─────────────────────────────────────────────────────────────────────

/// k-NN via Repeated ρ-NN search (CAKES Algorithm 4).
///
/// Strategy: start with a small radius, do ρ-NN, if we found fewer
/// than k hits, increase radius guided by LFD and retry.
///
/// ```text
/// radius_increase_factor = min(2, (k / hits_so_far)^μ)
/// where μ = mean(1/LFD) of overlapping clusters
/// ```
///
/// This is NOT the fastest CAKES algorithm (Depth-First Sieve is),
/// but it's the simplest and demonstrates the LFD-guided radius ratchet.
pub fn knn_repeated_rho(
    tree: &ClamTree,
    data: &[u8],
    vec_len: usize,
    query: &[u8],
    k: usize,
) -> KnnResult {
    let root = tree.root();
    if root.cardinality == 0 {
        return KnnResult {
            hits: Vec::new(),
            distance_calls: 0,
            clusters_pruned: 0,
        };
    }
    let mut rho = root.radius / root.cardinality as u64;
    if rho == 0 {
        rho = 1;
    }

    let mut total_distance_calls = 0;
    let mut total_pruned = 0;

    loop {
        let result = rho_nn(tree, data, vec_len, query, rho);
        total_distance_calls += result.distance_calls;
        total_pruned += result.clusters_pruned;

        if result.hits.len() >= k {
            // Found enough — return top k
            let mut hits = result.hits;
            hits.truncate(k);
            return KnnResult {
                hits,
                distance_calls: total_distance_calls,
                clusters_pruned: total_pruned,
            };
        }

        // Increase radius using LFD-guided ratchet (Equation 3 from CAKES)
        if result.hits.is_empty() {
            rho *= 2;
        } else {
            // Collect LFDs from overlapping leaf clusters
            let mean_inv_lfd = estimate_local_lfd(tree, data, vec_len, query, rho);
            let ratio = k as f64 / result.hits.len() as f64;
            let factor = ratio.powf(mean_inv_lfd).clamp(1.1, 2.0);
            rho = ((rho as f64) * factor).ceil() as u64;
        }

        // Safety: don't exceed root radius
        if rho > root.radius {
            rho = root.radius;
            // One final search at max radius
            let result = rho_nn(tree, data, vec_len, query, rho);
            total_distance_calls += result.distance_calls;
            total_pruned += result.clusters_pruned;
            let mut hits = result.hits;
            hits.truncate(k);
            return KnnResult {
                hits,
                distance_calls: total_distance_calls,
                clusters_pruned: total_pruned,
            };
        }
    }
}

/// Estimate the mean inverse LFD of leaf clusters near the query.
/// Used for the radius ratchet in Repeated ρ-NN (CAKES Equation 3).
///
/// μ = (1/|Q|) · Σ(1/LFD(C)) for C in overlapping leaves
fn estimate_local_lfd(tree: &ClamTree, data: &[u8], vec_len: usize, query: &[u8], rho: u64) -> f64 {
    let mut sum_inv_lfd = 0.0;
    let mut count = 0usize;

    let mut stack = vec![0usize];
    while let Some(node_idx) = stack.pop() {
        let cluster = &tree.nodes[node_idx];
        let center = tree.center_data(cluster, data, vec_len);
        let delta = hamming_inline(query, center);

        if cluster.delta_minus(delta) > rho {
            continue;
        }

        if cluster.is_leaf() {
            let lfd = cluster.lfd.value.max(0.1); // avoid div-by-zero
            sum_inv_lfd += 1.0 / lfd;
            count += 1;
        } else {
            if let Some(left) = cluster.left {
                stack.push(left);
            }
            if let Some(right) = cluster.right {
                stack.push(right);
            }
        }
    }

    if count == 0 {
        1.0 // default: assume LFD = 1
    } else {
        sum_inv_lfd / count as f64
    }
}

// ─────────────────────────────────────────────────────────────────────
// Depth-First Sieve (Algorithm 6 from CAKES) — fastest empirically
// ─────────────────────────────────────────────────────────────────────

/// Depth-First Sieve k-NN search (CAKES Algorithm 6).
///
/// Uses two priority queues:
/// - Q: min-heap of clusters by δ⁻ (closest possible point first)
/// - H: max-heap of k best hits by distance (worst current hit on top)
///
/// Terminates when the best remaining cluster's δ⁻ exceeds the worst
/// hit's distance — at that point, no remaining cluster can improve H.
///
/// ## Complexity
///
/// O(⌈d⌉ · log 𝒩 · log(⌈d⌉ · log 𝒩) + k · (1 + 2·(|C̄|/k)^(d⁻¹))^d · log k)
///
/// where d = LFD, 𝒩 = metric entropy, |C̄| = mean leaf cardinality.
/// This is sublinear in n when LFD ≪ embedding dimension.
pub fn knn_dfs_sieve(
    tree: &ClamTree,
    data: &[u8],
    vec_len: usize,
    query: &[u8],
    k: usize,
) -> KnnResult {
    let mut distance_calls = 0usize;
    let mut clusters_pruned = 0usize;

    // Q: min-heap of (δ⁻, node_idx) — closest-first traversal
    let mut queue: BinaryHeap<Reverse<(u64, usize)>> = BinaryHeap::new();

    // H: max-heap of (distance, original_idx) — worst hit on top, capacity k
    let mut hits: BinaryHeap<(u64, usize)> = BinaryHeap::new();

    // Initialize with root
    let root = tree.root();
    let root_center = tree.center_data(root, data, vec_len);
    let root_delta = hamming_inline(query, root_center);
    distance_calls += 1;
    let root_d_minus = root.delta_minus(root_delta);
    queue.push(Reverse((root_d_minus, 0)));

    // Main loop: keep going while H isn't full or best remaining δ⁻
    // could beat our worst hit
    while let Some(&Reverse((d_minus, node_idx))) = queue.peek() {
        // Termination: H is full AND worst hit is closer than best remaining
        if hits.len() >= k {
            if let Some(&(worst_dist, _)) = hits.peek() {
                if worst_dist <= d_minus {
                    break;
                }
            }
        }

        queue.pop();
        let cluster = &tree.nodes[node_idx];

        if cluster.is_leaf() {
            // Leaf: scan all points
            for (orig_idx, point_data) in tree.cluster_points(cluster, data, vec_len) {
                let d = hamming_inline(query, point_data);
                distance_calls += 1;

                if hits.len() < k {
                    hits.push((d, orig_idx));
                } else if let Some(&(worst, _)) = hits.peek() {
                    if d < worst {
                        hits.pop();
                        hits.push((d, orig_idx));
                    }
                }
            }
        } else {
            // Internal: push children with their δ⁻ values
            for child_idx in [cluster.left, cluster.right].iter().flatten() {
                let child = &tree.nodes[*child_idx];
                let child_center = tree.center_data(child, data, vec_len);
                let child_delta = hamming_inline(query, child_center);
                distance_calls += 1;

                let child_d_minus = child.delta_minus(child_delta);

                // Prune: if H is full and child's δ⁻ exceeds worst hit
                if hits.len() >= k {
                    if let Some(&(worst, _)) = hits.peek() {
                        if child_d_minus > worst {
                            clusters_pruned += 1;
                            continue;
                        }
                    }
                }

                queue.push(Reverse((child_d_minus, *child_idx)));
            }
        }
    }

    // Drain hits into sorted vec
    let mut result: Vec<(usize, u64)> = hits.into_iter().map(|(d, idx)| (idx, d)).collect();
    result.sort_by_key(|&(_, d)| d);

    KnnResult {
        hits: result,
        distance_calls,
        clusters_pruned,
    }
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::{BuildConfig, ClamTree};
    use numrus_core::SplitMix64;

    fn make_test_data(n: usize, vec_len: usize, seed: u64) -> Vec<u8> {
        let mut rng = SplitMix64::new(seed);
        let mut data = vec![0u8; n * vec_len];
        for byte in data.iter_mut() {
            *byte = (rng.next_u64() & 0xFF) as u8;
        }
        data
    }

    /// Linear scan for ground truth.
    fn linear_knn(
        data: &[u8],
        vec_len: usize,
        count: usize,
        query: &[u8],
        k: usize,
    ) -> Vec<(usize, u64)> {
        let mut dists: Vec<(usize, u64)> = (0..count)
            .map(|i| {
                let point = &data[i * vec_len..(i + 1) * vec_len];
                (i, hamming_inline(query, point))
            })
            .collect();
        dists.sort_by_key(|&(_, d)| d);
        dists.truncate(k);
        dists
    }

    #[test]
    fn test_rho_nn_finds_close_points() {
        let vec_len = 64;
        let count = 200;
        let data = make_test_data(count, vec_len, 42);

        let config = BuildConfig {
            min_cardinality: 5,
            max_depth: 30,
            min_radius: 0,
        };
        let tree = ClamTree::build(&data, vec_len, count, &config);

        // Use first point as query
        let query = &data[0..vec_len];

        // Find all within rho=0 (just the query itself)
        let result = rho_nn(&tree, &data, vec_len, query, 0);
        assert!(!result.hits.is_empty());
        assert_eq!(result.hits[0].1, 0); // distance 0 = exact match

        println!(
            "ρ-NN(ρ=0): {} hits, {} distance calls, {} clusters pruned",
            result.hits.len(),
            result.distance_calls,
            result.clusters_pruned
        );
    }

    #[test]
    fn test_rho_nn_exact_recall() {
        let vec_len = 64;
        let count = 200;
        let data = make_test_data(count, vec_len, 123);

        let config = BuildConfig {
            min_cardinality: 3,
            max_depth: 30,
            min_radius: 0,
        };
        let tree = ClamTree::build(&data, vec_len, count, &config);

        let query = &data[0..vec_len];
        let rho = 200; // reasonable radius

        let result = rho_nn(&tree, &data, vec_len, query, rho);

        // Ground truth: linear scan
        let ground_truth: Vec<(usize, u64)> = (0..count)
            .map(|i| {
                let point = &data[i * vec_len..(i + 1) * vec_len];
                (i, hamming_inline(query, point))
            })
            .filter(|&(_, d)| d <= rho)
            .collect();

        // Hamming is a metric → exact recall
        assert_eq!(
            result.hits.len(),
            ground_truth.len(),
            "ρ-NN should have perfect recall for metric distances"
        );
    }

    #[test]
    fn test_knn_repeated_rho() {
        let vec_len = 64;
        let count = 200;
        let data = make_test_data(count, vec_len, 77);

        let config = BuildConfig {
            min_cardinality: 3,
            max_depth: 30,
            min_radius: 0,
        };
        let tree = ClamTree::build(&data, vec_len, count, &config);

        let query = &data[0..vec_len];
        let k = 10;

        let result = knn_repeated_rho(&tree, &data, vec_len, query, k);
        let ground_truth = linear_knn(&data, vec_len, count, query, k);

        assert_eq!(result.hits.len(), k);

        // Check exact recall: our k-th hit should match linear scan's k-th hit distance
        let our_max_dist = result.hits.last().unwrap().1;
        let gt_max_dist = ground_truth.last().unwrap().1;
        assert_eq!(
            our_max_dist, gt_max_dist,
            "k-NN should find exact same max distance as linear scan"
        );

        println!(
            "Repeated ρ-NN: {} distance calls, {} pruned (vs {} linear)",
            result.distance_calls, result.clusters_pruned, count
        );
    }

    #[test]
    fn test_knn_dfs_sieve() {
        let vec_len = 64;
        let count = 200;
        let data = make_test_data(count, vec_len, 99);

        let config = BuildConfig {
            min_cardinality: 3,
            max_depth: 30,
            min_radius: 0,
        };
        let tree = ClamTree::build(&data, vec_len, count, &config);

        let query = &data[0..vec_len];
        let k = 10;

        let result = knn_dfs_sieve(&tree, &data, vec_len, query, k);
        let ground_truth = linear_knn(&data, vec_len, count, query, k);

        assert_eq!(result.hits.len(), k);

        // Verify exact recall
        let our_max_dist = result.hits.last().unwrap().1;
        let gt_max_dist = ground_truth.last().unwrap().1;
        assert_eq!(
            our_max_dist, gt_max_dist,
            "DFS Sieve should find exact same max distance as linear scan"
        );

        println!(
            "DFS Sieve: {} distance calls, {} pruned (vs {} linear)",
            result.distance_calls, result.clusters_pruned, count
        );
    }

    #[test]
    fn test_dfs_sieve_speedup_over_linear() {
        // Larger test to demonstrate pruning advantage
        let vec_len = 256; // 2048-bit fingerprints
        let count = 1000;
        let data = make_test_data(count, vec_len, 42);

        let config = BuildConfig {
            min_cardinality: 5,
            max_depth: 40,
            min_radius: 0,
        };
        let tree = ClamTree::build(&data, vec_len, count, &config);

        let query = &data[0..vec_len];
        let k = 10;

        let result = knn_dfs_sieve(&tree, &data, vec_len, query, k);

        let speedup = count as f64 / result.distance_calls as f64;
        println!(
            "DFS Sieve speedup: {:.1}x ({} calls vs {} linear), {} pruned",
            speedup, result.distance_calls, count, result.clusters_pruned
        );

        // With random data, speedup may be modest, but should prune something
        assert!(
            result.clusters_pruned > 0,
            "should prune at least some clusters"
        );
    }

    #[test]
    fn test_all_three_agree() {
        let vec_len = 64;
        let count = 100;
        let data = make_test_data(count, vec_len, 55);

        let config = BuildConfig {
            min_cardinality: 2,
            max_depth: 30,
            min_radius: 0,
        };
        let tree = ClamTree::build(&data, vec_len, count, &config);

        let query = &data[32 * vec_len..33 * vec_len]; // use point 32 as query
        let k = 5;

        let result_repeated = knn_repeated_rho(&tree, &data, vec_len, query, k);
        let result_dfs = knn_dfs_sieve(&tree, &data, vec_len, query, k);
        let ground_truth = linear_knn(&data, vec_len, count, query, k);

        // All should agree on the max distance of k-th neighbor
        let gt_max = ground_truth.last().unwrap().1;
        assert_eq!(result_repeated.hits.last().unwrap().1, gt_max);
        assert_eq!(result_dfs.hits.last().unwrap().1, gt_max);
    }
}
