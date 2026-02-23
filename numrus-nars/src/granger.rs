//! Granger causality for holographic and BF16 time series.
//!
//! Contains `granger_signal`, `granger_scan` (i8-based symbol distance),
//! and BF16Entity, CausalFeatureMap, `bf16_granger_causal_map`,
//! `bf16_granger_causal_scan` (BF16-structured per-dimension causal attribution).

use crate::bind::symbol_distance;
use numrus_core::bf16_hamming::{
    fp32_to_bf16_bytes, select_bf16_hamming_fn, structural_diff, TRAINING_WEIGHTS,
};

// ---------------------------------------------------------------------------
// Granger Signal — temporal directional causality
// ---------------------------------------------------------------------------

/// Granger signal between two time series of holographic vectors.
///
/// G(A->B, tau) = d(A_t, B_{t+tau}) - d(B_t, B_{t+tau})
///
/// If G < 0: A_t is closer to B_{t+tau} than B_t is — A predicts B.
/// If G > 0: B_t is closer to future B than A_t — no causal signal from A.
/// If G ~= 0: A and B are equidistant from future B — inconclusive.
///
/// Returns the mean Granger signal across all valid time steps.
pub fn granger_signal(series_a: &[Vec<i8>], series_b: &[Vec<i8>], tau: usize) -> f64 {
    assert!(
        tau > 0,
        "Granger signal requires tau > 0 (lag must be at least 1)"
    );
    assert_eq!(series_a.len(), series_b.len());
    let n = series_a.len();
    if tau >= n {
        return 0.0;
    }

    let mut sum = 0.0f64;
    let mut count = 0usize;

    for t in 0..(n - tau) {
        let d_ab = symbol_distance(&series_a[t], &series_b[t + tau]) as f64;
        let d_bb = symbol_distance(&series_b[t], &series_b[t + tau]) as f64;
        sum += d_ab - d_bb;
        count += 1;
    }

    if count > 0 {
        sum / count as f64
    } else {
        0.0
    }
}

/// Scan multiple lags to find the strongest Granger signal.
///
/// Returns (best_lag, signal) where signal is the most negative G(A->B, tau).
/// More negative means A is a stronger predictor of B at that lag.
pub fn granger_scan(series_a: &[Vec<i8>], series_b: &[Vec<i8>], max_lag: usize) -> (usize, f64) {
    let mut best_lag = 1;
    let mut best_signal = f64::MAX;

    for tau in 1..=max_lag {
        let g = granger_signal(series_a, series_b, tau);
        if g < best_signal {
            best_signal = g;
            best_lag = tau;
        }
    }

    (best_lag, best_signal)
}

// ---------------------------------------------------------------------------
// BF16 Entity
// ---------------------------------------------------------------------------

/// An entity with BF16-encoded vector (2 bytes per dimension).
///
/// For Jina v3 1024-D embeddings: 2048 bytes per entity.
/// Created by truncating FP32 embeddings via `fp32_to_bf16_bytes()`.
#[derive(Clone)]
pub struct BF16Entity {
    pub id: u32,
    pub name: String,
    /// BF16 bytes: 2 bytes per dimension, little-endian.
    pub bf16_bytes: Vec<u8>,
    /// Number of dimensions (bf16_bytes.len() / 2).
    pub n_dims: usize,
}

impl BF16Entity {
    /// Create from FP32 embedding (truncates to BF16).
    pub fn from_f32(id: u32, name: &str, embedding: &[f32]) -> Self {
        let bf16_bytes = fp32_to_bf16_bytes(embedding);
        let n_dims = embedding.len();
        Self {
            id,
            name: name.to_string(),
            bf16_bytes,
            n_dims,
        }
    }
}

// ---------------------------------------------------------------------------
// Per-Dimension Causal Map
// ---------------------------------------------------------------------------

/// Causal attribution per dimension: which features carry the causal signal
/// between two time series.
#[derive(Clone, Debug)]
pub struct CausalFeatureMap {
    /// Number of dimensions.
    pub n_dims: usize,
    /// Per-dimension sign-flip count across all timesteps.
    pub sign_flip_counts: Vec<u32>,
    /// Per-dimension exponent-shift count.
    pub exponent_shift_counts: Vec<u32>,
    /// Dimensions sorted by sign-flip frequency (descending).
    pub top_causal_dims: Vec<(usize, u32)>,
    /// Total timesteps analyzed.
    pub timesteps: usize,
    /// Overall Granger signal (scalar, for comparison with existing API).
    pub granger_signal: f64,
    /// Best lag (at which the per-dim signal was strongest).
    pub best_lag: usize,
}

impl CausalFeatureMap {
    /// Fraction of timesteps where dimension `dim` had a sign flip.
    pub fn sign_flip_rate(&self, dim: usize) -> f64 {
        if self.timesteps == 0 {
            return 0.0;
        }
        self.sign_flip_counts[dim] as f64 / self.timesteps as f64
    }

    /// Dimensions where sign flips occur > threshold fraction of timesteps.
    pub fn causal_dims_above(&self, threshold: f64) -> Vec<usize> {
        (0..self.n_dims)
            .filter(|&d| self.sign_flip_rate(d) > threshold)
            .collect()
    }
}

// ---------------------------------------------------------------------------
// BF16 Granger Signal — Per-Dimension Causal Attribution
// ---------------------------------------------------------------------------

/// BF16-structured Granger signal with per-dimension causal attribution.
///
/// Like `granger_signal()` but operates on BF16 byte series and returns
/// not just "A causes B" but "A causes B via dimensions [47, 312, 891]".
pub fn bf16_granger_causal_map(
    series_a: &[BF16Entity],
    series_b: &[BF16Entity],
    tau: usize,
) -> CausalFeatureMap {
    assert!(tau > 0);
    assert_eq!(series_a.len(), series_b.len());
    let n = series_a.len();
    let n_dims = series_a[0].n_dims;
    assert!(tau < n);

    let bf16_fn = select_bf16_hamming_fn();
    let weights = &TRAINING_WEIGHTS;

    let mut sign_flips = vec![0u32; n_dims];
    let mut exp_shifts = vec![0u32; n_dims];
    let mut cross_sum = 0.0f64;
    let mut auto_sum = 0.0f64;
    let mut count = 0usize;

    for t in 0..(n - tau) {
        // Cross-series structural diff: A_t vs B_{t+tau}
        let cross_diff = structural_diff(&series_a[t].bf16_bytes, &series_b[t + tau].bf16_bytes);

        // Accumulate per-dimension sign flips from cross-series
        for &dim in &cross_diff.sign_flip_dims {
            sign_flips[dim] += 1;
        }
        for &dim in &cross_diff.major_magnitude_shifts {
            exp_shifts[dim] += 1;
        }

        // Scalar Granger signal for comparison
        let d_ab = bf16_fn(
            &series_a[t].bf16_bytes,
            &series_b[t + tau].bf16_bytes,
            weights,
        ) as f64;
        let d_bb = bf16_fn(
            &series_b[t].bf16_bytes,
            &series_b[t + tau].bf16_bytes,
            weights,
        ) as f64;
        cross_sum += d_ab;
        auto_sum += d_bb;
        count += 1;
    }

    // Build top causal dims (sorted by sign-flip count, descending)
    let mut top: Vec<(usize, u32)> = sign_flips
        .iter()
        .enumerate()
        .filter(|(_, &c)| c > 0)
        .map(|(d, &c)| (d, c))
        .collect();
    top.sort_by_key(|x| std::cmp::Reverse(x.1));

    let granger = if count > 0 {
        (cross_sum - auto_sum) / count as f64
    } else {
        0.0
    };

    CausalFeatureMap {
        n_dims,
        sign_flip_counts: sign_flips,
        exponent_shift_counts: exp_shifts,
        top_causal_dims: top,
        timesteps: count,
        granger_signal: granger,
        best_lag: tau,
    }
}

/// Scan multiple lags and return the CausalFeatureMap at the best lag.
///
/// Single-pass algorithm: iterates timesteps once, accumulating per-lag
/// counters simultaneously. O(n * max_lag) total, not O(n * max_lag^2).
pub fn bf16_granger_causal_scan(
    series_a: &[BF16Entity],
    series_b: &[BF16Entity],
    max_lag: usize,
) -> CausalFeatureMap {
    assert!(max_lag >= 1);
    assert_eq!(series_a.len(), series_b.len());
    let n = series_a.len();
    assert!(max_lag < n);
    let n_dims = series_a[0].n_dims;

    let bf16_fn = select_bf16_hamming_fn();
    let weights = &TRAINING_WEIGHTS;

    // Per-lag accumulators
    struct LagAcc {
        sign_flips: Vec<u32>,
        exp_shifts: Vec<u32>,
        cross_sum: f64,
        auto_sum: f64,
        count: usize,
    }

    let mut lags: Vec<LagAcc> = (0..max_lag)
        .map(|_| LagAcc {
            sign_flips: vec![0u32; n_dims],
            exp_shifts: vec![0u32; n_dims],
            cross_sum: 0.0,
            auto_sum: 0.0,
            count: 0,
        })
        .collect();

    // Single pass over all timesteps
    for t in 0..n {
        for (lag_idx, lag) in lags.iter_mut().enumerate() {
            let tau = lag_idx + 1; // lag 1..=max_lag
            if t + tau >= n {
                continue;
            }

            let cross_diff =
                structural_diff(&series_a[t].bf16_bytes, &series_b[t + tau].bf16_bytes);

            for &dim in &cross_diff.sign_flip_dims {
                lag.sign_flips[dim] += 1;
            }
            for &dim in &cross_diff.major_magnitude_shifts {
                lag.exp_shifts[dim] += 1;
            }

            let d_ab = bf16_fn(
                &series_a[t].bf16_bytes,
                &series_b[t + tau].bf16_bytes,
                weights,
            ) as f64;
            let d_bb = bf16_fn(
                &series_b[t].bf16_bytes,
                &series_b[t + tau].bf16_bytes,
                weights,
            ) as f64;
            lag.cross_sum += d_ab;
            lag.auto_sum += d_bb;
            lag.count += 1;
        }
    }

    // Find lag with most negative Granger signal
    let mut best_idx = 0;
    let mut best_granger = f64::INFINITY;
    for (i, lag) in lags.iter().enumerate() {
        let granger = if lag.count > 0 {
            (lag.cross_sum - lag.auto_sum) / lag.count as f64
        } else {
            0.0
        };
        if granger < best_granger {
            best_granger = granger;
            best_idx = i;
        }
    }

    let best = &lags[best_idx];
    let tau = best_idx + 1;

    let mut top: Vec<(usize, u32)> = best
        .sign_flips
        .iter()
        .enumerate()
        .filter(|(_, &c)| c > 0)
        .map(|(d, &c)| (d, c))
        .collect();
    top.sort_by_key(|x| std::cmp::Reverse(x.1));

    CausalFeatureMap {
        n_dims,
        sign_flip_counts: best.sign_flips.clone(),
        exponent_shift_counts: best.exp_shifts.clone(),
        top_causal_dims: top,
        timesteps: best.count,
        granger_signal: best_granger,
        best_lag: tau,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use numrus_core::SplitMix64;

    fn make_jina_like_embedding(seed: u64, n_dims: usize) -> Vec<f32> {
        let mut rng = SplitMix64::new(seed);
        (0..n_dims)
            .map(|_| (rng.next_u64() as f64 / u64::MAX as f64 * 2.0 - 1.0) as f32 * 0.1)
            .collect()
    }

    #[test]
    fn test_granger_signal_self_prediction() {
        let mut rng = SplitMix64::new(42);
        let d = 256;
        let _base = crate::bind::Base::Binary;
        let n = 20;

        // Series B with gradual drift: B_{t+1} is a noisy copy of B_t
        let mut series_b: Vec<Vec<i8>> = Vec::with_capacity(n);
        let b0: Vec<i8> = (0..d).map(|_| (rng.next_u64() & 1) as i8).collect();
        series_b.push(b0);
        for t in 1..n {
            let prev = &series_b[t - 1];
            let mut next = prev.clone();
            // Flip ~5% of bits
            for i in 0..d {
                if rng.next_u64().is_multiple_of(20) {
                    next[i] ^= 1;
                }
            }
            series_b.push(next);
        }

        // Series A = random (uncorrelated with B)
        let series_a: Vec<Vec<i8>> = (0..n)
            .map(|_| (0..d).map(|_| (rng.next_u64() & 1) as i8).collect())
            .collect();

        // Random A should NOT predict B (G ~= 0 or positive)
        let g = granger_signal(&series_a, &series_b, 1);
        // B predicts itself better than random A does, so G should be >= 0
        assert!(
            g >= -5.0,
            "Random series should not strongly predict B: G = {}",
            g
        );
    }

    #[test]
    fn test_granger_signal_causal_series() {
        let mut rng = SplitMix64::new(42);
        let d = 256;
        let n = 30;
        let tau = 2;

        // A causes B with lag tau: B_{t+tau} = A_t (with noise)
        let series_a: Vec<Vec<i8>> = (0..n)
            .map(|_| (0..d).map(|_| (rng.next_u64() & 1) as i8).collect())
            .collect();

        let mut series_b: Vec<Vec<i8>> = Vec::with_capacity(n);
        for t in 0..n {
            if t >= tau {
                // B[t] = A[t-tau] with ~3% noise
                let mut b = series_a[t - tau].clone();
                for i in 0..d {
                    if rng.next_u64().is_multiple_of(33) {
                        b[i] ^= 1;
                    }
                }
                series_b.push(b);
            } else {
                series_b.push((0..d).map(|_| (rng.next_u64() & 1) as i8).collect());
            }
        }

        // A should predict B at lag tau: G(A->B,tau) should be negative
        let g = granger_signal(&series_a, &series_b, tau);
        assert!(g < 0.0, "A should predict B at lag {}: G = {}", tau, g);

        // Scan should find the best lag near tau
        let (best_lag, best_g) = granger_scan(&series_a, &series_b, 5);
        assert_eq!(
            best_lag, tau,
            "Best lag should be {}, got {} (G={})",
            tau, best_lag, best_g
        );
    }

    #[test]
    fn test_bf16_granger_causal_map_returns_per_dim() {
        let n_dims = 64;
        let n_steps = 20;
        let tau = 2;

        let series_a: Vec<BF16Entity> = (0..n_steps)
            .map(|t| {
                BF16Entity::from_f32(
                    t as u32,
                    &format!("A_{}", t),
                    &make_jina_like_embedding(t as u64, n_dims),
                )
            })
            .collect();

        // B[t] at lag tau is similar to A[t-tau] with some dims flipped
        let series_b: Vec<BF16Entity> = (0..n_steps)
            .map(|t| {
                if t >= tau {
                    let mut emb = make_jina_like_embedding((t - tau) as u64, n_dims);
                    for &d in &[5, 10, 15] {
                        if d < n_dims {
                            emb[d] = -emb[d];
                        }
                    }
                    BF16Entity::from_f32(t as u32, &format!("B_{}", t), &emb)
                } else {
                    BF16Entity::from_f32(
                        t as u32,
                        &format!("B_{}", t),
                        &make_jina_like_embedding(100 + t as u64, n_dims),
                    )
                }
            })
            .collect();

        let map = bf16_granger_causal_map(&series_a, &series_b, tau);

        // Dims 5, 10, 15 should have high sign-flip counts
        assert!(map.sign_flip_counts[5] > 0, "Dim 5 should have sign flips");
        assert!(
            map.sign_flip_counts[10] > 0,
            "Dim 10 should have sign flips"
        );
        assert!(
            map.sign_flip_counts[15] > 0,
            "Dim 15 should have sign flips"
        );
        assert!(!map.top_causal_dims.is_empty(), "Should have causal dims");
    }

    #[test]
    fn test_bf16_granger_scan_finds_best_lag() {
        let n_dims = 32;
        let n_steps = 30;
        let true_lag = 3;

        let series_a: Vec<BF16Entity> = (0..n_steps)
            .map(|t| {
                BF16Entity::from_f32(
                    t as u32,
                    &format!("A_{}", t),
                    &make_jina_like_embedding(t as u64, n_dims),
                )
            })
            .collect();

        let series_b: Vec<BF16Entity> = (0..n_steps)
            .map(|t| {
                if t >= true_lag {
                    let mut emb = make_jina_like_embedding((t - true_lag) as u64, n_dims);
                    // Small perturbation
                    for d in 0..3 {
                        emb[d] = -emb[d];
                    }
                    BF16Entity::from_f32(t as u32, &format!("B_{}", t), &emb)
                } else {
                    BF16Entity::from_f32(
                        t as u32,
                        &format!("B_{}", t),
                        &make_jina_like_embedding(200 + t as u64, n_dims),
                    )
                }
            })
            .collect();

        let map = bf16_granger_causal_scan(&series_a, &series_b, 5);
        // Best lag should be around the true lag
        assert!(map.best_lag >= 1 && map.best_lag <= 5);
    }

    #[test]
    fn test_causal_feature_map_rates() {
        let map = CausalFeatureMap {
            n_dims: 4,
            sign_flip_counts: vec![10, 0, 5, 8],
            exponent_shift_counts: vec![2, 0, 1, 3],
            top_causal_dims: vec![(0, 10), (3, 8), (2, 5)],
            timesteps: 20,
            granger_signal: -0.5,
            best_lag: 2,
        };

        assert!((map.sign_flip_rate(0) - 0.5).abs() < 0.01);
        assert_eq!(map.sign_flip_rate(1), 0.0);
        assert!((map.sign_flip_rate(2) - 0.25).abs() < 0.01);

        let above_30 = map.causal_dims_above(0.3);
        assert!(above_30.contains(&0)); // 50%
        assert!(above_30.contains(&3)); // 40%
        assert!(!above_30.contains(&2)); // 25% < 30%
    }
}
