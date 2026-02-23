//! Deterministic PRNG for reproducible tests and seeded algorithms.
//!
//! `SplitMix64` is a fast, high-quality 64-bit PRNG with no external deps.
//! Used across the numrus ecosystem wherever deterministic randomness is
//! needed: CLAM tree construction, test data generation, LSH projections,
//! ghost discovery templates.
//!
//! Consolidates 5 prior copies (3× splitmix64 in numrus-clam, 2× xorshift
//! in numrus-oracle) into a single canonical implementation.

/// SplitMix64 PRNG — deterministic, fast, statistically strong.
///
/// Period: 2^64. Passes BigCrush. Single u64 state.
///
/// # Example
/// ```ignore
/// let mut rng = SplitMix64::new(42);
/// let val = rng.next_u64();
/// let frac = rng.next_f64(); // [0, 1)
/// ```
pub struct SplitMix64(u64);

impl SplitMix64 {
    /// Create a new PRNG with the given seed.
    #[inline]
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// Next raw u64.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    /// Uniform f64 in [0, 1).
    ///
    /// Uses the top 53 bits for a full mantissa.
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Standard normal sample via Box-Muller transform.
    pub fn next_gaussian(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-15); // avoid log(0)
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// Random i8 in [min, max] (inclusive).
    #[inline]
    pub fn gen_range_i8(&mut self, min: i8, max: i8) -> i8 {
        let range = (max as i16 - min as i16 + 1) as u64;
        (min as i64 + (self.next_u64() % range) as i64) as i8
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deterministic() {
        let mut a = SplitMix64::new(42);
        let mut b = SplitMix64::new(42);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn test_different_seeds() {
        let mut a = SplitMix64::new(1);
        let mut b = SplitMix64::new(2);
        assert_ne!(a.next_u64(), b.next_u64());
    }

    #[test]
    fn test_f64_range() {
        let mut rng = SplitMix64::new(42);
        for _ in 0..1000 {
            let v = rng.next_f64();
            assert!((0.0..1.0).contains(&v), "next_f64() = {} out of [0, 1)", v);
        }
    }

    #[test]
    fn test_gaussian_distribution() {
        let mut rng = SplitMix64::new(12345);
        let n = 10_000;
        let samples: Vec<f64> = (0..n).map(|_| rng.next_gaussian()).collect();

        let mean: f64 = samples.iter().sum::<f64>() / n as f64;
        let variance: f64 = samples.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n as f64;

        assert!(mean.abs() < 0.1, "Gaussian mean = {}, expected ~0.0", mean);
        assert!(
            (variance - 1.0).abs() < 0.2,
            "Gaussian variance = {}, expected ~1.0",
            variance
        );
    }

    #[test]
    fn test_gen_range_i8() {
        let mut rng = SplitMix64::new(42);
        for _ in 0..1000 {
            let v = rng.gen_range_i8(-3, 3);
            assert!((-3..=3).contains(&v), "gen_range_i8(-3, 3) = {}", v);
        }
    }
}
