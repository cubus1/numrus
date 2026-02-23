//! SimHash-as-GEMM projection: batch f32 embeddings → binary containers.
//!
//! Insight: SimHash IS a matrix multiply followed by sign extraction:
//!   `containers = sign(embeddings × hyperplane_matrix)`
//!
//! Where `hyperplane_matrix` is a (D × container_bits) random matrix
//! seeded deterministically. This replaces per-vector random dot products
//! (25 min for 100K × 512D) with a single batch call (~4 sec).
//!
//! The matrix multiply uses numrus_blas::level3::sgemm for SIMD-accelerated GEMM.

use super::NumArrayU8;
use numrus_core::layout::{Layout, Transpose};

// ============================================================================
// Deterministic random matrix generation (SplitMix64)
// ============================================================================

/// SplitMix64 PRNG — simple, fast, deterministic.
/// Same seed always produces the same hyperplane matrix.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    /// Returns a random f32 in [-1.0, 1.0).
    #[inline]
    fn next_f32(&mut self) -> f32 {
        let u = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32;
        u * 2.0 - 1.0
    }
}

// ============================================================================
// SimHash batch projection
// ============================================================================

/// Batch SimHash projection.
///
/// Projects `n` embeddings of dimension `d` into binary containers of
/// `container_bits` bits each.
///
/// Algorithm:
/// 1. Generate deterministic random hyperplane matrix from `seed`
/// 2. Compute `projections = embeddings × hyperplanes` via tiled matmul
/// 3. Extract sign bits: bit[j] = 1 if projections[i][j] >= 0
///
/// # Arguments
/// * `embeddings` - Flat f32 array, `n × d` row-major
/// * `n` - Number of embeddings
/// * `d` - Embedding dimension (e.g., 512 for CLIP, 1024 for Jina)
/// * `container_bits` - Output container size in bits (e.g., 16384)
/// * `seed` - Deterministic seed for hyperplane generation
///
/// # Returns
/// Vec of `n` NumArrayU8 containers, each `container_bits / 8` bytes.
pub fn simhash_batch_project(
    embeddings: &[f32],
    n: usize,
    d: usize,
    container_bits: usize,
    seed: u64,
) -> Vec<NumArrayU8> {
    assert_eq!(embeddings.len(), n * d);
    assert_eq!(container_bits % 8, 0, "container_bits must be byte-aligned");

    let container_bytes = container_bits / 8;

    // Generate random hyperplane matrix: d × container_bits
    let hyperplanes = generate_hyperplane_matrix(d, container_bits, seed);

    // GEMM: projections = embeddings × hyperplanes
    // (n × d) × (d × container_bits) → (n × container_bits)
    let mut projections = vec![0.0f32; n * container_bits];
    numrus_blas::level3::sgemm(
        Layout::RowMajor,
        Transpose::NoTrans,
        Transpose::NoTrans,
        n,
        container_bits,
        d,
        1.0,
        embeddings,
        d,
        &hyperplanes,
        container_bits,
        0.0,
        &mut projections,
        container_bits,
    );

    // Extract sign bits into packed binary containers
    let mut containers = Vec::with_capacity(n);
    for i in 0..n {
        let row = &projections[i * container_bits..(i + 1) * container_bits];
        let packed = pack_sign_bits(row, container_bytes);
        containers.push(NumArrayU8::new(packed));
    }

    containers
}

/// Single-vector SimHash projection (convenience wrapper).
pub fn simhash_project(embedding: &[f32], container_bits: usize, seed: u64) -> NumArrayU8 {
    let mut result = simhash_batch_project(embedding, 1, embedding.len(), container_bits, seed);
    result.pop().unwrap()
}

/// Generate the deterministic hyperplane matrix.
fn generate_hyperplane_matrix(d: usize, container_bits: usize, seed: u64) -> Vec<f32> {
    let mut rng = SplitMix64::new(seed);
    (0..d * container_bits).map(|_| rng.next_f32()).collect()
}

/// Pack sign bits from f32 projections into a byte array.
///
/// bit[j] = 1 if projection[j] >= 0.0, else 0.
/// Bits are packed MSB-first within each byte.
fn pack_sign_bits(projections: &[f32], container_bytes: usize) -> Vec<u8> {
    let mut packed = vec![0u8; container_bytes];

    for byte_idx in 0..container_bytes {
        let mut byte_val = 0u8;
        for bit_idx in 0..8 {
            let proj_idx = byte_idx * 8 + bit_idx;
            if proj_idx < projections.len() && projections[proj_idx] >= 0.0 {
                byte_val |= 1 << (7 - bit_idx);
            }
        }
        packed[byte_idx] = byte_val;
    }

    packed
}

// ============================================================================
// INT8 SimHash projection (for Container 3 int8 embeddings)
// ============================================================================

/// Project int8 embeddings directly to binary containers.
///
/// Avoids f32 intermediate — uses int8 dot products against
/// random sign hyperplanes for maximum throughput.
pub fn simhash_int8_project(embedding_i8: &[i8], container_bits: usize, seed: u64) -> NumArrayU8 {
    let d = embedding_i8.len();
    let container_bytes = container_bits / 8;

    let mut rng = SplitMix64::new(seed);
    let mut packed = vec![0u8; container_bytes];

    for byte_idx in 0..container_bytes {
        let mut byte_val = 0u8;
        for bit_idx in 0..8 {
            let mut dot: i64 = 0;
            for j in 0..d {
                let h = if rng.next_f32() >= 0.0 { 1i8 } else { -1i8 };
                dot += embedding_i8[j] as i64 * h as i64;
            }
            if dot >= 0 {
                byte_val |= 1 << (7 - bit_idx);
            }
        }
        packed[byte_idx] = byte_val;
    }

    NumArrayU8::new(packed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simhash_deterministic() {
        let emb = vec![1.0f32; 512];
        let c1 = simhash_project(&emb, 8192, 42);
        let c2 = simhash_project(&emb, 8192, 42);
        assert_eq!(c1.get_data(), c2.get_data());
    }

    #[test]
    fn test_simhash_different_seeds() {
        let emb = vec![1.0f32; 512];
        let c1 = simhash_project(&emb, 8192, 42);
        let c2 = simhash_project(&emb, 8192, 123);
        assert_ne!(c1.get_data(), c2.get_data());
    }

    #[test]
    fn test_simhash_similar_inputs_close() {
        let emb1: Vec<f32> = (0..512).map(|i| i as f32 / 512.0).collect();
        let mut emb2 = emb1.clone();
        emb2[0] += 0.001;

        let c1 = simhash_project(&emb1, 8192, 42);
        let c2 = simhash_project(&emb2, 8192, 42);

        let dist = c1.hamming_distance(&c2);
        assert!(
            dist < 1000,
            "Similar inputs should have low Hamming distance, got {}",
            dist
        );
    }

    #[test]
    fn test_simhash_batch() {
        let n = 10;
        let d = 128;
        let embs: Vec<f32> = (0..n * d).map(|i| (i as f32).sin()).collect();
        let containers = simhash_batch_project(&embs, n, d, 8192, 42);

        assert_eq!(containers.len(), n);
        for c in &containers {
            assert_eq!(c.len(), 8192 / 8);
        }
    }

    #[test]
    fn test_simhash_container_size_16384() {
        let emb = vec![0.5f32; 1024];
        let c = simhash_project(&emb, 16384, 42);
        assert_eq!(c.len(), 16384 / 8); // 2048 bytes = 1 CogRecord container
    }

    #[test]
    fn test_pack_sign_bits() {
        let proj = vec![1.0f32; 8];
        let packed = pack_sign_bits(&proj, 1);
        assert_eq!(packed, vec![0xFF]);

        let proj = vec![-1.0f32; 8];
        let packed = pack_sign_bits(&proj, 1);
        assert_eq!(packed, vec![0x00]);

        let proj = vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0f32];
        let packed = pack_sign_bits(&proj, 1);
        assert_eq!(packed, vec![0b10101010]);
    }
}
