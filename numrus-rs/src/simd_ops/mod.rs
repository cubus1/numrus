use std::simd::cmp::SimdOrd;
use std::simd::f32x16;
use std::simd::f64x8;
use std::simd::i32x16;
use std::simd::i64x8;
use std::simd::num::SimdFloat;
use std::simd::num::SimdInt;
use std::simd::num::SimdUint;
use std::simd::u8x64;

const LANES_8: usize = 64;
const LANES_32: usize = 16;
const LANES_64: usize = 8;

/// Threshold (in elements) above which bitwise operations are parallelized across threads.
const PARALLEL_THRESHOLD: usize = 65_536;

/// Parallel write into disjoint output slices — zero synchronization.
/// Each thread gets exclusive `&mut` ownership of its slice via `split_at_mut`.
/// Replaces `Arc<Mutex<&mut [T]>>` pattern throughout simd_ops.
#[inline]
fn parallel_into_slices<T: Send>(
    out: &mut [T],
    chunk_size: usize,
    f: impl Fn(usize, &mut [T]) + Send + Sync,
) {
    if chunk_size == 0 || out.is_empty() {
        return;
    }
    let f = &f;
    std::thread::scope(|s| {
        let mut remaining = out;
        let mut offset = 0;
        while !remaining.is_empty() {
            let take = chunk_size.min(remaining.len());
            let (chunk, rest) = remaining.split_at_mut(take);
            remaining = rest;
            let off = offset;
            s.spawn(move || f(off, chunk));
            offset += take;
        }
    });
}

/// Parallel reduction — each thread computes a partial result, summed at the end.
/// Zero synchronization during computation.
#[inline]
fn parallel_reduce_sum(len: usize, f: impl Fn(usize, usize) -> u64 + Send + Sync) -> u64 {
    let n_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let chunk_size = len.div_ceil(n_threads);
    let f = &f;
    std::thread::scope(|s| {
        let mut handles = Vec::new();
        for start in (0..len).step_by(chunk_size) {
            let end = (start + chunk_size).min(len);
            handles.push(s.spawn(move || f(start, end)));
        }
        handles.into_iter().map(|h| h.join().unwrap()).sum()
    })
}

pub trait SimdOps<T> {
    fn matrix_multiply(a: &[T], b: &[T], c: &mut [T], m: usize, k: usize, n: usize);
    fn dot_product(a: &[T], b: &[T]) -> T;
    fn transpose(src: &[T], dst: &mut [T], n: usize, k: usize);
    fn sum(a: &[T]) -> T;
    fn min_simd(a: &[T]) -> T;
    fn max_simd(a: &[T]) -> T;
    fn l1_norm(a: &[T]) -> T;
    fn l2_norm(a: &[T]) -> T;

    /// Element-wise: out[i] = a[i] + scalar
    fn add_scalar(a: &[T], scalar: T, out: &mut [T]);
    /// Element-wise: out[i] = a[i] - scalar
    fn sub_scalar(a: &[T], scalar: T, out: &mut [T]);
    /// Element-wise: out[i] = a[i] * scalar
    fn mul_scalar(a: &[T], scalar: T, out: &mut [T]);
    /// Element-wise: out[i] = a[i] / scalar
    fn div_scalar(a: &[T], scalar: T, out: &mut [T]);
    /// Element-wise: out[i] = a[i] + b[i]
    fn add_array(a: &[T], b: &[T], out: &mut [T]);
    /// Element-wise: out[i] = a[i] - b[i]
    fn sub_array(a: &[T], b: &[T], out: &mut [T]);
    /// Element-wise: out[i] = a[i] * b[i]
    fn mul_array(a: &[T], b: &[T], out: &mut [T]);
    /// Element-wise: out[i] = a[i] / b[i]
    fn div_array(a: &[T], b: &[T], out: &mut [T]);

    /// Batch element-wise exp: out[i] = exp(a[i]).
    /// Uses SIMD polynomial approximation via VML for full bus utilization.
    fn exp_batch(a: &[T], out: &mut [T]);

    /// Batch element-wise ln: out[i] = ln(a[i]).
    /// Uses SIMD Padé series via VML for full bus utilization.
    fn log_batch(a: &[T], out: &mut [T]);
}

/// AVX512-optimized bitwise operations for integer SIMD types.
///
/// All methods write directly into pre-allocated output slices to avoid allocation
/// in the hot path. Implementations use 4x loop unrolling for maximum instruction-level
/// parallelism, and automatically parallelize across threads for arrays larger than
/// `PARALLEL_THRESHOLD`.
pub trait BitwiseSimdOps<T: Copy> {
    /// Element-wise bitwise AND: `out[i] = a[i] & b[i]`
    fn bitwise_and(a: &[T], b: &[T], out: &mut [T]);
    /// Element-wise bitwise XOR: `out[i] = a[i] ^ b[i]`
    fn bitwise_xor(a: &[T], b: &[T], out: &mut [T]);
    /// Element-wise bitwise OR: `out[i] = a[i] | b[i]`
    fn bitwise_or(a: &[T], b: &[T], out: &mut [T]);
    /// Element-wise bitwise NOT: `out[i] = !a[i]`
    fn bitwise_not(a: &[T], out: &mut [T]);
    /// Scalar bitwise AND: `out[i] = a[i] & scalar`
    fn bitwise_and_scalar(a: &[T], scalar: T, out: &mut [T]);
    /// Scalar bitwise XOR: `out[i] = a[i] ^ scalar`
    fn bitwise_xor_scalar(a: &[T], scalar: T, out: &mut [T]);
    /// Scalar bitwise OR: `out[i] = a[i] | scalar`
    fn bitwise_or_scalar(a: &[T], scalar: T, out: &mut [T]);
}

/// Fused hamming-distance operations on bitpacked u8 arrays.
///
/// These are purpose-built for bitpacked hamming distance (XOR + popcount)
/// on arrays whose lengths are multiples of 8192. Every vector is a full
/// 512-bit zmm register (u8×64), and 8192 bytes = exactly 128 vectors.
/// With 4× unrolling, that means 32 iterations per 8192 block—zero waste.
pub trait HammingSimdOps {
    /// Bitpacked hamming distance: popcount(a XOR b).
    /// Returns the number of differing bits between the two byte slices.
    fn hamming_distance(a: &[u8], b: &[u8]) -> u64;

    /// Popcount: count the total number of set bits in the byte slice.
    fn popcount(a: &[u8]) -> u64;

    /// Bitpacked hamming distance on multiple pairs.
    /// `a_vecs` and `b_vecs` are each `count` vectors of length `vec_len`.
    /// Returns a Vec of hamming distances, one per pair.
    /// Uses parallel processing on 16-core machines.
    fn hamming_distance_batch(
        a_vecs: &[u8],
        b_vecs: &[u8],
        vec_len: usize,
        count: usize,
    ) -> Vec<u64>;
}

// TODO(simd): REFACTOR — dot_product_scalar is a scalar fallback for non-SIMD types.
// Used by u8 transpose path. Should be replaced with SIMD dot for supported types.
#[inline(always)]
fn dot_product_scalar<T>(a: &[T], b: &[T]) -> T
where
    T: std::ops::Mul<Output = T> + std::iter::Sum + Copy,
{
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

// TODO(simd): REFACTOR — all 5 transpose() implementations (u8, f32, f64, i32, i64) are scalar
// double-nested loops. Use SIMD 8x8 or 16x16 block transpose with shuffle/unpack intrinsics.
impl SimdOps<u8> for u8x64 {
    fn transpose(src: &[u8], dst: &mut [u8], rows: usize, cols: usize) {
        for i in 0..rows {
            for j in 0..cols {
                dst[j * rows + i] = src[i * cols + j];
            }
        }
    }

    fn matrix_multiply(a: &[u8], b: &[u8], c: &mut [u8], m: usize, k: usize, n: usize) {
        assert_eq!(a.len(), m * k);
        assert_eq!(b.len(), k * n);
        assert_eq!(c.len(), m * n);

        c.fill(0);

        let mut b_transposed = vec![0u8; n * k];
        Self::transpose(b, &mut b_transposed, k, n);

        let n_threads = std::thread::available_parallelism()
            .map(|t| t.get())
            .unwrap_or(4);
        let rows_per_thread = m.div_ceil(n_threads);
        parallel_into_slices(c, rows_per_thread * n, |offset, chunk| {
            let row_start = offset / n;
            let rows_this = chunk.len() / n;
            for i in 0..rows_this {
                let global_row = row_start + i;
                let a_row = &a[global_row * k..(global_row + 1) * k];
                let mut c_row = vec![0u8; n];
                for j in 0..n {
                    let b_col = &b_transposed[j * k..(j + 1) * k];
                    c_row[j] = Self::dot_product(a_row, b_col);
                }
                chunk[i * n..(i + 1) * n].copy_from_slice(&c_row);
            }
        });
    }

    fn dot_product(a: &[u8], b: &[u8]) -> u8 {
        assert_eq!(a.len(), b.len());
        let len = a.len();

        let chunks = len / LANES_8;

        let mut sum1 = u8x64::splat(0);
        let mut sum2 = u8x64::splat(0);

        // Main loop with manual unrolling
        for i in (0..chunks).step_by(2) {
            let a1 = u8x64::from_slice(&a[i * LANES_8..]);
            let b1 = u8x64::from_slice(&b[i * LANES_8..]);
            sum1 += a1 * b1;

            if i + 1 < chunks {
                let a2 = u8x64::from_slice(&a[(i + 1) * LANES_8..]);
                let b2 = u8x64::from_slice(&b[(i + 1) * LANES_8..]);
                sum2 += a2 * b2;
            }
        }

        let mut scalar_sum = (sum1 + sum2).reduce_sum();

        // Efficient tail handling
        let remainder = len % LANES_8;
        if remainder > 0 {
            let tail_start = len - remainder;
            scalar_sum += dot_product_scalar(&a[tail_start..], &b[tail_start..]);
        }

        scalar_sum
    }

    fn sum(a: &[u8]) -> u8 {
        let mut sum = u8x64::splat(0);
        let chunks = a.len() / 64;

        for i in 0..chunks {
            let simd_chunk = u8x64::from_slice(&a[i * 64..]);
            sum += simd_chunk;
        }

        let mut scalar_sum = sum.reduce_sum();
        // Sum any remaining elements that didn't fit into a SIMD chunk
        for i in (chunks * LANES_8)..a.len() {
            scalar_sum += a[i];
        }

        scalar_sum
    }

    fn min_simd(a: &[u8]) -> u8 {
        let simd_min_initial_value = u8::MAX;
        let mut simd_min = u8x64::splat(simd_min_initial_value);
        let chunks = a.len() / LANES_8;
        for i in 0..chunks {
            let simd_chunk = u8x64::from_slice(&a[i * LANES_8..]);
            simd_min = simd_min.simd_min(simd_chunk);
        }
        let mut final_min = simd_min.reduce_min();
        // Handle remaining elements
        for i in chunks * LANES_8..a.len() {
            final_min = final_min.min(a[i]);
        }
        final_min
    }

    fn max_simd(a: &[u8]) -> u8 {
        let simd_max_initial_value = u8::MIN;
        let mut simd_max = u8x64::splat(simd_max_initial_value);
        let chunks = a.len() / LANES_8;
        for i in 0..chunks {
            let simd_chunk = u8x64::from_slice(&a[i * LANES_8..]);
            simd_max = simd_max.simd_max(simd_chunk);
        }
        let mut final_max = simd_max.reduce_max();
        // Handle remaining elements
        for i in chunks * LANES_8..a.len() {
            final_max = final_max.max(a[i]);
        }
        final_max
    }

    fn l1_norm(_a: &[u8]) -> u8 {
        unimplemented!("Norm operations are not supported for u8")
    }

    fn l2_norm(_a: &[u8]) -> u8 {
        unimplemented!("Norm operations are not supported for u8")
    }

    fn add_scalar(a: &[u8], scalar: u8, out: &mut [u8]) {
        let s = u8x64::splat(scalar);
        let chunks = a.len() / LANES_8;
        for i in 0..chunks {
            let base = i * LANES_8;
            let v = u8x64::from_slice(&a[base..]);
            (v + s).copy_to_slice(&mut out[base..base + LANES_8]);
        }
        for i in (chunks * LANES_8)..a.len() {
            out[i] = a[i].wrapping_add(scalar);
        }
    }

    fn sub_scalar(a: &[u8], scalar: u8, out: &mut [u8]) {
        let s = u8x64::splat(scalar);
        let chunks = a.len() / LANES_8;
        for i in 0..chunks {
            let base = i * LANES_8;
            let v = u8x64::from_slice(&a[base..]);
            (v - s).copy_to_slice(&mut out[base..base + LANES_8]);
        }
        for i in (chunks * LANES_8)..a.len() {
            out[i] = a[i].wrapping_sub(scalar);
        }
    }

    fn mul_scalar(a: &[u8], scalar: u8, out: &mut [u8]) {
        // u8 SIMD mul not directly available — scalar loop (auto-vectorized)
        for i in 0..a.len() {
            out[i] = a[i].wrapping_mul(scalar);
        }
    }

    fn div_scalar(a: &[u8], scalar: u8, out: &mut [u8]) {
        for i in 0..a.len() {
            out[i] = a[i] / scalar;
        }
    }

    fn add_array(a: &[u8], b: &[u8], out: &mut [u8]) {
        let chunks = a.len() / LANES_8;
        for i in 0..chunks {
            let base = i * LANES_8;
            let va = u8x64::from_slice(&a[base..]);
            let vb = u8x64::from_slice(&b[base..]);
            (va + vb).copy_to_slice(&mut out[base..base + LANES_8]);
        }
        for i in (chunks * LANES_8)..a.len() {
            out[i] = a[i].wrapping_add(b[i]);
        }
    }

    fn sub_array(a: &[u8], b: &[u8], out: &mut [u8]) {
        let chunks = a.len() / LANES_8;
        for i in 0..chunks {
            let base = i * LANES_8;
            let va = u8x64::from_slice(&a[base..]);
            let vb = u8x64::from_slice(&b[base..]);
            (va - vb).copy_to_slice(&mut out[base..base + LANES_8]);
        }
        for i in (chunks * LANES_8)..a.len() {
            out[i] = a[i].wrapping_sub(b[i]);
        }
    }

    fn mul_array(a: &[u8], b: &[u8], out: &mut [u8]) {
        for i in 0..a.len() {
            out[i] = a[i].wrapping_mul(b[i]);
        }
    }

    fn div_array(a: &[u8], b: &[u8], out: &mut [u8]) {
        for i in 0..a.len() {
            out[i] = a[i] / b[i];
        }
    }

    fn exp_batch(_a: &[u8], _out: &mut [u8]) {
        unimplemented!("exp not defined for u8");
    }

    fn log_batch(_a: &[u8], _out: &mut [u8]) {
        unimplemented!("log not defined for u8");
    }
}

impl SimdOps<f32> for f32x16 {
    fn transpose(src: &[f32], dst: &mut [f32], rows: usize, cols: usize) {
        for i in 0..rows {
            for j in 0..cols {
                dst[j * rows + i] = src[i * cols + j];
            }
        }
    }

    fn matrix_multiply(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
        assert_eq!(a.len(), m * k);
        assert_eq!(b.len(), k * n);
        assert_eq!(c.len(), m * n);

        c.fill(0.0);

        let mut b_transposed = vec![0.0f32; n * k];
        Self::transpose(b, &mut b_transposed, k, n);

        let n_threads = std::thread::available_parallelism()
            .map(|t| t.get())
            .unwrap_or(4);
        let rows_per_thread = m.div_ceil(n_threads);
        parallel_into_slices(c, rows_per_thread * n, |offset, chunk| {
            let row_start = offset / n;
            let rows_this = chunk.len() / n;
            for i in 0..rows_this {
                let global_row = row_start + i;
                let a_row = &a[global_row * k..(global_row + 1) * k];
                let mut c_row = vec![0.0; n];
                for j in 0..n {
                    let b_col = &b_transposed[j * k..(j + 1) * k];
                    c_row[j] = Self::dot_product(a_row, b_col);
                }
                chunk[i * n..(i + 1) * n].copy_from_slice(&c_row);
            }
        });
    }

    fn dot_product(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        let len = a.len();

        let chunks = len / LANES_32;

        let mut sum1 = f32x16::splat(0.0);
        let mut sum2 = f32x16::splat(0.0);

        // Main loop with manual unrolling
        for i in (0..chunks).step_by(2) {
            let a1 = f32x16::from_slice(&a[i * LANES_32..]);
            let b1 = f32x16::from_slice(&b[i * LANES_32..]);
            sum1 += a1 * b1;

            if i + 1 < chunks {
                let a2 = f32x16::from_slice(&a[(i + 1) * LANES_32..]);
                let b2 = f32x16::from_slice(&b[(i + 1) * LANES_32..]);
                sum2 += a2 * b2;
            }
        }

        let mut scalar_sum = (sum1 + sum2).reduce_sum();

        // Efficient tail handling
        let remainder = len % LANES_32;
        if remainder > 0 {
            let tail_start = len - remainder;
            scalar_sum += dot_product_scalar(&a[tail_start..], &b[tail_start..]);
        }

        scalar_sum
    }

    fn sum(a: &[f32]) -> f32 {
        let mut sum = f32x16::splat(0.0);
        let chunks = a.len() / LANES_32;

        for i in 0..chunks {
            let simd_chunk = f32x16::from_slice(&a[i * LANES_32..]);
            sum += simd_chunk;
        }

        let mut scalar_sum = sum.reduce_sum();
        // Sum any remaining elements that didn't fit into a SIMD chunk
        for i in (chunks * LANES_32)..a.len() {
            scalar_sum += a[i];
        }

        scalar_sum
    }

    fn min_simd(a: &[f32]) -> f32 {
        let simd_min_initial_value = f32::MAX;
        let mut simd_min = f32x16::splat(simd_min_initial_value);
        let chunks = a.len() / LANES_32;
        for i in 0..chunks {
            let simd_chunk = f32x16::from_slice(&a[i * LANES_32..]);
            simd_min = simd_min.simd_min(simd_chunk);
        }
        let mut final_min = simd_min.reduce_min();
        // Handle remaining elements
        for i in chunks * LANES_32..a.len() {
            final_min = final_min.min(a[i]);
        }
        final_min
    }

    fn max_simd(a: &[f32]) -> f32 {
        let simd_max_initial_value = f32::MIN;
        let mut simd_max = f32x16::splat(simd_max_initial_value);
        let chunks = a.len() / LANES_32;
        for i in 0..chunks {
            let simd_chunk = f32x16::from_slice(&a[i * LANES_32..]);
            simd_max = simd_max.simd_max(simd_chunk);
        }
        let mut final_max = simd_max.reduce_max();
        // Handle remaining elements
        for i in chunks * LANES_32..a.len() {
            final_max = final_max.max(a[i]);
        }
        final_max
    }

    fn l1_norm(a: &[f32]) -> f32 {
        let mut sum = f32x16::splat(0.0);
        let chunks = a.len() / LANES_32;

        // Main SIMD processing
        for i in 0..chunks {
            let simd_chunk = f32x16::from_slice(&a[i * LANES_32..(i + 1) * LANES_32]);
            sum += simd_chunk.abs();
        }

        // Scalar tail processing
        let mut scalar_sum = sum.reduce_sum();
        for i in chunks * LANES_32..a.len() {
            scalar_sum += a[i].abs();
        }

        scalar_sum
    }

    fn l2_norm(a: &[f32]) -> f32 {
        let mut sum = f32x16::splat(0.0);
        let chunks = a.len() / LANES_32;

        // Main SIMD processing with fused multiply-add
        for i in 0..chunks {
            let simd_chunk = f32x16::from_slice(&a[i * LANES_32..(i + 1) * LANES_32]);
            sum += simd_chunk * simd_chunk;
        }

        // Scalar tail processing
        let mut scalar_sum = sum.reduce_sum();
        for i in chunks * LANES_32..a.len() {
            scalar_sum += a[i] * a[i];
        }

        scalar_sum.sqrt()
    }

    fn add_scalar(a: &[f32], scalar: f32, out: &mut [f32]) {
        let s = f32x16::splat(scalar);
        let chunks = a.len() / LANES_32;
        for i in 0..chunks {
            let base = i * LANES_32;
            let v = f32x16::from_slice(&a[base..]);
            (v + s).copy_to_slice(&mut out[base..base + LANES_32]);
        }
        for i in (chunks * LANES_32)..a.len() {
            out[i] = a[i] + scalar;
        }
    }

    fn sub_scalar(a: &[f32], scalar: f32, out: &mut [f32]) {
        let s = f32x16::splat(scalar);
        let chunks = a.len() / LANES_32;
        for i in 0..chunks {
            let base = i * LANES_32;
            let v = f32x16::from_slice(&a[base..]);
            (v - s).copy_to_slice(&mut out[base..base + LANES_32]);
        }
        for i in (chunks * LANES_32)..a.len() {
            out[i] = a[i] - scalar;
        }
    }

    fn mul_scalar(a: &[f32], scalar: f32, out: &mut [f32]) {
        let s = f32x16::splat(scalar);
        let chunks = a.len() / LANES_32;
        for i in 0..chunks {
            let base = i * LANES_32;
            let v = f32x16::from_slice(&a[base..]);
            (v * s).copy_to_slice(&mut out[base..base + LANES_32]);
        }
        for i in (chunks * LANES_32)..a.len() {
            out[i] = a[i] * scalar;
        }
    }

    fn div_scalar(a: &[f32], scalar: f32, out: &mut [f32]) {
        let s = f32x16::splat(scalar);
        let chunks = a.len() / LANES_32;
        for i in 0..chunks {
            let base = i * LANES_32;
            let v = f32x16::from_slice(&a[base..]);
            (v / s).copy_to_slice(&mut out[base..base + LANES_32]);
        }
        for i in (chunks * LANES_32)..a.len() {
            out[i] = a[i] / scalar;
        }
    }

    fn add_array(a: &[f32], b: &[f32], out: &mut [f32]) {
        let chunks = a.len() / LANES_32;
        for i in 0..chunks {
            let base = i * LANES_32;
            let va = f32x16::from_slice(&a[base..]);
            let vb = f32x16::from_slice(&b[base..]);
            (va + vb).copy_to_slice(&mut out[base..base + LANES_32]);
        }
        for i in (chunks * LANES_32)..a.len() {
            out[i] = a[i] + b[i];
        }
    }

    fn sub_array(a: &[f32], b: &[f32], out: &mut [f32]) {
        let chunks = a.len() / LANES_32;
        for i in 0..chunks {
            let base = i * LANES_32;
            let va = f32x16::from_slice(&a[base..]);
            let vb = f32x16::from_slice(&b[base..]);
            (va - vb).copy_to_slice(&mut out[base..base + LANES_32]);
        }
        for i in (chunks * LANES_32)..a.len() {
            out[i] = a[i] - b[i];
        }
    }

    fn mul_array(a: &[f32], b: &[f32], out: &mut [f32]) {
        let chunks = a.len() / LANES_32;
        for i in 0..chunks {
            let base = i * LANES_32;
            let va = f32x16::from_slice(&a[base..]);
            let vb = f32x16::from_slice(&b[base..]);
            (va * vb).copy_to_slice(&mut out[base..base + LANES_32]);
        }
        for i in (chunks * LANES_32)..a.len() {
            out[i] = a[i] * b[i];
        }
    }

    fn div_array(a: &[f32], b: &[f32], out: &mut [f32]) {
        let chunks = a.len() / LANES_32;
        for i in 0..chunks {
            let base = i * LANES_32;
            let va = f32x16::from_slice(&a[base..]);
            let vb = f32x16::from_slice(&b[base..]);
            (va / vb).copy_to_slice(&mut out[base..base + LANES_32]);
        }
        for i in (chunks * LANES_32)..a.len() {
            out[i] = a[i] / b[i];
        }
    }

    fn exp_batch(a: &[f32], out: &mut [f32]) {
        numrus_mkl::vml::vsexp(a, out);
    }

    fn log_batch(a: &[f32], out: &mut [f32]) {
        numrus_mkl::vml::vsln(a, out);
    }
}

impl SimdOps<f64> for f64x8 {
    fn transpose(src: &[f64], dst: &mut [f64], rows: usize, cols: usize) {
        for i in 0..rows {
            for j in 0..cols {
                dst[j * rows + i] = src[i * cols + j];
            }
        }
    }

    fn matrix_multiply(a: &[f64], b: &[f64], c: &mut [f64], m: usize, k: usize, n: usize) {
        assert_eq!(a.len(), m * k);
        assert_eq!(b.len(), k * n);
        assert_eq!(c.len(), m * n);

        c.fill(0.0);

        let mut b_transposed = vec![0.0f64; n * k];
        Self::transpose(b, &mut b_transposed, k, n);

        let n_threads = std::thread::available_parallelism()
            .map(|t| t.get())
            .unwrap_or(4);
        let rows_per_thread = m.div_ceil(n_threads);
        parallel_into_slices(c, rows_per_thread * n, |offset, chunk| {
            let row_start = offset / n;
            let rows_this = chunk.len() / n;
            for i in 0..rows_this {
                let global_row = row_start + i;
                let a_row = &a[global_row * k..(global_row + 1) * k];
                let mut c_row = vec![0.0; n];
                for j in 0..n {
                    let b_col = &b_transposed[j * k..(j + 1) * k];
                    c_row[j] = Self::dot_product(a_row, b_col);
                }
                chunk[i * n..(i + 1) * n].copy_from_slice(&c_row);
            }
        });
    }

    fn dot_product(a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), b.len());
        let len = a.len();

        let chunks = len / LANES_64;

        let mut sum1 = f64x8::splat(0.0);
        let mut sum2 = f64x8::splat(0.0);

        // Main loop with manual unrolling
        for i in (0..chunks).step_by(2) {
            let a1 = f64x8::from_slice(&a[i * LANES_64..]);
            let b1 = f64x8::from_slice(&b[i * LANES_64..]);
            sum1 += a1 * b1;

            if i + 1 < chunks {
                let a2 = f64x8::from_slice(&a[(i + 1) * LANES_64..]);
                let b2 = f64x8::from_slice(&b[(i + 1) * LANES_64..]);
                sum2 += a2 * b2;
            }
        }

        let mut scalar_sum = (sum1 + sum2).reduce_sum();

        // Efficient tail handling
        let remainder = len % LANES_64;
        if remainder > 0 {
            let tail_start = len - remainder;
            scalar_sum += dot_product_scalar(&a[tail_start..], &b[tail_start..]);
        }

        scalar_sum
    }

    fn sum(a: &[f64]) -> f64 {
        let mut sum = f64x8::splat(0.0);
        let chunks = a.len() / LANES_64;

        for i in 0..chunks {
            let simd_chunk = f64x8::from_slice(&a[i * LANES_64..]);
            sum += simd_chunk;
        }

        let mut scalar_sum = sum.reduce_sum();
        // Sum any remaining elements that didn't fit into a SIMD chunk
        for i in (chunks * LANES_64)..a.len() {
            scalar_sum += a[i];
        }

        scalar_sum
    }

    fn min_simd(a: &[f64]) -> f64 {
        let simd_min_initial_value = f64::MAX;
        let mut simd_min = f64x8::splat(simd_min_initial_value);
        let chunks = a.len() / LANES_64;
        for i in 0..chunks {
            let simd_chunk = f64x8::from_slice(&a[i * LANES_64..]);
            simd_min = simd_min.simd_min(simd_chunk);
        }
        let mut final_min = simd_min.reduce_min();
        // Handle remaining elements
        for i in chunks * LANES_64..a.len() {
            final_min = final_min.min(a[i]);
        }
        final_min
    }

    fn max_simd(a: &[f64]) -> f64 {
        let simd_max_initial_value = f64::MIN;
        let mut simd_max = f64x8::splat(simd_max_initial_value);
        let chunks = a.len() / LANES_64;
        for i in 0..chunks {
            let simd_chunk = f64x8::from_slice(&a[i * LANES_64..]);
            simd_max = simd_max.simd_max(simd_chunk);
        }
        let mut final_max = simd_max.reduce_max();
        // Handle remaining elements
        for i in chunks * LANES_64..a.len() {
            final_max = final_max.max(a[i]);
        }
        final_max
    }

    fn l1_norm(a: &[f64]) -> f64 {
        let mut sum = f64x8::splat(0.0);
        let chunks = a.len() / LANES_64;

        for i in 0..chunks {
            let simd_chunk = f64x8::from_slice(&a[i * LANES_64..(i + 1) * LANES_64]);
            sum += simd_chunk.abs();
        }

        let mut scalar_sum = sum.reduce_sum();
        for i in chunks * LANES_64..a.len() {
            scalar_sum += a[i].abs();
        }

        scalar_sum
    }

    fn l2_norm(a: &[f64]) -> f64 {
        let mut sum = f64x8::splat(0.0);
        let chunks = a.len() / LANES_64;

        for i in 0..chunks {
            let simd_chunk = f64x8::from_slice(&a[i * LANES_64..(i + 1) * LANES_64]);
            sum += simd_chunk * simd_chunk;
        }

        let mut scalar_sum = sum.reduce_sum();
        for i in chunks * LANES_64..a.len() {
            scalar_sum += a[i] * a[i];
        }

        scalar_sum.sqrt()
    }

    fn add_scalar(a: &[f64], scalar: f64, out: &mut [f64]) {
        let s = f64x8::splat(scalar);
        let chunks = a.len() / LANES_64;
        for i in 0..chunks {
            let base = i * LANES_64;
            let v = f64x8::from_slice(&a[base..]);
            (v + s).copy_to_slice(&mut out[base..base + LANES_64]);
        }
        for i in (chunks * LANES_64)..a.len() {
            out[i] = a[i] + scalar;
        }
    }

    fn sub_scalar(a: &[f64], scalar: f64, out: &mut [f64]) {
        let s = f64x8::splat(scalar);
        let chunks = a.len() / LANES_64;
        for i in 0..chunks {
            let base = i * LANES_64;
            let v = f64x8::from_slice(&a[base..]);
            (v - s).copy_to_slice(&mut out[base..base + LANES_64]);
        }
        for i in (chunks * LANES_64)..a.len() {
            out[i] = a[i] - scalar;
        }
    }

    fn mul_scalar(a: &[f64], scalar: f64, out: &mut [f64]) {
        let s = f64x8::splat(scalar);
        let chunks = a.len() / LANES_64;
        for i in 0..chunks {
            let base = i * LANES_64;
            let v = f64x8::from_slice(&a[base..]);
            (v * s).copy_to_slice(&mut out[base..base + LANES_64]);
        }
        for i in (chunks * LANES_64)..a.len() {
            out[i] = a[i] * scalar;
        }
    }

    fn div_scalar(a: &[f64], scalar: f64, out: &mut [f64]) {
        let s = f64x8::splat(scalar);
        let chunks = a.len() / LANES_64;
        for i in 0..chunks {
            let base = i * LANES_64;
            let v = f64x8::from_slice(&a[base..]);
            (v / s).copy_to_slice(&mut out[base..base + LANES_64]);
        }
        for i in (chunks * LANES_64)..a.len() {
            out[i] = a[i] / scalar;
        }
    }

    fn add_array(a: &[f64], b: &[f64], out: &mut [f64]) {
        let chunks = a.len() / LANES_64;
        for i in 0..chunks {
            let base = i * LANES_64;
            let va = f64x8::from_slice(&a[base..]);
            let vb = f64x8::from_slice(&b[base..]);
            (va + vb).copy_to_slice(&mut out[base..base + LANES_64]);
        }
        for i in (chunks * LANES_64)..a.len() {
            out[i] = a[i] + b[i];
        }
    }

    fn sub_array(a: &[f64], b: &[f64], out: &mut [f64]) {
        let chunks = a.len() / LANES_64;
        for i in 0..chunks {
            let base = i * LANES_64;
            let va = f64x8::from_slice(&a[base..]);
            let vb = f64x8::from_slice(&b[base..]);
            (va - vb).copy_to_slice(&mut out[base..base + LANES_64]);
        }
        for i in (chunks * LANES_64)..a.len() {
            out[i] = a[i] - b[i];
        }
    }

    fn mul_array(a: &[f64], b: &[f64], out: &mut [f64]) {
        let chunks = a.len() / LANES_64;
        for i in 0..chunks {
            let base = i * LANES_64;
            let va = f64x8::from_slice(&a[base..]);
            let vb = f64x8::from_slice(&b[base..]);
            (va * vb).copy_to_slice(&mut out[base..base + LANES_64]);
        }
        for i in (chunks * LANES_64)..a.len() {
            out[i] = a[i] * b[i];
        }
    }

    fn div_array(a: &[f64], b: &[f64], out: &mut [f64]) {
        let chunks = a.len() / LANES_64;
        for i in 0..chunks {
            let base = i * LANES_64;
            let va = f64x8::from_slice(&a[base..]);
            let vb = f64x8::from_slice(&b[base..]);
            (va / vb).copy_to_slice(&mut out[base..base + LANES_64]);
        }
        for i in (chunks * LANES_64)..a.len() {
            out[i] = a[i] / b[i];
        }
    }

    fn exp_batch(a: &[f64], out: &mut [f64]) {
        numrus_mkl::vml::vdexp(a, out);
    }

    fn log_batch(a: &[f64], out: &mut [f64]) {
        numrus_mkl::vml::vdln(a, out);
    }
}

// ---------------------------------------------------------------------------
// SimdOps for i32x16 (AVX-512 width: 16 × 32-bit = 512 bits)
// ---------------------------------------------------------------------------
impl SimdOps<i32> for i32x16 {
    fn transpose(src: &[i32], dst: &mut [i32], rows: usize, cols: usize) {
        for i in 0..rows {
            for j in 0..cols {
                dst[j * rows + i] = src[i * cols + j];
            }
        }
    }

    fn matrix_multiply(a: &[i32], b: &[i32], c: &mut [i32], m: usize, k: usize, n: usize) {
        assert_eq!(a.len(), m * k);
        assert_eq!(b.len(), k * n);
        assert_eq!(c.len(), m * n);
        c.fill(0);
        let mut b_transposed = vec![0i32; n * k];
        Self::transpose(b, &mut b_transposed, k, n);
        let n_threads = std::thread::available_parallelism()
            .map(|t| t.get())
            .unwrap_or(4);
        let rows_per_thread = m.div_ceil(n_threads);
        parallel_into_slices(c, rows_per_thread * n, |offset, chunk| {
            let row_start = offset / n;
            let rows_this = chunk.len() / n;
            for i in 0..rows_this {
                let global_row = row_start + i;
                let a_row = &a[global_row * k..(global_row + 1) * k];
                let mut c_row = vec![0i32; n];
                for j in 0..n {
                    let b_col = &b_transposed[j * k..(j + 1) * k];
                    c_row[j] = Self::dot_product(a_row, b_col);
                }
                chunk[i * n..(i + 1) * n].copy_from_slice(&c_row);
            }
        });
    }

    fn dot_product(a: &[i32], b: &[i32]) -> i32 {
        assert_eq!(a.len(), b.len());
        let len = a.len();
        let chunks = len / LANES_32;
        let mut sum1 = i32x16::splat(0);
        let mut sum2 = i32x16::splat(0);
        for i in (0..chunks).step_by(2) {
            let a1 = i32x16::from_slice(&a[i * LANES_32..]);
            let b1 = i32x16::from_slice(&b[i * LANES_32..]);
            sum1 += a1 * b1;
            if i + 1 < chunks {
                let a2 = i32x16::from_slice(&a[(i + 1) * LANES_32..]);
                let b2 = i32x16::from_slice(&b[(i + 1) * LANES_32..]);
                sum2 += a2 * b2;
            }
        }
        let mut scalar_sum = (sum1 + sum2).reduce_sum();
        let remainder = len % LANES_32;
        if remainder > 0 {
            let tail_start = len - remainder;
            scalar_sum += dot_product_scalar(&a[tail_start..], &b[tail_start..]);
        }
        scalar_sum
    }

    fn sum(a: &[i32]) -> i32 {
        let mut sum = i32x16::splat(0);
        let chunks = a.len() / LANES_32;
        for i in 0..chunks {
            let simd_chunk = i32x16::from_slice(&a[i * LANES_32..]);
            sum += simd_chunk;
        }
        let mut scalar_sum = sum.reduce_sum();
        for i in (chunks * LANES_32)..a.len() {
            scalar_sum += a[i];
        }
        scalar_sum
    }

    fn min_simd(a: &[i32]) -> i32 {
        let mut simd_min = i32x16::splat(i32::MAX);
        let chunks = a.len() / LANES_32;
        for i in 0..chunks {
            let simd_chunk = i32x16::from_slice(&a[i * LANES_32..]);
            simd_min = simd_min.simd_min(simd_chunk);
        }
        let mut final_min = simd_min.reduce_min();
        for i in chunks * LANES_32..a.len() {
            final_min = final_min.min(a[i]);
        }
        final_min
    }

    fn max_simd(a: &[i32]) -> i32 {
        let mut simd_max = i32x16::splat(i32::MIN);
        let chunks = a.len() / LANES_32;
        for i in 0..chunks {
            let simd_chunk = i32x16::from_slice(&a[i * LANES_32..]);
            simd_max = simd_max.simd_max(simd_chunk);
        }
        let mut final_max = simd_max.reduce_max();
        for i in chunks * LANES_32..a.len() {
            final_max = final_max.max(a[i]);
        }
        final_max
    }

    fn l1_norm(a: &[i32]) -> i32 {
        let mut sum = i32x16::splat(0);
        let chunks = a.len() / LANES_32;
        for i in 0..chunks {
            let simd_chunk = i32x16::from_slice(&a[i * LANES_32..(i + 1) * LANES_32]);
            sum += simd_chunk.abs();
        }
        let mut scalar_sum = sum.reduce_sum();
        for i in chunks * LANES_32..a.len() {
            scalar_sum += a[i].abs();
        }
        scalar_sum
    }

    fn l2_norm(a: &[i32]) -> i32 {
        let mut sum = i32x16::splat(0);
        let chunks = a.len() / LANES_32;
        for i in 0..chunks {
            let simd_chunk = i32x16::from_slice(&a[i * LANES_32..(i + 1) * LANES_32]);
            sum += simd_chunk * simd_chunk;
        }
        let mut scalar_sum = sum.reduce_sum();
        for i in chunks * LANES_32..a.len() {
            scalar_sum += a[i] * a[i];
        }
        (scalar_sum as f64).sqrt() as i32
    }

    fn add_scalar(a: &[i32], scalar: i32, out: &mut [i32]) {
        let s = i32x16::splat(scalar);
        let chunks = a.len() / LANES_32;
        for i in 0..chunks {
            let base = i * LANES_32;
            let v = i32x16::from_slice(&a[base..]);
            (v + s).copy_to_slice(&mut out[base..base + LANES_32]);
        }
        for i in (chunks * LANES_32)..a.len() {
            out[i] = a[i] + scalar;
        }
    }

    fn sub_scalar(a: &[i32], scalar: i32, out: &mut [i32]) {
        let s = i32x16::splat(scalar);
        let chunks = a.len() / LANES_32;
        for i in 0..chunks {
            let base = i * LANES_32;
            let v = i32x16::from_slice(&a[base..]);
            (v - s).copy_to_slice(&mut out[base..base + LANES_32]);
        }
        for i in (chunks * LANES_32)..a.len() {
            out[i] = a[i] - scalar;
        }
    }

    fn mul_scalar(a: &[i32], scalar: i32, out: &mut [i32]) {
        let s = i32x16::splat(scalar);
        let chunks = a.len() / LANES_32;
        for i in 0..chunks {
            let base = i * LANES_32;
            let v = i32x16::from_slice(&a[base..]);
            (v * s).copy_to_slice(&mut out[base..base + LANES_32]);
        }
        for i in (chunks * LANES_32)..a.len() {
            out[i] = a[i] * scalar;
        }
    }

    fn div_scalar(a: &[i32], scalar: i32, out: &mut [i32]) {
        let s = i32x16::splat(scalar);
        let chunks = a.len() / LANES_32;
        for i in 0..chunks {
            let base = i * LANES_32;
            let v = i32x16::from_slice(&a[base..]);
            (v / s).copy_to_slice(&mut out[base..base + LANES_32]);
        }
        for i in (chunks * LANES_32)..a.len() {
            out[i] = a[i] / scalar;
        }
    }

    fn add_array(a: &[i32], b: &[i32], out: &mut [i32]) {
        let chunks = a.len() / LANES_32;
        for i in 0..chunks {
            let base = i * LANES_32;
            let va = i32x16::from_slice(&a[base..]);
            let vb = i32x16::from_slice(&b[base..]);
            (va + vb).copy_to_slice(&mut out[base..base + LANES_32]);
        }
        for i in (chunks * LANES_32)..a.len() {
            out[i] = a[i] + b[i];
        }
    }

    fn sub_array(a: &[i32], b: &[i32], out: &mut [i32]) {
        let chunks = a.len() / LANES_32;
        for i in 0..chunks {
            let base = i * LANES_32;
            let va = i32x16::from_slice(&a[base..]);
            let vb = i32x16::from_slice(&b[base..]);
            (va - vb).copy_to_slice(&mut out[base..base + LANES_32]);
        }
        for i in (chunks * LANES_32)..a.len() {
            out[i] = a[i] - b[i];
        }
    }

    fn mul_array(a: &[i32], b: &[i32], out: &mut [i32]) {
        let chunks = a.len() / LANES_32;
        for i in 0..chunks {
            let base = i * LANES_32;
            let va = i32x16::from_slice(&a[base..]);
            let vb = i32x16::from_slice(&b[base..]);
            (va * vb).copy_to_slice(&mut out[base..base + LANES_32]);
        }
        for i in (chunks * LANES_32)..a.len() {
            out[i] = a[i] * b[i];
        }
    }

    fn div_array(a: &[i32], b: &[i32], out: &mut [i32]) {
        let chunks = a.len() / LANES_32;
        for i in 0..chunks {
            let base = i * LANES_32;
            let va = i32x16::from_slice(&a[base..]);
            let vb = i32x16::from_slice(&b[base..]);
            (va / vb).copy_to_slice(&mut out[base..base + LANES_32]);
        }
        for i in (chunks * LANES_32)..a.len() {
            out[i] = a[i] / b[i];
        }
    }

    fn exp_batch(_a: &[i32], _out: &mut [i32]) {
        unimplemented!("exp not defined for i32");
    }

    fn log_batch(_a: &[i32], _out: &mut [i32]) {
        unimplemented!("log not defined for i32");
    }
}

// ---------------------------------------------------------------------------
// SimdOps for i64x8 (AVX-512 width: 8 × 64-bit = 512 bits)
// ---------------------------------------------------------------------------
impl SimdOps<i64> for i64x8 {
    fn transpose(src: &[i64], dst: &mut [i64], rows: usize, cols: usize) {
        for i in 0..rows {
            for j in 0..cols {
                dst[j * rows + i] = src[i * cols + j];
            }
        }
    }

    fn matrix_multiply(a: &[i64], b: &[i64], c: &mut [i64], m: usize, k: usize, n: usize) {
        assert_eq!(a.len(), m * k);
        assert_eq!(b.len(), k * n);
        assert_eq!(c.len(), m * n);
        c.fill(0);
        let mut b_transposed = vec![0i64; n * k];
        Self::transpose(b, &mut b_transposed, k, n);
        let n_threads = std::thread::available_parallelism()
            .map(|t| t.get())
            .unwrap_or(4);
        let rows_per_thread = m.div_ceil(n_threads);
        parallel_into_slices(c, rows_per_thread * n, |offset, chunk| {
            let row_start = offset / n;
            let rows_this = chunk.len() / n;
            for i in 0..rows_this {
                let global_row = row_start + i;
                let a_row = &a[global_row * k..(global_row + 1) * k];
                let mut c_row = vec![0i64; n];
                for j in 0..n {
                    let b_col = &b_transposed[j * k..(j + 1) * k];
                    c_row[j] = Self::dot_product(a_row, b_col);
                }
                chunk[i * n..(i + 1) * n].copy_from_slice(&c_row);
            }
        });
    }

    fn dot_product(a: &[i64], b: &[i64]) -> i64 {
        assert_eq!(a.len(), b.len());
        let len = a.len();
        let chunks = len / LANES_64;
        let mut sum1 = i64x8::splat(0);
        let mut sum2 = i64x8::splat(0);
        for i in (0..chunks).step_by(2) {
            let a1 = i64x8::from_slice(&a[i * LANES_64..]);
            let b1 = i64x8::from_slice(&b[i * LANES_64..]);
            sum1 += a1 * b1;
            if i + 1 < chunks {
                let a2 = i64x8::from_slice(&a[(i + 1) * LANES_64..]);
                let b2 = i64x8::from_slice(&b[(i + 1) * LANES_64..]);
                sum2 += a2 * b2;
            }
        }
        let mut scalar_sum = (sum1 + sum2).reduce_sum();
        let remainder = len % LANES_64;
        if remainder > 0 {
            let tail_start = len - remainder;
            scalar_sum += dot_product_scalar(&a[tail_start..], &b[tail_start..]);
        }
        scalar_sum
    }

    fn sum(a: &[i64]) -> i64 {
        let mut sum = i64x8::splat(0);
        let chunks = a.len() / LANES_64;
        for i in 0..chunks {
            let simd_chunk = i64x8::from_slice(&a[i * LANES_64..]);
            sum += simd_chunk;
        }
        let mut scalar_sum = sum.reduce_sum();
        for i in (chunks * LANES_64)..a.len() {
            scalar_sum += a[i];
        }
        scalar_sum
    }

    fn min_simd(a: &[i64]) -> i64 {
        let mut simd_min = i64x8::splat(i64::MAX);
        let chunks = a.len() / LANES_64;
        for i in 0..chunks {
            let simd_chunk = i64x8::from_slice(&a[i * LANES_64..]);
            simd_min = simd_min.simd_min(simd_chunk);
        }
        let mut final_min = simd_min.reduce_min();
        for i in chunks * LANES_64..a.len() {
            final_min = final_min.min(a[i]);
        }
        final_min
    }

    fn max_simd(a: &[i64]) -> i64 {
        let mut simd_max = i64x8::splat(i64::MIN);
        let chunks = a.len() / LANES_64;
        for i in 0..chunks {
            let simd_chunk = i64x8::from_slice(&a[i * LANES_64..]);
            simd_max = simd_max.simd_max(simd_chunk);
        }
        let mut final_max = simd_max.reduce_max();
        for i in chunks * LANES_64..a.len() {
            final_max = final_max.max(a[i]);
        }
        final_max
    }

    fn l1_norm(a: &[i64]) -> i64 {
        let mut sum = i64x8::splat(0);
        let chunks = a.len() / LANES_64;
        for i in 0..chunks {
            let simd_chunk = i64x8::from_slice(&a[i * LANES_64..(i + 1) * LANES_64]);
            sum += simd_chunk.abs();
        }
        let mut scalar_sum = sum.reduce_sum();
        for i in chunks * LANES_64..a.len() {
            scalar_sum += a[i].abs();
        }
        scalar_sum
    }

    fn l2_norm(a: &[i64]) -> i64 {
        let mut sum = i64x8::splat(0);
        let chunks = a.len() / LANES_64;
        for i in 0..chunks {
            let simd_chunk = i64x8::from_slice(&a[i * LANES_64..(i + 1) * LANES_64]);
            sum += simd_chunk * simd_chunk;
        }
        let mut scalar_sum = sum.reduce_sum();
        for i in chunks * LANES_64..a.len() {
            scalar_sum += a[i] * a[i];
        }
        (scalar_sum as f64).sqrt() as i64
    }

    fn add_scalar(a: &[i64], scalar: i64, out: &mut [i64]) {
        let s = i64x8::splat(scalar);
        let chunks = a.len() / LANES_64;
        for i in 0..chunks {
            let base = i * LANES_64;
            let v = i64x8::from_slice(&a[base..]);
            (v + s).copy_to_slice(&mut out[base..base + LANES_64]);
        }
        for i in (chunks * LANES_64)..a.len() {
            out[i] = a[i] + scalar;
        }
    }

    fn sub_scalar(a: &[i64], scalar: i64, out: &mut [i64]) {
        let s = i64x8::splat(scalar);
        let chunks = a.len() / LANES_64;
        for i in 0..chunks {
            let base = i * LANES_64;
            let v = i64x8::from_slice(&a[base..]);
            (v - s).copy_to_slice(&mut out[base..base + LANES_64]);
        }
        for i in (chunks * LANES_64)..a.len() {
            out[i] = a[i] - scalar;
        }
    }

    fn mul_scalar(a: &[i64], scalar: i64, out: &mut [i64]) {
        let s = i64x8::splat(scalar);
        let chunks = a.len() / LANES_64;
        for i in 0..chunks {
            let base = i * LANES_64;
            let v = i64x8::from_slice(&a[base..]);
            (v * s).copy_to_slice(&mut out[base..base + LANES_64]);
        }
        for i in (chunks * LANES_64)..a.len() {
            out[i] = a[i] * scalar;
        }
    }

    fn div_scalar(a: &[i64], scalar: i64, out: &mut [i64]) {
        let s = i64x8::splat(scalar);
        let chunks = a.len() / LANES_64;
        for i in 0..chunks {
            let base = i * LANES_64;
            let v = i64x8::from_slice(&a[base..]);
            (v / s).copy_to_slice(&mut out[base..base + LANES_64]);
        }
        for i in (chunks * LANES_64)..a.len() {
            out[i] = a[i] / scalar;
        }
    }

    fn add_array(a: &[i64], b: &[i64], out: &mut [i64]) {
        let chunks = a.len() / LANES_64;
        for i in 0..chunks {
            let base = i * LANES_64;
            let va = i64x8::from_slice(&a[base..]);
            let vb = i64x8::from_slice(&b[base..]);
            (va + vb).copy_to_slice(&mut out[base..base + LANES_64]);
        }
        for i in (chunks * LANES_64)..a.len() {
            out[i] = a[i] + b[i];
        }
    }

    fn sub_array(a: &[i64], b: &[i64], out: &mut [i64]) {
        let chunks = a.len() / LANES_64;
        for i in 0..chunks {
            let base = i * LANES_64;
            let va = i64x8::from_slice(&a[base..]);
            let vb = i64x8::from_slice(&b[base..]);
            (va - vb).copy_to_slice(&mut out[base..base + LANES_64]);
        }
        for i in (chunks * LANES_64)..a.len() {
            out[i] = a[i] - b[i];
        }
    }

    fn mul_array(a: &[i64], b: &[i64], out: &mut [i64]) {
        let chunks = a.len() / LANES_64;
        for i in 0..chunks {
            let base = i * LANES_64;
            let va = i64x8::from_slice(&a[base..]);
            let vb = i64x8::from_slice(&b[base..]);
            (va * vb).copy_to_slice(&mut out[base..base + LANES_64]);
        }
        for i in (chunks * LANES_64)..a.len() {
            out[i] = a[i] * b[i];
        }
    }

    fn div_array(a: &[i64], b: &[i64], out: &mut [i64]) {
        let chunks = a.len() / LANES_64;
        for i in 0..chunks {
            let base = i * LANES_64;
            let va = i64x8::from_slice(&a[base..]);
            let vb = i64x8::from_slice(&b[base..]);
            (va / vb).copy_to_slice(&mut out[base..base + LANES_64]);
        }
        for i in (chunks * LANES_64)..a.len() {
            out[i] = a[i] / b[i];
        }
    }

    fn exp_batch(_a: &[i64], _out: &mut [i64]) {
        unimplemented!("exp not defined for i64");
    }

    fn log_batch(_a: &[i64], _out: &mut [i64]) {
        unimplemented!("log not defined for i64");
    }
}

// ===========================================================================
// BitwiseSimdOps – AVX-512 accelerated bitwise AND / XOR / OR / NOT
// ===========================================================================
//
// Design mirrors numpy's ufunc inner loop for bitwise operations:
//   1. Process the bulk of data in 512-bit SIMD chunks (u8×64, i32×16, i64×8).
//   2. Use 4× loop unrolling so the CPU can pipeline independent SIMD instructions.
//   3. Handle the tail with scalar ops (no masking overhead).
//   4. For arrays > PARALLEL_THRESHOLD elements, split across threads.
//
// Each SIMD vector maps to one AVX-512 register (zmm), so every iteration
// processes 64 bytes of data with a single vpandd / vpxord / vpord instruction.

// ---- u8x64 (64 bytes per vector = 512 bits) ----

impl BitwiseSimdOps<u8> for u8x64 {
    #[inline]
    fn bitwise_and(a: &[u8], b: &[u8], out: &mut [u8]) {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), out.len());
        let len = a.len();

        if len >= PARALLEL_THRESHOLD {
            let n_threads = std::thread::available_parallelism()
                .map(|t| t.get())
                .unwrap_or(4);
            let chunk_size = len.div_ceil(n_threads);
            parallel_into_slices(out, chunk_size, |offset, chunk| {
                bitwise_and_chunk_u8(
                    &a[offset..offset + chunk.len()],
                    &b[offset..offset + chunk.len()],
                    chunk,
                );
            });
            return;
        }

        bitwise_and_chunk_u8(a, b, out);
    }

    #[inline]
    fn bitwise_xor(a: &[u8], b: &[u8], out: &mut [u8]) {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), out.len());
        let len = a.len();

        if len >= PARALLEL_THRESHOLD {
            let n_threads = std::thread::available_parallelism()
                .map(|t| t.get())
                .unwrap_or(4);
            let chunk_size = len.div_ceil(n_threads);
            parallel_into_slices(out, chunk_size, |offset, chunk| {
                bitwise_xor_chunk_u8(
                    &a[offset..offset + chunk.len()],
                    &b[offset..offset + chunk.len()],
                    chunk,
                );
            });
            return;
        }

        bitwise_xor_chunk_u8(a, b, out);
    }

    #[inline]
    fn bitwise_or(a: &[u8], b: &[u8], out: &mut [u8]) {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), out.len());
        let len = a.len();

        if len >= PARALLEL_THRESHOLD {
            let n_threads = std::thread::available_parallelism()
                .map(|t| t.get())
                .unwrap_or(4);
            let chunk_size = len.div_ceil(n_threads);
            parallel_into_slices(out, chunk_size, |offset, chunk| {
                bitwise_or_chunk_u8(
                    &a[offset..offset + chunk.len()],
                    &b[offset..offset + chunk.len()],
                    chunk,
                );
            });
            return;
        }

        bitwise_or_chunk_u8(a, b, out);
    }

    #[inline]
    fn bitwise_not(a: &[u8], out: &mut [u8]) {
        assert_eq!(a.len(), out.len());
        let len = a.len();

        if len >= PARALLEL_THRESHOLD {
            let n_threads = std::thread::available_parallelism()
                .map(|t| t.get())
                .unwrap_or(4);
            let chunk_size = len.div_ceil(n_threads);
            parallel_into_slices(out, chunk_size, |offset, chunk| {
                bitwise_not_chunk_u8(&a[offset..offset + chunk.len()], chunk);
            });
            return;
        }

        bitwise_not_chunk_u8(a, out);
    }

    #[inline]
    fn bitwise_and_scalar(a: &[u8], scalar: u8, out: &mut [u8]) {
        assert_eq!(a.len(), out.len());
        let len = a.len();
        let splat = u8x64::splat(scalar);
        let chunks = len / LANES_8;
        // 4× unrolled main loop
        let full_quads = chunks / 4;
        for q in 0..full_quads {
            let base = q * 4 * LANES_8;
            let r0 = u8x64::from_slice(&a[base..]);
            let r1 = u8x64::from_slice(&a[base + LANES_8..]);
            let r2 = u8x64::from_slice(&a[base + 2 * LANES_8..]);
            let r3 = u8x64::from_slice(&a[base + 3 * LANES_8..]);
            (r0 & splat).copy_to_slice(&mut out[base..]);
            (r1 & splat).copy_to_slice(&mut out[base + LANES_8..]);
            (r2 & splat).copy_to_slice(&mut out[base + 2 * LANES_8..]);
            (r3 & splat).copy_to_slice(&mut out[base + 3 * LANES_8..]);
        }
        // Remaining full vectors
        for i in full_quads * 4..chunks {
            let off = i * LANES_8;
            let r = u8x64::from_slice(&a[off..]);
            (r & splat).copy_to_slice(&mut out[off..]);
        }
        // Scalar tail
        for i in chunks * LANES_8..len {
            out[i] = a[i] & scalar;
        }
    }

    #[inline]
    fn bitwise_xor_scalar(a: &[u8], scalar: u8, out: &mut [u8]) {
        assert_eq!(a.len(), out.len());
        let len = a.len();
        let splat = u8x64::splat(scalar);
        let chunks = len / LANES_8;
        let full_quads = chunks / 4;
        for q in 0..full_quads {
            let base = q * 4 * LANES_8;
            let r0 = u8x64::from_slice(&a[base..]);
            let r1 = u8x64::from_slice(&a[base + LANES_8..]);
            let r2 = u8x64::from_slice(&a[base + 2 * LANES_8..]);
            let r3 = u8x64::from_slice(&a[base + 3 * LANES_8..]);
            (r0 ^ splat).copy_to_slice(&mut out[base..]);
            (r1 ^ splat).copy_to_slice(&mut out[base + LANES_8..]);
            (r2 ^ splat).copy_to_slice(&mut out[base + 2 * LANES_8..]);
            (r3 ^ splat).copy_to_slice(&mut out[base + 3 * LANES_8..]);
        }
        for i in full_quads * 4..chunks {
            let off = i * LANES_8;
            let r = u8x64::from_slice(&a[off..]);
            (r ^ splat).copy_to_slice(&mut out[off..]);
        }
        for i in chunks * LANES_8..len {
            out[i] = a[i] ^ scalar;
        }
    }

    #[inline]
    fn bitwise_or_scalar(a: &[u8], scalar: u8, out: &mut [u8]) {
        assert_eq!(a.len(), out.len());
        let len = a.len();
        let splat = u8x64::splat(scalar);
        let chunks = len / LANES_8;
        let full_quads = chunks / 4;
        for q in 0..full_quads {
            let base = q * 4 * LANES_8;
            let r0 = u8x64::from_slice(&a[base..]);
            let r1 = u8x64::from_slice(&a[base + LANES_8..]);
            let r2 = u8x64::from_slice(&a[base + 2 * LANES_8..]);
            let r3 = u8x64::from_slice(&a[base + 3 * LANES_8..]);
            (r0 | splat).copy_to_slice(&mut out[base..]);
            (r1 | splat).copy_to_slice(&mut out[base + LANES_8..]);
            (r2 | splat).copy_to_slice(&mut out[base + 2 * LANES_8..]);
            (r3 | splat).copy_to_slice(&mut out[base + 3 * LANES_8..]);
        }
        for i in full_quads * 4..chunks {
            let off = i * LANES_8;
            let r = u8x64::from_slice(&a[off..]);
            (r | splat).copy_to_slice(&mut out[off..]);
        }
        for i in chunks * LANES_8..len {
            out[i] = a[i] | scalar;
        }
    }
}

// Inner chunk processors – 4× unrolled SIMD, no allocation, no branches in the hot path.

#[inline(always)]
fn bitwise_and_chunk_u8(a: &[u8], b: &[u8], out: &mut [u8]) {
    let len = a.len();
    let chunks = len / LANES_8;
    let full_quads = chunks / 4;
    for q in 0..full_quads {
        let base = q * 4 * LANES_8;
        let a0 = u8x64::from_slice(&a[base..]);
        let b0 = u8x64::from_slice(&b[base..]);
        let a1 = u8x64::from_slice(&a[base + LANES_8..]);
        let b1 = u8x64::from_slice(&b[base + LANES_8..]);
        let a2 = u8x64::from_slice(&a[base + 2 * LANES_8..]);
        let b2 = u8x64::from_slice(&b[base + 2 * LANES_8..]);
        let a3 = u8x64::from_slice(&a[base + 3 * LANES_8..]);
        let b3 = u8x64::from_slice(&b[base + 3 * LANES_8..]);
        (a0 & b0).copy_to_slice(&mut out[base..]);
        (a1 & b1).copy_to_slice(&mut out[base + LANES_8..]);
        (a2 & b2).copy_to_slice(&mut out[base + 2 * LANES_8..]);
        (a3 & b3).copy_to_slice(&mut out[base + 3 * LANES_8..]);
    }
    for i in full_quads * 4..chunks {
        let off = i * LANES_8;
        let va = u8x64::from_slice(&a[off..]);
        let vb = u8x64::from_slice(&b[off..]);
        (va & vb).copy_to_slice(&mut out[off..]);
    }
    for i in chunks * LANES_8..len {
        out[i] = a[i] & b[i];
    }
}

#[inline(always)]
fn bitwise_xor_chunk_u8(a: &[u8], b: &[u8], out: &mut [u8]) {
    let len = a.len();
    let chunks = len / LANES_8;
    let full_quads = chunks / 4;
    for q in 0..full_quads {
        let base = q * 4 * LANES_8;
        let a0 = u8x64::from_slice(&a[base..]);
        let b0 = u8x64::from_slice(&b[base..]);
        let a1 = u8x64::from_slice(&a[base + LANES_8..]);
        let b1 = u8x64::from_slice(&b[base + LANES_8..]);
        let a2 = u8x64::from_slice(&a[base + 2 * LANES_8..]);
        let b2 = u8x64::from_slice(&b[base + 2 * LANES_8..]);
        let a3 = u8x64::from_slice(&a[base + 3 * LANES_8..]);
        let b3 = u8x64::from_slice(&b[base + 3 * LANES_8..]);
        (a0 ^ b0).copy_to_slice(&mut out[base..]);
        (a1 ^ b1).copy_to_slice(&mut out[base + LANES_8..]);
        (a2 ^ b2).copy_to_slice(&mut out[base + 2 * LANES_8..]);
        (a3 ^ b3).copy_to_slice(&mut out[base + 3 * LANES_8..]);
    }
    for i in full_quads * 4..chunks {
        let off = i * LANES_8;
        let va = u8x64::from_slice(&a[off..]);
        let vb = u8x64::from_slice(&b[off..]);
        (va ^ vb).copy_to_slice(&mut out[off..]);
    }
    for i in chunks * LANES_8..len {
        out[i] = a[i] ^ b[i];
    }
}

#[inline(always)]
fn bitwise_or_chunk_u8(a: &[u8], b: &[u8], out: &mut [u8]) {
    let len = a.len();
    let chunks = len / LANES_8;
    let full_quads = chunks / 4;
    for q in 0..full_quads {
        let base = q * 4 * LANES_8;
        let a0 = u8x64::from_slice(&a[base..]);
        let b0 = u8x64::from_slice(&b[base..]);
        let a1 = u8x64::from_slice(&a[base + LANES_8..]);
        let b1 = u8x64::from_slice(&b[base + LANES_8..]);
        let a2 = u8x64::from_slice(&a[base + 2 * LANES_8..]);
        let b2 = u8x64::from_slice(&b[base + 2 * LANES_8..]);
        let a3 = u8x64::from_slice(&a[base + 3 * LANES_8..]);
        let b3 = u8x64::from_slice(&b[base + 3 * LANES_8..]);
        (a0 | b0).copy_to_slice(&mut out[base..]);
        (a1 | b1).copy_to_slice(&mut out[base + LANES_8..]);
        (a2 | b2).copy_to_slice(&mut out[base + 2 * LANES_8..]);
        (a3 | b3).copy_to_slice(&mut out[base + 3 * LANES_8..]);
    }
    for i in full_quads * 4..chunks {
        let off = i * LANES_8;
        let va = u8x64::from_slice(&a[off..]);
        let vb = u8x64::from_slice(&b[off..]);
        (va | vb).copy_to_slice(&mut out[off..]);
    }
    for i in chunks * LANES_8..len {
        out[i] = a[i] | b[i];
    }
}

#[inline(always)]
fn bitwise_not_chunk_u8(a: &[u8], out: &mut [u8]) {
    let len = a.len();
    let chunks = len / LANES_8;
    let full_quads = chunks / 4;
    for q in 0..full_quads {
        let base = q * 4 * LANES_8;
        let a0 = u8x64::from_slice(&a[base..]);
        let a1 = u8x64::from_slice(&a[base + LANES_8..]);
        let a2 = u8x64::from_slice(&a[base + 2 * LANES_8..]);
        let a3 = u8x64::from_slice(&a[base + 3 * LANES_8..]);
        (!a0).copy_to_slice(&mut out[base..]);
        (!a1).copy_to_slice(&mut out[base + LANES_8..]);
        (!a2).copy_to_slice(&mut out[base + 2 * LANES_8..]);
        (!a3).copy_to_slice(&mut out[base + 3 * LANES_8..]);
    }
    for i in full_quads * 4..chunks {
        let off = i * LANES_8;
        let va = u8x64::from_slice(&a[off..]);
        (!va).copy_to_slice(&mut out[off..]);
    }
    for i in chunks * LANES_8..len {
        out[i] = !a[i];
    }
}

// ---- i32x16 (16 × i32 = 512 bits) ----

impl BitwiseSimdOps<i32> for i32x16 {
    #[inline]
    fn bitwise_and(a: &[i32], b: &[i32], out: &mut [i32]) {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), out.len());
        let len = a.len();
        if len >= PARALLEL_THRESHOLD {
            let n_threads = std::thread::available_parallelism()
                .map(|t| t.get())
                .unwrap_or(4);
            let chunk_size = len.div_ceil(n_threads);
            parallel_into_slices(out, chunk_size, |offset, chunk| {
                bitwise_and_chunk_i32(
                    &a[offset..offset + chunk.len()],
                    &b[offset..offset + chunk.len()],
                    chunk,
                );
            });
            return;
        }
        bitwise_and_chunk_i32(a, b, out);
    }

    #[inline]
    fn bitwise_xor(a: &[i32], b: &[i32], out: &mut [i32]) {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), out.len());
        let len = a.len();
        if len >= PARALLEL_THRESHOLD {
            let n_threads = std::thread::available_parallelism()
                .map(|t| t.get())
                .unwrap_or(4);
            let chunk_size = len.div_ceil(n_threads);
            parallel_into_slices(out, chunk_size, |offset, chunk| {
                bitwise_xor_chunk_i32(
                    &a[offset..offset + chunk.len()],
                    &b[offset..offset + chunk.len()],
                    chunk,
                );
            });
            return;
        }
        bitwise_xor_chunk_i32(a, b, out);
    }

    #[inline]
    fn bitwise_or(a: &[i32], b: &[i32], out: &mut [i32]) {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), out.len());
        let len = a.len();
        if len >= PARALLEL_THRESHOLD {
            let n_threads = std::thread::available_parallelism()
                .map(|t| t.get())
                .unwrap_or(4);
            let chunk_size = len.div_ceil(n_threads);
            parallel_into_slices(out, chunk_size, |offset, chunk| {
                bitwise_or_chunk_i32(
                    &a[offset..offset + chunk.len()],
                    &b[offset..offset + chunk.len()],
                    chunk,
                );
            });
            return;
        }
        bitwise_or_chunk_i32(a, b, out);
    }

    #[inline]
    fn bitwise_not(a: &[i32], out: &mut [i32]) {
        assert_eq!(a.len(), out.len());
        let len = a.len();
        if len >= PARALLEL_THRESHOLD {
            let n_threads = std::thread::available_parallelism()
                .map(|t| t.get())
                .unwrap_or(4);
            let chunk_size = len.div_ceil(n_threads);
            parallel_into_slices(out, chunk_size, |offset, chunk| {
                bitwise_not_chunk_i32(&a[offset..offset + chunk.len()], chunk);
            });
            return;
        }
        bitwise_not_chunk_i32(a, out);
    }

    #[inline]
    fn bitwise_and_scalar(a: &[i32], scalar: i32, out: &mut [i32]) {
        assert_eq!(a.len(), out.len());
        let len = a.len();
        let splat = i32x16::splat(scalar);
        let chunks = len / LANES_32;
        let full_quads = chunks / 4;
        for q in 0..full_quads {
            let base = q * 4 * LANES_32;
            let r0 = i32x16::from_slice(&a[base..]);
            let r1 = i32x16::from_slice(&a[base + LANES_32..]);
            let r2 = i32x16::from_slice(&a[base + 2 * LANES_32..]);
            let r3 = i32x16::from_slice(&a[base + 3 * LANES_32..]);
            (r0 & splat).copy_to_slice(&mut out[base..]);
            (r1 & splat).copy_to_slice(&mut out[base + LANES_32..]);
            (r2 & splat).copy_to_slice(&mut out[base + 2 * LANES_32..]);
            (r3 & splat).copy_to_slice(&mut out[base + 3 * LANES_32..]);
        }
        for i in full_quads * 4..chunks {
            let off = i * LANES_32;
            let r = i32x16::from_slice(&a[off..]);
            (r & splat).copy_to_slice(&mut out[off..]);
        }
        for i in chunks * LANES_32..len {
            out[i] = a[i] & scalar;
        }
    }

    #[inline]
    fn bitwise_xor_scalar(a: &[i32], scalar: i32, out: &mut [i32]) {
        assert_eq!(a.len(), out.len());
        let len = a.len();
        let splat = i32x16::splat(scalar);
        let chunks = len / LANES_32;
        let full_quads = chunks / 4;
        for q in 0..full_quads {
            let base = q * 4 * LANES_32;
            let r0 = i32x16::from_slice(&a[base..]);
            let r1 = i32x16::from_slice(&a[base + LANES_32..]);
            let r2 = i32x16::from_slice(&a[base + 2 * LANES_32..]);
            let r3 = i32x16::from_slice(&a[base + 3 * LANES_32..]);
            (r0 ^ splat).copy_to_slice(&mut out[base..]);
            (r1 ^ splat).copy_to_slice(&mut out[base + LANES_32..]);
            (r2 ^ splat).copy_to_slice(&mut out[base + 2 * LANES_32..]);
            (r3 ^ splat).copy_to_slice(&mut out[base + 3 * LANES_32..]);
        }
        for i in full_quads * 4..chunks {
            let off = i * LANES_32;
            let r = i32x16::from_slice(&a[off..]);
            (r ^ splat).copy_to_slice(&mut out[off..]);
        }
        for i in chunks * LANES_32..len {
            out[i] = a[i] ^ scalar;
        }
    }

    #[inline]
    fn bitwise_or_scalar(a: &[i32], scalar: i32, out: &mut [i32]) {
        assert_eq!(a.len(), out.len());
        let len = a.len();
        let splat = i32x16::splat(scalar);
        let chunks = len / LANES_32;
        let full_quads = chunks / 4;
        for q in 0..full_quads {
            let base = q * 4 * LANES_32;
            let r0 = i32x16::from_slice(&a[base..]);
            let r1 = i32x16::from_slice(&a[base + LANES_32..]);
            let r2 = i32x16::from_slice(&a[base + 2 * LANES_32..]);
            let r3 = i32x16::from_slice(&a[base + 3 * LANES_32..]);
            (r0 | splat).copy_to_slice(&mut out[base..]);
            (r1 | splat).copy_to_slice(&mut out[base + LANES_32..]);
            (r2 | splat).copy_to_slice(&mut out[base + 2 * LANES_32..]);
            (r3 | splat).copy_to_slice(&mut out[base + 3 * LANES_32..]);
        }
        for i in full_quads * 4..chunks {
            let off = i * LANES_32;
            let r = i32x16::from_slice(&a[off..]);
            (r | splat).copy_to_slice(&mut out[off..]);
        }
        for i in chunks * LANES_32..len {
            out[i] = a[i] | scalar;
        }
    }
}

#[inline(always)]
fn bitwise_and_chunk_i32(a: &[i32], b: &[i32], out: &mut [i32]) {
    let len = a.len();
    let chunks = len / LANES_32;
    let full_quads = chunks / 4;
    for q in 0..full_quads {
        let base = q * 4 * LANES_32;
        let a0 = i32x16::from_slice(&a[base..]);
        let b0 = i32x16::from_slice(&b[base..]);
        let a1 = i32x16::from_slice(&a[base + LANES_32..]);
        let b1 = i32x16::from_slice(&b[base + LANES_32..]);
        let a2 = i32x16::from_slice(&a[base + 2 * LANES_32..]);
        let b2 = i32x16::from_slice(&b[base + 2 * LANES_32..]);
        let a3 = i32x16::from_slice(&a[base + 3 * LANES_32..]);
        let b3 = i32x16::from_slice(&b[base + 3 * LANES_32..]);
        (a0 & b0).copy_to_slice(&mut out[base..]);
        (a1 & b1).copy_to_slice(&mut out[base + LANES_32..]);
        (a2 & b2).copy_to_slice(&mut out[base + 2 * LANES_32..]);
        (a3 & b3).copy_to_slice(&mut out[base + 3 * LANES_32..]);
    }
    for i in full_quads * 4..chunks {
        let off = i * LANES_32;
        let va = i32x16::from_slice(&a[off..]);
        let vb = i32x16::from_slice(&b[off..]);
        (va & vb).copy_to_slice(&mut out[off..]);
    }
    for i in chunks * LANES_32..len {
        out[i] = a[i] & b[i];
    }
}

#[inline(always)]
fn bitwise_xor_chunk_i32(a: &[i32], b: &[i32], out: &mut [i32]) {
    let len = a.len();
    let chunks = len / LANES_32;
    let full_quads = chunks / 4;
    for q in 0..full_quads {
        let base = q * 4 * LANES_32;
        let a0 = i32x16::from_slice(&a[base..]);
        let b0 = i32x16::from_slice(&b[base..]);
        let a1 = i32x16::from_slice(&a[base + LANES_32..]);
        let b1 = i32x16::from_slice(&b[base + LANES_32..]);
        let a2 = i32x16::from_slice(&a[base + 2 * LANES_32..]);
        let b2 = i32x16::from_slice(&b[base + 2 * LANES_32..]);
        let a3 = i32x16::from_slice(&a[base + 3 * LANES_32..]);
        let b3 = i32x16::from_slice(&b[base + 3 * LANES_32..]);
        (a0 ^ b0).copy_to_slice(&mut out[base..]);
        (a1 ^ b1).copy_to_slice(&mut out[base + LANES_32..]);
        (a2 ^ b2).copy_to_slice(&mut out[base + 2 * LANES_32..]);
        (a3 ^ b3).copy_to_slice(&mut out[base + 3 * LANES_32..]);
    }
    for i in full_quads * 4..chunks {
        let off = i * LANES_32;
        let va = i32x16::from_slice(&a[off..]);
        let vb = i32x16::from_slice(&b[off..]);
        (va ^ vb).copy_to_slice(&mut out[off..]);
    }
    for i in chunks * LANES_32..len {
        out[i] = a[i] ^ b[i];
    }
}

#[inline(always)]
fn bitwise_or_chunk_i32(a: &[i32], b: &[i32], out: &mut [i32]) {
    let len = a.len();
    let chunks = len / LANES_32;
    let full_quads = chunks / 4;
    for q in 0..full_quads {
        let base = q * 4 * LANES_32;
        let a0 = i32x16::from_slice(&a[base..]);
        let b0 = i32x16::from_slice(&b[base..]);
        let a1 = i32x16::from_slice(&a[base + LANES_32..]);
        let b1 = i32x16::from_slice(&b[base + LANES_32..]);
        let a2 = i32x16::from_slice(&a[base + 2 * LANES_32..]);
        let b2 = i32x16::from_slice(&b[base + 2 * LANES_32..]);
        let a3 = i32x16::from_slice(&a[base + 3 * LANES_32..]);
        let b3 = i32x16::from_slice(&b[base + 3 * LANES_32..]);
        (a0 | b0).copy_to_slice(&mut out[base..]);
        (a1 | b1).copy_to_slice(&mut out[base + LANES_32..]);
        (a2 | b2).copy_to_slice(&mut out[base + 2 * LANES_32..]);
        (a3 | b3).copy_to_slice(&mut out[base + 3 * LANES_32..]);
    }
    for i in full_quads * 4..chunks {
        let off = i * LANES_32;
        let va = i32x16::from_slice(&a[off..]);
        let vb = i32x16::from_slice(&b[off..]);
        (va | vb).copy_to_slice(&mut out[off..]);
    }
    for i in chunks * LANES_32..len {
        out[i] = a[i] | b[i];
    }
}

#[inline(always)]
fn bitwise_not_chunk_i32(a: &[i32], out: &mut [i32]) {
    let len = a.len();
    let chunks = len / LANES_32;
    let full_quads = chunks / 4;
    for q in 0..full_quads {
        let base = q * 4 * LANES_32;
        let a0 = i32x16::from_slice(&a[base..]);
        let a1 = i32x16::from_slice(&a[base + LANES_32..]);
        let a2 = i32x16::from_slice(&a[base + 2 * LANES_32..]);
        let a3 = i32x16::from_slice(&a[base + 3 * LANES_32..]);
        (!a0).copy_to_slice(&mut out[base..]);
        (!a1).copy_to_slice(&mut out[base + LANES_32..]);
        (!a2).copy_to_slice(&mut out[base + 2 * LANES_32..]);
        (!a3).copy_to_slice(&mut out[base + 3 * LANES_32..]);
    }
    for i in full_quads * 4..chunks {
        let off = i * LANES_32;
        let va = i32x16::from_slice(&a[off..]);
        (!va).copy_to_slice(&mut out[off..]);
    }
    for i in chunks * LANES_32..len {
        out[i] = !a[i];
    }
}

// ---- i64x8 (8 × i64 = 512 bits) ----

impl BitwiseSimdOps<i64> for i64x8 {
    #[inline]
    fn bitwise_and(a: &[i64], b: &[i64], out: &mut [i64]) {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), out.len());
        let len = a.len();
        if len >= PARALLEL_THRESHOLD {
            let n_threads = std::thread::available_parallelism()
                .map(|t| t.get())
                .unwrap_or(4);
            let chunk_size = len.div_ceil(n_threads);
            parallel_into_slices(out, chunk_size, |offset, chunk| {
                bitwise_and_chunk_i64(
                    &a[offset..offset + chunk.len()],
                    &b[offset..offset + chunk.len()],
                    chunk,
                );
            });
            return;
        }
        bitwise_and_chunk_i64(a, b, out);
    }

    #[inline]
    fn bitwise_xor(a: &[i64], b: &[i64], out: &mut [i64]) {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), out.len());
        let len = a.len();
        if len >= PARALLEL_THRESHOLD {
            let n_threads = std::thread::available_parallelism()
                .map(|t| t.get())
                .unwrap_or(4);
            let chunk_size = len.div_ceil(n_threads);
            parallel_into_slices(out, chunk_size, |offset, chunk| {
                bitwise_xor_chunk_i64(
                    &a[offset..offset + chunk.len()],
                    &b[offset..offset + chunk.len()],
                    chunk,
                );
            });
            return;
        }
        bitwise_xor_chunk_i64(a, b, out);
    }

    #[inline]
    fn bitwise_or(a: &[i64], b: &[i64], out: &mut [i64]) {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), out.len());
        let len = a.len();
        if len >= PARALLEL_THRESHOLD {
            let n_threads = std::thread::available_parallelism()
                .map(|t| t.get())
                .unwrap_or(4);
            let chunk_size = len.div_ceil(n_threads);
            parallel_into_slices(out, chunk_size, |offset, chunk| {
                bitwise_or_chunk_i64(
                    &a[offset..offset + chunk.len()],
                    &b[offset..offset + chunk.len()],
                    chunk,
                );
            });
            return;
        }
        bitwise_or_chunk_i64(a, b, out);
    }

    #[inline]
    fn bitwise_not(a: &[i64], out: &mut [i64]) {
        assert_eq!(a.len(), out.len());
        let len = a.len();
        if len >= PARALLEL_THRESHOLD {
            let n_threads = std::thread::available_parallelism()
                .map(|t| t.get())
                .unwrap_or(4);
            let chunk_size = len.div_ceil(n_threads);
            parallel_into_slices(out, chunk_size, |offset, chunk| {
                bitwise_not_chunk_i64(&a[offset..offset + chunk.len()], chunk);
            });
            return;
        }
        bitwise_not_chunk_i64(a, out);
    }

    #[inline]
    fn bitwise_and_scalar(a: &[i64], scalar: i64, out: &mut [i64]) {
        assert_eq!(a.len(), out.len());
        let len = a.len();
        let splat = i64x8::splat(scalar);
        let chunks = len / LANES_64;
        let full_quads = chunks / 4;
        for q in 0..full_quads {
            let base = q * 4 * LANES_64;
            let r0 = i64x8::from_slice(&a[base..]);
            let r1 = i64x8::from_slice(&a[base + LANES_64..]);
            let r2 = i64x8::from_slice(&a[base + 2 * LANES_64..]);
            let r3 = i64x8::from_slice(&a[base + 3 * LANES_64..]);
            (r0 & splat).copy_to_slice(&mut out[base..]);
            (r1 & splat).copy_to_slice(&mut out[base + LANES_64..]);
            (r2 & splat).copy_to_slice(&mut out[base + 2 * LANES_64..]);
            (r3 & splat).copy_to_slice(&mut out[base + 3 * LANES_64..]);
        }
        for i in full_quads * 4..chunks {
            let off = i * LANES_64;
            let r = i64x8::from_slice(&a[off..]);
            (r & splat).copy_to_slice(&mut out[off..]);
        }
        for i in chunks * LANES_64..len {
            out[i] = a[i] & scalar;
        }
    }

    #[inline]
    fn bitwise_xor_scalar(a: &[i64], scalar: i64, out: &mut [i64]) {
        assert_eq!(a.len(), out.len());
        let len = a.len();
        let splat = i64x8::splat(scalar);
        let chunks = len / LANES_64;
        let full_quads = chunks / 4;
        for q in 0..full_quads {
            let base = q * 4 * LANES_64;
            let r0 = i64x8::from_slice(&a[base..]);
            let r1 = i64x8::from_slice(&a[base + LANES_64..]);
            let r2 = i64x8::from_slice(&a[base + 2 * LANES_64..]);
            let r3 = i64x8::from_slice(&a[base + 3 * LANES_64..]);
            (r0 ^ splat).copy_to_slice(&mut out[base..]);
            (r1 ^ splat).copy_to_slice(&mut out[base + LANES_64..]);
            (r2 ^ splat).copy_to_slice(&mut out[base + 2 * LANES_64..]);
            (r3 ^ splat).copy_to_slice(&mut out[base + 3 * LANES_64..]);
        }
        for i in full_quads * 4..chunks {
            let off = i * LANES_64;
            let r = i64x8::from_slice(&a[off..]);
            (r ^ splat).copy_to_slice(&mut out[off..]);
        }
        for i in chunks * LANES_64..len {
            out[i] = a[i] ^ scalar;
        }
    }

    #[inline]
    fn bitwise_or_scalar(a: &[i64], scalar: i64, out: &mut [i64]) {
        assert_eq!(a.len(), out.len());
        let len = a.len();
        let splat = i64x8::splat(scalar);
        let chunks = len / LANES_64;
        let full_quads = chunks / 4;
        for q in 0..full_quads {
            let base = q * 4 * LANES_64;
            let r0 = i64x8::from_slice(&a[base..]);
            let r1 = i64x8::from_slice(&a[base + LANES_64..]);
            let r2 = i64x8::from_slice(&a[base + 2 * LANES_64..]);
            let r3 = i64x8::from_slice(&a[base + 3 * LANES_64..]);
            (r0 | splat).copy_to_slice(&mut out[base..]);
            (r1 | splat).copy_to_slice(&mut out[base + LANES_64..]);
            (r2 | splat).copy_to_slice(&mut out[base + 2 * LANES_64..]);
            (r3 | splat).copy_to_slice(&mut out[base + 3 * LANES_64..]);
        }
        for i in full_quads * 4..chunks {
            let off = i * LANES_64;
            let r = i64x8::from_slice(&a[off..]);
            (r | splat).copy_to_slice(&mut out[off..]);
        }
        for i in chunks * LANES_64..len {
            out[i] = a[i] | scalar;
        }
    }
}

#[inline(always)]
fn bitwise_and_chunk_i64(a: &[i64], b: &[i64], out: &mut [i64]) {
    let len = a.len();
    let chunks = len / LANES_64;
    let full_quads = chunks / 4;
    for q in 0..full_quads {
        let base = q * 4 * LANES_64;
        let a0 = i64x8::from_slice(&a[base..]);
        let b0 = i64x8::from_slice(&b[base..]);
        let a1 = i64x8::from_slice(&a[base + LANES_64..]);
        let b1 = i64x8::from_slice(&b[base + LANES_64..]);
        let a2 = i64x8::from_slice(&a[base + 2 * LANES_64..]);
        let b2 = i64x8::from_slice(&b[base + 2 * LANES_64..]);
        let a3 = i64x8::from_slice(&a[base + 3 * LANES_64..]);
        let b3 = i64x8::from_slice(&b[base + 3 * LANES_64..]);
        (a0 & b0).copy_to_slice(&mut out[base..]);
        (a1 & b1).copy_to_slice(&mut out[base + LANES_64..]);
        (a2 & b2).copy_to_slice(&mut out[base + 2 * LANES_64..]);
        (a3 & b3).copy_to_slice(&mut out[base + 3 * LANES_64..]);
    }
    for i in full_quads * 4..chunks {
        let off = i * LANES_64;
        let va = i64x8::from_slice(&a[off..]);
        let vb = i64x8::from_slice(&b[off..]);
        (va & vb).copy_to_slice(&mut out[off..]);
    }
    for i in chunks * LANES_64..len {
        out[i] = a[i] & b[i];
    }
}

#[inline(always)]
fn bitwise_xor_chunk_i64(a: &[i64], b: &[i64], out: &mut [i64]) {
    let len = a.len();
    let chunks = len / LANES_64;
    let full_quads = chunks / 4;
    for q in 0..full_quads {
        let base = q * 4 * LANES_64;
        let a0 = i64x8::from_slice(&a[base..]);
        let b0 = i64x8::from_slice(&b[base..]);
        let a1 = i64x8::from_slice(&a[base + LANES_64..]);
        let b1 = i64x8::from_slice(&b[base + LANES_64..]);
        let a2 = i64x8::from_slice(&a[base + 2 * LANES_64..]);
        let b2 = i64x8::from_slice(&b[base + 2 * LANES_64..]);
        let a3 = i64x8::from_slice(&a[base + 3 * LANES_64..]);
        let b3 = i64x8::from_slice(&b[base + 3 * LANES_64..]);
        (a0 ^ b0).copy_to_slice(&mut out[base..]);
        (a1 ^ b1).copy_to_slice(&mut out[base + LANES_64..]);
        (a2 ^ b2).copy_to_slice(&mut out[base + 2 * LANES_64..]);
        (a3 ^ b3).copy_to_slice(&mut out[base + 3 * LANES_64..]);
    }
    for i in full_quads * 4..chunks {
        let off = i * LANES_64;
        let va = i64x8::from_slice(&a[off..]);
        let vb = i64x8::from_slice(&b[off..]);
        (va ^ vb).copy_to_slice(&mut out[off..]);
    }
    for i in chunks * LANES_64..len {
        out[i] = a[i] ^ b[i];
    }
}

#[inline(always)]
fn bitwise_or_chunk_i64(a: &[i64], b: &[i64], out: &mut [i64]) {
    let len = a.len();
    let chunks = len / LANES_64;
    let full_quads = chunks / 4;
    for q in 0..full_quads {
        let base = q * 4 * LANES_64;
        let a0 = i64x8::from_slice(&a[base..]);
        let b0 = i64x8::from_slice(&b[base..]);
        let a1 = i64x8::from_slice(&a[base + LANES_64..]);
        let b1 = i64x8::from_slice(&b[base + LANES_64..]);
        let a2 = i64x8::from_slice(&a[base + 2 * LANES_64..]);
        let b2 = i64x8::from_slice(&b[base + 2 * LANES_64..]);
        let a3 = i64x8::from_slice(&a[base + 3 * LANES_64..]);
        let b3 = i64x8::from_slice(&b[base + 3 * LANES_64..]);
        (a0 | b0).copy_to_slice(&mut out[base..]);
        (a1 | b1).copy_to_slice(&mut out[base + LANES_64..]);
        (a2 | b2).copy_to_slice(&mut out[base + 2 * LANES_64..]);
        (a3 | b3).copy_to_slice(&mut out[base + 3 * LANES_64..]);
    }
    for i in full_quads * 4..chunks {
        let off = i * LANES_64;
        let va = i64x8::from_slice(&a[off..]);
        let vb = i64x8::from_slice(&b[off..]);
        (va | vb).copy_to_slice(&mut out[off..]);
    }
    for i in chunks * LANES_64..len {
        out[i] = a[i] | b[i];
    }
}

#[inline(always)]
fn bitwise_not_chunk_i64(a: &[i64], out: &mut [i64]) {
    let len = a.len();
    let chunks = len / LANES_64;
    let full_quads = chunks / 4;
    for q in 0..full_quads {
        let base = q * 4 * LANES_64;
        let a0 = i64x8::from_slice(&a[base..]);
        let a1 = i64x8::from_slice(&a[base + LANES_64..]);
        let a2 = i64x8::from_slice(&a[base + 2 * LANES_64..]);
        let a3 = i64x8::from_slice(&a[base + 3 * LANES_64..]);
        (!a0).copy_to_slice(&mut out[base..]);
        (!a1).copy_to_slice(&mut out[base + LANES_64..]);
        (!a2).copy_to_slice(&mut out[base + 2 * LANES_64..]);
        (!a3).copy_to_slice(&mut out[base + 3 * LANES_64..]);
    }
    for i in full_quads * 4..chunks {
        let off = i * LANES_64;
        let va = i64x8::from_slice(&a[off..]);
        (!va).copy_to_slice(&mut out[off..]);
    }
    for i in chunks * LANES_64..len {
        out[i] = !a[i];
    }
}

// ===========================================================================
// HammingSimdOps – Fused XOR+popcount for bitpacked hamming distance
// ===========================================================================
//
// AVX-512 doesn't have a native VPOPCNT on all microarchitectures, so we
// use the classic SIMD popcount technique:
//   1. XOR the two vectors to get the diff bits.
//   2. Use a nibble-lookup (vpshufb equivalent) to count bits per byte.
//   3. Sum the byte counts with a horizontal add (vpsadbw against zero).
//
// For arrays that are multiples of 8192 bytes, each 8192-byte block is
// exactly 128 zmm vectors. With 4× unrolling that's 32 iterations—
// perfect pipeline saturation with zero tail handling.

impl HammingSimdOps for u8x64 {
    #[inline]
    fn hamming_distance(a: &[u8], b: &[u8]) -> u64 {
        assert_eq!(a.len(), b.len());
        let len = a.len();

        if len >= PARALLEL_THRESHOLD {
            return parallel_reduce_sum(len, |start, end| {
                hamming_chunk(&a[start..end], &b[start..end])
            });
        }

        hamming_chunk(a, b)
    }

    #[inline]
    fn popcount(a: &[u8]) -> u64 {
        debug_assert!(!a.is_empty());
        let len = a.len();

        if len >= PARALLEL_THRESHOLD {
            return parallel_reduce_sum(len, |start, end| popcount_chunk(&a[start..end]));
        }

        popcount_chunk(a)
    }

    fn hamming_distance_batch(
        a_vecs: &[u8],
        b_vecs: &[u8],
        vec_len: usize,
        count: usize,
    ) -> Vec<u64> {
        assert_eq!(a_vecs.len(), vec_len * count);
        assert_eq!(b_vecs.len(), vec_len * count);

        let mut results = vec![0u64; count];

        if count >= 16 {
            let n_threads = std::thread::available_parallelism()
                .map(|t| t.get())
                .unwrap_or(4);
            let chunk_size = count.div_ceil(n_threads);
            parallel_into_slices(&mut results, chunk_size, |offset, chunk| {
                for i in 0..chunk.len() {
                    let idx = offset + i;
                    let a_slice = &a_vecs[idx * vec_len..(idx + 1) * vec_len];
                    let b_slice = &b_vecs[idx * vec_len..(idx + 1) * vec_len];
                    chunk[i] = hamming_chunk(a_slice, b_slice);
                }
            });
        } else {
            for i in 0..count {
                let a_slice = &a_vecs[i * vec_len..(i + 1) * vec_len];
                let b_slice = &b_vecs[i * vec_len..(i + 1) * vec_len];
                results[i] = hamming_chunk(a_slice, b_slice);
            }
        }

        results
    }
}

/// Fused XOR+popcount with 3-tier SIMD dispatch via numrus_core.
///
/// Dispatch chain: VPOPCNTDQ (AVX-512) → Harley-Seal (AVX2) → scalar POPCNT.
/// All callers in this module get the fastest available path automatically.
#[inline(always)]
fn hamming_chunk(a: &[u8], b: &[u8]) -> u64 {
    numrus_core::simd::hamming_distance(a, b)
}

/// Popcount for a single chunk of u8 data.
///
/// Dispatch chain: VPOPCNTDQ (AVX-512) → Harley-Seal (AVX2) → scalar POPCNT.
/// For 2048-byte containers: 32 VPOPCNTDQ iterations, matching u64x8 width exactly.
#[inline(always)]
fn popcount_chunk(a: &[u8]) -> u64 {
    numrus_core::simd::popcount(a)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transpose_3x3_f32() {
        let src = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let rows = 3;
        let cols = 3;
        let mut dst = vec![0.0f32; src.len()];

        f32x16::transpose(&src, &mut dst, rows, cols);

        let expected = [1.0f32, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0];

        assert_eq!(
            dst, expected,
            "The transposed matrix does not match the expected result for a 3x3 matrix."
        );
    }

    #[test]
    fn test_dot_product_f32() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [4.0, 3.0, 2.0, 1.0];
        let result = f32x16::dot_product(&a, &b);
        assert_eq!(result, 20.0);
    }

    #[test]
    fn test_dot_product_u8() {
        let a = [1, 2, 3, 4, 5, 6, 7, 8];
        let b = [8, 7, 6, 5, 4, 3, 2, 1];
        let result = u8x64::dot_product(&a, &b);
        assert_eq!(result, 120);
    }

    #[test]
    fn test_matrix_multiply_small_known_result_f32() {
        let a = vec![
            1.0, 2.0, 3.0, // 2x3 matrix
            4.0, 5.0, 6.0,
        ];
        let b = vec![
            7.0, 8.0, // 3x2 matrix
            9.0, 10.0, 11.0, 12.0,
        ];
        let mut c = vec![0.0; 4]; // Result will be 2x2 matrix

        let m = 2;
        let k = 3;
        let n = 2;

        f32x16::matrix_multiply(&a, &b, &mut c, m, k, n);

        let expected = vec![58.0, 64.0, 139.0, 154.0];
        assert_eq!(c, expected);
    }

    #[test]
    fn test_matrix_multiply_small_known_result_u8() {
        let a = vec![
            1, 2, 3, // 2x3 matrix
            4, 5, 6,
        ];
        let b = vec![
            7, 8, // 3x2 matrix
            9, 10, 11, 12,
        ];
        let mut c = vec![0; 4]; // Result will be 2x2 matrix

        let m = 2;
        let k = 3;
        let n = 2;

        u8x64::matrix_multiply(&a, &b, &mut c, m, k, n);

        let expected = vec![58, 64, 139, 154];
        assert_eq!(c, expected);
    }

    #[test]
    fn test_matrix_multiply_non_square_matrices_f32() {
        let a = vec![
            1.0, 4.0, // 3x2 matrix
            2.0, 5.0, 3.0, 6.0,
        ];
        let b = vec![
            7.0, 9.0, 11.0, // 2x3 matrix
            8.0, 10.0, 12.0,
        ];
        let mut c = vec![0.0; 9]; // Result will be 3x3 matrix

        let m = 3;
        let k = 2;
        let n = 3;

        f32x16::matrix_multiply(&a, &b, &mut c, m, k, n);

        let expected = vec![39.0, 49.0, 59.0, 54.0, 68.0, 82.0, 69.0, 87.0, 105.0];
        assert_eq!(c, expected);
    }

    #[test]
    fn test_matrix_multiply_negative_and_zero_values_f32() {
        let a = vec![
            0.0, -2.0, 3.0, // 2x3 matrix
            -4.0, 5.0, -6.0,
        ];
        let b = vec![
            -1.0, 0.0, // 3x2 matrix
            2.0, -3.0, 0.0, 4.0,
        ];
        let mut c = vec![0.0; 4]; // Result will be 2x2 matrix

        let m = 2;
        let k = 3;
        let n = 2;

        f32x16::matrix_multiply(&a, &b, &mut c, m, k, n);

        let expected = vec![-4.0, 18.0, 14.0, -39.0];
        assert_eq!(c, expected);
    }

    #[test]
    fn test_sum_f32() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let result = f32x16::sum(&a);
        assert_eq!(result, 10.0);
    }

    #[test]
    fn test_min_f32() {
        let a = [4.0, 1.0, 3.0, 2.0];
        let result = f32x16::min_simd(&a);
        assert_eq!(result, 1.0);
    }

    #[test]
    fn test_max_f32() {
        let a = [4.0, 1.0, 3.0, 2.0];
        let result = f32x16::max_simd(&a);
        assert_eq!(result, 4.0);
    }

    #[test]
    fn test_transpose_3x3_f64() {
        let src = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let rows = 3;
        let cols = 3;
        let mut dst = vec![0.0f64; src.len()];

        f64x8::transpose(&src, &mut dst, rows, cols);

        let expected = [1.0f64, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0];

        assert_eq!(
            dst, expected,
            "The transposed matrix does not match the expected result for a 3x3 matrix."
        );
    }

    #[test]
    fn test_dot_product_f64() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [4.0, 3.0, 2.0, 1.0];
        let result = f64x8::dot_product(&a, &b);
        assert_eq!(result, 20.0);
    }

    #[test]
    fn test_matrix_multiply_small_known_result_f64() {
        let a = vec![
            1.0, 2.0, 3.0, // 2x3 matrix
            4.0, 5.0, 6.0,
        ];
        let b = vec![
            7.0, 8.0, // 3x2 matrix
            9.0, 10.0, 11.0, 12.0,
        ];
        let mut c = vec![0.0; 4]; // Result will be 2x2 matrix

        let m = 2;
        let k = 3;
        let n = 2;

        f64x8::matrix_multiply(&a, &b, &mut c, m, k, n);

        let expected = vec![58.0, 64.0, 139.0, 154.0];
        assert_eq!(c, expected);
    }

    #[test]
    fn test_matrix_multiply_non_square_matrices_f64() {
        let a = vec![
            1.0, 4.0, // 3x2 matrix
            2.0, 5.0, 3.0, 6.0,
        ];
        let b = vec![
            7.0, 9.0, 11.0, // 2x3 matrix
            8.0, 10.0, 12.0,
        ];
        let mut c = vec![0.0; 9]; // Result will be 3x3 matrix

        let m = 3;
        let k = 2;
        let n = 3;

        f64x8::matrix_multiply(&a, &b, &mut c, m, k, n);

        let expected = vec![39.0, 49.0, 59.0, 54.0, 68.0, 82.0, 69.0, 87.0, 105.0];
        assert_eq!(c, expected);
    }

    #[test]
    fn test_matrix_multiply_negative_and_zero_values_f64() {
        let a = vec![
            0.0, -2.0, 3.0, // 2x3 matrix
            -4.0, 5.0, -6.0,
        ];
        let b = vec![
            -1.0, 0.0, // 3x2 matrix
            2.0, -3.0, 0.0, 4.0,
        ];
        let mut c = vec![0.0; 4]; // Result will be 2x2 matrix

        let m = 2;
        let k = 3;
        let n = 2;

        f64x8::matrix_multiply(&a, &b, &mut c, m, k, n);

        let expected = vec![-4.0, 18.0, 14.0, -39.0];
        assert_eq!(c, expected);
    }

    #[test]
    fn test_sum_f64() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let result = f64x8::sum(&a);
        assert_eq!(result, 10.0);
    }

    #[test]
    fn test_min_f64() {
        let a = [4.0, 1.0, 3.0, 2.0];
        let result = f64x8::min_simd(&a);
        assert_eq!(result, 1.0);
    }

    #[test]
    fn test_max_f64() {
        let a = [4.0, 1.0, 3.0, 2.0];
        let result = f64x8::max_simd(&a);
        assert_eq!(result, 4.0);
    }

    #[test]
    fn test_l1_norm_f32() {
        let a = [-1.0f32, 2.0, -3.0, 4.0];
        let result = f32x16::l1_norm(&a);
        assert_eq!(result, 10.0);
    }

    #[test]
    fn test_l2_norm_f32() {
        let a = [3.0f32, 4.0];
        let result = f32x16::l2_norm(&a);
        assert_eq!(result, 5.0);
    }

    #[test]
    fn test_l1_norm_f64() {
        let a = [-1.0f64, 2.0, -3.0, 4.0];
        let result = f64x8::l1_norm(&a);
        assert_eq!(result, 10.0);
    }

    #[test]
    fn test_l2_norm_f64() {
        let a = [3.0f64, 4.0];
        let result = f64x8::l2_norm(&a);
        assert_eq!(result, 5.0);
    }

    #[test]
    fn test_l1_norm_2d_axis0() {
        let a = [
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, // Shape (2, 3)
        ];
        let result = f32x16::l1_norm(&a[0..3]); // First row
        assert_eq!(result, 6.0);
        let result = f32x16::l1_norm(&a[3..6]); // Second row
        assert_eq!(result, 15.0);
    }

    #[test]
    fn test_l2_norm_2d_axis1() {
        let a = [
            3.0f32, 4.0, 5.0, 12.0, // Shape (2, 2)
        ];
        let result = f32x16::l2_norm(&a[0..2]);
        assert_eq!(result, 5.0);
        let result = f32x16::l2_norm(&a[2..4]);
        assert_eq!(result, 13.0);
    }

    #[test]
    fn test_l1_norm_empty() {
        let a: [f32; 0] = [];
        let result = f32x16::l1_norm(&a);
        assert_eq!(result, 0.0);
    }

    // ---- i32x16 SimdOps tests ----

    #[test]
    fn test_dot_product_i32() {
        let a = [1i32, 2, 3, 4];
        let b = [4i32, 3, 2, 1];
        let result = i32x16::dot_product(&a, &b);
        assert_eq!(result, 20);
    }

    #[test]
    fn test_sum_i32() {
        let a = [1i32, 2, 3, 4];
        assert_eq!(i32x16::sum(&a), 10);
    }

    #[test]
    fn test_min_max_i32() {
        let a = [4i32, 1, 3, 2];
        assert_eq!(i32x16::min_simd(&a), 1);
        assert_eq!(i32x16::max_simd(&a), 4);
    }

    // ---- i64x8 SimdOps tests ----

    #[test]
    fn test_dot_product_i64() {
        let a = [1i64, 2, 3, 4];
        let b = [4i64, 3, 2, 1];
        let result = i64x8::dot_product(&a, &b);
        assert_eq!(result, 20);
    }

    #[test]
    fn test_sum_i64() {
        let a = [1i64, 2, 3, 4];
        assert_eq!(i64x8::sum(&a), 10);
    }

    // ---- BitwiseSimdOps tests (u8) ----

    #[test]
    fn test_bitwise_and_u8_small() {
        let a = [0xFFu8, 0x0F, 0xF0, 0xAA, 0x55];
        let b = [0x0Fu8, 0xFF, 0x0F, 0x55, 0xAA];
        let mut out = [0u8; 5];
        u8x64::bitwise_and(&a, &b, &mut out);
        assert_eq!(out, [0x0F, 0x0F, 0x00, 0x00, 0x00]);
    }

    #[test]
    fn test_bitwise_xor_u8_small() {
        let a = [0xFFu8, 0x0F, 0xF0, 0xAA, 0x55];
        let b = [0x0Fu8, 0xFF, 0x0F, 0x55, 0xAA];
        let mut out = [0u8; 5];
        u8x64::bitwise_xor(&a, &b, &mut out);
        assert_eq!(out, [0xF0, 0xF0, 0xFF, 0xFF, 0xFF]);
    }

    #[test]
    fn test_bitwise_or_u8_small() {
        let a = [0xF0u8, 0x0F, 0xA0, 0x05];
        let b = [0x0Fu8, 0xF0, 0x50, 0x0A];
        let mut out = [0u8; 4];
        u8x64::bitwise_or(&a, &b, &mut out);
        assert_eq!(out, [0xFF, 0xFF, 0xF0, 0x0F]);
    }

    #[test]
    fn test_bitwise_not_u8_small() {
        let a = [0x00u8, 0xFF, 0x0F, 0xF0];
        let mut out = [0u8; 4];
        u8x64::bitwise_not(&a, &mut out);
        assert_eq!(out, [0xFF, 0x00, 0xF0, 0x0F]);
    }

    #[test]
    fn test_bitwise_and_u8_large() {
        // Test with exactly 8192 elements (multiple of 8192)
        let n = 8192;
        let a: Vec<u8> = (0..n).map(|i| (i % 256) as u8).collect();
        let b: Vec<u8> = (0..n).map(|i| ((i * 7) % 256) as u8).collect();
        let mut out = vec![0u8; n];
        u8x64::bitwise_and(&a, &b, &mut out);
        for i in 0..n {
            assert_eq!(out[i], a[i] & b[i], "mismatch at index {}", i);
        }
    }

    #[test]
    fn test_bitwise_xor_u8_large() {
        let n = 8192;
        let a: Vec<u8> = (0..n).map(|i| (i % 256) as u8).collect();
        let b: Vec<u8> = (0..n).map(|i| ((i * 13) % 256) as u8).collect();
        let mut out = vec![0u8; n];
        u8x64::bitwise_xor(&a, &b, &mut out);
        for i in 0..n {
            assert_eq!(out[i], a[i] ^ b[i], "mismatch at index {}", i);
        }
    }

    #[test]
    fn test_bitwise_and_scalar_u8() {
        let a = [0xFFu8, 0x0F, 0xF0, 0xAA, 0x55];
        let mut out = [0u8; 5];
        u8x64::bitwise_and_scalar(&a, 0x0F, &mut out);
        assert_eq!(out, [0x0F, 0x0F, 0x00, 0x0A, 0x05]);
    }

    #[test]
    fn test_bitwise_xor_scalar_u8() {
        let a = [0xFFu8, 0x0F, 0xF0, 0xAA];
        let mut out = [0u8; 4];
        u8x64::bitwise_xor_scalar(&a, 0xFF, &mut out);
        assert_eq!(out, [0x00, 0xF0, 0x0F, 0x55]);
    }

    // ---- BitwiseSimdOps tests (i32) ----

    #[test]
    fn test_bitwise_and_i32_small() {
        let a = [0x0F0F0F0Fi32, -1, 0, 0x12345678];
        let b = [0x00FF00FFi32, 0x0F0F0F0F, -1, 0x0000FFFF];
        let mut out = [0i32; 4];
        i32x16::bitwise_and(&a, &b, &mut out);
        assert_eq!(
            out,
            [
                0x0F0F0F0Fi32 & 0x00FF00FFi32,
                0x0F0F0F0Fi32,
                0,
                0x12345678i32 & 0x0000FFFFi32,
            ]
        );
    }

    #[test]
    fn test_bitwise_xor_i32_small() {
        let a = [0i32, -1, 0x12345678, 0xFF];
        let b = [0i32, -1, 0x12345678, 0xFF00];
        let mut out = [0i32; 4];
        i32x16::bitwise_xor(&a, &b, &mut out);
        assert_eq!(out, [0, 0, 0, 0xFF ^ 0xFF00]);
    }

    #[test]
    fn test_bitwise_and_i32_large() {
        let n = 8192;
        let a: Vec<i32> = (0..n).map(|i| i as i32).collect();
        let b: Vec<i32> = (0..n).map(|i| (i * 3) as i32).collect();
        let mut out = vec![0i32; n];
        i32x16::bitwise_and(&a, &b, &mut out);
        for i in 0..n {
            assert_eq!(out[i], a[i] & b[i]);
        }
    }

    // ---- BitwiseSimdOps tests (i64) ----

    #[test]
    fn test_bitwise_and_i64_small() {
        let a = [0x0F0F0F0Fi64, -1, 0, 0x12345678];
        let b = [0x00FF00FFi64, 0x0F0F0F0F, -1, 0x0000FFFF];
        let mut out = [0i64; 4];
        i64x8::bitwise_and(&a, &b, &mut out);
        for i in 0..4 {
            assert_eq!(out[i], a[i] & b[i]);
        }
    }

    #[test]
    fn test_bitwise_xor_i64_large() {
        let n = 8192;
        let a: Vec<i64> = (0..n).map(|i| i as i64).collect();
        let b: Vec<i64> = (0..n).map(|i| (i * 7) as i64).collect();
        let mut out = vec![0i64; n];
        i64x8::bitwise_xor(&a, &b, &mut out);
        for i in 0..n {
            assert_eq!(out[i], a[i] ^ b[i]);
        }
    }

    // ---- Hamming distance tests ----

    #[test]
    fn test_hamming_distance_identical() {
        let n = 8192;
        let a: Vec<u8> = (0..n).map(|i| (i % 256) as u8).collect();
        let b = a.clone();
        assert_eq!(u8x64::hamming_distance(&a, &b), 0);
    }

    #[test]
    fn test_hamming_distance_all_ones() {
        // All bits differ: 8192 bytes × 8 bits = 65536
        let n = 8192;
        let a = vec![0x00u8; n];
        let b = vec![0xFFu8; n];
        assert_eq!(u8x64::hamming_distance(&a, &b), n as u64 * 8);
    }

    #[test]
    fn test_hamming_distance_single_bit() {
        let n = 8192;
        let a = vec![0u8; n];
        let mut b = vec![0u8; n];
        b[0] = 1; // one bit differs
        assert_eq!(u8x64::hamming_distance(&a, &b), 1);
    }

    #[test]
    fn test_hamming_distance_known_pattern() {
        // 0xAA = 10101010, 0x55 = 01010101 → XOR = 0xFF → 8 bits per byte
        let n = 8192;
        let a = vec![0xAAu8; n];
        let b = vec![0x55u8; n];
        assert_eq!(u8x64::hamming_distance(&a, &b), n as u64 * 8);
    }

    #[test]
    fn test_hamming_distance_small() {
        let a = [0b11001100u8, 0b10101010];
        let b = [0b11110000u8, 0b01010101];
        // XOR: 0b00111100 (4 bits), 0b11111111 (8 bits) = 12
        assert_eq!(u8x64::hamming_distance(&a, &b), 12);
    }

    #[test]
    fn test_popcount_u8() {
        let a = [0xFFu8; 8192];
        // Each byte has 8 set bits → 8192 × 8 = 65536
        assert_eq!(u8x64::popcount(&a), 65536);
    }

    #[test]
    fn test_popcount_u8_small() {
        let a = [0b10101010u8, 0b01010101, 0b11111111, 0b00000000];
        // 4 + 4 + 8 + 0 = 16
        assert_eq!(u8x64::popcount(&a), 16);
    }

    #[test]
    fn test_hamming_distance_batch() {
        let vec_len = 8192;
        let count = 4;
        let a_vecs = vec![0xAAu8; vec_len * count];
        let mut b_vecs = vec![0xAAu8; vec_len * count];
        // Make 2nd vector differ completely
        for i in vec_len..2 * vec_len {
            b_vecs[i] = 0x55;
        }
        let results = u8x64::hamming_distance_batch(&a_vecs, &b_vecs, vec_len, count);
        assert_eq!(results[0], 0);
        assert_eq!(results[1], vec_len as u64 * 8);
        assert_eq!(results[2], 0);
        assert_eq!(results[3], 0);
    }

    #[test]
    fn test_hamming_distance_multiple_of_8192() {
        // Test various multiples of 8192
        for mult in [1, 2, 4, 8] {
            let n = 8192 * mult;
            let a = vec![0xF0u8; n];
            let b = vec![0x0Fu8; n];
            // XOR = 0xFF → 8 bits per byte
            assert_eq!(
                u8x64::hamming_distance(&a, &b),
                n as u64 * 8,
                "Failed for n={}",
                n
            );
        }
    }
}
