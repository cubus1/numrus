//! BF16 (Brain Float 16) GEMM for ML inference workloads.
//!
//! BFloat16 uses 8-bit exponent (same range as f32) + 7-bit mantissa.
//! Key advantage: trivial conversion to/from f32 (just truncate/pad mantissa).
//!
//! SIMD conversion: f32x16 <-> bf16 via u32x16 bit manipulation.
//! GEMM kernel: converts BF16 tiles to f32 in L1, uses SIMD dot_f32 microkernel.
//!
//! On CPUs with AVX-512 BF16 support (`avx512_bf16`):
//! - `vcvtne2ps2bf16`: convert two f32x16 -> one bf16x32
//! - `vdpbf16ps`: dot product bf16 pairs -> f32 accumulate
//!
//! On CPUs with AMX-BF16 (`amx_bf16`):
//! - `tdpbf16ps`: tile dot product, 16x32 x 32x16 -> 16x16 f32 in one instruction
//!
//! This module provides:
//! - BF16 <-> f32 conversion (SIMD-accelerated bulk ops)
//! - BF16 GEMM with f32 accumulation (SIMD dot product microkernel)
//! - Mixed-precision GEMM: inputs in BF16, output in f32

use numrus_core::simd::{dot_f32, F32_LANES};

#[cfg(feature = "avx512")]
use std::simd::{f32x16 as F32Simd, u32x16 as U32Simd};
#[cfg(not(feature = "avx512"))]
use std::simd::{f32x8 as F32Simd, u32x8 as U32Simd};

use std::simd::num::SimdFloat;

/// BFloat16 stored as raw u16 bits.
/// Layout: [1 sign][8 exponent][7 mantissa]
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(transparent)]
pub struct BF16(pub u16);

impl BF16 {
    /// Convert f32 -> BF16 by truncating lower 16 mantissa bits.
    /// This is "round toward zero" -- fast but loses precision.
    #[inline(always)]
    pub fn from_f32_truncate(v: f32) -> Self {
        BF16((v.to_bits() >> 16) as u16)
    }

    /// Convert f32 -> BF16 with round-to-nearest-even.
    #[inline(always)]
    pub fn from_f32(v: f32) -> Self {
        let bits = v.to_bits();
        // Round to nearest even: add rounding bias
        let round_bit = 0x0000_8000u32; // bit 15
        let lsb = (bits >> 16) & 1;
        let rounded = bits.wrapping_add(round_bit - 1 + lsb);
        BF16((rounded >> 16) as u16)
    }

    /// Convert BF16 -> f32 by padding with 16 zero bits.
    #[inline(always)]
    pub fn to_f32(self) -> f32 {
        f32::from_bits((self.0 as u32) << 16)
    }

    pub const ZERO: BF16 = BF16(0);
    pub const ONE: BF16 = BF16(0x3F80); // 1.0f32 >> 16
}

impl From<f32> for BF16 {
    fn from(v: f32) -> Self {
        BF16::from_f32(v)
    }
}

impl From<BF16> for f32 {
    fn from(v: BF16) -> f32 {
        v.to_f32()
    }
}

// ============================================================================
// Bulk conversion (SIMD-accelerated)
// ============================================================================

/// Convert f32 slice to BF16 (truncation, fastest).
///
/// SIMD path: loads f32x16, extracts bits via `to_bits()`, shifts right by 16,
/// truncates to u16. Processes F32_LANES (16) elements per iteration.
pub fn f32_to_bf16_slice(src: &[f32], dst: &mut [BF16]) {
    assert!(dst.len() >= src.len());
    let len = src.len();
    let chunks = len / F32_LANES;

    for i in 0..chunks {
        let base = i * F32_LANES;
        let fv = F32Simd::from_slice(&src[base..]);
        let bits = fv.to_bits(); // u32x16
        let shifted = bits >> U32Simd::splat(16); // upper 16 bits
        let arr = shifted.to_array();
        for j in 0..F32_LANES {
            dst[base + j] = BF16(arr[j] as u16);
        }
    }

    // Scalar tail
    for i in (chunks * F32_LANES)..len {
        dst[i] = BF16::from_f32_truncate(src[i]);
    }
}

/// Convert f32 slice to BF16 (round-to-nearest-even).
///
/// SIMD path: loads f32x16, extracts bits, adds rounding bias
/// (0x7FFF + LSB of bf16), shifts right by 16, truncates to u16.
pub fn f32_to_bf16_rounded(src: &[f32], dst: &mut [BF16]) {
    assert!(dst.len() >= src.len());
    let len = src.len();
    let chunks = len / F32_LANES;

    let round_base = U32Simd::splat(0x0000_7FFFu32); // round_bit - 1 = 0x8000 - 1
    let one = U32Simd::splat(1);
    let shift16 = U32Simd::splat(16);

    for i in 0..chunks {
        let base = i * F32_LANES;
        let fv = F32Simd::from_slice(&src[base..]);
        let bits = fv.to_bits(); // u32x16
                                 // LSB of the bf16 result = bit 16 of the f32 bits
        let lsb = (bits >> shift16) & one;
        // rounding bias = 0x7FFF + lsb (round-to-nearest-even)
        let bias = round_base + lsb;
        // Add bias (wrapping to handle overflow correctly)
        let rounded = simd_wrapping_add_u32(bits, bias);
        let shifted = rounded >> shift16;
        let arr = shifted.to_array();
        for j in 0..F32_LANES {
            dst[base + j] = BF16(arr[j] as u16);
        }
    }

    // Scalar tail
    for i in (chunks * F32_LANES)..len {
        dst[i] = BF16::from_f32(src[i]);
    }
}

/// SIMD wrapping add for u32 lanes (portable_simd does not expose wrapping_add directly).
#[inline(always)]
fn simd_wrapping_add_u32(a: U32Simd, b: U32Simd) -> U32Simd {
    // For u32x16, the standard + operator wraps in release mode.
    // In debug mode with overflow checks, we use the array fallback.
    let a_arr = a.to_array();
    let b_arr = b.to_array();
    let mut out = [0u32; F32_LANES];
    for i in 0..F32_LANES {
        out[i] = a_arr[i].wrapping_add(b_arr[i]);
    }
    U32Simd::from_array(out)
}

/// Convert BF16 slice to f32.
///
/// SIMD path: zero-extends u16 to u32, shifts left by 16,
/// reinterprets as f32 via `from_bits()`.
pub fn bf16_to_f32_slice(src: &[BF16], dst: &mut [f32]) {
    assert!(dst.len() >= src.len());
    let len = src.len();
    let chunks = len / F32_LANES;

    for i in 0..chunks {
        let base = i * F32_LANES;
        // Load F32_LANES u16 values, zero-extend to u32, shift left 16
        let mut u32_arr = [0u32; F32_LANES];
        for j in 0..F32_LANES {
            u32_arr[j] = (src[base + j].0 as u32) << 16;
        }
        let bits = U32Simd::from_array(u32_arr);
        let fv = F32Simd::from_bits(bits);
        fv.copy_to_slice(&mut dst[base..base + F32_LANES]);
    }

    // Scalar tail
    for i in (chunks * F32_LANES)..len {
        dst[i] = src[i].to_f32();
    }
}

/// Allocate and convert f32 -> BF16.
pub fn f32_vec_to_bf16(src: &[f32]) -> Vec<BF16> {
    let mut dst = vec![BF16::ZERO; src.len()];
    f32_to_bf16_rounded(src, &mut dst);
    dst
}

/// Allocate and convert BF16 -> f32.
pub fn bf16_vec_to_f32(src: &[BF16]) -> Vec<f32> {
    let mut dst = vec![0.0f32; src.len()];
    bf16_to_f32_slice(src, &mut dst);
    dst
}

// ============================================================================
// BF16 GEMM with f32 accumulation
// ============================================================================

/// BF16 GEMM: C_f32 += A_bf16 * B_bf16
///
/// Inputs are BF16 (half the memory bandwidth of f32), accumulation
/// happens in f32 for numerical stability.
///
/// A is M*K (row-major BF16), B is K*N (row-major BF16).
/// C is M*N (row-major f32).
///
/// This gives ~2x memory bandwidth improvement over f32 GEMM since
/// inputs are half the size, while maintaining f32 accumulation precision.
pub fn bf16_gemm_f32(
    a: &[BF16],
    b: &[BF16],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    beta: f32,
) {
    assert!(a.len() >= m * k);
    assert!(b.len() >= k * n);
    assert!(c.len() >= m * n);

    #[cfg(feature = "mkl")]
    {
        // BF16 is #[repr(transparent)] wrapping u16, safe to cast pointer
        unsafe {
            numrus_core::mkl_ffi::cblas_gemm_bf16bf16f32(
                101, // CblasRowMajor
                111, // CblasNoTrans
                111, // CblasNoTrans
                m as i32,
                n as i32,
                k as i32,
                alpha,
                a.as_ptr() as *const u16,
                k as i32,
                b.as_ptr() as *const u16,
                n as i32,
                beta,
                c.as_mut_ptr(),
                n as i32,
            );
        }
        return;
    }

    // Scale C by beta
    if beta == 0.0 {
        c[..m * n].fill(0.0);
    } else if beta != 1.0 {
        for v in c[..m * n].iter_mut() {
            *v *= beta;
        }
    }

    if alpha == 0.0 || m == 0 || n == 0 || k == 0 {
        return;
    }

    // For small matrices, use simple loop
    if m * n * k < 110_000 {
        bf16_gemm_simple(a, b, c, m, n, k, alpha);
        return;
    }

    // Cache-blocked BF16 GEMM with SIMD dot product microkernel
    bf16_gemm_blocked(a, b, c, m, n, k, alpha);
}

/// Simple BF16 GEMM with f32 accumulation.
/// Transposes B for sequential access, uses SIMD dot_f32 for inner product.
fn bf16_gemm_simple(
    a: &[BF16],
    b: &[BF16],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
) {
    // Transpose B for sequential access
    let mut b_t = vec![BF16::ZERO; n * k];
    for p in 0..k {
        for j in 0..n {
            b_t[j * k + p] = b[p * n + j];
        }
    }

    // Pre-convert transposed B rows to f32 for SIMD dot product
    let mut b_t_f32 = vec![0.0f32; n * k];
    bf16_to_f32_slice(&b_t, &mut b_t_f32);

    // Temporary buffer for one row of A in f32
    let mut a_row_f32 = vec![0.0f32; k];

    for i in 0..m {
        let a_row = &a[i * k..(i + 1) * k];
        bf16_to_f32_slice(a_row, &mut a_row_f32);

        for j in 0..n {
            let b_col_f32 = &b_t_f32[j * k..(j + 1) * k];
            let dot = dot_f32(&a_row_f32, b_col_f32);
            c[i * n + j] += alpha * dot;
        }
    }
}

/// Cache-blocked BF16 GEMM.
/// Converts BF16 tiles to f32 in L1-sized blocks, then uses SIMD dot_f32
/// microkernel for the inner product instead of a scalar triple loop.
fn bf16_gemm_blocked(
    a: &[BF16],
    b: &[BF16],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
) {
    const MC: usize = 128;
    const NC: usize = 256;
    const KC: usize = 256;

    // Temporary f32 buffers for converted tiles
    let mc = MC.min(m);
    let nc = NC.min(n);
    let kc = KC.min(k);

    let mut a_f32 = vec![0.0f32; mc * kc];
    // B tile stored in column-major order within the tile: b_col_f32[j][0..pb]
    // This allows using dot_f32 on contiguous row-of-A and column-of-B slices.
    let mut b_col_f32 = vec![0.0f32; nc * kc];

    for jc in (0..n).step_by(NC) {
        let jb = NC.min(n - jc);

        for pc in (0..k).step_by(KC) {
            let pb = KC.min(k - pc);

            // Convert B tile to f32, stored as transposed (column-major within tile):
            // b_col_f32[j * pb + p] = B[pc+p, jc+j].to_f32()
            // This makes each column of B contiguous for dot product.
            for j in 0..jb {
                for p in 0..pb {
                    b_col_f32[j * pb + p] = b[(pc + p) * n + (jc + j)].to_f32();
                }
            }

            for ic in (0..m).step_by(MC) {
                let ib = MC.min(m - ic);

                // Convert A tile to f32 (row-major: a_f32[i * pb + p])
                for i in 0..ib {
                    let a_src = &a[(ic + i) * k + pc..];
                    let a_dst = &mut a_f32[i * pb..i * pb + pb];
                    for p in 0..pb {
                        a_dst[p] = a_src[p].to_f32();
                    }
                }

                // Compute tile: C[ic:ic+ib, jc:jc+jb] += alpha * A_tile * B_tile
                // Use SIMD dot_f32 for the inner product over the K dimension.
                for i in 0..ib {
                    let a_row = &a_f32[i * pb..i * pb + pb];
                    for j in 0..jb {
                        let b_col = &b_col_f32[j * pb..j * pb + pb];
                        let dot = dot_f32(a_row, b_col);
                        c[(ic + i) * n + (jc + j)] += alpha * dot;
                    }
                }
            }
        }
    }
}

/// Mixed-precision GEMM: f32 inputs -> BF16 compute -> f32 output.
/// Quantizes inputs on-the-fly, useful for training where you want
/// reduced memory bandwidth but f32 gradients.
pub fn mixed_precision_gemm(
    a_f32: &[f32],
    b_f32: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    beta: f32,
) {
    let a_bf16 = f32_vec_to_bf16(a_f32);
    let b_bf16 = f32_vec_to_bf16(b_f32);
    bf16_gemm_f32(&a_bf16, &b_bf16, c, m, n, k, alpha, beta);
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bf16_roundtrip() {
        let values = [
            0.0f32,
            1.0,
            -1.0,
            std::f32::consts::PI,
            1e10,
            -1e-10,
            0.5,
            255.0,
        ];
        for &v in &values {
            let bf = BF16::from_f32(v);
            let back = bf.to_f32();
            let err = (back - v).abs();
            let tol = v.abs() * 0.01 + 1e-30; // ~0.8% relative error for BF16
            assert!(err < tol, "BF16 roundtrip: {} -> {} (err={})", v, back, err);
        }
    }

    #[test]
    fn test_bf16_truncate_vs_round() {
        // 1.5 in f32 = 0x3FC00000
        // BF16 truncate: 0x3FC0
        // BF16 round: 0x3FC0 (already exact)
        let v = 1.5f32;
        let t = BF16::from_f32_truncate(v);
        let r = BF16::from_f32(v);
        assert_eq!(t.to_f32(), 1.5);
        assert_eq!(r.to_f32(), 1.5);
    }

    #[test]
    fn test_bf16_gemm_identity() {
        let a_f32 = vec![1.0f32, 0.0, 0.0, 1.0];
        let b_f32 = vec![3.0f32, 7.0, 5.0, 11.0];
        let a = f32_vec_to_bf16(&a_f32);
        let b = f32_vec_to_bf16(&b_f32);
        let mut c = vec![0.0f32; 4];
        bf16_gemm_f32(&a, &b, &mut c, 2, 2, 2, 1.0, 0.0);
        assert!((c[0] - 3.0).abs() < 0.1);
        assert!((c[1] - 7.0).abs() < 0.1);
        assert!((c[2] - 5.0).abs() < 0.1);
        assert!((c[3] - 11.0).abs() < 0.1);
    }

    #[test]
    fn test_bf16_gemm_multiply() {
        let a_f32 = vec![1.0f32, 2.0, 3.0, 4.0];
        let b_f32 = vec![5.0f32, 6.0, 7.0, 8.0];
        let a = f32_vec_to_bf16(&a_f32);
        let b = f32_vec_to_bf16(&b_f32);
        let mut c = vec![0.0f32; 4];
        bf16_gemm_f32(&a, &b, &mut c, 2, 2, 2, 1.0, 0.0);
        // Expected: [[19, 22], [43, 50]]
        assert!((c[0] - 19.0).abs() < 0.5, "c[0]={}", c[0]);
        assert!((c[1] - 22.0).abs() < 0.5, "c[1]={}", c[1]);
        assert!((c[2] - 43.0).abs() < 0.5, "c[2]={}", c[2]);
        assert!((c[3] - 50.0).abs() < 0.5, "c[3]={}", c[3]);
    }

    #[test]
    fn test_mixed_precision_gemm() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![5.0f32, 6.0, 7.0, 8.0];
        let mut c = vec![10.0f32; 4];
        mixed_precision_gemm(&a, &b, &mut c, 2, 2, 2, 1.0, 1.0);
        // C = 1.0 * A*B + 1.0 * 10 = [[29, 32], [53, 60]]
        assert!((c[0] - 29.0).abs() < 1.0);
        assert!((c[1] - 32.0).abs() < 1.0);
        assert!((c[2] - 53.0).abs() < 1.0);
        assert!((c[3] - 60.0).abs() < 1.0);
    }

    #[test]
    fn test_bulk_conversion() {
        let src = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let bf16 = f32_vec_to_bf16(&src);
        let back = bf16_vec_to_f32(&bf16);
        for i in 0..src.len() {
            assert!((back[i] - src[i]).abs() < 0.1);
        }
    }
}
