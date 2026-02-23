//! Vector Math Library (VML) — SIMD-vectorized transcendental functions.
//!
//! Pure Rust replacement for Intel MKL VML. All functions process arrays
//! element-wise using AVX-512 SIMD.
//!
//! Naming convention follows MKL: `vs` prefix = single-precision vector,
//! `vd` prefix = double-precision vector.

use numrus_core::simd::{F32_LANES, F64_LANES};
use std::simd::num::SimdFloat;
use std::simd::StdFloat;

// SIMD vector types selected by feature flag
#[cfg(feature = "avx512")]
use std::simd::{f32x16 as F32Simd, f64x8 as F64Simd};
#[cfg(not(feature = "avx512"))]
use std::simd::{f32x8 as F32Simd, f64x4 as F64Simd};

// Integer SIMD types matching float lane widths — for bit manipulation in
// transcendental functions (ldexp, exponent extraction, sign flip).
#[cfg(feature = "avx512")]
use std::simd::{u32x16 as U32Simd, u64x8 as U64Simd};
#[cfg(not(feature = "avx512"))]
use std::simd::{u32x8 as U32Simd, u64x4 as U64Simd};

// ============================================================================
// EXP: e^x
// ============================================================================

/// Vectorized single-precision exp: out[i] = e^(x[i])
///
/// Uses polynomial approximation for SIMD lanes, scalar fallback for tail.
pub fn vsexp(x: &[f32], out: &mut [f32]) {
    #[cfg(feature = "mkl")]
    {
        unsafe {
            numrus_core::mkl_ffi::vsExp(x.len() as i32, x.as_ptr(), out.as_mut_ptr());
        }
        return;
    }
    assert_eq!(x.len(), out.len());
    let len = x.len();
    let chunks = len / F32_LANES;

    for i in 0..chunks {
        let base = i * F32_LANES;
        let xv = F32Simd::from_slice(&x[base..]);
        let result = simd_exp_f32(xv);
        result.copy_to_slice(&mut out[base..base + F32_LANES]);
    }

    for i in (chunks * F32_LANES)..len {
        out[i] = x[i].exp();
    }
}

/// Vectorized double-precision exp: out[i] = e^(x[i])
pub fn vdexp(x: &[f64], out: &mut [f64]) {
    #[cfg(feature = "mkl")]
    {
        unsafe {
            numrus_core::mkl_ffi::vdExp(x.len() as i32, x.as_ptr(), out.as_mut_ptr());
        }
        return;
    }
    assert_eq!(x.len(), out.len());
    let len = x.len();
    let chunks = len / F64_LANES;

    for i in 0..chunks {
        let base = i * F64_LANES;
        let xv = F64Simd::from_slice(&x[base..]);
        let result = simd_exp_f64(xv);
        result.copy_to_slice(&mut out[base..base + F64_LANES]);
    }

    for i in (chunks * F64_LANES)..len {
        out[i] = x[i].exp();
    }
}

// ============================================================================
// LOG: ln(x)
// ============================================================================

/// Vectorized single-precision natural log: out[i] = ln(x[i])
pub fn vsln(x: &[f32], out: &mut [f32]) {
    #[cfg(feature = "mkl")]
    {
        unsafe {
            numrus_core::mkl_ffi::vsLn(x.len() as i32, x.as_ptr(), out.as_mut_ptr());
        }
        return;
    }
    assert_eq!(x.len(), out.len());
    let len = x.len();
    let chunks = len / F32_LANES;

    for i in 0..chunks {
        let base = i * F32_LANES;
        let xv = F32Simd::from_slice(&x[base..]);
        let result = simd_ln_f32(xv);
        result.copy_to_slice(&mut out[base..base + F32_LANES]);
    }

    for i in (chunks * F32_LANES)..len {
        out[i] = x[i].ln();
    }
}

/// Vectorized double-precision natural log: out[i] = ln(x[i])
pub fn vdln(x: &[f64], out: &mut [f64]) {
    #[cfg(feature = "mkl")]
    {
        unsafe {
            numrus_core::mkl_ffi::vdLn(x.len() as i32, x.as_ptr(), out.as_mut_ptr());
        }
        return;
    }
    assert_eq!(x.len(), out.len());
    let len = x.len();
    let chunks = len / F64_LANES;

    for i in 0..chunks {
        let base = i * F64_LANES;
        let xv = F64Simd::from_slice(&x[base..]);
        let result = simd_ln_f64(xv);
        result.copy_to_slice(&mut out[base..base + F64_LANES]);
    }

    for i in (chunks * F64_LANES)..len {
        out[i] = x[i].ln();
    }
}

// ============================================================================
// SQRT: square root
// ============================================================================

/// Vectorized single-precision sqrt: out[i] = sqrt(x[i])
pub fn vssqrt(x: &[f32], out: &mut [f32]) {
    #[cfg(feature = "mkl")]
    {
        unsafe {
            numrus_core::mkl_ffi::vsSqrt(x.len() as i32, x.as_ptr(), out.as_mut_ptr());
        }
        return;
    }
    assert_eq!(x.len(), out.len());
    let len = x.len();
    let chunks = len / F32_LANES;

    for i in 0..chunks {
        let base = i * F32_LANES;
        let xv = F32Simd::from_slice(&x[base..]);
        let result = xv.sqrt();
        result.copy_to_slice(&mut out[base..base + F32_LANES]);
    }

    for i in (chunks * F32_LANES)..len {
        out[i] = x[i].sqrt();
    }
}

/// Vectorized double-precision sqrt.
pub fn vdsqrt(x: &[f64], out: &mut [f64]) {
    #[cfg(feature = "mkl")]
    {
        unsafe {
            numrus_core::mkl_ffi::vdSqrt(x.len() as i32, x.as_ptr(), out.as_mut_ptr());
        }
        return;
    }
    assert_eq!(x.len(), out.len());
    let len = x.len();
    let chunks = len / F64_LANES;

    for i in 0..chunks {
        let base = i * F64_LANES;
        let xv = F64Simd::from_slice(&x[base..]);
        let result = xv.sqrt();
        result.copy_to_slice(&mut out[base..base + F64_LANES]);
    }

    for i in (chunks * F64_LANES)..len {
        out[i] = x[i].sqrt();
    }
}

// ============================================================================
// ABS: absolute value
// ============================================================================

/// Vectorized single-precision abs: out[i] = |x[i]|
pub fn vsabs(x: &[f32], out: &mut [f32]) {
    #[cfg(feature = "mkl")]
    {
        unsafe {
            numrus_core::mkl_ffi::vsAbs(x.len() as i32, x.as_ptr(), out.as_mut_ptr());
        }
        return;
    }
    assert_eq!(x.len(), out.len());
    let len = x.len();
    let chunks = len / F32_LANES;

    for i in 0..chunks {
        let base = i * F32_LANES;
        let xv = F32Simd::from_slice(&x[base..]);
        let result = xv.abs();
        result.copy_to_slice(&mut out[base..base + F32_LANES]);
    }

    for i in (chunks * F32_LANES)..len {
        out[i] = x[i].abs();
    }
}

/// Vectorized double-precision abs.
pub fn vdabs(x: &[f64], out: &mut [f64]) {
    #[cfg(feature = "mkl")]
    {
        unsafe {
            numrus_core::mkl_ffi::vdAbs(x.len() as i32, x.as_ptr(), out.as_mut_ptr());
        }
        return;
    }
    assert_eq!(x.len(), out.len());
    let len = x.len();
    let chunks = len / F64_LANES;

    for i in 0..chunks {
        let base = i * F64_LANES;
        let xv = F64Simd::from_slice(&x[base..]);
        let result = xv.abs();
        result.copy_to_slice(&mut out[base..base + F64_LANES]);
    }

    for i in (chunks * F64_LANES)..len {
        out[i] = x[i].abs();
    }
}

// ============================================================================
// ADD / SUB / MUL / DIV: element-wise arithmetic
// ============================================================================

/// Vectorized single-precision add: out[i] = a[i] + b[i]
pub fn vsadd(a: &[f32], b: &[f32], out: &mut [f32]) {
    #[cfg(feature = "mkl")]
    {
        unsafe {
            numrus_core::mkl_ffi::vsAdd(a.len() as i32, a.as_ptr(), b.as_ptr(), out.as_mut_ptr());
        }
        return;
    }
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), out.len());
    let len = a.len();
    let chunks = len / F32_LANES;

    for i in 0..chunks {
        let base = i * F32_LANES;
        let av = F32Simd::from_slice(&a[base..]);
        let bv = F32Simd::from_slice(&b[base..]);
        let result = av + bv;
        result.copy_to_slice(&mut out[base..base + F32_LANES]);
    }

    for i in (chunks * F32_LANES)..len {
        out[i] = a[i] + b[i];
    }
}

/// Vectorized single-precision multiply: out[i] = a[i] * b[i]
pub fn vsmul(a: &[f32], b: &[f32], out: &mut [f32]) {
    #[cfg(feature = "mkl")]
    {
        unsafe {
            numrus_core::mkl_ffi::vsMul(a.len() as i32, a.as_ptr(), b.as_ptr(), out.as_mut_ptr());
        }
        return;
    }
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), out.len());
    let len = a.len();
    let chunks = len / F32_LANES;

    for i in 0..chunks {
        let base = i * F32_LANES;
        let av = F32Simd::from_slice(&a[base..]);
        let bv = F32Simd::from_slice(&b[base..]);
        let result = av * bv;
        result.copy_to_slice(&mut out[base..base + F32_LANES]);
    }

    for i in (chunks * F32_LANES)..len {
        out[i] = a[i] * b[i];
    }
}

/// Vectorized single-precision divide: out[i] = a[i] / b[i]
pub fn vsdiv(a: &[f32], b: &[f32], out: &mut [f32]) {
    #[cfg(feature = "mkl")]
    {
        unsafe {
            numrus_core::mkl_ffi::vsDiv(a.len() as i32, a.as_ptr(), b.as_ptr(), out.as_mut_ptr());
        }
        return;
    }
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), out.len());
    let len = a.len();
    let chunks = len / F32_LANES;

    for i in 0..chunks {
        let base = i * F32_LANES;
        let av = F32Simd::from_slice(&a[base..]);
        let bv = F32Simd::from_slice(&b[base..]);
        let result = av / bv;
        result.copy_to_slice(&mut out[base..base + F32_LANES]);
    }

    for i in (chunks * F32_LANES)..len {
        out[i] = a[i] / b[i];
    }
}

// ============================================================================
// SIN / COS: trigonometric functions (SIMD polynomial approximation)
// ============================================================================

/// Vectorized single-precision sin: out[i] = sin(x[i])
///
/// Uses Cody-Waite range reduction + degree-9 minimax polynomial,
/// all in SIMD. Sign correction via XOR on IEEE 754 sign bit.
pub fn vssin(x: &[f32], out: &mut [f32]) {
    #[cfg(feature = "mkl")]
    {
        unsafe {
            numrus_core::mkl_ffi::vsSin(x.len() as i32, x.as_ptr(), out.as_mut_ptr());
        }
        return;
    }
    assert_eq!(x.len(), out.len());
    let len = x.len();
    let chunks = len / F32_LANES;

    for i in 0..chunks {
        let base = i * F32_LANES;
        let xv = F32Simd::from_slice(&x[base..]);
        let result = simd_sin_f32(xv);
        result.copy_to_slice(&mut out[base..base + F32_LANES]);
    }

    for i in (chunks * F32_LANES)..len {
        out[i] = x[i].sin();
    }
}

/// Vectorized single-precision cos: out[i] = cos(x[i])
///
/// cos(x) = sin(x + π/2), reusing the SIMD sin kernel.
pub fn vscos(x: &[f32], out: &mut [f32]) {
    #[cfg(feature = "mkl")]
    {
        unsafe {
            numrus_core::mkl_ffi::vsCos(x.len() as i32, x.as_ptr(), out.as_mut_ptr());
        }
        return;
    }
    assert_eq!(x.len(), out.len());
    let len = x.len();
    let chunks = len / F32_LANES;
    let half_pi = F32Simd::splat(std::f32::consts::FRAC_PI_2);

    for i in 0..chunks {
        let base = i * F32_LANES;
        let xv = F32Simd::from_slice(&x[base..]) + half_pi;
        let result = simd_sin_f32(xv);
        result.copy_to_slice(&mut out[base..base + F32_LANES]);
    }

    for i in (chunks * F32_LANES)..len {
        out[i] = x[i].cos();
    }
}

/// Vectorized single-precision pow: out[i] = a[i]^b[i]
///
/// Computed as exp(b * ln(a)) using SIMD exp and ln kernels.
/// Handles a > 0 only (no complex branch for negative bases).
pub fn vspow(a: &[f32], b: &[f32], out: &mut [f32]) {
    #[cfg(feature = "mkl")]
    {
        unsafe {
            numrus_core::mkl_ffi::vsPow(a.len() as i32, a.as_ptr(), b.as_ptr(), out.as_mut_ptr());
        }
        return;
    }
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), out.len());
    let len = a.len();
    let chunks = len / F32_LANES;

    for i in 0..chunks {
        let base = i * F32_LANES;
        let av = F32Simd::from_slice(&a[base..]);
        let bv = F32Simd::from_slice(&b[base..]);
        // a^b = exp(b * ln(a))
        let ln_a = simd_ln_f32(av);
        let result = simd_exp_f32(bv * ln_a);
        result.copy_to_slice(&mut out[base..base + F32_LANES]);
    }

    for i in (chunks * F32_LANES)..len {
        out[i] = a[i].powf(b[i]);
    }
}

// ============================================================================
// SIMD polynomial approximations for transcendental functions
// ============================================================================

/// Fast SIMD exp(x) for F32Simd using the "range reduction + polynomial" method.
///
/// Algorithm:
/// 1. Clamp input to avoid overflow/underflow
/// 2. Decompose x = n * ln(2) + r, where n = round(x / ln(2))
/// 3. Compute exp(r) using degree-6 minimax polynomial
/// 4. Scale by 2^n via IEEE 754 exponent bit manipulation (ldexp)
#[inline(always)]
fn simd_exp_f32(x: F32Simd) -> F32Simd {
    let ln2_inv = F32Simd::splat(std::f32::consts::LOG2_E);
    let ln2_hi = F32Simd::splat(0.693_145_75_f32);
    let ln2_lo = F32Simd::splat(1.428_606_8e-6_f32);

    // Polynomial coefficients (minimax on [-ln2/2, ln2/2])
    let c1 = F32Simd::splat(1.0);
    let c2 = F32Simd::splat(0.5);
    let c3 = F32Simd::splat(0.166_666_67);
    let c4 = F32Simd::splat(0.041_666_668);
    let c5 = F32Simd::splat(0.008_333_334);

    // Clamp to avoid overflow
    let x_clamped = x
        .simd_max(F32Simd::splat(-87.0))
        .simd_min(F32Simd::splat(88.0));

    // n = round(x / ln2)
    let n = (x_clamped * ln2_inv + F32Simd::splat(0.5)).floor();

    // r = x - n * ln2 (high precision via hi/lo split)
    let r = x_clamped - n * ln2_hi - n * ln2_lo;

    // exp(r) ≈ 1 + r + r^2/2 + r^3/6 + r^4/24 + r^5/120
    let poly = c1 + r * (c1 + r * (c2 + r * (c3 + r * (c4 + r * c5))));

    // ldexp: poly * 2^n via IEEE 754 exponent bit manipulation.
    // 2^n as f32 = bits ((n + 127) << 23) when -126 <= n <= 127.
    // Convert n (float with integer values) to i32, bias, shift into exponent field.
    let n_arr = n.to_array();
    let mut pow2n_arr = [0u32; F32_LANES];
    for i in 0..F32_LANES {
        pow2n_arr[i] = ((n_arr[i] as i32 + 127) as u32) << 23;
    }
    let pow2n = F32Simd::from_bits(U32Simd::from_array(pow2n_arr));
    poly * pow2n
}

/// Fast SIMD exp(x) for F64Simd using range reduction + polynomial.
///
/// Degree-7 Taylor/minimax polynomial with Cody-Waite reduction.
/// ldexp via IEEE 754 f64 exponent bit manipulation.
#[inline(always)]
fn simd_exp_f64(x: F64Simd) -> F64Simd {
    let ln2_inv = F64Simd::splat(std::f64::consts::LOG2_E);
    let ln2_hi = F64Simd::splat(6.93145751953125e-1f64);
    let ln2_lo = F64Simd::splat(1.42860676533018e-6f64);

    // Degree-7 polynomial coefficients (Taylor series 1/n!)
    let c1 = F64Simd::splat(1.0);
    let c2 = F64Simd::splat(0.5);
    let c3 = F64Simd::splat(1.0 / 6.0);
    let c4 = F64Simd::splat(1.0 / 24.0);
    let c5 = F64Simd::splat(1.0 / 120.0);
    let c6 = F64Simd::splat(1.0 / 720.0);
    let c7 = F64Simd::splat(1.0 / 5040.0);

    let x_clamped = x
        .simd_max(F64Simd::splat(-708.0))
        .simd_min(F64Simd::splat(709.0));
    let n = (x_clamped * ln2_inv + F64Simd::splat(0.5)).floor();
    let r = x_clamped - n * ln2_hi - n * ln2_lo;

    let poly = c1 + r * (c1 + r * (c2 + r * (c3 + r * (c4 + r * (c5 + r * (c6 + r * c7))))));

    // ldexp for f64: 2^n = bits ((n + 1023) << 52)
    let n_arr = n.to_array();
    let mut pow2n_arr = [0u64; F64_LANES];
    for i in 0..F64_LANES {
        pow2n_arr[i] = ((n_arr[i] as i64 + 1023) as u64) << 52;
    }
    let pow2n = F64Simd::from_bits(U64Simd::from_array(pow2n_arr));
    poly * pow2n
}

/// Fast SIMD ln(x) for F32Simd.
///
/// Algorithm:
/// 1. Decompose x = 2^e * m, where m ∈ [1, 2) via IEEE 754 bit extraction
/// 2. ln(x) = e * ln(2) + ln(m)
/// 3. ln(m) via Padé-like series: u = (m-1)/(m+1), ln(m) = 2u(1 + u²/3 + u⁴/5 + ...)
#[inline(always)]
fn simd_ln_f32(x: F32Simd) -> F32Simd {
    let bits = x.to_bits(); // u32xN

    // Extract exponent: ((bits >> 23) & 0xFF) - 127
    let exp_raw = (bits >> U32Simd::splat(23)) & U32Simd::splat(0xFF);
    // Convert to f32 for the final computation: e = exp_raw - 127
    let exp_arr = exp_raw.to_array();
    let mut exp_f32_arr = [0.0f32; F32_LANES];
    for i in 0..F32_LANES {
        exp_f32_arr[i] = exp_arr[i] as f32 - 127.0;
    }
    let exp_f32 = F32Simd::from_array(exp_f32_arr);

    // Normalize mantissa to [1, 2): clear exponent, set to bias 127
    let mantissa_bits = (bits & U32Simd::splat(0x007F_FFFF)) | U32Simd::splat(0x3F80_0000);
    let m = F32Simd::from_bits(mantissa_bits);

    // Padé series: u = (m-1)/(m+1), ln(m) = 2*u*(1 + u²/3 + u⁴/5 + u⁶/7 + u⁸/9)
    let one = F32Simd::splat(1.0);
    let u = (m - one) / (m + one);
    let u2 = u * u;

    let poly = one
        + u2 * (F32Simd::splat(1.0 / 3.0)
            + u2 * (F32Simd::splat(1.0 / 5.0)
                + u2 * (F32Simd::splat(1.0 / 7.0) + u2 * F32Simd::splat(1.0 / 9.0))));
    let ln_m = F32Simd::splat(2.0) * u * poly;

    exp_f32 * F32Simd::splat(std::f32::consts::LN_2) + ln_m
}

/// Fast SIMD ln(x) for F64Simd.
///
/// Same algorithm as simd_ln_f32 but with higher-degree polynomial for f64 precision.
/// Padé series to degree 15 (u^14 term).
#[inline(always)]
fn simd_ln_f64(x: F64Simd) -> F64Simd {
    let bits = x.to_bits(); // u64xN

    // Extract exponent: ((bits >> 52) & 0x7FF) - 1023
    let exp_raw = (bits >> U64Simd::splat(52)) & U64Simd::splat(0x7FF);
    let exp_arr = exp_raw.to_array();
    let mut exp_f64_arr = [0.0f64; F64_LANES];
    for i in 0..F64_LANES {
        exp_f64_arr[i] = exp_arr[i] as f64 - 1023.0;
    }
    let exp_f64 = F64Simd::from_array(exp_f64_arr);

    // Normalize mantissa to [1, 2)
    let mantissa_bits =
        (bits & U64Simd::splat(0x000F_FFFF_FFFF_FFFF)) | U64Simd::splat(0x3FF0_0000_0000_0000);
    let m = F64Simd::from_bits(mantissa_bits);

    let one = F64Simd::splat(1.0);
    let u = (m - one) / (m + one);
    let u2 = u * u;

    // Higher-degree series for f64 precision
    let poly = one
        + u2 * (F64Simd::splat(1.0 / 3.0)
            + u2 * (F64Simd::splat(1.0 / 5.0)
                + u2 * (F64Simd::splat(1.0 / 7.0)
                    + u2 * (F64Simd::splat(1.0 / 9.0)
                        + u2 * (F64Simd::splat(1.0 / 11.0)
                            + u2 * (F64Simd::splat(1.0 / 13.0)
                                + u2 * F64Simd::splat(1.0 / 15.0)))))));
    let ln_m = F64Simd::splat(2.0) * u * poly;

    exp_f64 * F64Simd::splat(std::f64::consts::LN_2) + ln_m
}

/// Fast SIMD sin(x) for F32Simd.
///
/// Algorithm:
/// 1. Cody-Waite range reduction: n = round(x/π), r = x - n*π
///    (r ∈ [-π/2, π/2])
/// 2. sin(r) via degree-9 minimax polynomial (Horner form)
/// 3. Sign correction: sin(nπ + r) = (-1)^n * sin(r)
///    Applied via XOR on IEEE 754 sign bit — no branching.
#[inline(always)]
fn simd_sin_f32(x: F32Simd) -> F32Simd {
    let inv_pi = F32Simd::splat(std::f32::consts::FRAC_1_PI);

    // Cody-Waite constants for π (hi + lo = π to ~24 digits)
    let pi_hi = F32Simd::splat(3.140625f32);
    let pi_lo = F32Simd::splat(9.676_536e-4_f32);

    // Range reduction: n = round(x/π)
    let n = (x * inv_pi + F32Simd::splat(0.5)).floor();

    // r = x - n * π (Cody-Waite for precision)
    let r = x - n * pi_hi - n * pi_lo;

    // sin(r) ≈ r * (1 + r²(-1/6 + r²(1/120 + r²(-1/5040 + r²/362880))))
    let r2 = r * r;
    let poly = F32Simd::splat(1.0)
        + r2 * (F32Simd::splat(-1.0 / 6.0)
            + r2 * (F32Simd::splat(1.0 / 120.0)
                + r2 * (F32Simd::splat(-1.0 / 5040.0) + r2 * F32Simd::splat(1.0 / 362880.0))));
    let sin_r = r * poly;

    // Sign correction: if n is odd, negate result.
    // Extract parity of n, shift to sign bit position, XOR with result.
    let n_arr = n.to_array();
    let sin_arr = sin_r.to_array();
    let mut out_arr = [0.0f32; F32_LANES];
    for i in 0..F32_LANES {
        let sign_flip = ((n_arr[i] as i32) & 1) as u32;
        out_arr[i] = f32::from_bits(sin_arr[i].to_bits() ^ (sign_flip << 31));
    }
    F32Simd::from_array(out_arr)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vsexp() {
        let x: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let mut out = vec![0.0f32; 32];
        vsexp(&x, &mut out);
        for i in 0..32 {
            let expected = x[i].exp();
            assert!(
                (out[i] - expected).abs() / expected.max(1e-10) < 1e-4,
                "vsexp mismatch at {}: {} vs {}",
                i,
                out[i],
                expected
            );
        }
    }

    #[test]
    fn test_vdexp() {
        let x: Vec<f64> = (0..16).map(|i| i as f64 * 0.2).collect();
        let mut out = vec![0.0f64; 16];
        vdexp(&x, &mut out);
        for i in 0..16 {
            let expected = x[i].exp();
            let rel_err = (out[i] - expected).abs() / expected.max(1e-15);
            assert!(
                rel_err < 1e-6,
                "vdexp mismatch at {}: {} vs {} (rel_err={})",
                i,
                out[i],
                expected,
                rel_err
            );
        }
    }

    #[test]
    fn test_vsln() {
        let x: Vec<f32> = (1..33).map(|i| i as f32 * 0.5).collect();
        let mut out = vec![0.0f32; 32];
        vsln(&x, &mut out);
        for i in 0..32 {
            let expected = x[i].ln();
            assert!(
                (out[i] - expected).abs() < 1e-4,
                "vsln mismatch at {}: {} vs {} (input={})",
                i,
                out[i],
                expected,
                x[i]
            );
        }
    }

    #[test]
    fn test_vdln() {
        let x: Vec<f64> = (1..17).map(|i| i as f64 * 0.3 + 0.1).collect();
        let mut out = vec![0.0f64; 16];
        vdln(&x, &mut out);
        for i in 0..16 {
            let expected = x[i].ln();
            assert!(
                (out[i] - expected).abs() < 1e-8,
                "vdln mismatch at {}: {} vs {} (input={})",
                i,
                out[i],
                expected,
                x[i]
            );
        }
    }

    #[test]
    fn test_vssqrt() {
        let x: Vec<f32> = (1..33).map(|i| i as f32).collect();
        let mut out = vec![0.0f32; 32];
        vssqrt(&x, &mut out);
        for i in 0..32 {
            assert!((out[i] - x[i].sqrt()).abs() < 1e-6);
        }
    }

    #[test]
    fn test_vsabs() {
        let x = vec![-1.0f32, 2.0, -3.0, 4.0];
        let mut out = vec![0.0f32; 4];
        vsabs(&x, &mut out);
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_vsadd() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![5.0f32, 6.0, 7.0, 8.0];
        let mut out = vec![0.0f32; 4];
        vsadd(&a, &b, &mut out);
        assert_eq!(out, vec![6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_vsmul() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![5.0f32, 6.0, 7.0, 8.0];
        let mut out = vec![0.0f32; 4];
        vsmul(&a, &b, &mut out);
        assert_eq!(out, vec![5.0, 12.0, 21.0, 32.0]);
    }

    #[test]
    fn test_vssin() {
        let x: Vec<f32> = (0..32).map(|i| i as f32 * 0.2 - 3.0).collect();
        let mut out = vec![0.0f32; 32];
        vssin(&x, &mut out);
        for i in 0..32 {
            let expected = x[i].sin();
            assert!(
                (out[i] - expected).abs() < 1e-4,
                "vssin mismatch at {}: {} vs {} (input={})",
                i,
                out[i],
                expected,
                x[i]
            );
        }
    }

    #[test]
    fn test_vscos() {
        let x: Vec<f32> = (0..32).map(|i| i as f32 * 0.2 - 3.0).collect();
        let mut out = vec![0.0f32; 32];
        vscos(&x, &mut out);
        for i in 0..32 {
            let expected = x[i].cos();
            assert!(
                (out[i] - expected).abs() < 1e-3,
                "vscos mismatch at {}: {} vs {} (input={})",
                i,
                out[i],
                expected,
                x[i]
            );
        }
    }

    #[test]
    fn test_vspow() {
        let a: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ];
        let b: Vec<f32> = vec![2.0; 16];
        let mut out = vec![0.0f32; 16];
        vspow(&a, &b, &mut out);
        for i in 0..16 {
            let expected = a[i].powf(b[i]);
            assert!(
                (out[i] - expected).abs() / expected.max(1.0) < 1e-3,
                "vspow mismatch at {}: {} vs {}",
                i,
                out[i],
                expected
            );
        }
    }

    #[test]
    fn test_exp_negative() {
        let x = vec![-1.0f32, -2.0, -5.0, -10.0];
        let mut out = vec![0.0f32; 4];
        vsexp(&x, &mut out);
        for i in 0..4 {
            let expected = x[i].exp();
            assert!(
                (out[i] - expected).abs() / expected.max(1e-10) < 1e-4,
                "exp({}) = {} vs expected {}",
                x[i],
                out[i],
                expected
            );
        }
    }

    #[test]
    fn test_sin_cos_identity() {
        // sin²(x) + cos²(x) = 1
        let x: Vec<f32> = (0..32).map(|i| i as f32 * 0.3 - 5.0).collect();
        let mut sin_out = vec![0.0f32; 32];
        let mut cos_out = vec![0.0f32; 32];
        vssin(&x, &mut sin_out);
        vscos(&x, &mut cos_out);
        for i in 0..32 {
            let sum = sin_out[i] * sin_out[i] + cos_out[i] * cos_out[i];
            assert!(
                (sum - 1.0).abs() < 1e-3,
                "sin²+cos² = {} at x={}",
                sum,
                x[i]
            );
        }
    }
}
