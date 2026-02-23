// BLAS functions match CBLAS signatures — many parameters are inherent to the API.
// Numeric kernels use index loops on packed arrays where iterators hurt readability.
#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]

//! # RustyBLAS
//!
//! Pure Rust BLAS implementation with AVX-512 SIMD — drop-in OpenBLAS replacement.
//!
//! No FFI, no C dependencies. All operations use `portable_simd` with
//! cache-blocked microkernels for maximum throughput.
//!
//! ## BLAS Levels
//!
//! - **Level 1** (vector-vector): `sdot`, `ddot`, `saxpy`, `daxpy`, `sscal`, `dscal`,
//!   `snrm2`, `dnrm2`, `sasum`, `dasum`, `isamax`, `idamax`, `scopy`, `dcopy`, `sswap`, `dswap`
//! - **Level 2** (matrix-vector): `sgemv`, `dgemv`, `sger`, `dger`, `ssymv`, `dsymv`,
//!   `strmv`, `dtrmv`, `strsv`, `dtrsv`
//! - **Level 3** (matrix-matrix): `sgemm`, `dgemm`, `ssymm`, `dsymm`, `strmm`, `dtrmm`,
//!   `strsm`, `dtrsm`, `ssyrk`, `dsyrk`
//!
//! ## Memory Layout
//!
//! Both row-major and column-major layouts are supported via CBLAS-style API.
//! All operations accept a `Layout` parameter.
//!
//! ## Zero-Copy with Blackboard
//!
//! ```
//! use numrus_core::Blackboard;
//! use numrus_blas::level3;
//! use numrus_core::layout::{Layout, Transpose};
//!
//! let mut bb = numrus_core::Blackboard::new();
//! let m = 64;
//! let k = 64;
//! let n = 64;
//! bb.alloc_f32("A", m * k);
//! bb.alloc_f32("B", k * n);
//! bb.alloc_f32("C", m * n);
//!
//! // Fill A, B...
//! let (a, b, c) = bb.borrow_3_mut_f32("A", "B", "C").unwrap();
//! a.fill(1.0);
//! b.fill(1.0);
//!
//! // GEMM directly on blackboard memory — zero copies, zero serialization
//! level3::sgemm(Layout::RowMajor, Transpose::NoTrans, Transpose::NoTrans,
//!               m, n, k, 1.0, a, k, b, n, 0.0, c, n);
//! ```

#![feature(portable_simd)]

pub mod bf16_gemm;
pub mod int8_gemm;
pub mod level1;
pub mod level2;
pub mod level3;

// Re-export layout types for convenience
pub use numrus_core::layout::{Diag, Layout, Side, Transpose, Uplo};

// Re-export quantized GEMM entry points
pub use bf16_gemm::{bf16_gemm_f32, mixed_precision_gemm, BF16};
pub use int8_gemm::{
    dequantize_i4_to_f32, int8_gemm_f32, int8_gemm_i32, quantize_f32_to_i4, quantize_f32_to_i8,
    quantize_f32_to_u8,
};
