// LAPACK/FFT functions have many parameters matching standard API signatures.
// Numeric kernels use index loops for clarity.
#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]

//! # RustyMKL
//!
//! Pure Rust Intel MKL replacement with AVX-512 SIMD.
//!
//! No FFI, no Intel dependencies. Covers the key MKL domains:
//!
//! - **LAPACK**: Linear algebra (LU, Cholesky, QR, eigenvalues, SVD)
//! - **FFT**: Fast Fourier Transform (radix-2 Cooley-Tukey, split-radix)
//! - **VML**: Vector math (exp, log, sin, cos, sqrt, pow — vectorized)
//!
//! ## Zero-Copy with Blackboard
//!
//! All operations work directly on blackboard buffers from `numrus_core`,
//! eliminating serialization between crates.
//!
//! ```
//! use numrus_core::Blackboard;
//! use numrus_mkl::vml;
//!
//! let mut bb = Blackboard::new();
//! bb.alloc_f32("input", 1024);
//! bb.alloc_f32("output", 1024);
//!
//! // Fill input...
//! let (inp, out) = bb.borrow_2_mut_f32("input", "output").unwrap();
//! inp.iter_mut().enumerate().for_each(|(i, x)| *x = i as f32 * 0.01);
//!
//! // Vectorized exp — directly on blackboard memory
//! vml::vsexp(inp, out);
//! ```

#![feature(portable_simd)]

pub mod fft;
pub mod lapack;
pub mod vml;
