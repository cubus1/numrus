//! # Numrus Core
//!
//! Shared SIMD primitives and zero-copy blackboard for the numrus ecosystem.
//!
//! This crate provides:
//! - **Blackboard**: Zero-copy shared mutable memory arena with SIMD-aligned allocations.
//!   Eliminates serialization between numrus-rs, numrus_blas, and numrus_mkl — all crates
//!   operate directly on the same aligned memory.
//! - **SIMD primitives**: Portable SIMD helpers shared across all crates.
//! - **Parallel execution**: Thread pool utilities for data-parallel SIMD workloads.
//! - **CBLAS layout types**: Row-major / column-major layout abstractions.

#![cfg_attr(any(feature = "avx512", feature = "avx2"), feature(portable_simd))]

pub mod bf16_hamming;
pub mod blackboard;
pub mod compute;
pub mod fingerprint;
pub mod layout;
pub mod parallel;
pub mod rng;

#[cfg(any(feature = "avx512", feature = "avx2"))]
pub mod prefilter;

// SIMD backend selection: AVX-512 or AVX2 (requires nightly for portable_simd)
#[cfg(feature = "avx512")]
pub mod simd;
#[cfg(all(feature = "avx2", not(feature = "avx512")))]
#[path = "simd_avx2.rs"]
pub mod simd;

// Intel MKL FFI bindings (only compiled when --features mkl is enabled)
#[cfg(feature = "mkl")]
pub mod mkl_ffi;

pub use bf16_hamming::{
    bf16_bytes_to_fp32, bf16_hamming_scalar, fp32_to_bf16_bytes, select_bf16_hamming_fn,
    structural_diff, BF16StructuralDiff, BF16Weights, JINA_WEIGHTS, TRAINING_WEIGHTS,
};
pub use blackboard::Blackboard;
pub use compute::{ComputeCaps, ComputeTier, Precision};
pub use fingerprint::{Fingerprint, Fingerprint1K, Fingerprint2K, Fingerprint64K};
pub use layout::{Layout, Transpose};
pub use parallel::parallel_for_chunks;
pub use rng::SplitMix64;
