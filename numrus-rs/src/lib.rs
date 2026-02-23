// Numeric array operations use index loops and many-argument SIMD dispatch patterns.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

//! # Numrus-Rs
//!
//! `numrus-rs` is a numerical library for Rust, focusing on operations that can be vectorized using SIMD.
//! This crate provides efficient numerical arrays and operations, including basic arithmetic, dot products,
//! and transformations.

#![feature(portable_simd)]

mod helpers;
pub mod num_array;
pub mod simd_ops;

pub mod traits;

pub use num_array::{binding_popcount_3d, find_discriminative_spots, find_holographic_sweet_spot};
pub use num_array::{decode_target_explicit, encode_edge_explicit, VerbCodebook};
pub use num_array::{simhash_batch_project, simhash_project};
pub use num_array::{sweep_cogrecords, CogRecord, SweepMode, SweepResult};
pub use num_array::{NumArray, NumArrayF32, NumArrayF64, NumArrayI32, NumArrayI64, NumArrayU8};
pub use simd_ops::{BitwiseSimdOps, HammingSimdOps, SimdOps};
