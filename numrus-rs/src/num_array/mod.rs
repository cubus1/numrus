mod array_struct;
pub mod binding_matrix;
pub mod bitwise;
pub mod cogrecord;
mod constructors;
pub mod graph;
pub mod hdc;
mod impl_clone_from;
pub mod linalg;
mod manipulation;
pub mod operations;
pub mod projection;
mod statistics;

pub use array_struct::{NumArray, NumArrayF32, NumArrayF64, NumArrayI32, NumArrayI64, NumArrayU8};
pub use binding_matrix::{
    binding_popcount_3d, find_discriminative_spots, find_holographic_sweet_spot,
};
pub use cogrecord::{sweep_cogrecords, CogRecord, SweepMode, SweepResult};
pub use graph::{decode_target_explicit, encode_edge_explicit, VerbCodebook};
pub use projection::{simhash_batch_project, simhash_project};
