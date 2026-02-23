#![allow(clippy::needless_range_loop)]

pub mod absorb;
pub mod flush;

pub use absorb::{
    XTransPattern, MultiResPattern,
    receptivity, organic_write, organic_write_f32, organic_read,
    OrganicWAL, WriteResult, PlasticityTracker, AbsorptionTracker,
};

pub use flush::{FlushAction, FlushResult, organic_flush};
