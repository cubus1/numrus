//! Compute dispatch layer: route work to the cheapest capable device.
//!
//! Architecture (tiered, cheapest-first):
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │  Tier 0: INT8 Prefilter (AVX-512 VNNI / NPU if available)  │
//! │  - Approximate candidate selection                         │
//! │  - Cheap stats (mean, variance, SD) on quantized data      │
//! │  - ~4x throughput vs f32 (64 MACs/instruction vs 16 FMAs)  │
//! ├─────────────────────────────────────────────────────────────┤
//! │  Tier 1: AVX-512 VNNI INT8 GEMM                            │
//! │  - Full INT8 matrix multiply for inference                  │
//! │  - Per-channel dequantization                               │
//! ├─────────────────────────────────────────────────────────────┤
//! │  Tier 2: AVX-512 BF16 GEMM                                 │
//! │  - Half bandwidth, f32 accumulation                         │
//! │  - Training + inference mixed precision                     │
//! ├─────────────────────────────────────────────────────────────┤
//! │  Tier 3: AVX-512 FP32 GEMM (cache-blocked + MT)            │
//! │  - Full precision compute                                   │
//! │  - Only for rows/columns that survived prefiltering         │
//! ├─────────────────────────────────────────────────────────────┤
//! │  Tier 4: Intel Xe2 GPU (if available via Level Zero)        │
//! │  - Offload massive dense GEMM (>2048x2048)                 │
//! │  - Async dispatch with CPU doing other work meanwhile       │
//! └─────────────────────────────────────────────────────────────┘
//! ```

use std::sync::OnceLock;

/// Detected compute capabilities of the current hardware.
#[derive(Clone, Debug, Default)]
pub struct ComputeCaps {
    pub avx512f: bool,
    pub avx512bw: bool,
    pub avx512vnni: bool,
    pub avx512_bf16: bool,
    pub avx512_vpopcntdq: bool,
    pub amx_tile: bool,
    pub amx_int8: bool,
    pub amx_bf16: bool,
    pub npu_available: bool,
    pub gpu_available: bool,
    pub gpu_name: Option<String>,
    pub num_cores: usize,
}

static CAPS: OnceLock<ComputeCaps> = OnceLock::new();

/// Which CPUID output register to check.
#[derive(Clone, Copy)]
#[allow(dead_code)] // All four registers exist; only Edx used for current AMX checks.
enum CpuidReg {
    Eax,
    Ebx,
    Ecx,
    Edx,
}

/// Check a specific CPUID feature bit (for AMX detection which lacks stable Rust macros).
#[cfg(target_arch = "x86_64")]
fn detect_cpuid_feature(leaf: u32, sub_leaf: u32, reg: CpuidReg, bit: u32) -> bool {
    let result = core::arch::x86_64::__cpuid_count(leaf, sub_leaf);
    let val = match reg {
        CpuidReg::Eax => result.eax,
        CpuidReg::Ebx => result.ebx,
        CpuidReg::Ecx => result.ecx,
        CpuidReg::Edx => result.edx,
    };
    (val >> bit) & 1 != 0
}

/// Detect hardware capabilities (cached after first call).
pub fn detect() -> &'static ComputeCaps {
    CAPS.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            ComputeCaps {
                avx512f: is_x86_feature_detected!("avx512f"),
                avx512bw: is_x86_feature_detected!("avx512bw"),
                avx512vnni: is_x86_feature_detected!("avx512vnni"),
                avx512_bf16: is_x86_feature_detected!("avx512bf16"),
                avx512_vpopcntdq: is_x86_feature_detected!("avx512vpopcntdq"),
                // AMX detection via cpuid (unstable intrinsics, use raw cpuid leaf 7, sub-leaf 0)
                amx_tile: detect_cpuid_feature(7, 0, CpuidReg::Edx, 24),
                amx_int8: detect_cpuid_feature(7, 0, CpuidReg::Edx, 25),
                amx_bf16: detect_cpuid_feature(7, 0, CpuidReg::Edx, 22),
                npu_available: detect_npu(),
                gpu_available: detect_gpu().is_some(),
                gpu_name: detect_gpu(),
                num_cores: std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(1),
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            ComputeCaps {
                avx512f: false,
                avx512bw: false,
                avx512vnni: false,
                avx512_bf16: false,
                avx512_vpopcntdq: false,
                amx_tile: false,
                amx_int8: false,
                amx_bf16: false,
                npu_available: false,
                gpu_available: false,
                gpu_name: None,
                num_cores: std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(1),
            }
        }
    })
}

/// Check if Intel NPU is available (Meteor Lake+).
#[allow(dead_code)]
fn detect_npu() -> bool {
    // Check for /dev/accel* (Intel NPU device nodes on Linux)
    std::path::Path::new("/dev/accel/accel0").exists()
        || std::path::Path::new("/dev/accel0").exists()
}

/// Detect Intel GPU via DRI device nodes.
#[allow(dead_code)]
fn detect_gpu() -> Option<String> {
    // Check for /dev/dri/renderD128 (Intel GPU render node)
    if std::path::Path::new("/dev/dri/renderD128").exists() {
        // Try to read GPU name from sysfs
        if let Ok(name) = std::fs::read_to_string("/sys/class/drm/card0/device/product_name") {
            return Some(name.trim().to_string());
        }
        return Some("Intel GPU (unknown model)".to_string());
    }
    None
}

/// Compute tier recommendation for a given workload.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum ComputeTier {
    /// INT8 VNNI: cheap prefilter, approximate stats, candidate selection
    Int8Vnni,
    /// BF16: half-bandwidth GEMM with f32 accumulation
    Bf16,
    /// FP32 AVX-512: full precision, cache-blocked + multithreaded
    Fp32Avx512,
    /// AMX tiles: massive INT8/BF16 GEMM (if available)
    AmxTile,
    /// GPU offload: very large dense compute
    Gpu,
    /// Scalar fallback
    #[default]
    Scalar,
}

/// Recommend compute tier based on workload characteristics.
pub fn recommend_tier(m: usize, n: usize, k: usize, precision_required: Precision) -> ComputeTier {
    let caps = detect();
    let total_flops = 2 * m * n * k;

    match precision_required {
        Precision::Approximate => {
            // For approximate work (prefiltering, candidate selection, stats)
            if caps.avx512vnni {
                return ComputeTier::Int8Vnni;
            }
        }
        Precision::Half => {
            if caps.amx_bf16 && total_flops > 100_000_000 {
                return ComputeTier::AmxTile;
            }
            if caps.avx512_bf16 {
                return ComputeTier::Bf16;
            }
        }
        Precision::Full => {
            if caps.gpu_available && total_flops > 1_000_000_000 {
                return ComputeTier::Gpu;
            }
            if caps.avx512f {
                return ComputeTier::Fp32Avx512;
            }
        }
    }

    if caps.avx512f {
        ComputeTier::Fp32Avx512
    } else {
        ComputeTier::Scalar
    }
}

/// Required precision level.
#[derive(Clone, Copy, Debug, Default)]
pub enum Precision {
    /// ~1% error OK (prefiltering, candidate ranking, approximate stats)
    Approximate,
    /// BF16: ~0.4% relative error (training, inference)
    Half,
    /// Full f32 precision
    #[default]
    Full,
}

/// Print detected capabilities summary.
pub fn print_caps() {
    let caps = detect();
    println!("=== Compute Capabilities ===");
    println!("  CPU cores:     {}", caps.num_cores);
    println!("  AVX-512F:      {}", caps.avx512f);
    println!("  AVX-512BW:     {}", caps.avx512bw);
    println!("  AVX-512 VNNI:  {}", caps.avx512vnni);
    println!("  AVX-512 BF16:  {}", caps.avx512_bf16);
    println!("  AVX-512 VPOP:  {}", caps.avx512_vpopcntdq);
    println!("  AMX Tile:      {}", caps.amx_tile);
    println!("  AMX INT8:      {}", caps.amx_int8);
    println!("  AMX BF16:      {}", caps.amx_bf16);
    println!("  NPU:           {}", caps.npu_available);
    if let Some(ref name) = caps.gpu_name {
        println!("  GPU:           {}", name);
    } else {
        println!("  GPU:           not detected");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_detect_caps() {
        let caps = detect();
        // On this machine we know AVX-512 is available
        assert!(caps.avx512f);
        assert!(caps.avx512vnni);
        assert!(caps.num_cores > 0);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_recommend_tier() {
        let tier = recommend_tier(1024, 1024, 1024, Precision::Approximate);
        // Should recommend INT8 VNNI since we have it
        assert_eq!(tier, ComputeTier::Int8Vnni);

        let tier = recommend_tier(256, 256, 256, Precision::Full);
        assert_eq!(tier, ComputeTier::Fp32Avx512);
    }
}
