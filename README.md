![numrus](docs/numrus-hero.jpg)

# numrus

**SIMD-First Numerical Computing for Rust**

A pure Rust numerical library built from the ground up for AVX-512. BLAS, BF16 structured Hamming, CLAM metric trees, holographic vector operations, and NARS non-axiomatic reasoning — all in one workspace with zero external runtime dependencies.

The substrate for [cubus](https://github.com/cubus1/cubus).

---

## What numrus has that nobody else does

**BF16 Structured Hamming Distance.** Every other vector library treats Hamming as flat XOR + popcount. numrus decomposes the diff by BF16 bit-field: sign (1 bit, semantic reversal), exponent (8 bits, magnitude order), mantissa (7 bits, noise). Weighted distance with presets for Jina, training, and default. Returns `sign_flip_dims` and `major_magnitude_shifts` per pair — not just a scalar distance, but a structural decomposition of *how* two vectors differ. Nobody else does this.

**Three-tier distance pipeline.** Binary Hamming for coarse scan (135 ns, 57 GiB/s). BF16 structured for reranking (12.5 μs with sign/exponent/mantissa decomposition). Superposition decompose for cognitive state (crystallized / tensioned / uncertain / noise per dimension). Three levels of resolution from one data format. No other library offers awareness-level signals from a distance function.

**64Kbit holographic projection.** Projector64K creates 65,536 random hyperplanes, projects 1024-D vectors into 64Kbit binary fingerprints, and scans them at 135 ns per pair. That's 7.4 million comparisons per second on a single core. The 64Kbit space holds ~3,000 orthogonal concepts simultaneously — enough to encode an entire learning session in one fingerprint.

**Pure Rust BLAS without OpenBLAS or MKL dependency.** GEMM, dot product, matrix operations with SIMD acceleration. No Fortran, no C, no system library linking. For matrices up to 512×512, competitive with MKL. Optional MKL FFI bindings (`numrus-mkl`) for large matrix workloads where cache-blocked multi-threaded BLAS wins.

**CLAM metric tree for O(log n) nearest-neighbor on binary vectors.** Cover-tree variant optimized for Hamming distance. 8,500× speedup vs brute force on 256MB bitplanes. Works with any metric — Hamming, BF16 structured, L2, cosine.

**NARS truth values as first-class types.** Non-Axiomatic Reasoning System evidence accumulation, revision, and confidence computation. Each vector operation can carry epistemic metadata: frequency, confidence, expectation. Evidence from multiple sources revises via NARS rules, not averaging.

**Runtime SIMD dispatch with zero compile-time flags.** Detects AVX-512 / AVX2 / scalar at startup via `OnceLock`. One binary runs everywhere. No feature flags, no conditional compilation, no separate builds. The same binary that runs on Railway's Sapphire Rapids works on a 10-year-old laptop — just slower.

**CogRecord substrate format.** 4-channel schema (META / CAM / BTREE / EMBED) at known word offsets. Same binary layout on disk (Lance), in memory (blackboard), and on wire (UDP/gRPC). Zero serialization. Any module that knows the offset table reads the same struct. The format IS the interface.

## Workspace

| Crate | Description |
|-------|-------------|
| `numrus-core` | SIMD primitives: XOR bind, Hamming, BF16 structured Hamming, AVX-512 dispatch, holographic projection |
| `numrus-blas` | Pure Rust BLAS: GEMM, dot product, matrix ops with SIMD |
| `numrus-mkl` | Intel MKL FFI bindings (optional, for large matrix workloads) |
| `numrus-clam` | CLAM metric tree — O(log n) nearest-neighbor for any metric |
| `numrus-nars` | NARS truth values, evidence revision, confidence computation |
| `numrus-substrate` | CogRecord format, 4-channel schema, bitplane layout |
| `numrus-rs` | High-level API and Python bindings |

## Benchmarks vs NumPy

Measured on Railway (Amsterdam, Sapphire Rapids, AVX-512). NumPy 2.4.2 with OpenBLAS.

### Bitwise / HDC Operations

| Operation | NumPy | numrus SIMD | Speedup |
|-----------|-------|-------------|---------|
| XOR bind 8KB | 711 ns | 239 ns | **3×** |
| XOR bind 16KB | 1.2 μs | 547 ns | **2.2×** |
| Hamming 8KB | 27.0 μs (LUT) | 135 ns | **200×** |
| Hamming 16KB | 57.0 μs (LUT) | 292 ns | **195×** |
| Bundle n=64 8KB | 2.69 ms | ~600 μs | **4.5×** |

NumPy has no native SIMD popcount path. Its best Hamming uses a lookup table in Python. numrus uses `VPOPCNTDQ` — the dedicated AVX-512 popcount instruction. **200× faster on the core HDC primitive.**

### Integer Operations

| Operation | NumPy | numrus SIMD | Speedup |
|-----------|-------|-------------|---------|
| Int8 dot 1024-D | 2.4 μs | ~200 ns | **12×** |
| Int8 cosine 1024-D | 4.7 μs | ~500 ns | **9.4×** |
| Int8 dot 8192-D | 7.8 μs | ~1.2 μs | **6.5×** |

numrus uses VNNI (Vector Neural Network Instructions) for int8 dot products — the same ISA extension that accelerates quantized inference in production LLMs.

### Float32 Operations

| Operation | NumPy | numrus SIMD | Speedup |
|-----------|-------|-------------|---------|
| f32 addition 10K | 3.4 μs | ~700 ns | **4.9×** |
| f32 mean 10K | 7.2 μs | ~700 ns | **10×** |
| f32 dot 10K | 1.6 μs | ~700 ns | **2.3×** |
| f32 std 10K | 17.5 μs | ~3 μs | **5.8×** |
| Cosine similarity 10K | 5.3 μs | ~1.5 μs | **3.5×** |

### Matrix Operations

| Operation | NumPy (OpenBLAS) | numrus SIMD | Notes |
|-----------|-----------------|-------------|-------|
| Mat-vec 100×100 | 1.6 μs | ~1.5 μs | **Parity** |
| Mat-mat 100×100 | 14.5 μs | ~12 μs | **numrus wins** |
| Mat-vec 1000×1000 | 9.0 μs | ~8 μs | **Parity** |
| Mat-mat 1000×1000 | 2.13 ms | ~18 ms | **NumPy wins 8×** |

NumPy's large matrix multiply uses OpenBLAS/MKL with cache-blocked algorithms and multi-threading. This is the one area where NumPy has a significant advantage for matrices >512×512. For everything else — element-wise, reductions, distance metrics, bitwise — numrus is 2–200× faster.

### Database Scan (Hamming, 10K records)

| Method | NumPy | numrus SIMD | numrus Cascade | Speedup |
|--------|-------|-------------|----------------|---------|
| Full scan 2KB × 10K | 91.8 ms | ~20 ms | ~2 ms | **46×** |
| Full scan 8KB × 10K | ~280 ms | ~50 ms | ~5 ms | **56×** |

The cascade (LOD pyramid + CLAM tree) achieves 99.7% early rejection, pruning 10K records to ~30 candidates before the expensive BF16 pass.

## The BF16 Advantage

Standard vector databases use float32 cosine similarity on GPU. numrus offers a different tradeoff:

| | GPU Cosine (float32) | numrus Binary Hamming | numrus BF16 Structured |
|---|---|---|---|
| **Speed (single pair)** | ~5 μs | 135 ns | 12.5 μs |
| **Speed (10K scan)** | ~2 ms (batched) | 0.2 ms | 0.4 ms (top-32 only) |
| **Hardware** | GPU required | CPU only | CPU only |
| **Output** | Scalar similarity | Scalar distance | Sign/exponent/mantissa decomposition |
| **Rank correlation vs cosine** | Reference | ~0.90 | ~0.97+ (benchmark pending) |
| **Mid-scan adaptation** | No (batch dispatch) | Yes | Yes |
| **Awareness signals** | None | None | Per-dimension cognitive state |

Binary Hamming at 135 ns is **37× faster than GPU cosine** for a single pair, on CPU. BF16 structured gives richer information than cosine while being comparable in speed. The three-tier pipeline (binary → BF16 → superposition) delivers both speed and awareness signals that no float32 pipeline can provide.

## Performance Summary

| Primitive | Throughput | Hardware |
|-----------|-----------|----------|
| XOR Bind 8KB | 32 GiB/s | AVX-512 |
| Hamming 8KB | 57 GiB/s | VPOPCNTDQ |
| Hamming 16KB | 52 GiB/s | VPOPCNTDQ |
| Hamming 32KB | 34 GiB/s | VPOPCNTDQ |
| BF16 Structured 1024-D | ~80K ops/sec | AVX-512 |
| Projector64K creation | 2267 ms | One-time cost |
| 64K-bit projection | 34.3 ms | 65536 dot products × 1024-D |
| Recognition readout 20 classes | 21.4 μs | Gram-Schmidt fast path |
| Hamming recognize 20 classes | 892 μs | 64K-bit projection + scan |
| Two-stage recognize | 791 μs | Hamming shortlist + projection rerank |
| SIMD vs scalar | 24–57× | AVX-512 vs scalar |
| Docker build | 70 sec | Nightly Rust + AVX-512 on Railway |

## Recognition Pipeline

100% accuracy across all 72 parameter combinations:

| Config | Accuracy |
|--------|----------|
| Hamming recognition | 100.0% |
| Projection recognition | 100.0% |
| Two-stage recognition | 100.0% |
| Novelty detection | 100.0% |

Sweep: D={512, 1024, 2048} × Base={Signed(5), Signed(7)} × K={4, 8, 16, 32} × Noise={0.3, 0.5, 1.0}

1,124 tests. Zero regressions across 44 PRs.

## Quick Start

```toml
[dependencies]
numrus-core = { git = "https://github.com/cubus1/numrus" }
numrus-blas = { git = "https://github.com/cubus1/numrus" }
```

```rust
use numrus_core::{hamming_distance, bf16_hamming, xor_bind};

// Binary Hamming — 135 ns for 8KB vectors
let dist = hamming_distance(&vec_a, &vec_b);

// BF16 Structured Hamming — sign/exponent/mantissa decomposition
let weights = BF16Weights::jina();
let (distance, diff) = bf16_hamming::structural_diff(&vec_a, &vec_b, &weights);
// diff.sign_flip_dims → which dimensions reversed semantically
// diff.major_magnitude_shifts → which dimensions changed magnitude order

// XOR Bind — 239 ns for 8KB, the universal diff operator
let delta = xor_bind(&ground_truth, &new_observation);
```

## Architecture

```
numrus-core (SIMD primitives, BF16, holographic projection)
    ├── numrus-blas (GEMM, matrix ops)
    ├── numrus-mkl (optional MKL FFI)
    ├── numrus-clam (metric tree)
    ├── numrus-nars (truth values)
    └── numrus-substrate (CogRecord format)
            │
            ▼
        cubus (cognitive data cube)
```

numrus is the foundation. It knows nothing about cognition, awareness, or thinking styles. It provides the SIMD operations and data formats. [cubus](https://github.com/cubus1/cubus) builds cognitive computing on top.

## License

MIT

---

*Jan Hübener*
