![numrus](docs/numrus-hero.jpg)

# numrus

**SIMD-First Numerical Computing for Rust**

A pure Rust numerical library built from the ground up for AVX-512. BLAS, BF16 structured Hamming, CLAM metric trees, holographic vector operations, and NARS non-axiomatic reasoning — all in one workspace with zero external runtime dependencies.

The substrate for [cubus](https://github.com/cubus1/cubus).

---

## Workspace

| Crate | Description |
|-------|-------------|
| `numrus-core` | SIMD primitives: XOR bind, Hamming distance, BF16 structured Hamming, AVX-512 runtime dispatch |
| `numrus-blas` | Pure Rust BLAS: GEMM, dot product, matrix operations with SIMD acceleration |
| `numrus-mkl` | Intel MKL FFI bindings for hardware-optimized linear algebra |
| `numrus-clam` | CLAM (Clustered Learning of Approximate Manifolds) metric tree for O(log n) nearest-neighbor |
| `numrus-nars` | NARS (Non-Axiomatic Reasoning System) truth values, evidence accumulation, revision |
| `numrus-substrate` | CogRecord format, 4-channel schema (META/CAM/BTREE/EMBED), bitplane layout |
| `numrus-rs` | High-level API and Python bindings |

## Performance

| Operation | Speed | Notes |
|-----------|-------|-------|
| XOR Bind 8KB | 239 ns | 32 GiB/s, AVX-512 |
| XOR Bind 16KB | 547 ns | 28 GiB/s |
| Hamming 8KB (64Kbit) | 135 ns | 57 GiB/s, VPOPCNTDQ |
| Hamming 16KB | 292 ns | 52 GiB/s |
| BF16 Structured Hamming 1024-D | ~12.5 μs | Sign/exponent/mantissa weighted |
| SIMD vs Scalar | 24–57× | AVX-512 vs scalar fallback |

## BF16 Structured Hamming

Not flat XOR + popcount. Decomposes the diff by BF16 bit-field:

- **Sign bit** (1 bit) — semantic reversal, weighted highest
- **Exponent** (8 bits) — magnitude order, weighted medium
- **Mantissa** (7 bits) — fine precision, weighted lowest

Presets: `DEFAULT (1/4/2)`, `JINA (1/6/3)`, `TRAINING (2/3/4)`. Custom weights supported.

The `structural_diff` function returns `sign_flip_dims` and `major_magnitude_shifts` per pair — the same signals that drive the awareness substrate in cubus.

## Runtime SIMD Dispatch

```rust
// Detected once at startup via OnceLock
// AVX-512 → AVX2 → scalar fallback
// No compile-time feature flags needed
let distance = hamming_distance(a, b);  // dispatches automatically
```

Builds with `-C target-cpu=x86-64-v4` for full AVX-512 codegen. Runs on Railway (Amsterdam, Sapphire Rapids) in 70 seconds Docker build.

## Three-Tier Distance Pipeline

| Tier | Method | Cost | Output |
|------|--------|------|--------|
| 1 | Binary Hamming | 135 ns / 8KB pair | Scalar distance, 99% candidate pruning |
| 2 | BF16 Structured | 12.5 μs / 1024-D pair | Sign/exponent/mantissa decomposition |
| 3 | Superposition decompose | ~1.2 μs / 3 vectors | Crystallized / tensioned / uncertain / noise per dimension |

Tier 1 is faster than GPU cosine on float32. Tier 2 gives richer signals. Tier 3 gives cognitive state for free.

## Origin

Born because CLIP training on 120K images took 50 minutes — Rust had no numpy equivalent. Instead of dropping to Python, we built numrus. AVX-512 GEMM, SIMD Hamming, BF16 structured distance, CLAM trees — all in Rust, all zero-copy, all sharing the same binary layout that cubus uses for its cognitive blackboard.

## Usage

```toml
[dependencies]
numrus-core = { git = "https://github.com/cubus1/numrus" }
numrus-blas = { git = "https://github.com/cubus1/numrus" }
numrus-clam = { git = "https://github.com/cubus1/numrus" }
```

## License

MIT

---

*Jan Hübener*
