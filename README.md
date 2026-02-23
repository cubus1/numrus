# numrus

SIMD-accelerated numeric computing for Rust. AVX-512, AVX2, scalar fallback.

## Crates

| Crate | Description |
|-------|-------------|
| numrus-core | SIMD dispatch, Hamming distance, BF16 structured Hamming |
| numrus-blas | Pure Rust BLAS L1-L3, BF16/INT8 GEMM |
| numrus-mkl | Feature-gated Intel MKL FFI |
| numrus-clam | CLAM metric tree, LOD pyramid, exact k-NN |
| numrus-nars | Non-axiomatic reasoning, Granger causality, causal learning |
| numrus-substrate | Learning substrate with absorption/saturation dynamics |
| numrus-rs | NumArray API (numpy-like) |
| bindings/python | PyO3 bindings — `pip install numrus` |

## Quick Start

```rust
use numrus_core::{select_hamming_fn, bf16_hamming::select_bf16_hamming_fn};

// Binary Hamming distance (AVX-512 auto-dispatch)
let hamming = select_hamming_fn();
let dist = hamming(&vec_a, &vec_b);

// BF16 structured Hamming (sign/exponent/mantissa aware)
let bf16_fn = select_bf16_hamming_fn();
let structured_dist = bf16_fn(&bf16_a, &bf16_b, &JINA_WEIGHTS);
```

## Performance

| Operation | 1024-D | Hardware |
|-----------|--------|----------|
| Binary Hamming | ~3 us | AVX-512 VPOPCNTDQ |
| BF16 Structured Hamming | ~12.5 us | AVX-512 |
| LOD+CLAM search (256MB) | <2 ms | 8,500x vs brute force |
| SGEMM 1Kx1K | ~2 ms | MKL |

## Dependency Graph

```
numrus-mkl ──→ numrus-core ←── numrus-blas
                    ↑
              numrus-clam
                    ↑
              numrus-nars ──→ numrus-core (bf16_hamming)
                    ↑
            numrus-substrate
                    ↑
               numrus-rs ──→ numrus-core, numrus-blas, numrus-mkl
                    ↑
           bindings/python ──→ numrus-rs
```

No circular dependencies. Clean DAG.

## License

MIT OR Apache-2.0
