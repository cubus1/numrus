use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use numrus_rs::NumArrayU8;

/// All vector sizes we benchmark: 8K, 16K, 32K, 64K bytes
const VEC_SIZES: &[usize] = &[8192, 16384, 32768, 65536];

fn create_random_vector(seed: u64, len: usize) -> Vec<u8> {
    // Simple LCG for reproducible pseudo-random data
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (state >> 33) as u8
        })
        .collect()
}

// ============================================================================
// BIND (XOR): SIMD vs Naive
// ============================================================================

fn bench_bind(c: &mut Criterion) {
    let mut group = c.benchmark_group("HDC Bind (XOR)");

    for &vec_len in VEC_SIZES {
        let a = NumArrayU8::new(create_random_vector(42, vec_len));
        let b = NumArrayU8::new(create_random_vector(123, vec_len));

        group.throughput(Throughput::Bytes(vec_len as u64));

        // SIMD path (current optimized)
        group.bench_with_input(
            BenchmarkId::new("simd", vec_len),
            &vec_len,
            |bencher, &_| bencher.iter(|| black_box(&a) ^ black_box(&b)),
        );

        // Naive scalar baseline
        let a_data = a.get_data().to_vec();
        let b_data = b.get_data().to_vec();
        group.bench_with_input(
            BenchmarkId::new("naive_scalar", vec_len),
            &vec_len,
            |bencher, &_| {
                bencher.iter(|| {
                    let mut out = vec![0u8; vec_len];
                    for i in 0..vec_len {
                        out[i] = a_data[i] ^ b_data[i];
                    }
                    black_box(out)
                })
            },
        );
    }

    group.finish();
}

// ============================================================================
// DISTANCE (Hamming): SIMD POPCNT vs Naive
// ============================================================================

fn bench_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("HDC Distance (Hamming)");

    for &vec_len in VEC_SIZES {
        let a = NumArrayU8::new(create_random_vector(42, vec_len));
        let b = NumArrayU8::new(create_random_vector(123, vec_len));

        group.throughput(Throughput::Bytes(vec_len as u64));

        // SIMD POPCNT path (current optimized)
        group.bench_with_input(
            BenchmarkId::new("simd_popcnt", vec_len),
            &vec_len,
            |bencher, &_| bencher.iter(|| black_box(a.hamming_distance(&b))),
        );

        // Naive scalar: per-byte XOR + lookup popcount
        let a_data = a.get_data().to_vec();
        let b_data = b.get_data().to_vec();
        group.bench_with_input(
            BenchmarkId::new("naive_scalar", vec_len),
            &vec_len,
            |bencher, &_| {
                bencher.iter(|| {
                    let mut total: u64 = 0;
                    for i in 0..vec_len {
                        total += (a_data[i] ^ b_data[i]).count_ones() as u64;
                    }
                    black_box(total)
                })
            },
        );
    }

    group.finish();
}

// ============================================================================
// PERMUTE
// ============================================================================

fn bench_permute(c: &mut Criterion) {
    let mut group = c.benchmark_group("HDC Permute");

    for &vec_len in VEC_SIZES {
        let v = NumArrayU8::new(create_random_vector(42, vec_len));

        group.throughput(Throughput::Bytes(vec_len as u64));
        group.bench_with_input(BenchmarkId::new("k=1", vec_len), &vec_len, |bencher, &_| {
            bencher.iter(|| black_box(&v).permute(black_box(1)))
        });
    }

    group.finish();
}

// ============================================================================
// BUNDLE: Ripple-carry SIMD vs Naive
// ============================================================================

fn bench_bundle(c: &mut Criterion) {
    let mut group = c.benchmark_group("HDC Bundle (Majority Vote)");

    for &vec_len in VEC_SIZES {
        for &count in &[5, 16, 64, 256, 1024] {
            let vectors: Vec<NumArrayU8> = (0..count)
                .map(|i| NumArrayU8::new(create_random_vector(i as u64, vec_len)))
                .collect();
            let vec_refs: Vec<&NumArrayU8> = vectors.iter().collect();

            group.throughput(Throughput::Bytes((vec_len * count) as u64));

            group.bench_with_input(
                BenchmarkId::new(format!("ripple_{}", vec_len), count),
                &count,
                |bencher, &_| bencher.iter(|| NumArrayU8::bundle(black_box(&vec_refs))),
            );

            // Naive baseline only for 8192 bytes to keep benchmark time reasonable
            if vec_len == 8192 {
                group.bench_with_input(
                    BenchmarkId::new("naive_8192", count),
                    &count,
                    |bencher, &_| {
                        bencher.iter(|| {
                            let len = vec_len;
                            let n = vec_refs.len();
                            let threshold = n / 2;
                            let mut out = vec![0u8; len];
                            for (byte_idx, out_byte) in out.iter_mut().enumerate().take(len) {
                                let mut result_byte = 0u8;
                                for bit in 0..8u8 {
                                    let mut count = 0u32;
                                    for v in vec_refs.iter() {
                                        count += ((v.get_data()[byte_idx] >> bit) & 1) as u32;
                                    }
                                    if count as usize > threshold {
                                        result_byte |= 1 << bit;
                                    }
                                }
                                *out_byte = result_byte;
                            }
                            black_box(out)
                        })
                    },
                );
            }
        }
    }

    group.finish();
}

// ============================================================================
// EDGE ENCODE / DECODE
// ============================================================================

fn bench_edge_encode_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("HDC Edge Encode/Decode");

    for &vec_len in &[8192usize, 65536] {
        let src = NumArrayU8::new(create_random_vector(1, vec_len));
        let rel = NumArrayU8::new(create_random_vector(2, vec_len));
        let tgt = NumArrayU8::new(create_random_vector(3, vec_len));

        group.throughput(Throughput::Bytes(vec_len as u64 * 3));

        group.bench_with_input(
            BenchmarkId::new("encode", vec_len),
            &vec_len,
            |bencher, &_| {
                bencher.iter(|| {
                    let perm_rel = black_box(&rel).permute(1);
                    let perm_tgt = black_box(&tgt).permute(2);
                    let edge = &(black_box(&src) ^ &perm_rel) ^ &perm_tgt;
                    black_box(edge)
                })
            },
        );

        let perm_rel = rel.permute(1);
        let perm_tgt = tgt.permute(2);
        let edge = &(&src ^ &perm_rel) ^ &perm_tgt;
        let total_bits = vec_len * 8;

        group.bench_with_input(
            BenchmarkId::new("decode_target", vec_len),
            &vec_len,
            |bencher, &_| {
                bencher.iter(|| {
                    let recovered_perm =
                        &(black_box(&edge) ^ black_box(&src)) ^ black_box(&perm_rel);
                    let recovered = recovered_perm.permute(total_bits - 2);
                    black_box(recovered)
                })
            },
        );
    }

    group.finish();
}

// ============================================================================
// BATCH DISTANCE
// ============================================================================

fn bench_batch_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("HDC Batch Distance");

    for &vec_len in &[8192usize, 65536] {
        for &count in &[10, 100, 1000] {
            let a_data: Vec<u8> = (0..vec_len * count)
                .map(|i| ((i * 37 + 13) % 256) as u8)
                .collect();
            let b_data: Vec<u8> = (0..vec_len * count)
                .map(|i| ((i * 71 + 42) % 256) as u8)
                .collect();
            let a = NumArrayU8::new(a_data);
            let b = NumArrayU8::new(b_data);

            group.throughput(Throughput::Bytes((vec_len * count * 2) as u64));

            group.bench_with_input(
                BenchmarkId::new(format!("batch_{}", vec_len), count),
                &count,
                |bencher, &count| {
                    bencher.iter(|| black_box(a.hamming_distance_batch(&b, vec_len, count)))
                },
            );
        }
    }

    group.finish();
}

// ============================================================================
// INT8 DOT PRODUCT: SIMD (VNNI) vs Naive
// ============================================================================

fn bench_dot_i8(c: &mut Criterion) {
    let mut group = c.benchmark_group("HDC Int8 Dot Product (VNNI)");

    // CogRecord Container 3 sizes: 1024D (1KB), 2048D (2KB full container)
    for &dim in &[1024usize, 2048, 8192] {
        let a = NumArrayU8::new(create_random_vector(42, dim));
        let b = NumArrayU8::new(create_random_vector(123, dim));

        group.throughput(Throughput::Bytes((dim * 2) as u64));

        // SIMD dot_i8 (VNNI-targetable)
        group.bench_with_input(BenchmarkId::new("dot_i8_simd", dim), &dim, |bencher, &_| {
            bencher.iter(|| black_box(a.dot_i8(&b)))
        });

        // SIMD cosine_i8
        group.bench_with_input(
            BenchmarkId::new("cosine_i8_simd", dim),
            &dim,
            |bencher, &_| bencher.iter(|| black_box(a.cosine_i8(&b))),
        );

        // Naive scalar dot product baseline
        let a_data = a.get_data().to_vec();
        let b_data = b.get_data().to_vec();
        group.bench_with_input(
            BenchmarkId::new("dot_i8_naive", dim),
            &dim,
            |bencher, &_| {
                bencher.iter(|| {
                    let mut total: i64 = 0;
                    for i in 0..dim {
                        total += (a_data[i] as i8 as i64) * (b_data[i] as i8 as i64);
                    }
                    black_box(total)
                })
            },
        );

        // Naive scalar cosine baseline
        group.bench_with_input(
            BenchmarkId::new("cosine_i8_naive", dim),
            &dim,
            |bencher, &_| {
                bencher.iter(|| {
                    let mut dot: i64 = 0;
                    let mut norm_a: i64 = 0;
                    let mut norm_b: i64 = 0;
                    for i in 0..dim {
                        let ai = a_data[i] as i8 as i64;
                        let bi = b_data[i] as i8 as i64;
                        dot += ai * bi;
                        norm_a += ai * ai;
                        norm_b += bi * bi;
                    }
                    let cos = dot as f64 / ((norm_a as f64).sqrt() * (norm_b as f64).sqrt());
                    black_box(cos)
                })
            },
        );
    }

    group.finish();
}

// ============================================================================
// ADAPTIVE CASCADE vs FULL SCAN (Hamming)
// ============================================================================

fn bench_adaptive_hamming(c: &mut Criterion) {
    let mut group = c.benchmark_group("Adaptive Hamming Search");
    group.sample_size(20); // Lower sample count for large-scale scans

    for &vec_len in &[2048usize, 8192] {
        for &db_count in &[1000, 10000] {
            let query = NumArrayU8::new(create_random_vector(42, vec_len));

            // Build database: ~0.1% match rate (1-10 matches in db_count vectors)
            let mut db_data = Vec::with_capacity(vec_len * db_count);
            let query_data = query.get_data();
            for i in 0..db_count {
                if i == 0 || i == db_count / 2 {
                    // Near matches: same as query with a few flips
                    let mut v = query_data.to_vec();
                    let stride = vec_len / 5;
                    for j in 0..5 {
                        v[j * stride] ^= 0xFF;
                    }
                    db_data.extend(v);
                } else {
                    db_data.extend(create_random_vector(i as u64 + 1000, vec_len));
                }
            }
            let db = NumArrayU8::new(db_data.clone());

            // Hamming threshold: ~40 bits (tight, should accept the 2 near-matches)
            let threshold = 50u64;

            // Full scan: compute every distance, filter
            group.bench_with_input(
                BenchmarkId::new(format!("full_scan_{}B", vec_len), db_count),
                &db_count,
                |bencher, &count| {
                    bencher.iter(|| {
                        let mut results = Vec::new();
                        for i in 0..count {
                            let candidate =
                                NumArrayU8::new(db_data[i * vec_len..(i + 1) * vec_len].to_vec());
                            let d = query.hamming_distance(&candidate);
                            if d <= threshold {
                                results.push((i, d));
                            }
                        }
                        black_box(results)
                    })
                },
            );

            // Adaptive cascade search
            group.bench_with_input(
                BenchmarkId::new(format!("adaptive_{}B", vec_len), db_count),
                &db_count,
                |bencher, &_| {
                    bencher.iter(|| {
                        black_box(query.hamming_search_adaptive(&db, vec_len, db_count, threshold))
                    })
                },
            );
        }
    }

    group.finish();
}

// ============================================================================
// ADAPTIVE CASCADE vs FULL SCAN (Cosine)
// ============================================================================

fn bench_adaptive_cosine(c: &mut Criterion) {
    let mut group = c.benchmark_group("Adaptive Cosine Search");
    group.sample_size(20);

    for &vec_len in &[1024usize, 2048] {
        for &db_count in &[1000, 10000] {
            let query = NumArrayU8::new(create_random_vector(42, vec_len));

            // Build database: mostly random, a few near-identical
            let mut db_data = Vec::with_capacity(vec_len * db_count);
            let query_data = query.get_data();
            for i in 0..db_count {
                if i == 0 || i == db_count / 3 || i == db_count * 2 / 3 {
                    // Near-identical: copy query with slight noise
                    let mut v = query_data.to_vec();
                    for j in (0..vec_len).step_by(100) {
                        v[j] = v[j].wrapping_add(1);
                    }
                    db_data.extend(v);
                } else {
                    db_data.extend(create_random_vector(i as u64 + 5000, vec_len));
                }
            }
            let db = NumArrayU8::new(db_data.clone());
            let min_sim = 0.9;

            // Full scan: compute every cosine
            group.bench_with_input(
                BenchmarkId::new(format!("full_scan_{}D", vec_len), db_count),
                &db_count,
                |bencher, &count| {
                    bencher.iter(|| {
                        let mut results = Vec::new();
                        for i in 0..count {
                            let candidate =
                                NumArrayU8::new(db_data[i * vec_len..(i + 1) * vec_len].to_vec());
                            let cos = query.cosine_i8(&candidate);
                            if cos >= min_sim {
                                results.push((i, cos));
                            }
                        }
                        black_box(results)
                    })
                },
            );

            // Adaptive cascade
            group.bench_with_input(
                BenchmarkId::new(format!("adaptive_{}D", vec_len), db_count),
                &db_count,
                |bencher, &_| {
                    bencher.iter(|| {
                        black_box(query.cosine_search_adaptive(&db, vec_len, db_count, min_sim))
                    })
                },
            );
        }
    }

    group.finish();
}

// ============================================================================
// POPCOUNT: SIMD vs Naive
// ============================================================================

fn bench_popcount(c: &mut Criterion) {
    let mut group = c.benchmark_group("Popcount");

    for &vec_len in VEC_SIZES {
        let a = NumArrayU8::new(create_random_vector(42, vec_len));

        group.throughput(Throughput::Bytes(vec_len as u64));

        // SIMD popcount
        group.bench_with_input(
            BenchmarkId::new("simd", vec_len),
            &vec_len,
            |bencher, &_| bencher.iter(|| black_box(a.popcount())),
        );

        // Naive scalar popcount
        let data = a.get_data().to_vec();
        group.bench_with_input(
            BenchmarkId::new("naive_scalar", vec_len),
            &vec_len,
            |bencher, &_| {
                bencher.iter(|| {
                    let mut total: u64 = 0;
                    for &byte in data.iter() {
                        total += byte.count_ones() as u64;
                    }
                    black_box(total)
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_bind,
    bench_distance,
    bench_permute,
    bench_bundle,
    bench_edge_encode_decode,
    bench_batch_distance,
    bench_dot_i8,
    bench_adaptive_hamming,
    bench_adaptive_cosine,
    bench_popcount,
);
criterion_main!(benches);
