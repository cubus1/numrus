#![feature(portable_simd)]

use std::simd::f32x16;
use std::simd::num::SimdFloat;
use std::time::Instant;

/// Old numrus-rs approach: transpose B then dot product each row
fn gemm_transpose_dot(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    c.fill(0.0);
    let mut bt = vec![0.0f32; n * k];
    for i in 0..k {
        for j in 0..n {
            bt[j * k + i] = b[i * n + j];
        }
    }
    for i in 0..m {
        let a_row = &a[i * k..(i + 1) * k];
        for j in 0..n {
            let b_col = &bt[j * k..(j + 1) * k];
            let chunks = k / 16;
            let mut sum1 = f32x16::splat(0.0);
            let mut sum2 = f32x16::splat(0.0);
            for ci in (0..chunks).step_by(2) {
                let a1 = f32x16::from_slice(&a_row[ci * 16..]);
                let b1 = f32x16::from_slice(&b_col[ci * 16..]);
                sum1 += a1 * b1;
                if ci + 1 < chunks {
                    let a2 = f32x16::from_slice(&a_row[(ci + 1) * 16..]);
                    let b2 = f32x16::from_slice(&b_col[(ci + 1) * 16..]);
                    sum2 += a2 * b2;
                }
            }
            let mut s = (sum1 + sum2).reduce_sum();
            for ki in (chunks * 16)..k {
                s += a_row[ki] * b_col[ki];
            }
            c[i * n + j] = s;
        }
    }
}

fn bench_one(name: &str, m: usize, n: usize, k: usize, iters: usize) {
    let a: Vec<f32> = (0..m * k)
        .map(|i| ((i * 7 + 3) % 1000) as f32 * 0.001)
        .collect();
    let b: Vec<f32> = (0..k * n)
        .map(|i| ((i * 11 + 5) % 1000) as f32 * 0.001)
        .collect();
    let mut c_old = vec![0.0f32; m * n];
    let mut c_new = vec![0.0f32; m * n];

    // Warmup
    gemm_transpose_dot(&a, &b, &mut c_old, m, k, n);
    numrus_blas::level3::sgemm(
        numrus_blas::Layout::RowMajor,
        numrus_blas::Transpose::NoTrans,
        numrus_blas::Transpose::NoTrans,
        m,
        n,
        k,
        1.0,
        &a,
        k,
        &b,
        n,
        0.0,
        &mut c_new,
        n,
    );

    // Verify correctness
    let max_err = c_old
        .iter()
        .zip(c_new.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let max_val = c_old.iter().map(|x| x.abs()).fold(0.0f32, f32::max);

    // Benchmark old (transpose-dot)
    let t0 = Instant::now();
    for _ in 0..iters {
        gemm_transpose_dot(&a, &b, &mut c_old, m, k, n);
    }
    let old_time = t0.elapsed().as_secs_f64() / iters as f64;
    let old_gflops = 2.0 * m as f64 * n as f64 * k as f64 / old_time / 1e9;

    // Benchmark new (cache-blocked + multithreaded)
    let t1 = Instant::now();
    for _ in 0..iters {
        c_new.fill(0.0);
        numrus_blas::level3::sgemm(
            numrus_blas::Layout::RowMajor,
            numrus_blas::Transpose::NoTrans,
            numrus_blas::Transpose::NoTrans,
            m,
            n,
            k,
            1.0,
            &a,
            k,
            &b,
            n,
            0.0,
            &mut c_new,
            n,
        );
    }
    let new_time = t1.elapsed().as_secs_f64() / iters as f64;
    let new_gflops = 2.0 * m as f64 * n as f64 * k as f64 / new_time / 1e9;

    let speedup = old_time / new_time;

    println!(
        "  {:12} | old: {:8.2} ms ({:6.2} GFLOPS) | new: {:8.2} ms ({:6.2} GFLOPS) | speedup: {:5.2}x | err: {:.2e}",
        name,
        old_time * 1000.0, old_gflops,
        new_time * 1000.0, new_gflops,
        speedup,
        if max_val > 0.0 { max_err / max_val } else { max_err }
    );
}

fn main() {
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    println!(
        "=== GEMM Benchmark: old (transpose-dot) vs new (cache-blocked + {}T) ===",
        cores
    );
    println!(
        "  {:12} | {:>38} | {:>38} | {:>11} | err",
        "Size", "Old (transpose-dot)", "New (Goto+MT)", "Speedup"
    );
    println!("  {}", "-".repeat(130));

    bench_one("32x32", 32, 32, 32, 5000);
    bench_one("64x64", 64, 64, 64, 2000);
    bench_one("128x128", 128, 128, 128, 500);
    bench_one("256x256", 256, 256, 256, 100);
    bench_one("512x512", 512, 512, 512, 20);
    bench_one("1024x1024", 1024, 1024, 1024, 5);

    println!("\n  Sizes >= 512x512 were the gap where NumPy/OpenBLAS beat numrus-rs.");
    println!("  Cache-blocked GEMM with multithreading closes this gap.");
}
