use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use numrus_blas::{level1, level3, Layout, Transpose};

fn bench_sdot(c: &mut Criterion) {
    let mut group = c.benchmark_group("sdot");
    for &n in &[64, 256, 1024, 4096, 16384] {
        let x: Vec<f32> = (0..n).map(|i| i as f32 * 0.001).collect();
        let y: Vec<f32> = (0..n).map(|i| i as f32 * 0.002).collect();
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &_n| {
            b.iter(|| level1::sdot(n, &x, 1, &y, 1));
        });
    }
    group.finish();
}

fn bench_saxpy(c: &mut Criterion) {
    let mut group = c.benchmark_group("saxpy");
    for &n in &[64, 256, 1024, 4096, 16384] {
        let x: Vec<f32> = (0..n).map(|i| i as f32 * 0.001).collect();
        let mut y: Vec<f32> = (0..n).map(|i| i as f32 * 0.002).collect();
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &_n| {
            b.iter(|| {
                y.fill(0.0);
                level1::saxpy(n, 2.0, &x, 1, &mut y, 1);
            });
        });
    }
    group.finish();
}

fn bench_sgemm(c: &mut Criterion) {
    let mut group = c.benchmark_group("sgemm");
    for &n in &[32, 64, 128, 256] {
        let a: Vec<f32> = (0..n * n).map(|i| (i as f32 * 0.001).sin()).collect();
        let b: Vec<f32> = (0..n * n).map(|i| (i as f32 * 0.002).cos()).collect();
        let mut c_mat = vec![0.0f32; n * n];
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, &n| {
            bench.iter(|| {
                c_mat.fill(0.0);
                level3::sgemm(
                    Layout::RowMajor,
                    Transpose::NoTrans,
                    Transpose::NoTrans,
                    n,
                    n,
                    n,
                    1.0,
                    &a,
                    n,
                    &b,
                    n,
                    0.0,
                    &mut c_mat,
                    n,
                );
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_sdot, bench_saxpy, bench_sgemm);
criterion_main!(benches);
