use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use numrus_mkl::{fft, vml};

fn bench_vsexp(c: &mut Criterion) {
    let mut group = c.benchmark_group("vsexp");
    for &n in &[64, 256, 1024, 4096, 16384] {
        let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01).min(80.0)).collect();
        let mut out = vec![0.0f32; n];
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &_n| {
            b.iter(|| vml::vsexp(&x, &mut out));
        });
    }
    group.finish();
}

fn bench_fft(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_f32");
    for &n in &[64, 256, 1024, 4096] {
        let mut data: Vec<f32> = (0..2 * n).map(|i| (i as f32 * 0.01).sin()).collect();
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter(|| {
                // Reset data each iteration
                for (i, d) in data[..2 * n].iter_mut().enumerate() {
                    *d = (i as f32 * 0.01).sin();
                }
                fft::fft_f32(&mut data, n);
            });
        });
    }
    group.finish();
}

fn bench_vssqrt(c: &mut Criterion) {
    let mut group = c.benchmark_group("vssqrt");
    for &n in &[64, 256, 1024, 4096, 16384] {
        let x: Vec<f32> = (1..=n).map(|i| i as f32).collect();
        let mut out = vec![0.0f32; n];
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &_n| {
            b.iter(|| vml::vssqrt(&x, &mut out));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_vsexp, bench_fft, bench_vssqrt);
criterion_main!(benches);
