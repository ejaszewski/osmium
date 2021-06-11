use criterion::*;

use nalgebra::SVector;
use num_traits::Zero;
use osmium_nn::layers::Linear;

fn dense_16(c: &mut Criterion) {
    let layer = Linear::<f32, 16, 16>::new();
    let input = SVector::<f32, 16>::zero();
    c.bench_function("dense_16", |b| b.iter(|| layer.forward(black_box(input))));
}

fn dense_32(c: &mut Criterion) {
    let layer = Linear::<f32, 32, 32>::new();
    let input = SVector::<f32, 32>::zero();
    c.bench_function("dense_32", |b| b.iter(|| layer.forward(black_box(input))));
}

fn dense_64(c: &mut Criterion) {
    let layer = Linear::<f32, 64, 64>::new();
    let input = SVector::<f32, 64>::zero();
    c.bench_function("dense_64", |b| b.iter(|| layer.forward(black_box(input))));
}

fn dense_128(c: &mut Criterion) {
    let layer = Linear::<f32, 128, 128>::new();
    let input = SVector::<f32, 128>::zero();
    c.bench_function("dense_128", |b| b.iter(|| layer.forward(black_box(input))));
}

criterion_group!(benches, dense_16, dense_32, dense_64, dense_128);
criterion_main!(benches);
