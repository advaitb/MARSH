//! Criterion benchmark of the headline systems number: the B-replicate bootstrap run, swept
//! over replicate count B and source count K (spec §6.3 experiment 4).
//!
//! Run with: `cargo bench --bench bootstrap_bench`

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use marsh::bootstrap::{self, BootstrapConfig, IntervalMethod};
use marsh::estimate::Prepared;
use marsh::lp::GoodLpSolver;
use marsh::sim::{generate, SimConfig};
use marsh::unknown::UnknownMode;
use std::hint::black_box;

fn bench_replicates(c: &mut Criterion) {
    let cfg = SimConfig {
        num_taxa: 64,
        num_sources: 5,
        sink_depth: 20_000,
        source_depth: 20_000,
        ..Default::default()
    };
    let sc = generate(&cfg, 42);
    let prepared = Prepared::new(&sc.tree, &sc.sources, UnknownMode::None, 0.0);
    let src: Vec<Vec<f64>> = sc.sources.profiles.iter().map(|p| p.normalized()).collect();
    let point = prepared
        .solve(&GoodLpSolver, &src, &sc.sink.normalized())
        .unwrap()
        .weights;

    let mut group = c.benchmark_group("bootstrap_vs_B");
    for &b in &[100usize, 500, 1000] {
        group.bench_with_input(BenchmarkId::from_parameter(b), &b, |bencher, &b| {
            let bc = BootstrapConfig {
                replicates: b,
                seed: 0,
                interval: IntervalMethod::Percentile,
                alpha: 0.05,
            };
            bencher.iter(|| {
                let res =
                    bootstrap::run(&GoodLpSolver, &prepared, &sc.sources, &sc.sink, &point, &bc);
                black_box(res.replicates_used)
            });
        });
    }
    group.finish();
}

fn bench_num_sources(c: &mut Criterion) {
    let mut group = c.benchmark_group("bootstrap_vs_K");
    for &k in &[2usize, 5, 10, 20] {
        let cfg = SimConfig {
            num_taxa: 64,
            num_sources: k,
            sink_depth: 20_000,
            source_depth: 20_000,
            ..Default::default()
        };
        let sc = generate(&cfg, 7);
        let prepared = Prepared::new(&sc.tree, &sc.sources, UnknownMode::None, 0.0);
        let src: Vec<Vec<f64>> = sc.sources.profiles.iter().map(|p| p.normalized()).collect();
        let point = prepared
            .solve(&GoodLpSolver, &src, &sc.sink.normalized())
            .unwrap()
            .weights;
        let bc = BootstrapConfig {
            replicates: 500,
            seed: 0,
            interval: IntervalMethod::Percentile,
            alpha: 0.05,
        };
        group.bench_with_input(BenchmarkId::from_parameter(k), &k, |bencher, _| {
            bencher.iter(|| {
                let res =
                    bootstrap::run(&GoodLpSolver, &prepared, &sc.sources, &sc.sink, &point, &bc);
                black_box(res.replicates_used)
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_replicates, bench_num_sources);
criterion_main!(benches);
