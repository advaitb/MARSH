//! The four validation experiments (spec §6.3), run as integration tests.
//!
//! 1. Drift sweep (headline): as phylogenetic drift increases, the tree-Wasserstein estimator's
//!    error stays ~flat while the non-phylogenetic L2 baseline degrades.
//! 2. Depth sweep: with zero drift, error decreases as depth grows (~1/sqrt(n) + 1/sqrt(m)).
//! 3. Coverage: bootstrap 95% intervals contain the truth ≈95% of the time (matched process).
//! 4. Speed: the B-replicate run is fast (asserted loosely; criterion bench lives in benches/).
//!
//! These use averages over multiple seeds to be robust to sampling noise while staying fast
//! enough for CI. They assert the *directional* claims of the whitepaper, not exact numbers.

use otst::baseline::l2_deconvolve;
use otst::bootstrap::{self, BootstrapConfig, IntervalMethod};
use otst::estimate::Prepared;
use otst::lp::GoodLpSolver;
use otst::sim::{generate, l1_error, DriftModel, SimConfig};

/// Solve one scenario with the tree-Wasserstein OT estimator (v1, no unknown source), returning
/// the named-source weights.
fn solve_ot(scenario: &otst::sim::Scenario) -> Vec<f64> {
    let prepared = Prepared::new(&scenario.tree, &scenario.sources);
    let src_norm: Vec<Vec<f64>> = scenario
        .sources
        .profiles
        .iter()
        .map(|p| p.normalized())
        .collect();
    let sol = prepared
        .solve(&GoodLpSolver, &src_norm, &scenario.sink.normalized())
        .expect("OT solve");
    // No unknown source, so all LP weights are named sources.
    sol.weights
}

/// Solve one scenario with the non-phylogenetic L2 baseline.
fn solve_l2(scenario: &otst::sim::Scenario) -> Vec<f64> {
    let src_norm: Vec<Vec<f64>> = scenario
        .sources
        .profiles
        .iter()
        .map(|p| p.normalized())
        .collect();
    l2_deconvolve(&src_norm, &scenario.sink.normalized())
}

/// Mean L1 error of a method over `n_seeds` scenarios at a given config.
fn mean_error(config: &SimConfig, n_seeds: u64, method: impl Fn(&otst::sim::Scenario) -> Vec<f64>) -> f64 {
    let mut total = 0.0;
    for seed in 0..n_seeds {
        let sc = generate(config, seed * 100 + 1);
        let est = method(&sc);
        total += l1_error(&est, &sc.true_weights);
    }
    total / n_seeds as f64
}

// ---------------------------------------------------------------------------
// Experiment 1 — Drift sweep (headline claim)
// ---------------------------------------------------------------------------
#[test]
fn experiment_drift_sweep_ot_beats_l2() {
    let base = SimConfig {
        num_taxa: 48,
        num_sources: 4,
        dirichlet_alpha: 0.3,
        unknown_fraction: 0.0,
        drift: 0.0,
        drift_model: DriftModel::CherrySwap,
        sink_depth: 20_000,
        source_depth: 20_000,
    };
    let n_seeds = 12;
    let drift_levels = [0.0, 0.1, 0.2, 0.3, 0.4];

    let mut ot_errors = Vec::new();
    let mut l2_errors = Vec::new();
    for &drift in &drift_levels {
        let cfg = SimConfig { drift, ..base.clone() };
        let ot = mean_error(&cfg, n_seeds, solve_ot);
        let l2 = mean_error(&cfg, n_seeds, solve_l2);
        ot_errors.push(ot);
        l2_errors.push(l2);
        println!("drift={drift:.2}  OT_err={ot:.4}  L2_err={l2:.4}");
    }

    // Claim A: at the highest drift, OT is clearly better than L2.
    let last = drift_levels.len() - 1;
    assert!(
        ot_errors[last] < l2_errors[last],
        "at drift {}, OT error {:.4} should beat L2 {:.4}",
        drift_levels[last],
        ot_errors[last],
        l2_errors[last]
    );

    // Claim B: OT degrades more slowly than L2 as drift grows (its error rises less).
    let ot_growth = ot_errors[last] - ot_errors[0];
    let l2_growth = l2_errors[last] - l2_errors[0];
    assert!(
        ot_growth < l2_growth,
        "OT error growth under drift ({ot_growth:.4}) should be smaller than L2's ({l2_growth:.4})"
    );
}

// ---------------------------------------------------------------------------
// Experiment 2 — Depth sweep (convergence with sequencing depth)
// ---------------------------------------------------------------------------
#[test]
fn experiment_depth_sweep_converges() {
    let base = SimConfig {
        num_taxa: 32,
        num_sources: 3,
        dirichlet_alpha: 0.4,
        unknown_fraction: 0.0,
        drift: 0.0,
        drift_model: DriftModel::CherrySwap,
        sink_depth: 0,
        source_depth: 0,
    };
    let n_seeds = 12;
    let depths = [500u64, 2_000, 8_000, 32_000];
    let mut errs = Vec::new();
    for &depth in &depths {
        let cfg = SimConfig {
            sink_depth: depth,
            source_depth: depth,
            ..base.clone()
        };
        let e = mean_error(&cfg, n_seeds, solve_ot);
        errs.push(e);
        println!("depth={depth}  OT_err={e:.4}");
    }
    // Monotone-ish decrease: deepest error should be well below shallowest.
    assert!(
        errs[errs.len() - 1] < errs[0] * 0.6,
        "error at deepest ({:.4}) should be < 60% of shallowest ({:.4})",
        errs[errs.len() - 1],
        errs[0]
    );
    // And the deepest depth should give a small absolute error.
    assert!(
        *errs.last().unwrap() < 0.15,
        "deep-sequencing error {:.4} should be small",
        errs.last().unwrap()
    );
}

// ---------------------------------------------------------------------------
// Experiment 3 — Coverage of bootstrap 95% intervals (matched process)
// ---------------------------------------------------------------------------
#[test]
fn experiment_coverage_is_near_nominal() {
    let cfg = SimConfig {
        num_taxa: 24,
        num_sources: 3,
        dirichlet_alpha: 0.5,
        unknown_fraction: 0.0,
        drift: 0.0,
        drift_model: DriftModel::CherrySwap,
        sink_depth: 5_000,
        source_depth: 5_000,
    };
    let n_scenarios = 60u64;
    let mut covered = 0usize;
    let mut total = 0usize;

    for seed in 0..n_scenarios {
        let sc = generate(&cfg, seed * 7 + 3);
        let prepared = Prepared::new(&sc.tree, &sc.sources);
        let src_norm: Vec<Vec<f64>> = sc.sources.profiles.iter().map(|p| p.normalized()).collect();
        let point = prepared
            .solve(&GoodLpSolver, &src_norm, &sc.sink.normalized())
            .unwrap()
            .weights;

        let boot_cfg = BootstrapConfig {
            replicates: 200,
            seed: 1_000 + seed,
            interval: IntervalMethod::Percentile,
            alpha: 0.05,
        };
        let res = bootstrap::run(&GoodLpSolver, &prepared, &sc.sources, &sc.sink, &point, &boot_cfg);

        for (j, si) in res.sources.iter().enumerate() {
            let truth = sc.true_weights[j];
            total += 1;
            // one-sided intervals: count as covered if truth <= ci_high (lower bound is 0)
            let lo = if si.one_sided { 0.0 } else { si.ci_low };
            if truth >= lo - 1e-9 && truth <= si.ci_high + 1e-9 {
                covered += 1;
            }
        }
    }

    let coverage = covered as f64 / total as f64;
    println!("coverage = {coverage:.3} over {total} source-intervals");
    // Nominal 95%. Percentile intervals are known to under-cover somewhat, so accept a band;
    // this asserts we are in the right ballpark, not perfectly calibrated (M4 improves this).
    assert!(
        coverage > 0.80,
        "coverage {coverage:.3} should be reasonably close to nominal 0.95"
    );
}

// ---------------------------------------------------------------------------
// Experiment 4 — Speed (loose smoke assertion; criterion bench in benches/)
// ---------------------------------------------------------------------------
#[test]
fn experiment_speed_bootstrap_is_fast() {
    let cfg = SimConfig {
        num_taxa: 64,
        num_sources: 5,
        sink_depth: 20_000,
        source_depth: 20_000,
        ..Default::default()
    };
    let sc = generate(&cfg, 42);
    let prepared = Prepared::new(&sc.tree, &sc.sources);
    let src_norm: Vec<Vec<f64>> = sc.sources.profiles.iter().map(|p| p.normalized()).collect();
    let point = prepared
        .solve(&GoodLpSolver, &src_norm, &sc.sink.normalized())
        .unwrap()
        .weights;

    let boot_cfg = BootstrapConfig {
        replicates: 500,
        seed: 0,
        interval: IntervalMethod::Percentile,
        alpha: 0.05,
    };
    let start = std::time::Instant::now();
    let res = bootstrap::run(&GoodLpSolver, &prepared, &sc.sources, &sc.sink, &point, &boot_cfg);
    let elapsed = start.elapsed();
    println!(
        "500 replicates on D={} K={} in {:?} ({} used)",
        cfg.num_taxa, cfg.num_sources, elapsed, res.replicates_used
    );
    assert_eq!(res.replicates_used, 500);
    // Generous ceiling so this never flakes in CI, while still catching pathological slowdowns.
    assert!(
        elapsed.as_secs() < 30,
        "500 replicates took {elapsed:?}, expected well under 30s"
    );
}
