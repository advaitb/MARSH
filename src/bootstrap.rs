//! Multinomial bootstrap for uncertainty (spec §2.5).
//!
//! For each of `B` replicates, independently:
//! 1. Resample the sink: `y* ~ Multinomial(n, p)`, `p* = y*/n`.
//! 2. **Resample every source too:** `counts_k* ~ Multinomial(m_k, b_k)`, `b_k* = counts_k*/m_k`.
//!    Skipping this ignores source-profile noise and the intervals under-cover when sources
//!    are shallow — this is the single most important correctness requirement in the tool.
//! 3. Recompute cumulative masses and re-solve the LP → `w*(b)`.
//!
//! The fixed background source `b_0` (v1 unknown modes) is a *fixed profile*, not observed
//! data, so it is deliberately **not** resampled — it stays constant across replicates, which
//! keeps the per-replicate problem convex and matches the design boundary in §2.4.
//!
//! Replicates are embarrassingly parallel (rayon). Each replicate derives its own RNG from a
//! base seed + replicate index (splitmix64-mixed), so runs are fully reproducible and there is
//! no shared mutable RNG state across threads.

use crate::estimate::Prepared;
use crate::lp::LpSolver;
use crate::profile::{Profile, SourceSet};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Binomial, Distribution};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Interval-construction method.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum IntervalMethod {
    /// Plain percentile interval (M2). Under-covers near the simplex boundary — such
    /// intervals are flagged `one_sided`.
    Percentile,
    /// m-out-of-n bootstrap (M4).
    MOutOfN,
    /// Bias-corrected and accelerated (M4).
    Bca,
}

/// Bootstrap configuration.
#[derive(Debug, Clone)]
pub struct BootstrapConfig {
    /// Number of replicates `B`. 0 disables the bootstrap (point estimate only).
    pub replicates: usize,
    /// Base RNG seed (recorded in output for reproducibility).
    pub seed: u64,
    /// Interval method.
    pub interval: IntervalMethod,
    /// Two-sided interval miss rate (0.05 → 95% interval).
    pub alpha: f64,
}

impl Default for BootstrapConfig {
    fn default() -> Self {
        BootstrapConfig {
            replicates: 500,
            seed: 0,
            interval: IntervalMethod::Percentile,
            alpha: 0.05,
        }
    }
}

/// Per-source uncertainty summary.
#[derive(Debug, Clone)]
pub struct SourceInterval {
    pub name: String,
    /// Point estimate from the original (un-resampled) data.
    pub point: f64,
    /// Mean of the bootstrap distribution.
    pub mean: f64,
    pub ci_low: f64,
    pub ci_high: f64,
    /// True when the lower tail piles up at 0 (near-boundary); the interval should be read as
    /// one-sided `[0, ci_high]` since the percentile lower bound under-covers there.
    pub one_sided: bool,
}

/// Full bootstrap result.
#[derive(Debug, Clone)]
pub struct BootstrapResult {
    pub sources: Vec<SourceInterval>,
    /// Number of replicates that solved successfully (≤ `replicates`).
    pub replicates_used: usize,
    /// The base seed actually used.
    pub seed: u64,
}

/// SplitMix64 — good dispersion from a counter, used to derive independent per-replicate seeds
/// from `(base_seed, replicate_index)`.
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// Draw a `Multinomial(n, probs)` sample via the sequential-conditional-binomial method
/// (`rand_distr` has no `Multinomial`). Returns per-category counts (as f64) summing to `n`.
fn multinomial<R: rand::Rng>(rng: &mut R, n: u64, probs: &[f64]) -> Vec<f64> {
    let mut counts = vec![0.0f64; probs.len()];
    if n == 0 || probs.is_empty() {
        return counts;
    }
    let mut remaining_n = n;
    let mut remaining_p: f64 = probs.iter().sum();
    let last = probs.len() - 1;
    for i in 0..probs.len() {
        if remaining_n == 0 {
            break;
        }
        if i == last {
            counts[i] = remaining_n as f64;
            break;
        }
        // Guard against tiny/negative remaining_p from floating error.
        let p_i = if remaining_p > 0.0 {
            (probs[i] / remaining_p).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let draw = Binomial::new(remaining_n, p_i)
            .expect("binomial params valid")
            .sample(rng);
        counts[i] = draw as f64;
        remaining_n -= draw;
        remaining_p -= probs[i];
    }
    counts
}

/// Resample a profile: draw `Multinomial(depth, normalized_profile)` and renormalize. Depth is
/// rounded to the nearest integer. A zero-depth profile resamples to itself (uniform).
fn resample_profile<R: rand::Rng>(rng: &mut R, profile: &Profile) -> Vec<f64> {
    let n = profile.depth.round() as u64;
    let probs = profile.normalized();
    if n == 0 {
        return probs;
    }
    let counts = multinomial(rng, n, &probs);
    let total: f64 = counts.iter().sum();
    if total > 0.0 {
        counts.iter().map(|&c| c / total).collect()
    } else {
        probs
    }
}

/// Run the multinomial bootstrap. `prepared` holds the fixed tree/edges/background; `sources`
/// and `sink` supply the raw counts + depths that get resampled. `point_weights` are the LP
/// weights from the original data (length = `prepared.num_lp_sources()`), used as the reported
/// point estimate and to align source names.
pub fn run<S>(
    solver: &S,
    prepared: &Prepared,
    sources: &SourceSet,
    sink: &Profile,
    point_weights: &[f64],
    config: &BootstrapConfig,
) -> BootstrapResult
where
    S: LpSolver + Sync,
{
    let k_lp = prepared.num_lp_sources();
    assert_eq!(point_weights.len(), k_lp, "point_weights length must equal LP source count");

    if config.replicates == 0 {
        // No bootstrap: intervals collapse to the point estimate.
        let sources_out = prepared
            .source_names
            .iter()
            .zip(point_weights.iter())
            .map(|(name, &w)| SourceInterval {
                name: name.clone(),
                point: w,
                mean: w,
                ci_low: w,
                ci_high: w,
                one_sided: false,
            })
            .collect();
        return BootstrapResult {
            sources: sources_out,
            replicates_used: 0,
            seed: config.seed,
        };
    }

    // Parallel replicates. Each derives its own RNG → deterministic and thread-safe.
    let base = config.seed;
    let replicate_weights: Vec<Vec<f64>> = (0..config.replicates)
        .into_par_iter()
        .filter_map(|b| {
            let mut rng = ChaCha8Rng::seed_from_u64(splitmix64(base.wrapping_add(b as u64)));

            // 1) resample sink
            let sink_star = resample_profile(&mut rng, sink);
            // 2) resample every source
            let src_star: Vec<Vec<f64>> = sources
                .profiles
                .iter()
                .map(|p| resample_profile(&mut rng, p))
                .collect();

            // 3) re-solve
            prepared.solve(solver, &src_star, &sink_star).ok().map(|s| s.weights)
        })
        .collect();

    let replicates_used = replicate_weights.len();

    // Summarize per LP source.
    let mut sources_out = Vec::with_capacity(k_lp);
    for j in 0..k_lp {
        let mut col: Vec<f64> = replicate_weights.iter().map(|w| w[j]).collect();
        let (mean, ci_low, ci_high, one_sided) = if col.is_empty() {
            let p = point_weights[j];
            (p, p, p, false)
        } else {
            let mean = col.iter().sum::<f64>() / col.len() as f64;
            col.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let lo_q = config.alpha / 2.0;
            let hi_q = 1.0 - config.alpha / 2.0;
            let mut ci_low = percentile(&col, lo_q);
            let ci_high = percentile(&col, hi_q);

            // Boundary handling: if more than the lower tail mass is piled at ~0, the lower
            // percentile is unreliable — report a one-sided [0, ci_high] interval.
            let frac_zero = col.iter().filter(|&&v| v < 1e-6).count() as f64 / col.len() as f64;
            let one_sided = frac_zero > lo_q;
            if one_sided {
                ci_low = 0.0;
            }
            (mean, ci_low, ci_high, one_sided)
        };

        sources_out.push(SourceInterval {
            name: prepared.source_names[j].clone(),
            point: point_weights[j],
            mean,
            ci_low,
            ci_high,
            one_sided,
        });
    }

    BootstrapResult {
        sources: sources_out,
        replicates_used,
        seed: base,
    }
}

/// Type-7 (linear-interpolation) percentile of a sorted slice. `q` in [0, 1].
fn percentile(sorted: &[f64], q: f64) -> f64 {
    match sorted.len() {
        0 => f64::NAN,
        1 => sorted[0],
        n => {
            let h = (n as f64 - 1.0) * q.clamp(0.0, 1.0);
            let lo = h.floor() as usize;
            let hi = h.ceil() as usize;
            sorted[lo] + (h - lo as f64) * (sorted[hi] - sorted[lo])
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::estimate::Prepared;
    use crate::lp::GoodLpSolver;
    use crate::tree::Tree;
    use approx::assert_abs_diff_eq;

    fn balanced_tree() -> Tree {
        Tree::parse_newick("((A:1,B:1):1,(C:1,D:1):1);").unwrap()
    }

    #[test]
    fn multinomial_sums_to_n() {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let s = multinomial(&mut rng, 1000, &[0.1, 0.2, 0.3, 0.4]);
        assert_abs_diff_eq!(s.iter().sum::<f64>(), 1000.0, epsilon = 1e-9);
    }

    #[test]
    fn multinomial_is_reproducible() {
        let mut r1 = ChaCha8Rng::seed_from_u64(7);
        let mut r2 = ChaCha8Rng::seed_from_u64(7);
        let a = multinomial(&mut r1, 500, &[0.25, 0.25, 0.25, 0.25]);
        let b = multinomial(&mut r2, 500, &[0.25, 0.25, 0.25, 0.25]);
        assert_eq!(a, b);
    }

    #[test]
    fn multinomial_marginals_are_reasonable() {
        // Empirical mean of category counts should approach n * p over many draws.
        let mut rng = ChaCha8Rng::seed_from_u64(99);
        let probs = [0.1, 0.2, 0.3, 0.4];
        let n = 1000u64;
        let draws = 400;
        let mut acc = [0.0f64; 4];
        for _ in 0..draws {
            let s = multinomial(&mut rng, n, &probs);
            for i in 0..4 {
                acc[i] += s[i];
            }
        }
        for i in 0..4 {
            let empirical = acc[i] / draws as f64;
            let expected = n as f64 * probs[i];
            // within 5% relative
            assert!(
                (empirical - expected).abs() < 0.05 * expected + 5.0,
                "category {i}: empirical {empirical}, expected {expected}"
            );
        }
    }

    #[test]
    fn zero_replicates_reproduces_point_estimate() {
        let tree = balanced_tree();
        let sources = SourceSet {
            names: vec!["s1".into(), "s2".into()],
            profiles: vec![
                Profile::from_counts(vec![50.0, 50.0, 0.0, 0.0]),
                Profile::from_counts(vec![0.0, 0.0, 50.0, 50.0]),
            ],
        };
        let sink = Profile::from_counts(vec![350.0, 350.0, 150.0, 150.0]);
        let prepared = Prepared::new(&tree, &sources);
        let src_norm: Vec<Vec<f64>> = sources.profiles.iter().map(|p| p.normalized()).collect();
        let point = prepared
            .solve(&GoodLpSolver, &src_norm, &sink.normalized())
            .unwrap()
            .weights;

        let cfg = BootstrapConfig {
            replicates: 0,
            ..Default::default()
        };
        let res = run(&GoodLpSolver, &prepared, &sources, &sink, &point, &cfg);
        assert_eq!(res.replicates_used, 0);
        // intervals collapse to the point estimate
        for si in &res.sources {
            assert_abs_diff_eq!(si.ci_low, si.point, epsilon = 1e-12);
            assert_abs_diff_eq!(si.ci_high, si.point, epsilon = 1e-12);
        }
    }

    #[test]
    fn bootstrap_interval_brackets_point_and_is_reproducible() {
        let tree = balanced_tree();
        let sources = SourceSet {
            names: vec!["s1".into(), "s2".into()],
            profiles: vec![
                Profile::from_counts(vec![80.0, 20.0, 0.0, 0.0]),
                Profile::from_counts(vec![0.0, 0.0, 20.0, 80.0]),
            ],
        };
        // sink = 0.6 s1 + 0.4 s2 at depth 2000
        let sink = Profile::from_counts(vec![960.0, 240.0, 160.0, 640.0]);
        let prepared = Prepared::new(&tree, &sources);
        let src_norm: Vec<Vec<f64>> = sources.profiles.iter().map(|p| p.normalized()).collect();
        let point = prepared
            .solve(&GoodLpSolver, &src_norm, &sink.normalized())
            .unwrap()
            .weights;

        let cfg = BootstrapConfig {
            replicates: 200,
            seed: 12345,
            interval: IntervalMethod::Percentile,
            alpha: 0.05,
        };
        let res1 = run(&GoodLpSolver, &prepared, &sources, &sink, &point, &cfg);
        let res2 = run(&GoodLpSolver, &prepared, &sources, &sink, &point, &cfg);

        assert_eq!(res1.replicates_used, 200);
        // reproducible across runs with the same seed
        for (a, b) in res1.sources.iter().zip(res2.sources.iter()) {
            assert_abs_diff_eq!(a.ci_low, b.ci_low, epsilon = 1e-12);
            assert_abs_diff_eq!(a.ci_high, b.ci_high, epsilon = 1e-12);
            assert_abs_diff_eq!(a.mean, b.mean, epsilon = 1e-12);
        }
        // the 95% interval should bracket the point estimate (with a little tolerance)
        for si in &res1.sources {
            assert!(
                si.ci_low - 1e-6 <= si.point && si.point <= si.ci_high + 1e-6,
                "interval [{}, {}] should bracket point {}",
                si.ci_low,
                si.ci_high,
                si.point
            );
            // and the mean should be near the point estimate (0.6/0.4)
            assert!((si.mean - si.point).abs() < 0.1);
        }
    }

    #[test]
    fn percentile_interpolates() {
        let v = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        assert_abs_diff_eq!(percentile(&v, 0.0), 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(percentile(&v, 1.0), 4.0, epsilon = 1e-12);
        assert_abs_diff_eq!(percentile(&v, 0.5), 2.0, epsilon = 1e-12);
        assert_abs_diff_eq!(percentile(&v, 0.25), 1.0, epsilon = 1e-12);
    }
}
