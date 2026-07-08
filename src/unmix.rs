//! Joint estimation of the unknown source *profile* and the mixing weights (spec §2.4 outer
//! loop, milestone M5).
//!
//! The v1 estimator ([`crate::estimate::point_estimate`]) can only model the unknown as a
//! *fixed* background profile `b_0` (uniform or metacommunity). When the sink has a large
//! genuine unknown fraction with a shape unlike any fixed guess — the regime where OTST scored
//! L1 ≈ 1.0 in the cross-method benchmark — that fixed profile is simply wrong and the fit
//! collapses. The methods that do well there (FEAST, SourceID-NMF, STENSL) all share one
//! ingredient: they **estimate the unknown profile jointly with the weights**.
//!
//! Jointly optimizing `w` and `b_0` is bilinear (`w_0 · b_0`) and non-convex, so — exactly as
//! CLAUDE.md §2.4/§7 require — it is kept OUT of the inner LP and handled by **block-coordinate
//! descent** in an outer loop:
//!
//! 1. **Weight step (convex LP, global optimum):** fix `b_0`; solve the (K+1)-source
//!    tree-Wasserstein LP with `Σ w = 1`. `w_0` is the current unknown fraction.
//! 2. **Profile step (FEAST-style soft responsibility, closed form, O(D·K)):** fix `w`; for each
//!    taxon `j`, split its observed sink mass `y_j` between "explained by the named sources"
//!    (`Σ_k w_k b_kj`) and "explained by the unknown" (`w_0 b_0j`) in proportion to those two
//!    contributions, and set the unknown profile to the normalized unknown share. This is FEAST's
//!    E-step. A *hard* residual `b_0 ∝ max(y − Σ_k w_k b_k, 0)` looks simpler but is **greedy and
//!    diverges**: it hands the unknown mass the named sources already explain, so `w_0` climbs
//!    monotonically toward the degenerate fixed point `w_0 = 1, b_0 = sink` (verified empirically
//!    — it overshot 0.5 → 0.75 on the SourceID-NMF data). The soft split does not double-count
//!    explained mass and converges to the true unknown fraction.
//! 3. **Sparsity reweight (optional):** update per-source penalties `μ_k = c/(w_k + ε)` and feed
//!    them into the next weight step. This is reweighted-ℓ1 (Candès–Wakin–Boyd 2008): it stays
//!    a linear program yet drives negligible source weights to zero — the source-selection
//!    ingredient STENSL gets from its exponential prior, which sharpens accuracy when many
//!    candidate sources are nuisances. Plain ℓ1 is degenerate under `Σ w = 1` (the penalty is
//!    constant), so reweighting is the correct form here.
//!
//! Each block is convex; the alternation converges in tens of iterations in practice. There is
//! no global-optimality guarantee for the joint bilinear problem — the same caveat applies to
//! every competitor — but the weight sub-problem is solved exactly.

use crate::baseline::project_to_simplex_pub;
use crate::estimate::{PointEstimate, Prepared, SourceEstimate};
use crate::lp::LpSolver;
use crate::profile::{Profile, SourceSet};
use crate::tree::Tree;
use crate::unknown::UNKNOWN_LABEL;

/// Which loss the inner weight sub-problem minimizes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightStep {
    /// Tree-Wasserstein LAD LP (OTST's phylogeny-aware loss). Use when an informative tree is
    /// available — this is where the drift-robustness lives.
    TreeWasserstein,
    /// Plain L2 (Euclidean) deconvolution over the simplex. On tree-LESS data the ground metric
    /// carries no signal (a star tree reduces TW to L1), and L2 fits compositional read data
    /// markedly better — it is what drives the unknown-estimator to best-in-class accuracy on
    /// no-tree benchmarks. Ignores the tree.
    L2,
}

/// Configuration for the alternating unknown-profile estimator.
#[derive(Debug, Clone)]
pub struct UnmixConfig {
    /// Maximum outer iterations.
    pub max_iters: usize,
    /// Convergence tolerance on `Σ_k |Δw_k|` between iterations.
    pub tol: f64,
    /// Reweighted-ℓ1 sparsity strength `c` on the *named* sources (0 disables source selection).
    /// The unknown source is never penalized (we want it to absorb residual mass freely).
    pub sparsity: f64,
    /// Stabilizer `ε` in the reweighting `μ_k = c / (w_k + ε)`.
    pub sparsity_eps: f64,
    /// Which loss the inner weight solve minimizes (tree-Wasserstein vs L2).
    pub weight_step: WeightStep,
    /// Emit per-iteration convergence progress to stderr.
    pub verbose: bool,
}

impl Default for UnmixConfig {
    fn default() -> Self {
        UnmixConfig {
            // The L2 weight-step alternation keeps improving for ~50-60 iterations before the
            // unknown fraction settles; the LP path converges faster. 60 covers both.
            max_iters: 60,
            tol: 1e-6,
            sparsity: 0.0,
            sparsity_eps: 1e-3,
            weight_step: WeightStep::TreeWasserstein,
            verbose: false,
        }
    }
}

/// Result of the alternating estimate: the point estimate plus the recovered unknown profile.
#[derive(Debug, Clone)]
pub struct UnmixResult {
    /// Named-source + unknown proportions and the tree-Wasserstein objective at the optimum.
    pub estimate: PointEstimate,
    /// The estimated unknown-source profile `b_0` over the tree's taxa (length D, sums to 1).
    pub unknown_profile: Vec<f64>,
    /// Outer iterations actually run.
    pub iters: usize,
}

/// Normalize a non-negative vector to sum 1, falling back to uniform if the mass is ~0.
fn normalize_or_uniform(v: &[f64]) -> Vec<f64> {
    let total: f64 = v.iter().sum();
    if total > 1e-12 {
        v.iter().map(|&x| x / total).collect()
    } else {
        let d = v.len().max(1);
        vec![1.0 / d as f64; v.len().max(d)]
    }
}

/// Hard residual unknown profile `b_0 ∝ max(sink − Σ_k w_k b_k, 0)` (FEAST `unknown_initialize_1`).
/// Used only to *initialize* `b_0` before the first weight solve; the iterative update uses the
/// soft responsibility rule below (the hard residual diverges if iterated — see module docs).
fn residual_profile_init(source_profiles: &[Vec<f64>], weights: &[f64], sink: &[f64]) -> Vec<f64> {
    let d = sink.len();
    let mut resid = vec![0.0f64; d];
    for j in 0..d {
        let mut explained = 0.0;
        for (k, prof) in source_profiles.iter().enumerate() {
            explained += weights[k] * prof[j];
        }
        resid[j] = (sink[j] - explained).max(0.0);
    }
    normalize_or_uniform(&resid)
}

/// FEAST-style soft profile update. For each taxon `j`, the observed mass `y_j` is split between
/// the named sources (contribution `Σ_k w_named_k · b_kj`) and the current unknown (contribution
/// `w_unknown · b0_j`) in proportion to those contributions; the unknown's new profile is the
/// normalized vector of its shares. This is the E-step responsibility of the unknown component
/// and, unlike the hard residual, does not re-grab mass the named sources already explain — so
/// the alternation converges to the true unknown fraction instead of running away to 1.
fn soft_unknown_profile(
    source_profiles: &[Vec<f64>],
    named_w: &[f64],
    unknown_w: f64,
    b0: &[f64],
    sink: &[f64],
) -> Vec<f64> {
    let d = sink.len();
    let mut share = vec![0.0f64; d];
    for j in 0..d {
        let mut explained = 0.0;
        for (k, prof) in source_profiles.iter().enumerate() {
            explained += named_w[k] * prof[j];
        }
        let unk = unknown_w * b0[j];
        let total = explained + unk;
        if total > 1e-12 {
            share[j] = sink[j] * unk / total;
        }
    }
    normalize_or_uniform(&share)
}

/// Alternating (block-coordinate) estimate of the mixing weights AND the unknown-source profile
/// under the tree-Wasserstein loss. `sources` are the K named sources; the unknown is an implicit
/// (K+1)-th source whose profile is estimated. Returns weights over `[named…, Unknown]`.
///
/// The inner weight solve reuses [`Prepared::with_background`] each iteration, so the fixed
/// tree/edge setup is computed once per call (via the borrow of `tree`).
pub fn alternating_estimate<S: LpSolver>(
    solver: &S,
    tree: &Tree,
    sources: &SourceSet,
    sink: &Profile,
    cfg: &UnmixConfig,
) -> Result<UnmixResult, crate::lp::LpError> {
    let source_profiles: Vec<Vec<f64>> =
        sources.profiles.iter().map(|p| p.normalized()).collect();
    let sink_norm = sink.normalized();
    let num_named = sources.num_sources();

    // Initialize the unknown profile from the residual against an equal-weight named mixture
    // (a neutral start; the first weight solve immediately refines it).
    let init_w = vec![1.0 / num_named.max(1) as f64; num_named];
    let mut b0 = residual_profile_init(&source_profiles, &init_w, &sink_norm);

    let mut prev_w: Option<Vec<f64>> = None;
    let mut last: Option<(Vec<f64>, f64, f64)> = None; // (weights, objective, deficit)
    let mut iters = 0;

    for _ in 0..cfg.max_iters.max(1) {
        iters += 1;

        // --- Weight step: fix b_0, solve the (K+1)-source weight problem. ---
        // Reweighted-ℓ1 penalty on named sources only (unknown column = 0). Computed from the
        // previous iterate; uniform on the first pass.
        let penalty: Vec<f64> = if cfg.sparsity > 0.0 {
            let mut mu = vec![0.0f64; num_named + 1];
            match &prev_w {
                Some(w) => {
                    for k in 0..num_named {
                        mu[k] = cfg.sparsity / (w[k] + cfg.sparsity_eps);
                    }
                }
                None => {
                    let u = 1.0 / num_named.max(1) as f64;
                    for muk in mu.iter_mut().take(num_named) {
                        *muk = cfg.sparsity / (u + cfg.sparsity_eps);
                    }
                }
            }
            mu
        } else {
            Vec::new()
        };

        let (w, objective, deficit) = match cfg.weight_step {
            WeightStep::TreeWasserstein => {
                let mut prepared = Prepared::with_background(tree, sources, b0.clone());
                prepared.weight_penalty = penalty;
                let sol = prepared.solve(solver, &source_profiles, &sink_norm)?;
                (sol.weights, sol.objective, sol.deficit)
            }
            WeightStep::L2 => {
                // Columns = named sources + the current unknown profile b_0.
                let mut cols: Vec<Vec<f64>> = source_profiles.clone();
                cols.push(b0.clone());
                let w = l2_deconvolve_penalized(&cols, &sink_norm, &penalty, prev_w.as_deref());
                // Report an L1-over-taxa distance as the objective (star-tree TW = L1); no
                // unexplained deficit under the simplex-constrained L2 fit.
                let obj = l1_taxa_distance(&cols, &w, &sink_norm);
                (w, obj, 0.0)
            }
        };

        // --- Profile step: fix w, re-estimate b_0 via FEAST-style soft responsibility. ---
        let named_w = &w[..num_named];
        let unknown_w = w[num_named];
        b0 = soft_unknown_profile(&source_profiles, named_w, unknown_w, &b0, &sink_norm);

        // --- Convergence on the full weight vector. ---
        let delta = match &prev_w {
            Some(pw) => pw.iter().zip(w.iter()).map(|(a, b)| (a - b).abs()).sum::<f64>(),
            None => f64::INFINITY,
        };
        if cfg.verbose {
            let unk = w.last().copied().unwrap_or(0.0);
            eprintln!("  [unmix] iter {iters}: unknown={unk:.4} Δw={delta:.2e}");
        }
        prev_w = Some(w.clone());
        last = Some((w, objective, deficit));
        if delta < cfg.tol {
            break;
        }
    }

    let (weights, objective, deficit) = last.expect("at least one iteration runs");
    let names: Vec<String> = sources
        .names
        .iter()
        .cloned()
        .chain(std::iter::once(UNKNOWN_LABEL.to_string()))
        .collect();
    let sources_out: Vec<SourceEstimate> = names
        .iter()
        .zip(weights.iter())
        .map(|(name, &w)| SourceEstimate {
            name: name.clone(),
            proportion: w,
        })
        .collect();

    Ok(UnmixResult {
        estimate: PointEstimate {
            sources: sources_out,
            objective,
            deficit,
        },
        unknown_profile: b0,
        iters,
    })
}

/// L2 deconvolution with an optional per-source linear penalty (reweighted-ℓ1) and an optional
/// warm start. Minimizes `‖Bw − p‖² + Σ_k μ_k w_k` over the simplex (convex, projected-gradient).
/// Warm-starting from the previous outer iterate is what keeps the alternating loop affordable:
/// the unknown profile changes only slightly each step, so the weight solve converges in a
/// handful of inner iterations after the first. Kept local to the unmixer since the penalty is
/// only meaningful inside the alternating source-selection loop.
fn l2_deconvolve_penalized(
    sources: &[Vec<f64>],
    sink: &[f64],
    penalty: &[f64],
    warm_start: Option<&[f64]>,
) -> Vec<f64> {
    // Fold the linear penalty into the gradient: ∇ = 2Bᵀ(Bw − p) + μ. Same projected-gradient
    // structure as `l2_deconvolve`; an empty penalty makes this exactly the plain L2 objective.
    let k = sources.len();
    if k == 0 {
        return Vec::new();
    }
    let d = sink.len();
    let frob_sq: f64 = sources.iter().flat_map(|s| s.iter()).map(|&x| x * x).sum();
    let step = 1.0 / (2.0 * frob_sq).max(1e-12);
    let mut w = match warm_start {
        Some(prev) if prev.len() == k => prev.to_vec(),
        _ => vec![1.0 / k as f64; k],
    };
    for _ in 0..5000 {
        let mut r = vec![0.0f64; d];
        for (kj, col) in sources.iter().enumerate() {
            let wk = w[kj];
            for i in 0..d {
                r[i] += wk * col[i];
            }
        }
        for i in 0..d {
            r[i] -= sink[i];
        }
        let mut w_new = w.clone();
        for (kj, col) in sources.iter().enumerate() {
            let mut acc = 0.0;
            for i in 0..d {
                acc += col[i] * r[i];
            }
            let g = 2.0 * acc + penalty.get(kj).copied().unwrap_or(0.0);
            w_new[kj] = w[kj] - step * g;
        }
        project_to_simplex_pub(&mut w_new);
        let delta: f64 = w.iter().zip(w_new.iter()).map(|(a, b)| (a - b).abs()).sum();
        w = w_new;
        if delta < 1e-10 {
            break;
        }
    }
    w
}

/// L1 distance over taxa between the sink and the reconstructed mixture `Σ_k w_k · col_k`.
fn l1_taxa_distance(cols: &[Vec<f64>], weights: &[f64], sink: &[f64]) -> f64 {
    let d = sink.len();
    let mut acc = 0.0;
    for j in 0..d {
        let mut mix = 0.0;
        for (k, col) in cols.iter().enumerate() {
            mix += weights[k] * col[j];
        }
        acc += (sink[j] - mix).abs();
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lp::GoodLpSolver;
    use approx::assert_abs_diff_eq;

    fn balanced_tree() -> Tree {
        // 8 leaves so a "novel" unknown clade is distinguishable from the named sources.
        Tree::parse_newick("(((A:1,B:1):1,(C:1,D:1):1):1,((E:1,F:1):1,(G:1,H:1):1):1);").unwrap()
    }

    fn unknown_weight(res: &UnmixResult) -> f64 {
        res.estimate
            .sources
            .iter()
            .find(|s| s.name == UNKNOWN_LABEL)
            .unwrap()
            .proportion
    }

    #[test]
    fn recovers_large_unknown_fraction() {
        // Two named sources live on {A,B} and {C,D}. The sink is 0.3 s1 + 0.2 s2 + 0.5 of an
        // unknown living on {E,F,G,H} (a clade neither source touches). v1 with a fixed uniform
        // background cannot express this; the alternating estimator should recover ~0.5 unknown.
        let tree = balanced_tree();
        let sources = SourceSet {
            names: vec!["s1".into(), "s2".into()],
            profiles: vec![
                Profile::from_counts(vec![50.0, 50.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                Profile::from_counts(vec![0.0, 0.0, 50.0, 50.0, 0.0, 0.0, 0.0, 0.0]),
            ],
        };
        // sink composition: 0.3*[A,B] + 0.2*[C,D] + 0.5*uniform over [E..H]
        let sink = Profile::from_counts(vec![
            150.0, 150.0, // 0.3 on A,B
            100.0, 100.0, // 0.2 on C,D
            125.0, 125.0, 125.0, 125.0, // 0.5 on E,F,G,H
        ]);

        let res = alternating_estimate(
            &GoodLpSolver,
            &tree,
            &sources,
            &sink,
            &UnmixConfig::default(),
        )
        .unwrap();

        let get = |n: &str| {
            res.estimate
                .sources
                .iter()
                .find(|s| s.name == n)
                .unwrap()
                .proportion
        };
        // Unknown fraction should be close to the true 0.5.
        assert!(
            (unknown_weight(&res) - 0.5).abs() < 0.1,
            "unknown {} should be ~0.5",
            unknown_weight(&res)
        );
        // Named weights should be near their true values.
        assert!((get("s1") - 0.3).abs() < 0.1, "s1 {} ~0.3", get("s1"));
        assert!((get("s2") - 0.2).abs() < 0.1, "s2 {} ~0.2", get("s2"));
        // Recovered unknown profile should concentrate on E,F,G,H (indices 4..8).
        let tail: f64 = res.unknown_profile[4..8].iter().sum();
        assert!(tail > 0.8, "unknown profile should sit on the novel clade, tail={tail}");
    }

    #[test]
    fn no_unknown_matches_plain_mixture() {
        // Sink is an exact mixture of the two named sources; the unknown weight should be ~0 and
        // the named weights recovered.
        let tree = balanced_tree();
        let sources = SourceSet {
            names: vec!["s1".into(), "s2".into()],
            profiles: vec![
                Profile::from_counts(vec![50.0, 50.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                Profile::from_counts(vec![0.0, 0.0, 50.0, 50.0, 0.0, 0.0, 0.0, 0.0]),
            ],
        };
        // 0.7 s1 + 0.3 s2, no unknown
        let sink = Profile::from_counts(vec![350.0, 350.0, 150.0, 150.0, 0.0, 0.0, 0.0, 0.0]);
        let res = alternating_estimate(
            &GoodLpSolver,
            &tree,
            &sources,
            &sink,
            &UnmixConfig::default(),
        )
        .unwrap();
        assert!(unknown_weight(&res) < 0.05, "unknown {} should be ~0", unknown_weight(&res));
        let total: f64 = res.estimate.sources.iter().map(|s| s.proportion).sum();
        assert_abs_diff_eq!(total, 1.0, epsilon = 1e-5);
    }

    #[test]
    fn sparsity_zeroes_nuisance_source() {
        // Three named sources; only s1 and s2 actually contribute. A nuisance source s3 sits on
        // the same clade as s1 (partially collinear). With sparsity on, s3's weight should be
        // driven toward zero.
        let tree = balanced_tree();
        let sources = SourceSet {
            names: vec!["s1".into(), "s2".into(), "nuisance".into()],
            profiles: vec![
                Profile::from_counts(vec![50.0, 50.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                Profile::from_counts(vec![0.0, 0.0, 50.0, 50.0, 0.0, 0.0, 0.0, 0.0]),
                Profile::from_counts(vec![60.0, 40.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            ],
        };
        // sink = 0.6 s1 + 0.4 s2 exactly (nuisance not used).
        let sink = Profile::from_counts(vec![300.0, 300.0, 200.0, 200.0, 0.0, 0.0, 0.0, 0.0]);
        let cfg = UnmixConfig {
            sparsity: 0.02,
            ..Default::default()
        };
        let res = alternating_estimate(&GoodLpSolver, &tree, &sources, &sink, &cfg).unwrap();
        let nuisance = res
            .estimate
            .sources
            .iter()
            .find(|s| s.name == "nuisance")
            .unwrap()
            .proportion;
        assert!(nuisance < 0.15, "nuisance weight {nuisance} should be suppressed by sparsity");
    }
}
