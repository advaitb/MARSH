//! The tree-Wasserstein least-absolute-deviations (LAD) point-estimate LP.
//!
//! Given per-edge weights `ℓ_e`, sink cumulative masses `s_e`, and a source cumulative-mass
//! matrix `M` (E×K), find mixing weights `w` on the simplex minimizing the tree-Wasserstein-1
//! (weighted UniFrac) distance between the sink and the mixture:
//!
//! ```text
//! minimize_w   Σ_e ℓ_e · | s_e − Σ_k M_ek w_k |
//! s.t.         w_k ≥ 0,   Σ_k w_k = 1
//! ```
//!
//! Linearized with per-edge slacks `t_e ≥ 0`:
//!
//! ```text
//! minimize   Σ_e t_e
//! s.t.       t_e ≥  ℓ_e (s_e − Σ_k M_ek w_k)
//!            t_e ≥ −ℓ_e (s_e − Σ_k M_ek w_k)
//!            w_k ≥ 0,   Σ_k w_k = 1        (v1: SumToOne)
//!                       Σ_k w_k ≤ 1        (v2: AtMostOne, unbalanced)
//! ```
//!
//! This is a convex LP → global optimum, the headline contrast with EM/NMF methods. The
//! backend is kept behind [`LpSolver`] so it can be swapped without touching callers.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum LpError {
    #[error("LP is infeasible or unbounded")]
    NoSolution,
    #[error("dimension mismatch: {0}")]
    Dim(String),
    #[error("solver error: {0}")]
    Solver(String),
}

/// The simplex constraint mode for the mixing weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightConstraint {
    /// `Σ_k w_k = 1` (v1: all sink mass is explained, possibly by a fixed background source).
    SumToOne,
    /// `Σ_k w_k ≤ 1` (v2: unbalanced OT; the deficit is unexplained mass). The deficit is
    /// penalized by adding a slack term to the objective, wired by the caller.
    AtMostOne,
}

/// A fully specified LAD LP instance.
pub struct LadProblem<'a> {
    /// Edge weights `ℓ_e`, length E.
    pub edge_lengths: &'a [f64],
    /// Sink cumulative masses `s_e`, length E.
    pub sink_cum: &'a [f64],
    /// Source cumulative masses `M`, row-major E×K (`m[e][k]`).
    pub source_cum: &'a [Vec<f64>],
    /// Number of sources K.
    pub num_sources: usize,
    /// Weight-sum constraint mode.
    pub constraint: WeightConstraint,
    /// v2 only: penalty rate `λ` on the unexplained deficit `1 − Σ w_k`. Ignored when
    /// `constraint == SumToOne`.
    pub deficit_penalty: f64,
    /// Optional per-source linear penalty `μ_k`, added to the objective as `Σ_k μ_k w_k`. Empty
    /// slice = no penalty (the common case). Used by the alternating unmixer's reweighted-L1
    /// source selection (spec §2.4 outer loop): setting `μ_k = c/(w_k+ε)` from the previous
    /// iterate drives negligible weights toward zero while keeping the solve a linear program
    /// (the term is linear in `w`, so per-`μ` global optimality is preserved). When non-empty,
    /// its length must equal `num_sources`.
    pub weight_penalty: &'a [f64],
}

/// The result of solving a [`LadProblem`].
#[derive(Debug, Clone)]
pub struct LadSolution {
    /// Mixing weights `w`, length K.
    pub weights: Vec<f64>,
    /// Objective value `Σ_e ℓ_e |residual_e|` (the tree-Wasserstein distance at the optimum;
    /// for v2 this excludes the deficit-penalty term — see [`Self::deficit`]).
    pub objective: f64,
    /// v2 only: the unexplained deficit `1 − Σ w_k` (≈ 0 for v1).
    pub deficit: f64,
}

/// A swappable LP backend.
pub trait LpSolver {
    fn solve(&self, problem: &LadProblem) -> Result<LadSolution, LpError>;
}

/// Recompute the pure tree-Wasserstein objective `Σ_e ℓ_e |s_e − Σ_k M_ek w_k|` from a weight
/// vector. Used to report the distance independently of solver slack variables, and in tests.
pub fn tree_wasserstein_objective(
    edge_lengths: &[f64],
    sink_cum: &[f64],
    source_cum: &[Vec<f64>],
    weights: &[f64],
) -> f64 {
    edge_lengths
        .iter()
        .zip(sink_cum.iter())
        .zip(source_cum.iter())
        .map(|((&ell, &s), row)| {
            let mix: f64 = row.iter().zip(weights.iter()).map(|(&m, &w)| m * w).sum();
            ell * (s - mix).abs()
        })
        .sum()
}

/// The default backend: `good_lp` with the pure-Rust `microlp` solver.
pub struct GoodLpSolver;

impl LpSolver for GoodLpSolver {
    fn solve(&self, problem: &LadProblem) -> Result<LadSolution, LpError> {
        use good_lp::{
            constraint, default_solver, variable, variables, Expression, Solution, SolverModel,
        };

        let e = problem.edge_lengths.len();
        let k = problem.num_sources;
        if problem.sink_cum.len() != e || problem.source_cum.len() != e {
            return Err(LpError::Dim(format!(
                "edge count mismatch: ℓ={}, s={}, M rows={}",
                e,
                problem.sink_cum.len(),
                problem.source_cum.len()
            )));
        }
        for (i, row) in problem.source_cum.iter().enumerate() {
            if row.len() != k {
                return Err(LpError::Dim(format!(
                    "M row {i} has {} entries, expected K={k}",
                    row.len()
                )));
            }
        }

        let mut vars = variables!();
        let w: Vec<_> = (0..k).map(|_| vars.add(variable().min(0.0))).collect();
        let t: Vec<_> = (0..e).map(|_| vars.add(variable().min(0.0))).collect();

        // Objective: Σ t_e  (+ λ·deficit for v2).
        let mut objective: Expression = t.iter().sum();

        // deficit = 1 − Σ w_k  (only meaningful for AtMostOne; for SumToOne it is exactly 0).
        let wsum: Expression = w.iter().sum();
        if problem.constraint == WeightConstraint::AtMostOne && problem.deficit_penalty != 0.0 {
            // λ·(1 − Σ w_k) = λ − λ·Σ w_k ; constant λ doesn't affect the argmin, drop it.
            objective += problem.deficit_penalty * (Expression::from(1.0) - wsum.clone());
        }

        // Optional reweighted-L1 source-selection penalty: Σ_k μ_k w_k (linear → stays an LP).
        if !problem.weight_penalty.is_empty() {
            if problem.weight_penalty.len() != k {
                return Err(LpError::Dim(format!(
                    "weight_penalty has {} entries, expected K={k}",
                    problem.weight_penalty.len()
                )));
            }
            objective += (0..k).map(|j| problem.weight_penalty[j] * w[j]).sum::<Expression>();
        }

        let mut model = vars.minimise(objective).using(default_solver);

        match problem.constraint {
            WeightConstraint::SumToOne => {
                model = model.with(constraint!(wsum.clone() == 1.0));
            }
            WeightConstraint::AtMostOne => {
                model = model.with(constraint!(wsum.clone() <= 1.0));
            }
        }

        // Per-edge |·| linearization. Indexing by edge `i` reads several parallel arrays
        // (t, source_cum, edge_lengths, sink_cum), so an index loop is clearest here.
        #[allow(clippy::needless_range_loop)]
        for i in 0..e {
            let mix: Expression = (0..k).map(|j| problem.source_cum[i][j] * w[j]).sum();
            let residual: Expression =
                problem.edge_lengths[i] * (Expression::from(problem.sink_cum[i]) - mix);
            model = model.with(constraint!(t[i] >= residual.clone()));
            model = model.with(constraint!(t[i] >= -residual));
        }

        let sol = model.solve().map_err(|err| match err {
            good_lp::ResolutionError::Infeasible | good_lp::ResolutionError::Unbounded => {
                LpError::NoSolution
            }
            other => LpError::Solver(format!("{other:?}")),
        })?;

        let mut weights: Vec<f64> = w.iter().map(|&v| sol.value(v)).collect();
        // Clamp tiny negatives from solver tolerance.
        for wv in &mut weights {
            if *wv < 0.0 && *wv > -1e-9 {
                *wv = 0.0;
            }
        }
        let sum_w: f64 = weights.iter().sum();
        let deficit = (1.0 - sum_w).max(0.0);

        // Report the *pure* tree-Wasserstein distance, recomputed from weights (independent of
        // the solver's slack variables — robust to solver-internal scaling).
        let objective = tree_wasserstein_objective(
            problem.edge_lengths,
            problem.sink_cum,
            problem.source_cum,
            &weights,
        );

        Ok(LadSolution {
            weights,
            objective,
            deficit,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    /// The exact toy instance verified in the design probe: E=4, K=2, true w=[0.5,0.5],
    /// objective 0.
    #[test]
    fn recovers_toy_instance() {
        let ell = vec![1.0, 1.0, 1.0, 2.0];
        let s = vec![0.5, 0.3, 0.5, 1.0];
        let m = vec![
            vec![0.6, 0.4],
            vec![0.2, 0.4],
            vec![0.8, 0.2],
            vec![1.0, 1.0],
        ];
        let prob = LadProblem {
            edge_lengths: &ell,
            sink_cum: &s,
            source_cum: &m,
            num_sources: 2,
            constraint: WeightConstraint::SumToOne,
            deficit_penalty: 0.0,
            weight_penalty: &[],
        };
        let sol = GoodLpSolver.solve(&prob).unwrap();
        assert_abs_diff_eq!(sol.weights[0], 0.5, epsilon = 1e-6);
        assert_abs_diff_eq!(sol.weights[1], 0.5, epsilon = 1e-6);
        assert_abs_diff_eq!(sol.objective, 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(sol.weights.iter().sum::<f64>(), 1.0, epsilon = 1e-6);
    }

    /// A pure-mixture instance where the true weights are asymmetric: the LP must recover them
    /// exactly (objective 0) because the sink is an exact convex combination of the sources.
    #[test]
    fn recovers_asymmetric_exact_mixture() {
        // 3 edges, 2 sources. Construct M and set s = 0.7*M[:,0] + 0.3*M[:,1].
        let ell = vec![1.0, 1.0, 1.0];
        let m = vec![vec![0.9, 0.1], vec![0.5, 0.5], vec![0.2, 0.8]];
        let w_true = [0.7, 0.3];
        let s: Vec<f64> = m
            .iter()
            .map(|row| row[0] * w_true[0] + row[1] * w_true[1])
            .collect();
        let prob = LadProblem {
            edge_lengths: &ell,
            sink_cum: &s,
            source_cum: &m,
            num_sources: 2,
            constraint: WeightConstraint::SumToOne,
            deficit_penalty: 0.0,
            weight_penalty: &[],
        };
        let sol = GoodLpSolver.solve(&prob).unwrap();
        assert_abs_diff_eq!(sol.objective, 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(sol.weights[0], 0.7, epsilon = 1e-5);
        assert_abs_diff_eq!(sol.weights[1], 0.3, epsilon = 1e-5);
    }

    /// The LP optimum must match an independent brute-force grid search over the simplex on a
    /// small instance — confirming global optimality.
    #[test]
    fn matches_brute_force_grid() {
        let ell = vec![1.0, 2.0, 0.5, 1.5];
        // arbitrary sink, not an exact mixture -> nonzero objective, nontrivial argmin
        let s = vec![0.55, 0.4, 0.62, 0.33];
        let m = vec![
            vec![0.9, 0.2, 0.5],
            vec![0.3, 0.7, 0.4],
            vec![0.8, 0.1, 0.6],
            vec![0.2, 0.5, 0.35],
        ];
        let prob = LadProblem {
            edge_lengths: &ell,
            sink_cum: &s,
            source_cum: &m,
            num_sources: 3,
            constraint: WeightConstraint::SumToOne,
            deficit_penalty: 0.0,
            weight_penalty: &[],
        };
        let sol = GoodLpSolver.solve(&prob).unwrap();

        // brute-force grid over the 3-simplex at resolution 1/G
        let g = 200i32;
        let mut best = f64::INFINITY;
        for a in 0..=g {
            for b in 0..=(g - a) {
                let c = g - a - b;
                let w = [a as f64 / g as f64, b as f64 / g as f64, c as f64 / g as f64];
                let obj = tree_wasserstein_objective(&ell, &s, &m, &w);
                if obj < best {
                    best = obj;
                }
            }
        }
        // The LP optimum should be no worse than the best grid point (and typically slightly
        // better, since the grid is discrete).
        assert!(
            sol.objective <= best + 1e-6,
            "LP objective {} should be <= grid best {}",
            sol.objective,
            best
        );
        // And the LP objective recomputed from its weights is self-consistent.
        let recomputed = tree_wasserstein_objective(&ell, &s, &m, &sol.weights);
        assert_abs_diff_eq!(sol.objective, recomputed, epsilon = 1e-9);
    }

    #[test]
    fn unbalanced_allows_deficit() {
        // If no mixture explains the sink well and λ is small, v2 may leave a deficit.
        let ell = vec![1.0, 1.0];
        let s = vec![0.5, 0.5];
        let m = vec![vec![1.0], vec![1.0]]; // single source, cumulative mass 1 on both edges
        let prob = LadProblem {
            edge_lengths: &ell,
            sink_cum: &s,
            source_cum: &m,
            num_sources: 1,
            constraint: WeightConstraint::AtMostOne,
            deficit_penalty: 0.01, // cheap to leave mass unexplained
            weight_penalty: &[],
        };
        let sol = GoodLpSolver.solve(&prob).unwrap();
        // weight should be <= 1 and deficit >= 0
        assert!(sol.weights[0] <= 1.0 + 1e-6);
        assert!(sol.deficit >= -1e-6);
    }
}
