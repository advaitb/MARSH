//! Orchestration: assemble the tree-Wasserstein LP inputs from a tree + sources + sink and
//! produce a point estimate. The bootstrap (M2) builds on the [`Prepared`] problem so that the
//! expensive tree/edge setup is done once and only the resampled masses change per replicate.

use crate::lp::{LadProblem, LadSolution, LpSolver, WeightConstraint};
use crate::profile::{Profile, SourceSet};
use crate::tree::Tree;
use crate::unknown::{background_profile, UnknownMode, UNKNOWN_LABEL};

/// A source-tracking problem prepared against a fixed tree: edge lengths and the immutable
/// pieces are computed once. Cumulative masses (which change under resampling) are computed
/// per solve.
pub struct Prepared<'t> {
    pub tree: &'t Tree,
    /// Edge lengths `ℓ_e`, length E, in tree edge order.
    pub edge_lengths: Vec<f64>,
    /// Output source names, length K (named sources, plus the background source if any).
    pub source_names: Vec<String>,
    /// The unknown mode in effect.
    pub mode: UnknownMode,
    /// v2 penalty λ.
    pub lambda: f64,
    /// The fixed background profile `b_0` (v1 unknown modes), if any.
    pub background: Option<Vec<f64>>,
    /// Number of *named* sources (excludes the background source).
    pub num_named: usize,
}

impl<'t> Prepared<'t> {
    /// Build the prepared problem. `source_names` come from the source set; a background source
    /// label is appended when a v1 unknown mode is active.
    pub fn new(
        tree: &'t Tree,
        sources: &SourceSet,
        mode: UnknownMode,
        lambda: f64,
    ) -> Self {
        let edge_lengths = tree.edge_lengths();
        let num_taxa = tree.num_leaves();
        let background = background_profile(mode, sources, num_taxa);

        let mut source_names = sources.names.clone();
        if background.is_some() {
            source_names.push(UNKNOWN_LABEL.to_string());
        }
        let num_named = sources.num_sources();

        Prepared {
            tree,
            edge_lengths,
            source_names,
            mode,
            lambda,
            background,
            num_named,
        }
    }

    /// Total number of sources in the LP (named + background).
    pub fn num_lp_sources(&self) -> usize {
        self.num_named + if self.background.is_some() { 1 } else { 0 }
    }

    fn weight_constraint(&self) -> WeightConstraint {
        match self.mode {
            UnknownMode::Unbalanced => WeightConstraint::AtMostOne,
            _ => WeightConstraint::SumToOne,
        }
    }

    /// Compute the E×K source cumulative-mass matrix `M` for the given (already normalized)
    /// source profiles. Appends the fixed background column when present.
    ///
    /// `source_profiles` must be the K named sources' normalized profiles (each length D).
    pub fn source_cumulative(&self, source_profiles: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let e = self.edge_lengths.len();
        let k = self.num_lp_sources();
        // Per-source cumulative-mass column, then transpose into E×K rows.
        let mut cols: Vec<Vec<f64>> = Vec::with_capacity(k);
        for prof in source_profiles {
            cols.push(self.tree.cumulative_masses(prof));
        }
        if let Some(bg) = &self.background {
            cols.push(self.tree.cumulative_masses(bg));
        }
        // transpose cols (K each length E) -> rows (E each length K)
        let mut rows = vec![vec![0.0f64; k]; e];
        for (kj, col) in cols.iter().enumerate() {
            for ei in 0..e {
                rows[ei][kj] = col[ei];
            }
        }
        rows
    }

    /// Solve the point estimate given normalized source profiles and a normalized sink.
    pub fn solve<S: LpSolver>(
        &self,
        solver: &S,
        source_profiles: &[Vec<f64>],
        sink: &[f64],
    ) -> Result<LadSolution, crate::lp::LpError> {
        let sink_cum = self.tree.cumulative_masses(sink);
        let source_cum = self.source_cumulative(source_profiles);
        let problem = LadProblem {
            edge_lengths: &self.edge_lengths,
            sink_cum: &sink_cum,
            source_cum: &source_cum,
            num_sources: self.num_lp_sources(),
            constraint: self.weight_constraint(),
            deficit_penalty: self.lambda,
        };
        solver.solve(&problem)
    }
}

/// One estimated source contribution.
#[derive(Debug, Clone)]
pub struct SourceEstimate {
    pub name: String,
    pub proportion: f64,
}

/// The point-estimate result (uncertainty is added by the bootstrap layer in M2).
#[derive(Debug, Clone)]
pub struct PointEstimate {
    pub sources: Vec<SourceEstimate>,
    /// The tree-Wasserstein objective at the optimum.
    pub objective: f64,
    /// Unexplained deficit (v2) — 0 for v1.
    pub deficit: f64,
}

/// Convenience: run a single point estimate end-to-end from the raw inputs.
pub fn point_estimate<'t, S: LpSolver>(
    solver: &S,
    tree: &'t Tree,
    sources: &SourceSet,
    sink: &Profile,
    mode: UnknownMode,
    lambda: f64,
) -> Result<(PointEstimate, Prepared<'t>), crate::lp::LpError> {
    let prepared = Prepared::new(tree, sources, mode, lambda);
    let source_profiles: Vec<Vec<f64>> = sources.profiles.iter().map(|p| p.normalized()).collect();
    let sink_norm = sink.normalized();
    let sol = prepared.solve(solver, &source_profiles, &sink_norm)?;

    let sources_out = prepared
        .source_names
        .iter()
        .zip(sol.weights.iter())
        .map(|(name, &w)| SourceEstimate {
            name: name.clone(),
            proportion: w,
        })
        .collect();

    Ok((
        PointEstimate {
            sources: sources_out,
            objective: sol.objective,
            deficit: sol.deficit,
        },
        prepared,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lp::GoodLpSolver;
    use approx::assert_abs_diff_eq;

    #[test]
    fn recovers_known_mixture_end_to_end() {
        // 4-leaf balanced tree.
        let tree = Tree::parse_newick("((A:1,B:1):1,(C:1,D:1):1);").unwrap();
        // Two sources: s1 concentrated on {A,B}, s2 on {C,D}.
        let sources = SourceSet {
            names: vec!["s1".into(), "s2".into()],
            profiles: vec![
                Profile::from_counts(vec![50.0, 50.0, 0.0, 0.0]),
                Profile::from_counts(vec![0.0, 0.0, 50.0, 50.0]),
            ],
        };
        // Sink = 0.7 s1 + 0.3 s2, as counts (depth 1000).
        let sink = Profile::from_counts(vec![350.0, 350.0, 150.0, 150.0]);

        let (est, _prep) = point_estimate(
            &GoodLpSolver,
            &tree,
            &sources,
            &sink,
            UnknownMode::None,
            0.0,
        )
        .unwrap();

        // Find s1 and s2 in output.
        let get = |n: &str| est.sources.iter().find(|s| s.name == n).unwrap().proportion;
        assert_abs_diff_eq!(get("s1"), 0.7, epsilon = 1e-4);
        assert_abs_diff_eq!(get("s2"), 0.3, epsilon = 1e-4);
        assert_abs_diff_eq!(est.objective, 0.0, epsilon = 1e-4);
    }

    #[test]
    fn metacommunity_adds_unknown_source() {
        let tree = Tree::parse_newick("((A:1,B:1):1,(C:1,D:1):1);").unwrap();
        let sources = SourceSet {
            names: vec!["s1".into(), "s2".into()],
            profiles: vec![
                Profile::from_counts(vec![50.0, 50.0, 0.0, 0.0]),
                Profile::from_counts(vec![0.0, 0.0, 50.0, 50.0]),
            ],
        };
        let sink = Profile::from_counts(vec![350.0, 350.0, 150.0, 150.0]);
        let (est, prep) = point_estimate(
            &GoodLpSolver,
            &tree,
            &sources,
            &sink,
            UnknownMode::Metacommunity,
            0.0,
        )
        .unwrap();
        // The background/unknown source is present in the output.
        assert!(est.sources.iter().any(|s| s.name == UNKNOWN_LABEL));
        assert_eq!(prep.num_lp_sources(), 3);
        // Weights still sum to 1.
        let total: f64 = est.sources.iter().map(|s| s.proportion).sum();
        assert_abs_diff_eq!(total, 1.0, epsilon = 1e-5);
    }
}
