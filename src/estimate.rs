//! Orchestration: assemble the tree-Wasserstein LP inputs from a tree + sources + sink and
//! produce a point estimate. The bootstrap (M2) builds on the [`Prepared`] problem so that the
//! expensive tree/edge setup is done once and only the resampled masses change per replicate.

use crate::lp::{LadProblem, LadSolution, LpSolver};
use crate::profile::{Profile, SourceSet};
use crate::tree::Tree;
use crate::unknown::UNKNOWN_LABEL;

/// A source-tracking problem prepared against a fixed tree: edge lengths and the immutable
/// pieces are computed once. Cumulative masses (which change under resampling) are computed
/// per solve. The mixing weights are always on the simplex (`Σ w = 1`).
pub struct Prepared<'t> {
    pub tree: &'t Tree,
    /// Edge lengths `ℓ_e`, length E, in tree edge order.
    pub edge_lengths: Vec<f64>,
    /// Output source names, length K (named sources, plus the unknown source if `background` is
    /// set).
    pub source_names: Vec<String>,
    /// An estimated unknown-source profile `b_0`, appended as the (K+1)-th source, when the
    /// joint unmixer is in use. `None` for the plain no-unknown solve.
    pub background: Option<Vec<f64>>,
    /// Number of *named* sources (excludes the unknown source).
    pub num_named: usize,
    /// Optional per-LP-source reweighted-L1 penalty `μ_k` (length `num_lp_sources()`, empty =
    /// none). Threaded into every LP solve; set by the alternating unmixer between iterations.
    pub weight_penalty: Vec<f64>,
}

impl<'t> Prepared<'t> {
    /// Build the plain (no-unknown) prepared problem: the named sources must explain all sink
    /// mass under `Σ w = 1`.
    pub fn new(tree: &'t Tree, sources: &SourceSet) -> Self {
        Prepared {
            tree,
            edge_lengths: tree.edge_lengths(),
            source_names: sources.names.clone(),
            background: None,
            num_named: sources.num_sources(),
            weight_penalty: Vec::new(),
        }
    }

    /// Build a prepared problem with an *explicit* unknown profile `b_0` appended as the
    /// (K+1)-th source (`Σ w = 1`). This is how the alternating unmixer ([`crate::unmix`], spec
    /// §2.4 outer loop) re-solves against a freshly estimated unknown profile each iteration
    /// while reusing the fixed tree/edge setup.
    pub fn with_background(tree: &'t Tree, sources: &SourceSet, background: Vec<f64>) -> Self {
        let mut source_names = sources.names.clone();
        source_names.push(UNKNOWN_LABEL.to_string());
        Prepared {
            tree,
            edge_lengths: tree.edge_lengths(),
            source_names,
            background: Some(background),
            num_named: sources.num_sources(),
            weight_penalty: Vec::new(),
        }
    }

    /// Total number of sources in the LP (named + unknown).
    pub fn num_lp_sources(&self) -> usize {
        self.num_named + if self.background.is_some() { 1 } else { 0 }
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
            weight_penalty: &self.weight_penalty,
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
}

/// Convenience: run a single (no-unknown) point estimate end-to-end from the raw inputs.
pub fn point_estimate<'t, S: LpSolver>(
    solver: &S,
    tree: &'t Tree,
    sources: &SourceSet,
    sink: &Profile,
) -> Result<(PointEstimate, Prepared<'t>), crate::lp::LpError> {
    let prepared = Prepared::new(tree, sources);
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

        let (est, _prep) = point_estimate(&GoodLpSolver, &tree, &sources, &sink).unwrap();

        // Find s1 and s2 in output.
        let get = |n: &str| est.sources.iter().find(|s| s.name == n).unwrap().proportion;
        assert_abs_diff_eq!(get("s1"), 0.7, epsilon = 1e-4);
        assert_abs_diff_eq!(get("s2"), 0.3, epsilon = 1e-4);
        assert_abs_diff_eq!(est.objective, 0.0, epsilon = 1e-4);
    }

    #[test]
    fn with_background_adds_unknown_source() {
        // A supplied unknown profile appears as the (K+1)-th LP source and weights sum to 1.
        let tree = Tree::parse_newick("((A:1,B:1):1,(C:1,D:1):1);").unwrap();
        let sources = SourceSet {
            names: vec!["s1".into(), "s2".into()],
            profiles: vec![
                Profile::from_counts(vec![50.0, 50.0, 0.0, 0.0]),
                Profile::from_counts(vec![0.0, 0.0, 50.0, 50.0]),
            ],
        };
        let sink = Profile::from_counts(vec![350.0, 350.0, 150.0, 150.0]);
        let b0 = vec![0.25, 0.25, 0.25, 0.25];
        let prepared = Prepared::with_background(&tree, &sources, b0);
        assert_eq!(prepared.num_lp_sources(), 3);
        assert!(prepared.source_names.contains(&UNKNOWN_LABEL.to_string()));
        let src_norm: Vec<Vec<f64>> = sources.profiles.iter().map(|p| p.normalized()).collect();
        let sol = prepared.solve(&GoodLpSolver, &src_norm, &sink.normalized()).unwrap();
        assert_abs_diff_eq!(sol.weights.iter().sum::<f64>(), 1.0, epsilon = 1e-5);
    }
}
