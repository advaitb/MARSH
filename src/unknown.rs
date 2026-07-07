//! Unknown-source handling.
//!
//! v1 (convex, default): append a *fixed* background source `b_0` to the source set and solve
//! the (K+1)-source LP with `Σ w = 1`. The estimated `w_0` is the unknown fraction. Because
//! `b_0` is fixed, the problem stays a linear program — global optimum guaranteed.
//!
//! v2 (unbalanced OT): don't add a background source; instead relax to `Σ w_k ≤ 1` and
//! penalize the deficit at rate `λ`. See [`crate::lp::WeightConstraint::AtMostOne`].
//!
//! Jointly estimating the *shape* of `b_0` alongside its weight `w_0` is bilinear
//! (`w_0 · b_0`) and non-convex, so it is deliberately kept out of the inner LP (it can only
//! live in an outer alternating loop — a future extension).

use crate::profile::SourceSet;
use serde::{Deserialize, Serialize};

/// How to model sink mass not attributable to the named sources.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum UnknownMode {
    /// No unknown source; the named sources must explain all sink mass (`Σ w = 1`).
    None,
    /// Fixed background source with a uniform profile over taxa (v1).
    Uniform,
    /// Fixed background source = mean of all source profiles (v1).
    Metacommunity,
    /// Unbalanced OT: allow `Σ w_k ≤ 1`, penalize the deficit at rate `λ` (v2).
    Unbalanced,
}

/// Construct the fixed background profile `b_0` for a v1 unknown mode, or `None` when no
/// background source should be appended (`None`/`Unbalanced`).
pub fn background_profile(
    mode: UnknownMode,
    sources: &SourceSet,
    num_taxa: usize,
) -> Option<Vec<f64>> {
    match mode {
        UnknownMode::None | UnknownMode::Unbalanced => None,
        UnknownMode::Uniform => Some(vec![1.0 / num_taxa as f64; num_taxa]),
        UnknownMode::Metacommunity => Some(sources.metacommunity(num_taxa)),
    }
}

/// The label used for the unknown/background source in outputs.
pub const UNKNOWN_LABEL: &str = "Unknown";

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profile::Profile;
    use approx::assert_abs_diff_eq;

    fn toy_sources() -> SourceSet {
        SourceSet {
            names: vec!["s1".into(), "s2".into()],
            profiles: vec![
                Profile::from_counts(vec![10.0, 0.0, 0.0, 0.0]),
                Profile::from_counts(vec![0.0, 0.0, 5.0, 5.0]),
            ],
        }
    }

    #[test]
    fn uniform_background_sums_to_one() {
        let b0 = background_profile(UnknownMode::Uniform, &toy_sources(), 4).unwrap();
        assert_abs_diff_eq!(b0.iter().sum::<f64>(), 1.0, epsilon = 1e-12);
        for v in b0 {
            assert_abs_diff_eq!(v, 0.25, epsilon = 1e-12);
        }
    }

    #[test]
    fn metacommunity_background_is_mean() {
        let b0 = background_profile(UnknownMode::Metacommunity, &toy_sources(), 4).unwrap();
        assert_abs_diff_eq!(b0.iter().sum::<f64>(), 1.0, epsilon = 1e-12);
        // s1 = [1,0,0,0], s2 = [0,0,0.5,0.5]; mean = [0.5,0,0.25,0.25]
        assert_abs_diff_eq!(b0[0], 0.5, epsilon = 1e-12);
        assert_abs_diff_eq!(b0[2], 0.25, epsilon = 1e-12);
    }

    #[test]
    fn none_and_unbalanced_have_no_background() {
        assert!(background_profile(UnknownMode::None, &toy_sources(), 4).is_none());
        assert!(background_profile(UnknownMode::Unbalanced, &toy_sources(), 4).is_none());
    }
}
