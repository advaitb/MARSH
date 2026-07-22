//! Unknown-source handling.
//!
//! MARSH models an unknown source in exactly one way: the joint profile+weights estimator in
//! [`crate::unmix`] (`--unknown`), which re-estimates the unknown's *shape* in an outer
//! alternating loop while the inner weight solve stays convex. When `--unknown` is not requested,
//! no unknown source is modeled and the named sources must explain all sink mass (`Σ w = 1`).
//!
//! (Earlier fixed-background modes — `uniform`, `metacommunity` — and the unbalanced-OT deficit
//! were removed: the benchmark showed the joint estimator dominates them, and a fixed/guessed
//! background profile is wrong whenever the true unknown has its own composition.)

/// The label used for the unknown source in outputs.
pub const UNKNOWN_LABEL: &str = "Unknown";
