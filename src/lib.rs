//! # otst — optimal-transport microbial source tracking
//!
//! Estimates what fraction of a sink microbial community came from each candidate source,
//! using a **tree-Wasserstein (weighted UniFrac) loss**. The point estimate is a convex linear
//! program (global optimum). See `CLAUDE.md` for the full design.
//!
//! ## Pipeline
//! 1. Parse a rooted phylogenetic tree from Newick ([`tree`]).
//! 2. Read source/sink count tables and align taxa to the tree leaves ([`io`], [`profile`]).
//! 3. Compute per-edge cumulative masses and solve the LAD LP ([`estimate`], [`lp`]).
//! 4. (M2) Wrap in a multinomial bootstrap for uncertainty.

pub mod baseline;
pub mod bootstrap;
pub mod cli;
pub mod estimate;
pub mod io;
pub mod lp;
pub mod profile;
pub mod sim;
pub mod tree;
pub mod unknown;
pub mod unmix;

pub use bootstrap::{BootstrapConfig, BootstrapResult, IntervalMethod, SourceInterval};
pub use estimate::{point_estimate, PointEstimate, Prepared, SourceEstimate};
pub use lp::{GoodLpSolver, LadProblem, LadSolution, LpError, LpSolver};
pub use profile::{Profile, SourceSet};
pub use tree::{Tree, TreeError};
pub use unmix::{alternating_estimate, UnmixConfig, UnmixResult, WeightStep};
