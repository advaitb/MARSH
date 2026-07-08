//! Simulation harness (spec §6.2) — generate sinks from sources with ground-truth weights,
//! a hidden unknown source, and controlled *phylogenetic* drift, then sequence at configurable
//! depth. This is what validates the paper's claims; it is not optional.
//!
//! The design intent (spec §6.3, headline experiment): drift that moves a taxon's mass toward
//! a **phylogenetic neighbor** should barely move the tree-Wasserstein estimate (cheap under
//! the ground metric) while wrecking a non-phylogenetic baseline. So drift here is
//! tree-aware: perturbed mass flows to the sibling leaf under the same parent.

use crate::profile::{Profile, SourceSet};
use crate::tree::{NodeId, Tree};
use rand::seq::SliceRandom;
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Gamma};

/// Parameters for one simulated scenario.
#[derive(Debug, Clone)]
pub struct SimConfig {
    /// Number of leaves (taxa) in the generated tree.
    pub num_taxa: usize,
    /// Number of named sources K.
    pub num_sources: usize,
    /// Dirichlet concentration for source profiles (smaller → sparser/peakier sources).
    pub dirichlet_alpha: f64,
    /// True fraction of the sink coming from the hidden unknown source (0 disables it).
    pub unknown_fraction: f64,
    /// Fraction of each source's mass that drifts (0 = no drift).
    pub drift: f64,
    /// Sink sequencing depth n.
    pub sink_depth: u64,
    /// Source sequencing depth m (per source).
    pub source_depth: u64,
}

impl Default for SimConfig {
    fn default() -> Self {
        SimConfig {
            num_taxa: 32,
            num_sources: 4,
            dirichlet_alpha: 0.3,
            unknown_fraction: 0.0,
            drift: 0.0,
            sink_depth: 10_000,
            source_depth: 10_000,
        }
    }
}

/// A generated scenario: the tree, the sampled source counts, the sink counts, and the
/// ground-truth mixing weights used to build the sink.
pub struct Scenario {
    pub tree: Tree,
    pub sources: SourceSet,
    pub sink: Profile,
    /// Ground-truth weights over the K named sources (length K). These sum to
    /// `1 − unknown_fraction`.
    pub true_weights: Vec<f64>,
    /// The true unknown fraction actually used.
    pub true_unknown: f64,
    /// The *true* (noise-free) sink composition before sequencing, length D.
    pub true_sink_composition: Vec<f64>,
}

/// Generate a random rooted binary tree over `num_taxa` leaves by repeatedly joining random
/// nodes (a coalescent-style bottom-up merge). Edge lengths are unit — topology carries the
/// phylogenetic signal, which is what the drift experiment exercises.
pub fn random_tree(rng: &mut ChaCha8Rng, num_taxa: usize) -> Tree {
    assert!(num_taxa >= 1);
    // Build a Newick string by iteratively pairing.
    let mut clusters: Vec<String> = (0..num_taxa).map(|i| format!("T{i}:1")).collect();
    while clusters.len() > 1 {
        clusters.shuffle(rng);
        let a = clusters.pop().unwrap();
        let b = clusters.pop().unwrap();
        clusters.push(format!("({a},{b}):1"));
    }
    let newick = format!("{};", clusters.pop().unwrap());
    Tree::parse_newick(&newick).expect("generated Newick should parse")
}

/// Draw a Dirichlet(alpha,…,alpha) sample of length `d` via normalized Gammas.
fn dirichlet(rng: &mut ChaCha8Rng, d: usize, alpha: f64) -> Vec<f64> {
    let gamma = Gamma::new(alpha, 1.0).expect("alpha > 0");
    let raw: Vec<f64> = (0..d).map(|_| gamma.sample(rng).max(1e-12)).collect();
    let sum: f64 = raw.iter().sum();
    raw.into_iter().map(|x| x / sum).collect()
}

/// For each leaf, its sibling leaf (a leaf sharing the same parent), if any — the target of
/// phylogenetic drift. Falls back to `None` for leaves whose parent has no other leaf child.
fn leaf_siblings(tree: &Tree) -> Vec<Option<usize>> {
    // position in leaf order -> node id, and inverse
    let leaves = tree.leaves();
    let mut node_to_leafpos = std::collections::HashMap::new();
    for (i, &n) in leaves.iter().enumerate() {
        node_to_leafpos.insert(n, i);
    }
    let mut sib = vec![None; leaves.len()];
    for (i, &leaf) in leaves.iter().enumerate() {
        if let Some(parent) = tree.nodes[leaf].parent {
            // find a different child of parent that is also a leaf
            for &child in &tree.nodes[parent].children {
                if child != leaf {
                    if let Some(&pos) = node_to_leafpos.get(&child) {
                        sib[i] = Some(pos);
                        break;
                    }
                }
            }
        }
    }
    sib
}

/// Apply phylogenetic drift to a profile: move a `drift` fraction of each taxon's mass to its
/// phylogenetic sibling leaf. Mass with no sibling stays put. Result is renormalized.
fn apply_phylo_drift(profile: &[f64], siblings: &[Option<usize>], drift: f64) -> Vec<f64> {
    if drift <= 0.0 {
        return profile.to_vec();
    }
    let mut out = vec![0.0f64; profile.len()];
    for (i, &mass) in profile.iter().enumerate() {
        let moved = mass * drift;
        let kept = mass - moved;
        out[i] += kept;
        match siblings[i] {
            Some(j) => out[j] += moved,
            None => out[i] += moved, // no sibling: keep it
        }
    }
    let sum: f64 = out.iter().sum();
    if sum > 0.0 {
        for v in &mut out {
            *v /= sum;
        }
    }
    out
}

/// Multinomial draw (sequential-conditional-binomial), returning counts as f64.
fn multinomial(rng: &mut ChaCha8Rng, n: u64, probs: &[f64]) -> Vec<f64> {
    use rand_distr::Binomial;
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
        let p_i = if remaining_p > 0.0 {
            (probs[i] / remaining_p).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let draw = Binomial::new(remaining_n, p_i).unwrap().sample(rng);
        counts[i] = draw as f64;
        remaining_n -= draw;
        remaining_p -= probs[i];
    }
    counts
}

/// Generate a full scenario from a config and seed.
pub fn generate(config: &SimConfig, seed: u64) -> Scenario {
    use rand::SeedableRng;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let d = config.num_taxa;
    let k = config.num_sources;

    // 1) tree
    let tree = random_tree(&mut rng, d);
    let siblings = leaf_siblings(&tree);
    // Leaves are in tree leaf order; source/sink vectors use that same order.

    // 2) true source profiles (Dirichlet), sampled to counts at source_depth
    let true_source_profiles: Vec<Vec<f64>> =
        (0..k).map(|_| dirichlet(&mut rng, d, config.dirichlet_alpha)).collect();

    // 3) hidden unknown source profile (also Dirichlet, independent)
    let unknown_profile = dirichlet(&mut rng, d, config.dirichlet_alpha);

    // 4) ground-truth weights over K named sources (Dirichlet on the simplex), scaled to
    //    (1 − unknown_fraction).
    let raw_w = dirichlet(&mut rng, k, 1.0);
    let named_mass = 1.0 - config.unknown_fraction;
    let true_weights: Vec<f64> = raw_w.iter().map(|&w| w * named_mass).collect();

    // 5) true sink composition = Σ w_k · drift(b_k) + unknown_fraction · unknown
    //    Drift is applied to the source profiles as they contribute to the SINK only — the
    //    observed sources remain the un-drifted profiles. This models real drift: the sink
    //    community has evolved away from the reference sources.
    let mut true_sink = vec![0.0f64; d];
    for (kj, prof) in true_source_profiles.iter().enumerate() {
        let drifted = apply_phylo_drift(prof, &siblings, config.drift);
        for i in 0..d {
            true_sink[i] += true_weights[kj] * drifted[i];
        }
    }
    for i in 0..d {
        true_sink[i] += config.unknown_fraction * unknown_profile[i];
    }
    // renormalize (guard against fp drift)
    let ssum: f64 = true_sink.iter().sum();
    for v in &mut true_sink {
        *v /= ssum;
    }

    // 6) sequence: draw source counts (un-drifted profiles) and sink counts
    let source_profiles_counts: Vec<Profile> = true_source_profiles
        .iter()
        .map(|prof| Profile::from_counts(multinomial(&mut rng, config.source_depth, prof)))
        .collect();
    let sink_counts = multinomial(&mut rng, config.sink_depth, &true_sink);

    let sources = SourceSet {
        names: (0..k).map(|i| format!("S{i}")).collect(),
        profiles: source_profiles_counts,
    };

    Scenario {
        tree,
        sources,
        sink: Profile::from_counts(sink_counts),
        true_weights,
        true_unknown: config.unknown_fraction,
        true_sink_composition: true_sink,
    }
}

/// L1 error between an estimate and the ground-truth named-source weights (both length K).
/// This is the total-variation-style error used across the experiments.
pub fn l1_error(estimate: &[f64], truth: &[f64]) -> f64 {
    estimate
        .iter()
        .zip(truth.iter())
        .map(|(&e, &t)| (e - t).abs())
        .sum()
}

/// Utility: the sibling-leaf map for a tree (exposed for tests / external drift use).
pub fn sibling_map(tree: &Tree) -> Vec<Option<usize>> {
    leaf_siblings(tree)
}

/// Utility: pick a random leaf node id (used in tests).
pub fn random_leaf(rng: &mut impl Rng, tree: &Tree) -> NodeId {
    let leaves = tree.leaves();
    leaves[rng.random_range(0..leaves.len())]
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn random_tree_has_right_leaf_count() {
        use rand::SeedableRng;
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let t = random_tree(&mut rng, 16);
        assert_eq!(t.num_leaves(), 16);
    }

    #[test]
    fn dirichlet_sums_to_one() {
        use rand::SeedableRng;
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        let p = dirichlet(&mut rng, 20, 0.5);
        assert_abs_diff_eq!(p.iter().sum::<f64>(), 1.0, epsilon = 1e-9);
        assert!(p.iter().all(|&x| x >= 0.0));
    }

    #[test]
    fn drift_preserves_normalization_and_moves_to_sibling() {
        // tiny tree: ((A,B),(C,D)) -> A,B siblings; C,D siblings
        let tree = Tree::parse_newick("((A:1,B:1):1,(C:1,D:1):1);").unwrap();
        let sib = leaf_siblings(&tree);
        // every leaf should have a sibling here
        assert!(sib.iter().all(|s| s.is_some()));
        // all mass on leaf 0; drift 0.5 -> half moves to its sibling
        let mut prof = vec![0.0; 4];
        prof[0] = 1.0;
        let drifted = apply_phylo_drift(&prof, &sib, 0.5);
        assert_abs_diff_eq!(drifted.iter().sum::<f64>(), 1.0, epsilon = 1e-12);
        let sibling_of_0 = sib[0].unwrap();
        assert_abs_diff_eq!(drifted[0], 0.5, epsilon = 1e-12);
        assert_abs_diff_eq!(drifted[sibling_of_0], 0.5, epsilon = 1e-12);
    }

    #[test]
    fn scenario_weights_sum_correctly() {
        let cfg = SimConfig {
            num_taxa: 16,
            num_sources: 3,
            unknown_fraction: 0.2,
            ..Default::default()
        };
        let sc = generate(&cfg, 123);
        let named: f64 = sc.true_weights.iter().sum();
        assert_abs_diff_eq!(named, 0.8, epsilon = 1e-9);
        assert_abs_diff_eq!(sc.true_unknown, 0.2, epsilon = 1e-12);
        assert_abs_diff_eq!(sc.true_sink_composition.iter().sum::<f64>(), 1.0, epsilon = 1e-9);
        assert_eq!(sc.sources.num_sources(), 3);
    }

    #[test]
    fn generate_is_reproducible() {
        let cfg = SimConfig::default();
        let a = generate(&cfg, 7);
        let b = generate(&cfg, 7);
        assert_eq!(a.sink.counts, b.sink.counts);
        assert_eq!(a.true_weights, b.true_weights);
    }

    #[test]
    fn l1_error_basic() {
        assert_abs_diff_eq!(l1_error(&[0.5, 0.5], &[0.6, 0.4]), 0.2, epsilon = 1e-12);
    }
}
