//! Count tables and normalized profiles over taxa.
//!
//! A [`Profile`] stores raw integer counts *and* the sequencing depth `m = Σ counts`. The
//! depth is needed by the bootstrap (§2.5): each source is resampled `Multinomial(m_k, b_k)`,
//! so a shallow source contributes more profile noise. Discarding the depth and keeping only
//! the normalized profile would silently break the source-resampling step — the top
//! correctness requirement of the whole tool.

/// A distribution over `D` taxa, backed by raw counts.
#[derive(Debug, Clone)]
pub struct Profile {
    /// Raw per-taxon counts, length `D`, aligned to the tree's leaf order.
    pub counts: Vec<f64>,
    /// Sequencing depth `m = Σ counts`.
    pub depth: f64,
}

impl Profile {
    /// Build a profile from raw counts. Depth is their sum.
    pub fn from_counts(counts: Vec<f64>) -> Self {
        let depth = counts.iter().sum();
        Profile { counts, depth }
    }

    /// Number of taxa `D`.
    pub fn len(&self) -> usize {
        self.counts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.counts.is_empty()
    }

    /// Normalized profile `b = counts / depth`. If depth is zero (an all-zero column) the
    /// result is a uniform distribution, so downstream cumulative masses stay well-defined
    /// rather than producing NaNs.
    pub fn normalized(&self) -> Vec<f64> {
        if self.depth > 0.0 {
            self.counts.iter().map(|&c| c / self.depth).collect()
        } else {
            let d = self.counts.len().max(1);
            vec![1.0 / d as f64; self.counts.len()]
        }
    }
}

/// A set of `K` source profiles plus their taxon labels, all aligned to a common taxon order
/// (the tree's leaf order).
#[derive(Debug, Clone)]
pub struct SourceSet {
    /// Source sample names, length `K`.
    pub names: Vec<String>,
    /// Per-source profiles, length `K`, each of length `D`.
    pub profiles: Vec<Profile>,
}

impl SourceSet {
    pub fn num_sources(&self) -> usize {
        self.profiles.len()
    }

    /// The metacommunity profile: the mean of all normalized source profiles. Used as the
    /// fixed background source `b_0` under `--unknown metacommunity`.
    pub fn metacommunity(&self, num_taxa: usize) -> Vec<f64> {
        let mut acc = vec![0.0f64; num_taxa];
        let k = self.profiles.len().max(1);
        for prof in &self.profiles {
            for (i, b) in prof.normalized().iter().enumerate() {
                acc[i] += b;
            }
        }
        for v in &mut acc {
            *v /= k as f64;
        }
        acc
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn normalizes_to_unit_sum() {
        let p = Profile::from_counts(vec![1.0, 2.0, 1.0]);
        assert_abs_diff_eq!(p.depth, 4.0, epsilon = 1e-12);
        let n = p.normalized();
        assert_abs_diff_eq!(n.iter().sum::<f64>(), 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(n[1], 0.5, epsilon = 1e-12);
    }

    #[test]
    fn zero_depth_is_uniform() {
        let p = Profile::from_counts(vec![0.0, 0.0, 0.0, 0.0]);
        let n = p.normalized();
        assert_abs_diff_eq!(n.iter().sum::<f64>(), 1.0, epsilon = 1e-12);
        for v in n {
            assert_abs_diff_eq!(v, 0.25, epsilon = 1e-12);
        }
    }

    #[test]
    fn metacommunity_is_mean() {
        let ss = SourceSet {
            names: vec!["s1".into(), "s2".into()],
            profiles: vec![
                Profile::from_counts(vec![10.0, 0.0]),
                Profile::from_counts(vec![0.0, 10.0]),
            ],
        };
        let mc = ss.metacommunity(2);
        assert_abs_diff_eq!(mc[0], 0.5, epsilon = 1e-12);
        assert_abs_diff_eq!(mc[1], 0.5, epsilon = 1e-12);
    }
}
