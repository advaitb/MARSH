//! Build a tree from a count table when no phylogeny is available (OTST improvement).
//!
//! Motivation (from benchmarking): every published MST dataset ships raw count tables with **no
//! phylogenetic tree**, so OTST falls back to a star tree — under which the tree-Wasserstein
//! loss degenerates to plain L1 and the drift-robustness advantage vanishes. A *data-driven*
//! tree recovers real structure: taxa that co-vary across the source/sink samples are placed
//! close together, so the ground metric again makes "moving mass to a neighbor" cheap.
//!
//! Method: compute a pairwise distance between taxa from their abundance vectors across samples
//! (correlation distance `1 − ρ`, which groups co-occurring taxa), then agglomerate with UPGMA
//! (average linkage). The resulting dendrogram is emitted as Newick and parsed by [`Tree`].
//!
//! This is a heuristic surrogate for phylogeny — co-abundance ≠ evolutionary relatedness — but
//! it is far better than a star tree when no reference tree exists, and it needs no external
//! data. It is exposed via `--cluster-tree`.

use crate::tree::{Tree, TreeError};

/// Distance metric between taxa (rows of the sample matrix).
#[derive(Debug, Clone, Copy)]
pub enum TaxonDistance {
    /// `1 − Pearson correlation` of the two taxa's abundance vectors across samples. Co-varying
    /// taxa → small distance. This is the default: it captures "these taxa move together".
    Correlation,
    /// Euclidean distance between (relative) abundance vectors.
    Euclidean,
}

/// Build a tree by UPGMA clustering of taxa using their abundances across `samples`.
///
/// `taxa` are the leaf names (length D). `sample_abundances[s]` is the length-D abundance vector
/// of sample `s` (raw counts are fine; correlation is scale-free per taxon). At least one
/// sample is required. With fewer than 2 taxa a trivial tree is returned.
pub fn build_cluster_tree(
    taxa: &[String],
    sample_abundances: &[Vec<f64>],
    metric: TaxonDistance,
) -> Result<Tree, TreeError> {
    let d = taxa.len();
    if d == 0 {
        return Err(TreeError::NoLeaves);
    }
    if d == 1 {
        return Tree::parse_newick(&format!("({}:1.0);", sanitize(&taxa[0])));
    }

    // Build the D×S taxon-by-sample matrix (transpose of the sample vectors).
    let s = sample_abundances.len();
    let mut mat = vec![vec![0.0f64; s]; d];
    for (si, sample) in sample_abundances.iter().enumerate() {
        for ti in 0..d {
            mat[ti][si] = sample.get(ti).copied().unwrap_or(0.0);
        }
    }

    // Pairwise distance matrix.
    let mut dist = vec![vec![0.0f64; d]; d];
    for i in 0..d {
        for j in (i + 1)..d {
            let dij = match metric {
                TaxonDistance::Correlation => correlation_distance(&mat[i], &mat[j]),
                TaxonDistance::Euclidean => euclidean(&mat[i], &mat[j]),
            };
            dist[i][j] = dij;
            dist[j][i] = dij;
        }
    }

    let newick = upgma_newick(taxa, dist);
    Tree::parse_newick(&newick)
}

/// `1 − Pearson ρ`, clamped to [0, 2]. Zero-variance taxa (flat across samples) are treated as
/// maximally uninformative → distance 1 (uncorrelated).
fn correlation_distance(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    if n == 0.0 {
        return 1.0;
    }
    let ma = a.iter().sum::<f64>() / n;
    let mb = b.iter().sum::<f64>() / n;
    let mut cov = 0.0;
    let mut va = 0.0;
    let mut vb = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        let dx = x - ma;
        let dy = y - mb;
        cov += dx * dy;
        va += dx * dx;
        vb += dy * dy;
    }
    if va <= 1e-12 || vb <= 1e-12 {
        return 1.0;
    }
    let rho = cov / (va.sqrt() * vb.sqrt());
    (1.0 - rho).clamp(0.0, 2.0)
}

fn euclidean(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

/// Agglomerative UPGMA (average linkage). Produces a Newick string with branch lengths equal to
/// the half-difference of cluster heights (ultrametric), so path length between two leaves
/// reflects their clustering distance.
fn upgma_newick(taxa: &[String], mut dist: Vec<Vec<f64>>) -> String {
    let d = taxa.len();
    // Active clusters: each holds (newick_string, size, height).
    let mut active: Vec<Option<(String, usize, f64)>> = taxa
        .iter()
        .map(|t| Some((sanitize(t), 1usize, 0.0f64)))
        .collect();
    let mut alive: Vec<usize> = (0..d).collect();

    while alive.len() > 1 {
        // find the closest pair among alive clusters
        let mut best = f64::INFINITY;
        let (mut bi, mut bj) = (alive[0], alive[1]);
        for a_idx in 0..alive.len() {
            for b_idx in (a_idx + 1)..alive.len() {
                let (i, j) = (alive[a_idx], alive[b_idx]);
                if dist[i][j] < best {
                    best = dist[i][j];
                    bi = i;
                    bj = j;
                }
            }
        }

        let (ni, si, hi) = active[bi].clone().unwrap();
        let (nj, sj, hj) = active[bj].clone().unwrap();
        // new cluster height = half the linkage distance (ultrametric)
        let new_height = best / 2.0;
        // branch lengths from each child up to the merge node (non-negative)
        let bl_i = (new_height - hi).max(0.0);
        let bl_j = (new_height - hj).max(0.0);
        let merged = format!("({ni}:{bl_i:.6},{nj}:{bl_j:.6})");
        let new_size = si + sj;

        // UPGMA distance update: average by cluster size to all other alive clusters.
        for &k in &alive {
            if k == bi || k == bj {
                continue;
            }
            let dik = dist[bi][k];
            let djk = dist[bj][k];
            let new_d = (si as f64 * dik + sj as f64 * djk) / new_size as f64;
            dist[bi][k] = new_d;
            dist[k][bi] = new_d;
        }

        active[bi] = Some((merged, new_size, new_height));
        active[bj] = None;
        alive.retain(|&x| x != bj);
    }

    let root = alive[0];
    let (root_newick, _, _) = active[root].clone().unwrap();
    format!("{root_newick};")
}

/// Quote a taxon name for Newick if it contains characters the parser treats as structural.
fn sanitize(name: &str) -> String {
    let needs_quote = name
        .bytes()
        .any(|c| matches!(c, b'(' | b')' | b',' | b':' | b';' | b'\'' | b'[' | b']') || c.is_ascii_whitespace());
    if needs_quote {
        format!("'{}'", name.replace('\'', "''"))
    } else {
        name.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clusters_covarying_taxa_together() {
        // A and B co-vary (identical pattern); C and D co-vary; the two pairs are anti-phase.
        // Samples (rows) x taxa (cols A,B,C,D):
        let samples = vec![
            vec![10.0, 10.0, 0.0, 0.0],
            vec![8.0, 8.0, 1.0, 1.0],
            vec![0.0, 0.0, 10.0, 10.0],
            vec![1.0, 1.0, 8.0, 8.0],
        ];
        let taxa: Vec<String> = ["A", "B", "C", "D"].iter().map(|s| s.to_string()).collect();
        let tree = build_cluster_tree(&taxa, &samples, TaxonDistance::Correlation).unwrap();
        assert_eq!(tree.num_leaves(), 4);

        // A and B should share a parent (their LCA is deeper than A and C's LCA).
        let depth_to_lca = |x: &str, y: &str| -> usize {
            let path = |name: &str| -> Vec<usize> {
                let mut n = tree.leaf_by_name(name).unwrap();
                let mut p = vec![n];
                while let Some(par) = tree.nodes[n].parent {
                    p.push(par);
                    n = par;
                }
                p
            };
            let py: std::collections::HashSet<usize> = path(y).into_iter().collect();
            let px = path(x);
            // number of edges from x up to the LCA
            let mut steps = 0;
            for node in &px {
                if py.contains(node) {
                    break;
                }
                steps += 1;
            }
            steps
        };
        // A->B LCA should be reached in fewer steps than A->C LCA
        assert!(
            depth_to_lca("A", "B") <= depth_to_lca("A", "C"),
            "co-varying A,B should cluster closer than A,C"
        );
    }

    #[test]
    fn handles_two_taxa() {
        let taxa = vec!["X".to_string(), "Y".to_string()];
        let samples = vec![vec![1.0, 2.0], vec![3.0, 1.0]];
        let tree = build_cluster_tree(&taxa, &samples, TaxonDistance::Correlation).unwrap();
        assert_eq!(tree.num_leaves(), 2);
    }

    #[test]
    fn sanitizes_and_parses_odd_names() {
        let taxa = vec!["OTU 1".to_string(), "OTU:2".to_string(), "OTU_3".to_string()];
        let samples = vec![vec![1.0, 0.0, 2.0], vec![0.0, 3.0, 1.0], vec![2.0, 1.0, 0.0]];
        let tree = build_cluster_tree(&taxa, &samples, TaxonDistance::Correlation).unwrap();
        assert_eq!(tree.num_leaves(), 3);
    }
}
