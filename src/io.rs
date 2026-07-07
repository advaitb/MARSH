//! Reading count tables (TSV) and aligning taxa to the tree's leaves.
//!
//! Table format (tab-separated):
//! ```text
//! taxon    sampleA    sampleB    ...
//! OTU_1    10         0          ...
//! OTU_2    3          7          ...
//! ```
//! The first column is the taxon id; the header's first cell is a label we ignore. Remaining
//! columns are samples (sources, or a single sink).
//!
//! Alignment policy (spec §7 — fail loudly, document handling):
//! - Taxa present in the table but **absent from the tree** are, by default, an error
//!   (`OnMissing::Error`). With `OnMissing::Drop` they are dropped and reported.
//! - Taxa present in the tree but absent from the table are assigned count 0 for every sample
//!   (a taxon simply unobserved in these samples). This is always allowed.
//! - Everything is reordered to the tree's leaf order so cumulative-mass computation lines up.

use crate::profile::{Profile, SourceSet};
use crate::tree::Tree;
use std::collections::HashMap;
use std::path::Path;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum IoError {
    #[error("failed to read {path}: {source}")]
    Read {
        path: String,
        source: std::io::Error,
    },
    #[error("malformed table in {path}: {msg}")]
    Malformed { path: String, msg: String },
    #[error(
        "taxon {taxon:?} in {path} is not a leaf of the tree; \
         use --on-missing drop to drop such taxa"
    )]
    TaxonNotInTree { path: String, taxon: String },
    #[error("table {path} has no sample columns")]
    NoSamples { path: String },
}

/// What to do when a table row's taxon is not a leaf in the tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OnMissing {
    /// Abort with an error (default; safest).
    Error,
    /// Drop the offending row and continue.
    Drop,
}

/// A parsed count table: taxon ids (rows) × sample names (columns) of counts.
#[derive(Debug, Clone)]
pub struct CountTable {
    pub taxa: Vec<String>,
    pub sample_names: Vec<String>,
    /// `values[row][col]` — counts for taxon `row`, sample `col`.
    pub values: Vec<Vec<f64>>,
}

impl CountTable {
    /// Parse a TSV count table from a file.
    pub fn from_tsv_path<P: AsRef<Path>>(path: P) -> Result<Self, IoError> {
        let path_str = path.as_ref().display().to_string();
        let text = std::fs::read_to_string(&path).map_err(|source| IoError::Read {
            path: path_str.clone(),
            source,
        })?;
        Self::from_tsv_str(&text, &path_str)
    }

    /// Parse a TSV count table from a string (path used only for error messages).
    pub fn from_tsv_str(text: &str, path: &str) -> Result<Self, IoError> {
        let mut lines = text
            .lines()
            .filter(|l| !l.trim().is_empty() && !l.starts_with('#'));

        let header = lines.next().ok_or_else(|| IoError::Malformed {
            path: path.to_string(),
            msg: "empty table (no header)".into(),
        })?;
        let header_cells: Vec<&str> = header.split('\t').collect();
        if header_cells.len() < 2 {
            return Err(IoError::NoSamples {
                path: path.to_string(),
            });
        }
        let sample_names: Vec<String> = header_cells[1..].iter().map(|s| s.to_string()).collect();
        let ncol = sample_names.len();

        let mut taxa = Vec::new();
        let mut values = Vec::new();
        for (lineno, line) in lines.enumerate() {
            let cells: Vec<&str> = line.split('\t').collect();
            if cells.len() != ncol + 1 {
                return Err(IoError::Malformed {
                    path: path.to_string(),
                    msg: format!(
                        "row {} has {} columns, expected {}",
                        lineno + 2,
                        cells.len(),
                        ncol + 1
                    ),
                });
            }
            taxa.push(cells[0].to_string());
            let mut row = Vec::with_capacity(ncol);
            for (ci, cell) in cells[1..].iter().enumerate() {
                let v: f64 = cell.trim().parse().map_err(|_| IoError::Malformed {
                    path: path.to_string(),
                    msg: format!(
                        "row {} col {}: {:?} is not a number",
                        lineno + 2,
                        ci + 2,
                        cell
                    ),
                })?;
                if v < 0.0 {
                    return Err(IoError::Malformed {
                        path: path.to_string(),
                        msg: format!("row {} col {}: negative count {}", lineno + 2, ci + 2, v),
                    });
                }
                row.push(v);
            }
            values.push(row);
        }
        Ok(CountTable {
            taxa,
            sample_names,
            values,
        })
    }

    /// Number of sample columns.
    pub fn num_samples(&self) -> usize {
        self.sample_names.len()
    }
}

/// Report of how taxa aligned between a table and the tree.
#[derive(Debug, Clone, Default)]
pub struct AlignReport {
    /// Table taxa dropped because they weren't tree leaves (only with `OnMissing::Drop`).
    pub dropped_taxa: Vec<String>,
    /// Tree leaves that had no row in the table (assigned all-zero counts).
    pub unobserved_leaves: Vec<String>,
}

/// Align a source count table to the tree, producing one [`Profile`] per sample column with
/// counts in the tree's leaf order.
pub fn align_sources(
    table: &CountTable,
    tree: &Tree,
    on_missing: OnMissing,
    path: &str,
) -> Result<(SourceSet, AlignReport), IoError> {
    let (aligned, report) = align_columns(table, tree, on_missing, path)?;
    let profiles = aligned
        .into_iter()
        .map(Profile::from_counts)
        .collect::<Vec<_>>();
    Ok((
        SourceSet {
            names: table.sample_names.clone(),
            profiles,
        },
        report,
    ))
}

/// Align a sink count table (must have exactly one sample column) to the tree.
pub fn align_sink(
    table: &CountTable,
    tree: &Tree,
    on_missing: OnMissing,
    path: &str,
) -> Result<(Profile, AlignReport), IoError> {
    if table.num_samples() != 1 {
        return Err(IoError::Malformed {
            path: path.to_string(),
            msg: format!(
                "sink table must have exactly one sample column, found {}",
                table.num_samples()
            ),
        });
    }
    let (mut aligned, report) = align_columns(table, tree, on_missing, path)?;
    Ok((Profile::from_counts(aligned.remove(0)), report))
}

/// Core alignment: returns one count vector per sample column, each length D (tree leaf count),
/// in tree leaf order.
fn align_columns(
    table: &CountTable,
    tree: &Tree,
    on_missing: OnMissing,
    path: &str,
) -> Result<(Vec<Vec<f64>>, AlignReport), IoError> {
    let d = tree.num_leaves();
    let ncol = table.num_samples();
    let mut columns = vec![vec![0.0f64; d]; ncol];
    let mut report = AlignReport::default();

    // leaf name -> position in tree leaf order
    let leaf_pos: HashMap<&str, usize> = tree
        .leaves()
        .iter()
        .enumerate()
        .filter_map(|(i, &n)| tree.nodes[n].name.as_deref().map(|nm| (nm, i)))
        .collect();

    let mut seen = vec![false; d];
    for (row_idx, taxon) in table.taxa.iter().enumerate() {
        match leaf_pos.get(taxon.as_str()) {
            Some(&pos) => {
                seen[pos] = true;
                for (col, val) in columns.iter_mut().zip(table.values[row_idx].iter()) {
                    col[pos] += *val;
                }
            }
            None => match on_missing {
                OnMissing::Error => {
                    return Err(IoError::TaxonNotInTree {
                        path: path.to_string(),
                        taxon: taxon.clone(),
                    })
                }
                OnMissing::Drop => report.dropped_taxa.push(taxon.clone()),
            },
        }
    }

    // report tree leaves that never appeared in the table
    for (i, &n) in tree.leaves().iter().enumerate() {
        if !seen[i] {
            if let Some(name) = &tree.nodes[n].name {
                report.unobserved_leaves.push(name.clone());
            }
        }
    }

    Ok((columns, report))
}

#[cfg(test)]
mod tests {
    use super::*;

    const TREE: &str = "((A:0.1,B:0.2):0.3,(C:0.4,D:0.5):0.6);";

    #[test]
    fn parses_and_aligns_sources() {
        let tree = Tree::parse_newick(TREE).unwrap();
        let tsv = "taxon\ts1\ts2\nA\t10\t0\nB\t0\t5\nC\t2\t3\nD\t0\t0\n";
        let table = CountTable::from_tsv_str(tsv, "src").unwrap();
        assert_eq!(table.num_samples(), 2);
        let (ss, report) = align_sources(&table, &tree, OnMissing::Error, "src").unwrap();
        assert_eq!(ss.num_sources(), 2);
        // depth of s1 = 12, s2 = 8
        assert_eq!(ss.profiles[0].depth, 12.0);
        assert_eq!(ss.profiles[1].depth, 8.0);
        assert!(report.dropped_taxa.is_empty());
    }

    #[test]
    fn errors_on_taxon_not_in_tree() {
        let tree = Tree::parse_newick(TREE).unwrap();
        let tsv = "taxon\ts1\nA\t10\nZZZ\t5\n";
        let table = CountTable::from_tsv_str(tsv, "src").unwrap();
        let err = align_sources(&table, &tree, OnMissing::Error, "src");
        assert!(matches!(err, Err(IoError::TaxonNotInTree { .. })));
    }

    #[test]
    fn drops_taxon_not_in_tree_when_asked() {
        let tree = Tree::parse_newick(TREE).unwrap();
        let tsv = "taxon\ts1\nA\t10\nZZZ\t5\n";
        let table = CountTable::from_tsv_str(tsv, "src").unwrap();
        let (_ss, report) = align_sources(&table, &tree, OnMissing::Drop, "src").unwrap();
        assert_eq!(report.dropped_taxa, vec!["ZZZ".to_string()]);
    }

    #[test]
    fn reports_unobserved_leaves() {
        let tree = Tree::parse_newick(TREE).unwrap();
        // omit D
        let tsv = "taxon\ts1\nA\t10\nB\t1\nC\t2\n";
        let table = CountTable::from_tsv_str(tsv, "src").unwrap();
        let (_ss, report) = align_sources(&table, &tree, OnMissing::Error, "src").unwrap();
        assert_eq!(report.unobserved_leaves, vec!["D".to_string()]);
    }

    #[test]
    fn sink_requires_single_column() {
        let tree = Tree::parse_newick(TREE).unwrap();
        let tsv = "taxon\ts1\ts2\nA\t10\t0\n";
        let table = CountTable::from_tsv_str(tsv, "sink").unwrap();
        let err = align_sink(&table, &tree, OnMissing::Error, "sink");
        assert!(matches!(err, Err(IoError::Malformed { .. })));
    }
}
