//! Rooted phylogenetic tree: hand-rolled Newick parser + post-order cumulative-mass
//! computation.
//!
//! The key operation for the tree-Wasserstein loss is, for every edge `e`, the sum of a
//! per-leaf quantity over all leaves in the subtree below `e` (the *cumulative mass* `s_e`).
//! We compute it with a single post-order accumulation: each node's accumulated value is the
//! sum of its children's, and a leaf's is its own mass. The value flowing up the edge *above*
//! a node is exactly the subtree mass below that edge.
//!
//! We hand-roll the parser (per the project spec) to avoid a fragile external dependency —
//! Newick is a simple grammar.

use std::collections::HashMap;
use thiserror::Error;

/// Index of a node in [`Tree::nodes`].
pub type NodeId = usize;

/// A single node in the rooted tree.
#[derive(Debug, Clone)]
pub struct Node {
    /// Node label. Leaves carry taxon names; internal nodes may be unnamed (`None`).
    pub name: Option<String>,
    /// Length of the edge *above* this node (connecting it to its parent). The root has
    /// no parent edge; its value is unused (stored as 0.0).
    pub edge_length: f64,
    /// Parent node, or `None` for the root.
    pub parent: Option<NodeId>,
    /// Child nodes (empty for leaves).
    pub children: Vec<NodeId>,
}

impl Node {
    fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }
}

/// A rooted phylogenetic tree stored in a flat arena.
#[derive(Debug, Clone)]
pub struct Tree {
    pub nodes: Vec<Node>,
    pub root: NodeId,
    /// Nodes in post-order (children before parents). Precomputed once at parse time so
    /// repeated cumulative-mass computations (thousands, in the bootstrap) are cheap.
    post_order: Vec<NodeId>,
    /// Leaf node ids in a stable order.
    leaves: Vec<NodeId>,
    /// Map from leaf/taxon name to its node id.
    leaf_index: HashMap<String, NodeId>,
}

#[derive(Debug, Error)]
pub enum TreeError {
    #[error("Newick parse error at byte {pos}: {msg}")]
    Parse { pos: usize, msg: String },
    #[error("negative edge length {0} is not allowed")]
    NegativeEdge(f64),
    #[error("duplicate leaf name {0:?} in tree")]
    DuplicateLeaf(String),
    #[error("tree has no leaves")]
    NoLeaves,
}

impl Tree {
    /// Number of edges `E`. Every node except the root contributes exactly one edge (the one
    /// above it), so `E = nodes.len() - 1`. The root edge (total mass) contributes nothing to
    /// the tree-Wasserstein distance and is excluded here.
    pub fn num_edges(&self) -> usize {
        self.nodes.len() - 1
    }

    /// Leaf node ids, stable order.
    pub fn leaves(&self) -> &[NodeId] {
        &self.leaves
    }

    /// Number of leaves (taxa) `D`.
    pub fn num_leaves(&self) -> usize {
        self.leaves.len()
    }

    /// Look up a leaf node id by taxon name.
    pub fn leaf_by_name(&self, name: &str) -> Option<NodeId> {
        self.leaf_index.get(name).copied()
    }

    /// The non-root nodes in post-order — one per edge. Iterating this yields every edge
    /// exactly once, children before parents. This is the canonical edge order used to lay
    /// out the `ℓ_e`, `s_e`, and `M_ek` vectors/matrix.
    pub fn edge_order(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.post_order.iter().copied().filter(move |&n| n != self.root)
    }

    /// Edge lengths `ℓ_e` in [`Self::edge_order`] order.
    pub fn edge_lengths(&self) -> Vec<f64> {
        self.edge_order().map(|n| self.nodes[n].edge_length).collect()
    }

    /// Compute cumulative masses `s_e` (sum of `leaf_mass` over leaves below edge `e`) for
    /// every edge, in [`Self::edge_order`] order.
    ///
    /// `leaf_mass[i]` is the mass on the i-th leaf in [`Self::leaves`] order. A single
    /// post-order pass accumulates each node's subtree mass; the value at node `n` (for a
    /// non-root `n`) is the cumulative mass flowing up the edge above `n`.
    pub fn cumulative_masses(&self, leaf_mass: &[f64]) -> Vec<f64> {
        assert_eq!(
            leaf_mass.len(),
            self.leaves.len(),
            "leaf_mass length must equal number of leaves"
        );

        // acc[node] = sum of leaf_mass over leaves in the subtree rooted at node.
        let mut acc = vec![0.0f64; self.nodes.len()];
        // seed leaves
        for (li, &leaf_node) in self.leaves.iter().enumerate() {
            acc[leaf_node] = leaf_mass[li];
        }
        // post-order: add each node's accumulated value into its parent
        for &n in &self.post_order {
            if let Some(p) = self.nodes[n].parent {
                acc[p] += acc[n];
            }
        }
        // read out non-root nodes in edge order
        self.edge_order().map(|n| acc[n]).collect()
    }

    /// Serialize the tree back to a Newick string (with branch lengths, trailing `;`). Round-trips
    /// with [`Self::parse_newick`]. Used to export simulated trees to a `.nwk` file so the same
    /// tree the simulator built can be handed to the CLI (`--tree`) and to competing methods.
    pub fn to_newick(&self) -> String {
        let mut s = String::new();
        self.write_node(self.root, &mut s, true);
        s.push(';');
        s
    }

    fn write_node(&self, n: NodeId, out: &mut String, is_root: bool) {
        let node = &self.nodes[n];
        if !node.children.is_empty() {
            out.push('(');
            for (i, &c) in node.children.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                self.write_node(c, out, false);
            }
            out.push(')');
        }
        if let Some(name) = &node.name {
            out.push_str(name);
        }
        if !is_root {
            out.push(':');
            out.push_str(&format!("{}", node.edge_length));
        }
    }

    /// Parse a Newick string into a rooted tree.
    ///
    /// Handles: named/unnamed internal nodes, quoted labels (`'A_1'`), missing branch
    /// lengths (default 0.0), nested clades, a trailing semicolon, and whitespace. Rejects
    /// negative edge lengths and duplicate leaf names.
    pub fn parse_newick(input: &str) -> Result<Tree, TreeError> {
        let mut parser = NewickParser::new(input);
        let mut nodes: Vec<Node> = Vec::new();
        let root = parser.parse_subtree(&mut nodes, None)?;
        parser.skip_ws();
        // optional trailing ';'
        if parser.peek() == Some(b';') {
            parser.bump();
        }
        parser.skip_ws();
        if let Some(c) = parser.peek() {
            return Err(TreeError::Parse {
                pos: parser.pos,
                msg: format!("unexpected trailing character {:?}", c as char),
            });
        }

        Tree::finalize(nodes, root)
    }

    /// Build a **star tree**: every taxon hangs directly off the root with unit edge length.
    ///
    /// This is the explicit "no phylogeny" fallback. Under a star tree, each leaf edge's
    /// cumulative mass is just that leaf's own mass, so the tree-Wasserstein objective reduces
    /// to `Σ_j |p_j − q_j|` — plain L1 deconvolution. **Drift-robustness is lost** (there is no
    /// notion of "close relative"), so callers must surface this to the user. Useful when no
    /// tree is available or taxon IDs are non-informative (they need only be unique strings).
    pub fn build_star(taxa: &[String]) -> Result<Tree, TreeError> {
        if taxa.is_empty() {
            return Err(TreeError::NoLeaves);
        }
        let mut nodes = Vec::with_capacity(taxa.len() + 1);
        // root
        nodes.push(Node {
            name: None,
            edge_length: 0.0,
            parent: None,
            children: Vec::new(),
        });
        let root = 0;
        for name in taxa {
            let id = nodes.len();
            nodes.push(Node {
                name: Some(name.clone()),
                edge_length: 1.0,
                parent: Some(root),
                children: Vec::new(),
            });
            nodes[root].children.push(id);
        }
        Tree::finalize(nodes, root)
    }


    /// Shared construction tail: validate edge lengths, collect leaves + name index, and
    /// precompute the post-order. Used by every tree builder.
    fn finalize(nodes: Vec<Node>, root: NodeId) -> Result<Tree, TreeError> {
        for node in &nodes {
            if node.edge_length < 0.0 {
                return Err(TreeError::NegativeEdge(node.edge_length));
            }
        }

        let mut leaves = Vec::new();
        let mut leaf_index = HashMap::new();
        for (id, node) in nodes.iter().enumerate() {
            if node.is_leaf() {
                leaves.push(id);
                if let Some(name) = &node.name {
                    if leaf_index.insert(name.clone(), id).is_some() {
                        return Err(TreeError::DuplicateLeaf(name.clone()));
                    }
                }
            }
        }
        if leaves.is_empty() {
            return Err(TreeError::NoLeaves);
        }

        let post_order = compute_post_order(&nodes, root);

        Ok(Tree {
            nodes,
            root,
            post_order,
            leaves,
            leaf_index,
        })
    }
}

/// Iterative post-order traversal (avoids recursion-depth limits on deep trees).
fn compute_post_order(nodes: &[Node], root: NodeId) -> Vec<NodeId> {
    let mut order = Vec::with_capacity(nodes.len());
    // (node, next_child_index)
    let mut stack: Vec<(NodeId, usize)> = vec![(root, 0)];
    while let Some((node, child_idx)) = stack.last_mut().copied() {
        if child_idx < nodes[node].children.len() {
            stack.last_mut().unwrap().1 += 1;
            stack.push((nodes[node].children[child_idx], 0));
        } else {
            order.push(node);
            stack.pop();
        }
    }
    order
}

/// Recursive-descent Newick parser.
struct NewickParser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> NewickParser<'a> {
    fn new(input: &'a str) -> Self {
        NewickParser {
            bytes: input.as_bytes(),
            pos: 0,
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn bump(&mut self) -> Option<u8> {
        let c = self.peek();
        if c.is_some() {
            self.pos += 1;
        }
        c
    }

    fn skip_ws(&mut self) {
        while let Some(c) = self.peek() {
            if c.is_ascii_whitespace() {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    /// Parse a subtree, appending nodes to `nodes`, and return the id of its root node.
    fn parse_subtree(
        &mut self,
        nodes: &mut Vec<Node>,
        parent: Option<NodeId>,
    ) -> Result<NodeId, TreeError> {
        self.skip_ws();

        // Reserve this node's id up front so children can point to it.
        let my_id = nodes.len();
        nodes.push(Node {
            name: None,
            edge_length: 0.0,
            parent,
            children: Vec::new(),
        });

        if self.peek() == Some(b'(') {
            // internal node: parse a comma-separated child list
            self.bump(); // consume '('
            loop {
                let child = self.parse_subtree(nodes, Some(my_id))?;
                nodes[my_id].children.push(child);
                self.skip_ws();
                match self.peek() {
                    Some(b',') => {
                        self.bump();
                    }
                    Some(b')') => {
                        self.bump();
                        break;
                    }
                    other => {
                        return Err(TreeError::Parse {
                            pos: self.pos,
                            msg: format!(
                                "expected ',' or ')' in clade, found {:?}",
                                other.map(|c| c as char)
                            ),
                        });
                    }
                }
            }
        }

        // optional label (leaf name or internal-node label)
        let label = self.parse_label()?;
        if let Some(label) = label {
            nodes[my_id].name = Some(label);
        }

        // optional ':branch_length'
        self.skip_ws();
        if self.peek() == Some(b':') {
            self.bump();
            let len = self.parse_number()?;
            nodes[my_id].edge_length = len;
        }

        Ok(my_id)
    }

    /// Parse a node label: either a single-quoted string (with `''` escapes) or an unquoted
    /// token terminated by Newick punctuation. Returns `None` if no label present.
    fn parse_label(&mut self) -> Result<Option<String>, TreeError> {
        self.skip_ws();
        match self.peek() {
            Some(b'\'') => {
                // quoted label
                self.bump(); // opening quote
                let mut s = String::new();
                loop {
                    match self.bump() {
                        Some(b'\'') => {
                            // '' is an escaped quote
                            if self.peek() == Some(b'\'') {
                                self.bump();
                                s.push('\'');
                            } else {
                                break;
                            }
                        }
                        Some(c) => s.push(c as char),
                        None => {
                            return Err(TreeError::Parse {
                                pos: self.pos,
                                msg: "unterminated quoted label".into(),
                            })
                        }
                    }
                }
                Ok(Some(s))
            }
            Some(c) if is_unquoted_label_char(c) => {
                let start = self.pos;
                while let Some(c) = self.peek() {
                    if is_unquoted_label_char(c) {
                        self.pos += 1;
                    } else {
                        break;
                    }
                }
                let raw = &self.bytes[start..self.pos];
                // In unquoted Newick labels, '_' represents a space.
                let s: String = raw.iter().map(|&b| if b == b'_' { ' ' } else { b as char }).collect();
                Ok(Some(s.trim().to_string()))
            }
            _ => Ok(None),
        }
    }

    fn parse_number(&mut self) -> Result<f64, TreeError> {
        self.skip_ws();
        let start = self.pos;
        while let Some(c) = self.peek() {
            if c.is_ascii_digit()
                || c == b'.'
                || c == b'-'
                || c == b'+'
                || c == b'e'
                || c == b'E'
            {
                self.pos += 1;
            } else {
                break;
            }
        }
        let text = std::str::from_utf8(&self.bytes[start..self.pos]).unwrap_or("");
        text.parse::<f64>().map_err(|_| TreeError::Parse {
            pos: start,
            msg: format!("invalid branch length {:?}", text),
        })
    }
}

/// Characters allowed in an unquoted Newick label (anything that isn't structural punctuation
/// or whitespace).
fn is_unquoted_label_char(c: u8) -> bool {
    !matches!(
        c,
        b'(' | b')' | b',' | b':' | b';' | b'\'' | b'[' | b']'
    ) && !c.is_ascii_whitespace()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn parses_basic_tree() {
        let t = Tree::parse_newick("((A:0.1,B:0.2):0.3,(C:0.4,D:0.5):0.6);").unwrap();
        assert_eq!(t.num_leaves(), 4);
        // 7 nodes total (4 leaves + 2 internal + root), 6 edges.
        assert_eq!(t.nodes.len(), 7);
        assert_eq!(t.num_edges(), 6);
        for name in ["A", "B", "C", "D"] {
            assert!(t.leaf_by_name(name).is_some(), "missing leaf {name}");
        }
    }

    #[test]
    fn to_newick_round_trips() {
        // Re-parsing a serialized tree must reproduce leaf set, edge count, and cumulative masses.
        let t = Tree::parse_newick("((A:0.1,B:0.2):0.3,(C:0.4,D:0.5):0.6);").unwrap();
        let t2 = Tree::parse_newick(&t.to_newick()).unwrap();
        assert_eq!(t.num_leaves(), t2.num_leaves());
        assert_eq!(t.num_edges(), t2.num_edges());
        // masses over the same leaf order must match (both in their own leaves() order)
        let mass = |tr: &Tree| {
            let d = tr.num_leaves();
            let m: Vec<f64> = (0..d).map(|i| (i + 1) as f64).collect();
            tr.cumulative_masses(&m)
        };
        // Leaf order is deterministic from structure; the serialized tree keeps child order.
        assert_eq!(mass(&t), mass(&t2));
    }

    #[test]
    fn root_to_leaf_path_lengths() {
        let t = Tree::parse_newick("((A:0.1,B:0.2):0.3,(C:0.4,D:0.5):0.6);").unwrap();
        // path length A = 0.1 + 0.3 = 0.4 ; D = 0.5 + 0.6 = 1.1
        let path_len = |name: &str| -> f64 {
            let mut n = t.leaf_by_name(name).unwrap();
            let mut total = 0.0;
            while let Some(p) = t.nodes[n].parent {
                total += t.nodes[n].edge_length;
                n = p;
            }
            total
        };
        assert_abs_diff_eq!(path_len("A"), 0.4, epsilon = 1e-12);
        assert_abs_diff_eq!(path_len("D"), 1.1, epsilon = 1e-12);
    }

    #[test]
    fn cumulative_masses_sum_correctly() {
        let t = Tree::parse_newick("((A:1,B:1):1,(C:1,D:1):1);").unwrap();
        // leaves order is deterministic; find indices
        let order: Vec<String> = t
            .leaves()
            .iter()
            .map(|&n| t.nodes[n].name.clone().unwrap())
            .collect();
        // build a leaf mass vector p with the four leaves
        let mut p = vec![0.0; 4];
        // put 0.25 on each
        for i in 0..4 {
            p[i] = 0.25;
        }
        let s = t.cumulative_masses(&p);
        // every cumulative mass should be between 0 and 1
        for &v in &s {
            assert!(v >= -1e-12 && v <= 1.0 + 1e-12);
        }
        // The two internal clades (AB) and (CD) each carry 0.5.
        // Find them: the two edges whose cumulative mass is 0.5.
        let half_count = s.iter().filter(|&&v| (v - 0.5).abs() < 1e-9).count();
        assert_eq!(half_count, 2, "two clades should each carry mass 0.5 (order={order:?})");
        // The four leaf edges each carry 0.25.
        let quarter_count = s.iter().filter(|&&v| (v - 0.25).abs() < 1e-9).count();
        assert_eq!(quarter_count, 4);
    }

    #[test]
    fn handles_quoted_labels_and_missing_lengths() {
        // unnamed root, quoted label with underscore-as-literal, a leaf with no branch length
        let t = Tree::parse_newick("('A 1':0.1,B,C:0.5);").unwrap();
        assert_eq!(t.num_leaves(), 3);
        assert!(t.leaf_by_name("A 1").is_some(), "quoted label should be preserved");
        assert!(t.leaf_by_name("B").is_some());
        // B has no branch length -> defaults to 0.0
        let b = t.leaf_by_name("B").unwrap();
        assert_abs_diff_eq!(t.nodes[b].edge_length, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn underscore_becomes_space_in_unquoted() {
        let t = Tree::parse_newick("(Homo_sapiens:0.1,Pan:0.2);").unwrap();
        assert!(t.leaf_by_name("Homo sapiens").is_some());
    }

    #[test]
    fn rejects_negative_edge() {
        let err = Tree::parse_newick("(A:-0.1,B:0.2);");
        assert!(matches!(err, Err(TreeError::NegativeEdge(_))));
    }

    #[test]
    fn rejects_duplicate_leaf() {
        let err = Tree::parse_newick("(A:0.1,A:0.2);");
        assert!(matches!(err, Err(TreeError::DuplicateLeaf(_))));
    }

    #[test]
    fn single_leaf_tree() {
        // A degenerate tree: one leaf under a root.
        let t = Tree::parse_newick("(A:1.0);").unwrap();
        assert_eq!(t.num_leaves(), 1);
        let s = t.cumulative_masses(&[1.0]);
        // one edge (A above root), cumulative mass 1.0
        assert_eq!(s.len(), t.num_edges());
        assert_abs_diff_eq!(s[0], 1.0, epsilon = 1e-12);
    }

    #[test]
    fn star_tree_reduces_to_l1() {
        // Under a star tree, each edge's cumulative mass is a single leaf mass, so the
        // tree-Wasserstein objective equals the L1 distance between the two distributions.
        let taxa: Vec<String> = ["A", "B", "C", "D"].iter().map(|s| s.to_string()).collect();
        let t = Tree::build_star(&taxa).unwrap();
        assert_eq!(t.num_leaves(), 4);
        assert_eq!(t.num_edges(), 4); // 4 leaf edges, no internal edges
        let p = [0.4, 0.3, 0.2, 0.1];
        let q = [0.1, 0.2, 0.3, 0.4];
        let ell = t.edge_lengths(); // all 1.0
        let sp = t.cumulative_masses(&p);
        let sq = t.cumulative_masses(&q);
        // objective = Σ ℓ_e |sp_e - sq_e|  should equal Σ_j |p_j - q_j|
        let tw: f64 = ell
            .iter()
            .zip(sp.iter().zip(sq.iter()))
            .map(|(l, (a, b))| l * (a - b).abs())
            .sum();
        let l1: f64 = p.iter().zip(q.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert_abs_diff_eq!(tw, l1, epsilon = 1e-12);
    }

    #[test]
    fn star_tree_accepts_noninformative_ids() {
        // Arbitrary unique OTU ids work as leaf names.
        let taxa: Vec<String> = ["OTU_1", "OTU_2", "abc123hash"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let t = Tree::build_star(&taxa).unwrap();
        assert!(t.leaf_by_name("OTU_1").is_some());
        assert!(t.leaf_by_name("abc123hash").is_some());
    }

}
