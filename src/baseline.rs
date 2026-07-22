//! Non-OT ablation baseline: plain L2 (Euclidean) deconvolution on raw taxon abundances.
//!
//! Minimizes `‖ B w − p ‖²` over the simplex `{ w ≥ 0, Σ w = 1 }`, where `B` is the D×K matrix
//! of source profiles (columns) and `p` is the sink composition. This is the standard
//! least-squares deconvolution used (in spirit) by non-phylogenetic methods. It **ignores the
//! tree entirely** — mass moved to a phylogenetic neighbor is penalized exactly as much as mass
//! moved anywhere else. Comparing MARSH against this isolates the contribution of the
//! tree-Wasserstein ground metric (spec §6.4).
//!
//! Solved by projected gradient descent: the objective is convex with Lipschitz-continuous
//! gradient `∇f(w) = 2 Bᵀ(Bw − p)`, and Euclidean projection onto the probability simplex is
//! exact (Duchi et al., 2008). No extra solver dependency.

/// Deconvolve `sink` into non-negative simplex weights over `sources` under an L2 loss.
///
/// `sources[k]` is the length-D profile of source k (need not be normalized; treated as the
/// column `B[:,k]`). `sink` is length D. Returns weights of length K on the simplex.
pub fn l2_deconvolve(sources: &[Vec<f64>], sink: &[f64]) -> Vec<f64> {
    let k = sources.len();
    if k == 0 {
        return Vec::new();
    }
    let d = sink.len();
    debug_assert!(sources.iter().all(|s| s.len() == d));

    // Lipschitz constant of ∇f: L = 2·λ_max(BᵀB) ≤ 2·‖B‖_F². Use the Frobenius bound for a
    // safe (if slightly conservative) fixed step size 1/L.
    let frob_sq: f64 = sources.iter().flat_map(|s| s.iter()).map(|&x| x * x).sum();
    let l = (2.0 * frob_sq).max(1e-12);
    let step = 1.0 / l;

    // Start at the simplex centroid.
    let mut w = vec![1.0 / k as f64; k];
    let max_iters = 5000;
    let tol = 1e-10;

    for _ in 0..max_iters {
        // residual r = B w − p  (length D)
        let mut r = vec![0.0f64; d];
        for (kj, col) in sources.iter().enumerate() {
            let wk = w[kj];
            for i in 0..d {
                r[i] += wk * col[i];
            }
        }
        for i in 0..d {
            r[i] -= sink[i];
        }
        // gradient g_k = 2 · Σ_i B[i,k] · r[i]
        let mut g = vec![0.0f64; k];
        for (kj, col) in sources.iter().enumerate() {
            let mut acc = 0.0;
            for i in 0..d {
                acc += col[i] * r[i];
            }
            g[kj] = 2.0 * acc;
        }
        // gradient step + projection
        let mut w_new: Vec<f64> = w.iter().zip(g.iter()).map(|(&wk, &gk)| wk - step * gk).collect();
        project_to_simplex(&mut w_new);

        // convergence check
        let delta: f64 = w
            .iter()
            .zip(w_new.iter())
            .map(|(&a, &b)| (a - b).abs())
            .sum();
        w = w_new;
        if delta < tol {
            break;
        }
    }
    w
}

/// Public wrapper over [`project_to_simplex`] for reuse by the alternating unmixer's penalized
/// L2 weight step.
pub fn project_to_simplex_pub(v: &mut [f64]) {
    project_to_simplex(v);
}

/// Euclidean projection of `v` onto the probability simplex `{ x ≥ 0, Σ x = 1 }`, in place
/// (Duchi, Shalev-Shwartz, Singer & Chandra, 2008).
fn project_to_simplex(v: &mut [f64]) {
    let n = v.len();
    if n == 0 {
        return;
    }
    let mut u: Vec<f64> = v.to_vec();
    u.sort_by(|a, b| b.partial_cmp(a).unwrap()); // descending
    let mut css = 0.0;
    let mut rho = 0usize;
    let mut theta = 0.0;
    for (j, &uj) in u.iter().enumerate() {
        css += uj;
        let t = (css - 1.0) / (j as f64 + 1.0);
        if uj - t > 0.0 {
            rho = j + 1;
            theta = t;
        }
    }
    // theta from the largest rho satisfying the condition
    let _ = rho;
    for x in v.iter_mut() {
        *x = (*x - theta).max(0.0);
    }
}

/// FEAST-style EM baseline (a faithful reimplementation of the FEAST estimator's core, for
/// side-by-side comparison — spec §6.4).
///
/// FEAST (Shenhav et al., Nature Methods 2019) models the sink counts `y` (length D) as a
/// multinomial mixture of K known source profiles plus one **unknown** source whose profile is
/// estimated jointly. Mixing proportions `w` (length K+1, last entry = unknown) live on the
/// simplex. It maximizes the multinomial likelihood by Expectation-Maximization:
///
/// - E-step: for each taxon j, split its observed mass across components in proportion to
///   `w_c · B_cj` (responsibilities).
/// - M-step: `w_c ∝ Σ_j resp_cj`; the unknown profile `B_unknown,j ∝ Σ (mass assigned to
///   unknown at taxon j)`.
///
/// Like all non-phylogenetic methods it operates on raw taxon abundances with **no tree** —
/// which is exactly why it is not drift-robust. Returns the K named-source weights (the unknown
/// weight is dropped so the output aligns with the other estimators; use `feast_em_full` for the
/// unknown component).
pub fn feast_em(sources: &[Vec<f64>], sink: &[f64]) -> Vec<f64> {
    let full = feast_em_full(sources, sink, true);
    // drop the trailing unknown component
    full[..sources.len()].to_vec()
}

/// FEAST-style EM returning the full `[w_1..w_K, w_unknown]` weight vector.
/// If `with_unknown` is false, no unknown source is modeled (pure mixture of the K sources).
///
/// Faithful to the FEAST reference (`cozygene/FEAST`, `Infer_LatentVariables.R`). Two details
/// are what make FEAST stable — and are exactly what a naive EM gets wrong:
///
/// 1. **Unknown initialized to the RESIDUAL**, not uniform/sink:
///    `u_j = max(sink_j − Σ_k source_{k,j}, 0)` (FEAST `unknown_initialize_1`). A uniform or
///    sink-shaped init lets the unknown explain everything on iteration 1 → `w_unknown → 1`.
/// 2. **Known source profiles stay ANCHORED to their observed counts** every M-step
///    (`num_list[[i]] <- num + observed[[i]]`, with `observed = 0` only for the unknown). The
///    known profiles are effectively frozen to the input data; only the unknown profile is
///    freely re-estimated, from residual responsibility. Without this anchor the free unknown
///    drifts to match the sink and absorbs all mass.
pub fn feast_em_full(sources: &[Vec<f64>], sink: &[f64], with_unknown: bool) -> Vec<f64> {
    let k = sources.len();
    if k == 0 {
        return Vec::new();
    }
    let d = sink.len();
    let n_comp = if with_unknown { k + 1 } else { k };

    // Sink as a distribution (FEAST: rel_sink <- sink / sum(sink)).
    let sink_total: f64 = sink.iter().sum();
    let y: Vec<f64> = if sink_total > 0.0 {
        sink.iter().map(|&v| v / sink_total).collect()
    } else {
        vec![1.0 / d as f64; d]
    };

    // Known source profiles, normalized to sum 1. These stay FIXED (FEAST anchors them to the
    // observed counts via the `+ observed` term, which pins them to their input distribution).
    let b_known: Vec<Vec<f64>> = sources
        .iter()
        .map(|s| {
            let t: f64 = s.iter().sum();
            if t > 0.0 {
                s.iter().map(|&v| v / t).collect()
            } else {
                vec![1.0 / d as f64; d]
            }
        })
        .collect();

    // Per-taxon total across known sources (for the residual init).
    let mut src_sum = vec![0.0f64; d];
    for prof in &b_known {
        for j in 0..d {
            src_sum[j] += prof[j];
        }
    }

    // Working profiles: known (fixed) + unknown (re-estimated).
    let mut b = b_known.clone();
    if with_unknown {
        // FEAST unknown_initialize_1: residual of the sink the known sources can't explain.
        let mut u: Vec<f64> = (0..d).map(|j| (y[j] - src_sum[j]).max(0.0)).collect();
        let usum: f64 = u.iter().sum();
        if usum > 0.0 {
            for v in &mut u {
                *v /= usum;
            }
        } else {
            // sink fully explained by knowns at init → start unknown uniform (it will shrink)
            u = vec![1.0 / d as f64; d];
        }
        b.push(u);
    }

    // Mixing weights, initialized uniform (FEAST uses random; uniform is deterministic and the
    // EM is convex-ish enough here that the fixed point is the same).
    let mut w = vec![1.0 / n_comp as f64; n_comp];

    let max_iters = 2000;
    let tol = 1e-6; // FEAST convergence tolerance on the proportions
    let eps = 1e-12;

    for _ in 0..max_iters {
        // E-step responsibilities r[c][j] ∝ w_c * b_c[j]; accumulate M-step statistics.
        let mut w_new = vec![0.0f64; n_comp];
        let mut unknown_mass = vec![0.0f64; d]; // responsibility-weighted mass to the unknown

        for j in 0..d {
            let yj = y[j];
            if yj <= 0.0 {
                continue;
            }
            let mut denom = 0.0;
            let mut resp = vec![0.0f64; n_comp];
            for c in 0..n_comp {
                let val = w[c] * b[c][j];
                resp[c] = val;
                denom += val;
            }
            if denom <= eps {
                continue;
            }
            for c in 0..n_comp {
                let assigned = yj * resp[c] / denom;
                w_new[c] += assigned;
                if with_unknown && c == n_comp - 1 {
                    unknown_mass[j] += assigned;
                }
            }
        }

        // M-step: normalize the mixing weights.
        let wsum: f64 = w_new.iter().sum();
        if wsum > 0.0 {
            for c in 0..n_comp {
                w_new[c] /= wsum;
            }
        }

        // M-step: re-estimate ONLY the unknown profile from its residual responsibility
        // (known profiles remain anchored to their input distributions — the FEAST `+observed`
        // effect). `observed = 0` for the unknown, so its new profile ∝ unknown_mass.
        if with_unknown {
            let um: f64 = unknown_mass.iter().sum();
            if um > eps {
                let last = n_comp - 1;
                for j in 0..d {
                    b[last][j] = unknown_mass[j] / um;
                }
            }
        }

        // Convergence on the unknown/first proportion (FEAST tracks |Δ alpha| ≤ 1e-6).
        let delta: f64 = w.iter().zip(w_new.iter()).map(|(&a, &c)| (a - c).abs()).sum();
        w = w_new;
        if delta < tol {
            break;
        }
    }
    w
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn projection_sums_to_one_and_nonneg() {
        let mut v = vec![0.5, 0.2, -0.3, 1.4];
        project_to_simplex(&mut v);
        assert_abs_diff_eq!(v.iter().sum::<f64>(), 1.0, epsilon = 1e-9);
        assert!(v.iter().all(|&x| x >= -1e-12));
    }

    #[test]
    fn projection_of_simplex_point_is_identity() {
        let mut v = vec![0.3, 0.3, 0.4];
        let orig = v.clone();
        project_to_simplex(&mut v);
        for (a, b) in v.iter().zip(orig.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-9);
        }
    }

    #[test]
    fn recovers_exact_l2_mixture() {
        // sources on 4 taxa; sink = 0.7 s1 + 0.3 s2 exactly.
        let s1 = vec![0.8, 0.2, 0.0, 0.0];
        let s2 = vec![0.0, 0.0, 0.2, 0.8];
        let sink: Vec<f64> = (0..4).map(|i| 0.7 * s1[i] + 0.3 * s2[i]).collect();
        let w = l2_deconvolve(&[s1, s2], &sink);
        assert_abs_diff_eq!(w[0], 0.7, epsilon = 1e-3);
        assert_abs_diff_eq!(w[1], 0.3, epsilon = 1e-3);
    }

    #[test]
    fn weights_stay_on_simplex() {
        let s1 = vec![0.6, 0.3, 0.1];
        let s2 = vec![0.1, 0.3, 0.6];
        let s3 = vec![0.3, 0.4, 0.3];
        let sink = vec![0.4, 0.3, 0.3];
        let w = l2_deconvolve(&[s1, s2, s3], &sink);
        assert_abs_diff_eq!(w.iter().sum::<f64>(), 1.0, epsilon = 1e-6);
        assert!(w.iter().all(|&x| x >= -1e-9));
    }

    #[test]
    fn feast_recovers_exact_mixture_no_unknown() {
        // sink = 0.7 s1 + 0.3 s2 exactly; distinct supports -> identifiable.
        let s1 = vec![0.8, 0.2, 0.0, 0.0];
        let s2 = vec![0.0, 0.0, 0.2, 0.8];
        let sink: Vec<f64> = (0..4).map(|i| 0.7 * s1[i] + 0.3 * s2[i]).collect();
        let w = feast_em_full(&[s1, s2], &sink, false);
        assert_abs_diff_eq!(w[0], 0.7, epsilon = 1e-3);
        assert_abs_diff_eq!(w[1], 0.3, epsilon = 1e-3);
        assert_abs_diff_eq!(w.iter().sum::<f64>(), 1.0, epsilon = 1e-9);
    }

    #[test]
    fn feast_weights_on_simplex_with_unknown() {
        let s1 = vec![0.7, 0.2, 0.1, 0.0];
        let s2 = vec![0.0, 0.1, 0.2, 0.7];
        let sink = vec![0.3, 0.2, 0.2, 0.3];
        let w = feast_em_full(&[s1, s2], &sink, true);
        assert_eq!(w.len(), 3); // K + unknown
        assert_abs_diff_eq!(w.iter().sum::<f64>(), 1.0, epsilon = 1e-6);
        assert!(w.iter().all(|&x| x >= -1e-9));
    }

    #[test]
    fn feast_detects_unknown_when_sink_has_novel_taxon() {
        // Sink has mass on taxon 3 that neither source covers -> unknown should absorb it.
        let s1 = vec![0.5, 0.5, 0.0, 0.0];
        let s2 = vec![0.5, 0.5, 0.0, 0.0];
        let sink = vec![0.25, 0.25, 0.0, 0.5]; // half on the novel taxon 3
        let w = feast_em_full(&[s1, s2], &sink, true);
        // unknown weight (last) should be substantial (~0.5)
        assert!(w[2] > 0.3, "unknown weight {} should capture novel mass", w[2]);
    }

    #[test]
    fn feast_em_matches_l2_public_api() {
        // feast_em drops the unknown component and returns K weights
        let s1 = vec![0.8, 0.2, 0.0, 0.0];
        let s2 = vec![0.0, 0.0, 0.2, 0.8];
        let sink: Vec<f64> = (0..4).map(|i| 0.6 * s1[i] + 0.4 * s2[i]).collect();
        let w = feast_em(&[s1, s2], &sink);
        assert_eq!(w.len(), 2);
    }

    #[test]
    fn feast_unknown_captures_genuine_residual() {
        // Sanity check on the in-repo EM ablation: when ~15% of the sink lies on a novel taxon
        // that no known source contains, the unknown should capture a meaningful share of it
        // (and not collapse to ~1.0). This is an ablation baseline, not a substitute for the
        // real FEAST R package (which the benchmark runs directly).
        let s1 = vec![0.8, 0.2, 0.0, 0.0, 0.0];
        let s2 = vec![0.0, 0.0, 0.7, 0.3, 0.0];
        let mut sink = vec![0.0; 5];
        for i in 0..5 {
            sink[i] = 0.5 * s1[i] + 0.35 * s2[i];
        }
        sink[4] += 0.15; // novel-taxon residual
        let w = feast_em_full(&[s1, s2], &sink, true);
        assert!(w[2] > 0.05, "unknown should capture novel-taxon mass, got {}", w[2]);
        assert!(w[2] < 0.95, "unknown must not swallow everything, got {}", w[2]);
    }
}
