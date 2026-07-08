//! Non-OT ablation baseline: plain L2 (Euclidean) deconvolution on raw taxon abundances.
//!
//! Minimizes `‖ B w − p ‖²` over the simplex `{ w ≥ 0, Σ w = 1 }`, where `B` is the D×K matrix
//! of source profiles (columns) and `p` is the sink composition. This is the standard
//! least-squares deconvolution used (in spirit) by non-phylogenetic methods. It **ignores the
//! tree entirely** — mass moved to a phylogenetic neighbor is penalized exactly as much as mass
//! moved anywhere else. Comparing OTST against this isolates the contribution of the
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
}
