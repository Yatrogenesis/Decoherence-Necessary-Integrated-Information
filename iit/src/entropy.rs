//! Information-Theoretic Measures
//!
//! Implements Shannon entropy, mutual information, and related quantities
//! for computing integrated information.
//!
//! ## Measures
//!
//! - **Shannon entropy**: H(X) = -Σ p(x) log₂ p(x)
//! - **Joint entropy**: H(X,Y) = -Σ p(x,y) log₂ p(x,y)
//! - **Conditional entropy**: H(X|Y) = H(X,Y) - H(Y)
//! - **Mutual information**: I(X;Y) = H(X) + H(Y) - H(X,Y)
//!
//! ## References
//!
//! ### Information Theory Fundamentals
//! - Shannon, C. E. (1948).
//!   "A Mathematical Theory of Communication."
//!   Bell System Technical Journal, 27(3), 379-423.
//!   DOI: 10.1002/j.1538-7305.1948.tb01338.x
//!
//! - Cover, T. M., & Thomas, J. A. (2006).
//!   "Elements of Information Theory" (2nd ed.).
//!   Wiley-Interscience. ISBN: 978-0471241959
//!
//! ### Application to Consciousness
//! - Tononi, G. (2008).
//!   "Consciousness as Integrated Information: a Provisional Manifesto."
//!   Biological Bulletin, 215(3), 216-242.
//!   DOI: 10.2307/25470707
//!
//! ## Author
//!
//! Francisco Molina-Burgos
//! Avermex Research Division
//! fmolina@avermex.com

use crate::{IITError, IITResult};

/// Compute Shannon entropy: H(X) = -Σ p(x) log₂ p(x)
///
/// # Arguments
/// * `distribution` - Probability distribution (must sum to 1)
///
/// # Returns
/// Entropy in bits
///
/// # Reference
/// - Shannon, C. E. (1948). Bell Syst. Tech. J. 27, 379-423.
/// - Cover & Thomas (2006), Eq. 2.1
pub fn shannon_entropy(distribution: &[f64]) -> IITResult<f64> {
    validate_distribution(distribution)?;

    let mut h = 0.0;
    for &p in distribution {
        if p > 1e-15 {
            h -= p * p.log2();
        }
    }

    Ok(h.max(0.0))  // Numerical safety
}

/// Compute joint entropy: H(X,Y) = -Σ p(x,y) log₂ p(x,y)
///
/// # Arguments
/// * `joint` - Joint probability distribution p(x,y) as flattened array
/// * `dim_x` - Dimension of X
/// * `dim_y` - Dimension of Y
///
/// # Reference
/// Cover & Thomas (2006), Eq. 2.8
pub fn joint_entropy(joint: &[f64], dim_x: usize, dim_y: usize) -> IITResult<f64> {
    if joint.len() != dim_x * dim_y {
        return Err(IITError::DimensionMismatch {
            expected: dim_x * dim_y,
            actual: joint.len(),
        });
    }

    shannon_entropy(joint)
}

/// Compute conditional entropy: H(X|Y) = H(X,Y) - H(Y)
///
/// Measures the uncertainty in X given knowledge of Y.
///
/// # Arguments
/// * `joint` - Joint distribution p(x,y)
/// * `marginal_y` - Marginal distribution p(y)
/// * `dim_x` - Dimension of X
/// * `dim_y` - Dimension of Y
///
/// # Reference
/// Cover & Thomas (2006), Eq. 2.10
pub fn conditional_entropy(
    joint: &[f64],
    marginal_y: &[f64],
    dim_x: usize,
    dim_y: usize,
) -> IITResult<f64> {
    let h_xy = joint_entropy(joint, dim_x, dim_y)?;
    let h_y = shannon_entropy(marginal_y)?;

    Ok((h_xy - h_y).max(0.0))
}

/// Compute mutual information: I(X;Y) = H(X) + H(Y) - H(X,Y)
///
/// Measures the total correlation between X and Y.
/// This is the foundation for computing integrated information.
///
/// # Properties
/// - I(X;Y) ≥ 0
/// - I(X;Y) = I(Y;X)
/// - I(X;Y) = 0 iff X and Y are independent
///
/// # Reference
/// Cover & Thomas (2006), Eq. 2.28
pub fn mutual_information(
    joint: &[f64],
    marginal_x: &[f64],
    marginal_y: &[f64],
) -> IITResult<f64> {
    let h_x = shannon_entropy(marginal_x)?;
    let h_y = shannon_entropy(marginal_y)?;
    let h_xy = shannon_entropy(joint)?;

    Ok((h_x + h_y - h_xy).max(0.0))
}

/// Compute mutual information directly from joint distribution
///
/// I(X;Y) = Σ p(x,y) log₂[p(x,y) / (p(x)p(y))]
///
/// More numerically stable for sparse distributions.
///
/// # Reference
/// Cover & Thomas (2006), Eq. 2.24
pub fn mutual_information_direct(
    joint: &[f64],
    dim_x: usize,
    dim_y: usize,
) -> IITResult<f64> {
    if joint.len() != dim_x * dim_y {
        return Err(IITError::DimensionMismatch {
            expected: dim_x * dim_y,
            actual: joint.len(),
        });
    }

    // Compute marginals
    let marginal_x = marginalize_y(joint, dim_x, dim_y);
    let marginal_y = marginalize_x(joint, dim_x, dim_y);

    let mut mi = 0.0;
    for x in 0..dim_x {
        for y in 0..dim_y {
            let p_xy = joint[x * dim_y + y];
            let p_x = marginal_x[x];
            let p_y = marginal_y[y];

            if p_xy > 1e-15 && p_x > 1e-15 && p_y > 1e-15 {
                mi += p_xy * (p_xy / (p_x * p_y)).log2();
            }
        }
    }

    Ok(mi.max(0.0))
}

/// Marginalize over Y: p(x) = Σ_y p(x,y)
fn marginalize_y(joint: &[f64], dim_x: usize, dim_y: usize) -> Vec<f64> {
    let mut marginal = vec![0.0; dim_x];
    for x in 0..dim_x {
        for y in 0..dim_y {
            marginal[x] += joint[x * dim_y + y];
        }
    }
    marginal
}

/// Marginalize over X: p(y) = Σ_x p(x,y)
fn marginalize_x(joint: &[f64], dim_x: usize, dim_y: usize) -> Vec<f64> {
    let mut marginal = vec![0.0; dim_y];
    for x in 0..dim_x {
        for y in 0..dim_y {
            marginal[y] += joint[x * dim_y + y];
        }
    }
    marginal
}

/// Compute KL divergence: D_KL(P || Q) = Σ p(x) log₂[p(x) / q(x)]
///
/// Measures how different Q is from P (asymmetric!).
///
/// # Reference
/// Cover & Thomas (2006), Eq. 2.24
pub fn kl_divergence(p: &[f64], q: &[f64]) -> IITResult<f64> {
    if p.len() != q.len() {
        return Err(IITError::DimensionMismatch {
            expected: p.len(),
            actual: q.len(),
        });
    }

    let mut kl = 0.0;
    for (&pi, &qi) in p.iter().zip(q.iter()) {
        if pi > 1e-15 {
            if qi < 1e-15 {
                return Ok(f64::INFINITY);
            }
            kl += pi * (pi / qi).log2();
        }
    }

    Ok(kl)
}

/// Compute Jensen-Shannon divergence: JSD(P || Q)
///
/// Symmetric version of KL divergence:
/// JSD(P || Q) = ½ D_KL(P || M) + ½ D_KL(Q || M)
/// where M = ½(P + Q)
///
/// # Properties
/// - 0 ≤ JSD ≤ 1 (in bits)
/// - JSD(P || Q) = JSD(Q || P)
pub fn js_divergence(p: &[f64], q: &[f64]) -> IITResult<f64> {
    if p.len() != q.len() {
        return Err(IITError::DimensionMismatch {
            expected: p.len(),
            actual: q.len(),
        });
    }

    // M = (P + Q) / 2
    let m: Vec<f64> = p.iter().zip(q.iter()).map(|(&pi, &qi)| (pi + qi) / 2.0).collect();

    let kl_pm = kl_divergence(p, &m)?;
    let kl_qm = kl_divergence(q, &m)?;

    Ok(0.5 * (kl_pm + kl_qm))
}

/// Validate probability distribution
fn validate_distribution(dist: &[f64]) -> IITResult<()> {
    if dist.is_empty() {
        return Err(IITError::InvalidDistribution("Empty distribution".to_string()));
    }

    let sum: f64 = dist.iter().sum();
    if (sum - 1.0).abs() > 1e-6 {
        return Err(IITError::InvalidDistribution(
            format!("Distribution sums to {}, expected 1.0", sum)
        ));
    }

    for &p in dist {
        if p < -1e-10 {
            return Err(IITError::InvalidDistribution(
                format!("Negative probability: {}", p)
            ));
        }
    }

    Ok(())
}

/// Compute total correlation (multi-information): TC(X₁,...,Xₙ)
///
/// TC = Σᵢ H(Xᵢ) - H(X₁,...,Xₙ)
///
/// Measures total statistical dependence among all variables.
/// NOT the same as integrated information (does not involve partitioning).
///
/// # Arguments
/// * `joint` - Joint distribution of all variables
/// * `marginals` - Marginal distributions of each variable
///
/// # Reference
/// Watanabe, S. (1960). IBM Journal of Research and Development, 4(1), 66-82.
pub fn total_correlation(joint: &[f64], marginals: &[Vec<f64>]) -> IITResult<f64> {
    let h_joint = shannon_entropy(joint)?;

    let mut h_marginals_sum = 0.0;
    for marginal in marginals {
        h_marginals_sum += shannon_entropy(marginal)?;
    }

    Ok((h_marginals_sum - h_joint).max(0.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_entropy_uniform() {
        // Uniform over 4 states: H = log₂(4) = 2 bits
        let p = vec![0.25, 0.25, 0.25, 0.25];
        let h = shannon_entropy(&p).unwrap();
        assert_relative_eq!(h, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_entropy_deterministic() {
        // Deterministic: H = 0
        let p = vec![1.0, 0.0, 0.0, 0.0];
        let h = shannon_entropy(&p).unwrap();
        assert_relative_eq!(h, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_mutual_information_independent() {
        // Independent: I(X;Y) = 0
        let joint = vec![0.25, 0.25, 0.25, 0.25];  // 2×2, uniform
        let marginal_x = vec![0.5, 0.5];
        let marginal_y = vec![0.5, 0.5];

        let mi = mutual_information(&joint, &marginal_x, &marginal_y).unwrap();
        assert_relative_eq!(mi, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_mutual_information_perfect_correlation() {
        // Perfect correlation: X = Y
        // p(0,0) = p(1,1) = 0.5, others = 0
        let joint = vec![0.5, 0.0, 0.0, 0.5];
        let marginal_x = vec![0.5, 0.5];
        let marginal_y = vec![0.5, 0.5];

        let mi = mutual_information(&joint, &marginal_x, &marginal_y).unwrap();
        assert_relative_eq!(mi, 1.0, epsilon = 1e-10);  // 1 bit
    }

    #[test]
    fn test_kl_divergence_same() {
        let p = vec![0.5, 0.5];
        let kl = kl_divergence(&p, &p).unwrap();
        assert_relative_eq!(kl, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_js_divergence_symmetric() {
        let p = vec![0.8, 0.2];
        let q = vec![0.2, 0.8];

        let js_pq = js_divergence(&p, &q).unwrap();
        let js_qp = js_divergence(&q, &p).unwrap();

        assert_relative_eq!(js_pq, js_qp, epsilon = 1e-10);
        assert!(js_pq > 0.0);
        assert!(js_pq <= 1.0);
    }
}
