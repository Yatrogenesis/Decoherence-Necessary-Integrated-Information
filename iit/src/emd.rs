//! Earth Mover's Distance (Wasserstein-1 Distance)
//!
//! Implements the REAL EMD via optimal transport using linear programming.
//! This is the correct metric for IIT calculations, not just L1 distance.
//!
//! ## Mathematical Definition
//!
//! ```text
//! EMD(P, Q) = min_{γ∈Γ(P,Q)} Σᵢⱼ γᵢⱼ cᵢⱼ
//! ```
//!
//! Subject to:
//! - Σⱼ γᵢⱼ = pᵢ (supply constraint)
//! - Σᵢ γᵢⱼ = qⱼ (demand constraint)
//! - γᵢⱼ ≥ 0
//!
//! Where cᵢⱼ is the ground distance (typically Hamming distance).
//!
//! ## Key Difference from Simple L1
//!
//! L1 distance: d(P,Q) = ½Σ|pᵢ - qᵢ|
//! This is NOT EMD unless the ground distance is identity matrix!
//!
//! ## References
//!
//! ### Optimal Transport
//! - Kantorovich, L. V. (1942).
//!   "On the translocation of masses."
//!   Doklady Akademii Nauk USSR, 37(7-8), 227-229.
//!
//! - Villani, C. (2008).
//!   "Optimal Transport: Old and New."
//!   Springer. ISBN: 978-3540710493
//!
//! ### Earth Mover's Distance
//! - Rubner, Y., Tomasi, C., & Guibas, L. J. (2000).
//!   "The Earth Mover's Distance as a Metric for Image Retrieval."
//!   International Journal of Computer Vision, 40(2), 99-121.
//!   DOI: 10.1023/A:1026543900054
//!
//! ### Computational Methods
//! - Peyré, G., & Cuturi, M. (2019).
//!   "Computational Optimal Transport."
//!   Foundations and Trends in Machine Learning, 11(5-6), 355-607.
//!   DOI: 10.1561/2200000073
//!
//! ### Application to IIT
//! - Oizumi, M., et al. (2014).
//!   "From the Phenomenology to the Mechanisms of Consciousness."
//!   PLOS Computational Biology, 10(5), e1003588.
//!
//! ## Author
//!
//! Francisco Molina-Burgos
//! Avermex Research Division
//! fmolina@avermex.com

use crate::{IITError, IITResult};
use minilp::{Problem, OptimizationDirection, ComparisonOp};

/// Ground distance metric for EMD
#[derive(Debug, Clone, Copy)]
pub enum GroundDistance {
    /// Hamming distance: number of differing bits
    /// This is the standard for IIT calculations
    Hamming,
    /// Euclidean distance in state space
    Euclidean,
    /// Discrete: 0 if same, 1 if different
    Discrete,
    /// Manhattan (L1) in decoded indices
    Manhattan,
}

/// Compute Earth Mover's Distance using linear programming
///
/// This solves the optimal transport problem EXACTLY.
///
/// # Arguments
/// * `p` - Source distribution
/// * `q` - Target distribution
/// * `cost_matrix` - Cost c[i][j] of moving mass from i to j
///
/// # Returns
/// EMD value (minimum transport cost)
///
/// # Complexity
/// O(n²) space, O(n³) time for LP solving
///
/// # Reference
/// Rubner et al. (2000), Section 3
pub fn emd_linear_programming(
    p: &[f64],
    q: &[f64],
    cost_matrix: &[Vec<f64>],
) -> IITResult<f64> {
    let n = p.len();
    let m = q.len();

    if cost_matrix.len() != n {
        return Err(IITError::DimensionMismatch {
            expected: n,
            actual: cost_matrix.len(),
        });
    }

    // Build LP problem
    // Minimize: Σᵢⱼ cᵢⱼ γᵢⱼ
    // Subject to:
    // Σⱼ γᵢⱼ = pᵢ for all i (supply)
    // Σᵢ γᵢⱼ = qⱼ for all j (demand)
    // γᵢⱼ ≥ 0

    let mut problem = Problem::new(OptimizationDirection::Minimize);

    // Create variables γᵢⱼ with costs cᵢⱼ
    let mut gamma = Vec::new();
    for i in 0..n {
        let mut row = Vec::new();
        for j in 0..m {
            let cost = cost_matrix[i][j];
            let var = problem.add_var(cost, (0.0, f64::INFINITY));
            row.push(var);
        }
        gamma.push(row);
    }

    // Supply constraints: Σⱼ γᵢⱼ = pᵢ
    for i in 0..n {
        let coeffs: Vec<_> = gamma[i].iter().map(|&var| (var, 1.0)).collect();
        problem.add_constraint(&coeffs, ComparisonOp::Eq, p[i]);
    }

    // Demand constraints: Σᵢ γᵢⱼ = qⱼ
    for j in 0..m {
        let coeffs: Vec<_> = (0..n).map(|i| (gamma[i][j], 1.0)).collect();
        problem.add_constraint(&coeffs, ComparisonOp::Eq, q[j]);
    }

    // Solve
    match problem.solve() {
        Ok(solution) => Ok(solution.objective()),
        Err(_) => Err(IITError::EMDError(
            "Linear programming solver failed".to_string()
        )),
    }
}

/// Compute EMD with automatic cost matrix construction
///
/// # Arguments
/// * `p` - Source distribution
/// * `q` - Target distribution
/// * `distance` - Ground distance metric
/// * `n_bits` - Number of bits for Hamming distance (if applicable)
pub fn emd(
    p: &[f64],
    q: &[f64],
    distance: GroundDistance,
    n_bits: Option<usize>,
) -> IITResult<f64> {
    if p.len() != q.len() {
        return Err(IITError::DimensionMismatch {
            expected: p.len(),
            actual: q.len(),
        });
    }

    let n = p.len();
    let n_bits = n_bits.unwrap_or_else(|| (n as f64).log2().ceil() as usize);

    // Build cost matrix
    let cost_matrix: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| ground_distance(i, j, distance, n_bits))
                .collect()
        })
        .collect();

    emd_linear_programming(p, q, &cost_matrix)
}

/// Compute ground distance between states
fn ground_distance(i: usize, j: usize, metric: GroundDistance, n_bits: usize) -> f64 {
    match metric {
        GroundDistance::Hamming => hamming_distance(i, j, n_bits) as f64,
        GroundDistance::Euclidean => {
            let diff = i as f64 - j as f64;
            diff.abs()
        }
        GroundDistance::Discrete => if i == j { 0.0 } else { 1.0 },
        GroundDistance::Manhattan => {
            (i as isize - j as isize).unsigned_abs() as f64
        }
    }
}

/// Hamming distance: number of differing bits
///
/// This is the standard ground distance for IIT calculations.
///
/// # Reference
/// Hamming, R. W. (1950). "Error detecting and error correcting codes."
/// Bell System Technical Journal, 29(2), 147-160.
fn hamming_distance(x: usize, y: usize, n_bits: usize) -> usize {
    let mut dist = 0;
    let mut diff = x ^ y;

    for _ in 0..n_bits {
        if diff & 1 != 0 {
            dist += 1;
        }
        diff >>= 1;
    }

    dist
}

/// 1D Wasserstein distance (closed form)
///
/// For 1D distributions, EMD equals the L1 distance between CDFs:
/// W₁(P, Q) = ∫ |F_P(x) - F_Q(x)| dx
///
/// For discrete distributions:
/// W₁ = Σᵢ |Σⱼ≤ᵢ (pⱼ - qⱼ)|
///
/// # Reference
/// Ramdas, A., et al. (2017). "On Wasserstein Two-Sample Testing."
/// Bernoulli, 23(3), 1728-1763.
pub fn wasserstein_1d(p: &[f64], q: &[f64]) -> IITResult<f64> {
    if p.len() != q.len() {
        return Err(IITError::DimensionMismatch {
            expected: p.len(),
            actual: q.len(),
        });
    }

    let mut cdf_diff = 0.0;
    let mut w1 = 0.0;

    for (&pi, &qi) in p.iter().zip(q.iter()) {
        cdf_diff += pi - qi;
        w1 += cdf_diff.abs();
    }

    Ok(w1)
}

/// L1 distance (total variation) - NOT EMD in general!
///
/// d_L1(P, Q) = ½ Σᵢ |pᵢ - qᵢ|
///
/// This equals EMD only when ground distance is discrete (0 or 1).
/// Kept for comparison and backwards compatibility.
pub fn l1_distance(p: &[f64], q: &[f64]) -> IITResult<f64> {
    if p.len() != q.len() {
        return Err(IITError::DimensionMismatch {
            expected: p.len(),
            actual: q.len(),
        });
    }

    let dist: f64 = p.iter().zip(q.iter()).map(|(pi, qi)| (pi - qi).abs()).sum();

    Ok(dist / 2.0)
}

/// Compute EMD between distributions over partitioned states
///
/// This is used in IIT to measure distance between whole-system
/// and partitioned repertoires.
///
/// # Arguments
/// * `whole` - Distribution from whole system
/// * `partitioned` - Distribution from partitioned system
/// * `n_elements` - Number of system elements
pub fn emd_for_iit(
    whole: &[f64],
    partitioned: &[f64],
    n_elements: usize,
) -> IITResult<f64> {
    emd(whole, partitioned, GroundDistance::Hamming, Some(n_elements))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_emd_identical_distributions() {
        let p = vec![0.25, 0.25, 0.25, 0.25];
        let cost = vec![
            vec![0.0, 1.0, 1.0, 2.0],
            vec![1.0, 0.0, 2.0, 1.0],
            vec![1.0, 2.0, 0.0, 1.0],
            vec![2.0, 1.0, 1.0, 0.0],
        ];

        let d = emd_linear_programming(&p, &p, &cost).unwrap();
        assert_relative_eq!(d, 0.0, epsilon = 1e-8);
    }

    #[test]
    fn test_emd_simple_transport() {
        // All mass at 0 needs to move to 1
        let p = vec![1.0, 0.0];
        let q = vec![0.0, 1.0];
        let cost = vec![
            vec![0.0, 1.0],
            vec![1.0, 0.0],
        ];

        let d = emd_linear_programming(&p, &q, &cost).unwrap();
        assert_relative_eq!(d, 1.0, epsilon = 1e-8);
    }

    #[test]
    fn test_emd_vs_l1_discrete() {
        // With discrete ground distance, EMD = L1
        let p = vec![0.5, 0.5, 0.0, 0.0];
        let q = vec![0.0, 0.0, 0.5, 0.5];

        let emd_val = emd(&p, &q, GroundDistance::Discrete, None).unwrap();
        let l1_val = l1_distance(&p, &q).unwrap();

        // For discrete metric, EMD = L1 = 1.0
        assert_relative_eq!(emd_val, 1.0, epsilon = 1e-8);
        assert_relative_eq!(l1_val, 1.0, epsilon = 1e-8);
    }

    #[test]
    fn test_emd_different_from_l1() {
        // With Hamming distance, EMD ≠ L1 in general
        // Move from |00⟩ to |11⟩ costs 2 (Hamming), but L1 just counts probability
        let p = vec![1.0, 0.0, 0.0, 0.0];  // |00⟩
        let q = vec![0.0, 0.0, 0.0, 1.0];  // |11⟩

        let emd_val = emd(&p, &q, GroundDistance::Hamming, Some(2)).unwrap();
        let l1_val = l1_distance(&p, &q).unwrap();

        // L1 = 1.0 (just counts different probability)
        // EMD = 2.0 (must move mass across Hamming distance 2)
        assert_relative_eq!(l1_val, 1.0, epsilon = 1e-8);
        assert_relative_eq!(emd_val, 2.0, epsilon = 1e-8);
    }

    #[test]
    fn test_wasserstein_1d() {
        // Shift distribution by 1
        let p = vec![0.5, 0.5, 0.0];
        let q = vec![0.0, 0.5, 0.5];

        let w1 = wasserstein_1d(&p, &q).unwrap();
        assert_relative_eq!(w1, 1.0, epsilon = 1e-8);
    }

    #[test]
    fn test_hamming_distance() {
        assert_eq!(hamming_distance(0b000, 0b111, 3), 3);
        assert_eq!(hamming_distance(0b010, 0b011, 3), 1);
        assert_eq!(hamming_distance(0b101, 0b101, 3), 0);
    }
}
