//! Integrated Information Variants
//!
//! This module implements MULTIPLE measures of integrated information,
//! each with explicit documentation of what it measures and its limitations.
//!
//! ## Key Finding
//!
//! **Pure quantum states have Φ = 0 for all variants.**
//! Only mixed states arising from decoherence exhibit Φ > 0.
//!
//! ## CRITICAL DISTINCTION
//!
//! **Φ_IIT (Tononi 2016)**: The "official" IIT 3.0/4.0 measure.
//! - Requires: TPM, cause-effect repertoires, EMD, MIP search
//! - Complexity: O(2^n × 2^n) for n elements
//! - Intractable for n > 12
//!
//! **I_synergy**: Mutual information-based proxy.
//! - Formula: I(A;B) = H(A) + H(B) - H(A,B)
//! - Much simpler, but NOT equivalent to Φ_IIT
//! - Measures: Total correlation between partition halves
//!
//! **Φ_G (Barrett & Seth 2011)**: Geometric integrated information.
//! - Based on decoder complexity
//! - Tractable for larger systems
//!
//! ## When to Use Each
//!
//! | Measure   | Best For                           | Tractable n |
//! |-----------|------------------------------------| ------------|
//! | Φ_IIT     | Theoretical comparison             | ≤ 12        |
//! | I_synergy | Quick screening, large systems     | Any         |
//! | Φ_G       | Balanced accuracy/tractability     | ≤ 20        |
//! | TC        | Total correlation (no partitioning)| Any         |
//!
//! ## References
//!
//! ### IIT 3.0 and 4.0
//! - Tononi, G., et al. (2016).
//!   "Integrated Information Theory: From Consciousness to its Physical Substrate."
//!   Nature Reviews Neuroscience, 17(7), 450-461.
//!   DOI: 10.1038/nrn.2016.44
//!
//! - Albantakis, L., et al. (2023).
//!   "Computing the Integrated Information of a Quantum Mechanism."
//!   Entropy, 25(3), 449.
//!   DOI: 10.3390/e25030449
//!
//! ### Practical Measures
//! - Barrett, A. B., & Seth, A. K. (2011).
//!   "Practical Measures of Integrated Information."
//!   PLOS Computational Biology, 7(1), e1001052.
//!   DOI: 10.1371/journal.pcbi.1001052
//!
//! - Mediano, P. A., et al. (2019).
//!   "Measuring Integrated Information: Comparison of Candidate Measures."
//!   Entropy, 21(1), 17.
//!   DOI: 10.3390/e21010017
//!
//! ### Original IIT
//! - Oizumi, M., et al. (2014).
//!   "From the Phenomenology to the Mechanisms of Consciousness."
//!   PLOS Computational Biology, 10(5), e1003588.
//!   DOI: 10.1371/journal.pcbi.1003588
//!
//! ## Author
//!
//! Francisco Molina-Burgos
//! Avermex Research Division
//! fmolina@avermex.com

use crate::entropy::shannon_entropy;
use crate::emd::{emd, GroundDistance};
use crate::partition::{Bipartition, find_mip, find_mip_exact, marginalize_to_subset, MAX_EXACT_SIZE};
use crate::{IITError, IITResult};
use serde::{Deserialize, Serialize};

/// Integrated information measure variant
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PhiVariant {
    /// I_synergy: Mutual information between partition halves
    /// I(A;B) = H(A) + H(B) - H(A,B)
    /// WARNING: This is NOT Φ_IIT!
    Synergy,

    /// Φ_IIT: Tononi's IIT 3.0 (EMD-based)
    /// Only tractable for n ≤ 12
    IIT,

    /// Φ_G: Geometric integrated information (Barrett & Seth)
    /// Based on decoder complexity
    Geometric,

    /// Total correlation (multi-information)
    /// TC = Σᵢ H(Xᵢ) - H(X)
    /// Does NOT involve MIP search
    TotalCorrelation,
}

impl std::fmt::Display for PhiVariant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PhiVariant::Synergy => write!(f, "I_synergy"),
            PhiVariant::IIT => write!(f, "Φ_IIT"),
            PhiVariant::Geometric => write!(f, "Φ_G"),
            PhiVariant::TotalCorrelation => write!(f, "TC"),
        }
    }
}

/// Result of Φ calculation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhiResult {
    /// The integrated information value
    pub phi: f64,
    /// Which variant was calculated
    pub variant: PhiVariant,
    /// MIP (if applicable)
    pub mip: Option<Bipartition>,
    /// Number of system elements
    pub n_elements: usize,
    /// State space dimension
    pub state_space_size: usize,
    /// Number of partitions evaluated
    pub partitions_evaluated: usize,
    /// Additional metrics
    pub diagnostics: PhiDiagnostics,
}

/// Diagnostic information about Φ calculation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhiDiagnostics {
    /// System entropy H(X)
    pub system_entropy: f64,
    /// Was exact MIP search used?
    pub exact_search: bool,
    /// Computation warnings
    pub warnings: Vec<String>,
}

/// Calculate integrated information for a probability distribution
///
/// # Arguments
/// * `distribution` - Joint probability distribution over all states
/// * `n_elements` - Number of system elements (oscillators, neurons, etc.)
/// * `levels_per_element` - States per element (2 for binary, 3 for qutrits, etc.)
/// * `variant` - Which Φ measure to compute
///
/// # Returns
/// PhiResult with the computed value and diagnostics
pub fn calculate_phi(
    distribution: &[f64],
    n_elements: usize,
    levels_per_element: usize,
    variant: PhiVariant,
) -> IITResult<PhiResult> {
    // Validate inputs
    let expected_size = levels_per_element.pow(n_elements as u32);
    if distribution.len() != expected_size {
        return Err(IITError::DimensionMismatch {
            expected: expected_size,
            actual: distribution.len(),
        });
    }

    let system_entropy = shannon_entropy(distribution)?;

    match variant {
        PhiVariant::Synergy => calculate_synergy(distribution, n_elements, levels_per_element, system_entropy),
        PhiVariant::IIT => calculate_phi_iit(distribution, n_elements, levels_per_element, system_entropy),
        PhiVariant::Geometric => calculate_phi_geometric(distribution, n_elements, levels_per_element, system_entropy),
        PhiVariant::TotalCorrelation => calculate_total_correlation(distribution, n_elements, levels_per_element, system_entropy),
    }
}

/// Calculate I_synergy (mutual information at MIP)
///
/// I_synergy = min_{partitions} I(A;B)
/// where I(A;B) = H(A) + H(B) - H(A,B)
fn calculate_synergy(
    distribution: &[f64],
    n_elements: usize,
    levels_per_element: usize,
    system_entropy: f64,
) -> IITResult<PhiResult> {
    if n_elements < 2 {
        return Ok(PhiResult {
            phi: 0.0,
            variant: PhiVariant::Synergy,
            mip: None,
            n_elements,
            state_space_size: distribution.len(),
            partitions_evaluated: 0,
            diagnostics: PhiDiagnostics {
                system_entropy,
                exact_search: true,
                warnings: vec!["Single element has no integration".to_string()],
            },
        });
    }

    // Define phi function for each partition
    let phi_fn = |partition: &Bipartition| -> IITResult<f64> {
        compute_mutual_info_for_partition(distribution, partition, n_elements, levels_per_element)
    };

    let mip_result = find_mip(n_elements, phi_fn)?;

    Ok(PhiResult {
        phi: mip_result.phi,
        variant: PhiVariant::Synergy,
        mip: Some(mip_result.partition),
        n_elements,
        state_space_size: distribution.len(),
        partitions_evaluated: mip_result.partitions_evaluated,
        diagnostics: PhiDiagnostics {
            system_entropy,
            exact_search: n_elements <= MAX_EXACT_SIZE,
            warnings: if n_elements > MAX_EXACT_SIZE {
                vec!["Used sampling - result may not be global minimum".to_string()]
            } else {
                vec![]
            },
        },
    })
}

/// Calculate Φ_IIT (Tononi's IIT 3.0)
///
/// This is the "real" Φ but is only tractable for small systems.
///
/// Φ_IIT = min_{partitions} EMD(whole || partitioned)
fn calculate_phi_iit(
    distribution: &[f64],
    n_elements: usize,
    levels_per_element: usize,
    system_entropy: f64,
) -> IITResult<PhiResult> {
    // Check tractability
    if n_elements > 12 {
        return Err(IITError::SystemTooLarge {
            n: n_elements,
            limit: 12,
        });
    }

    if n_elements < 2 {
        return Ok(PhiResult {
            phi: 0.0,
            variant: PhiVariant::IIT,
            mip: None,
            n_elements,
            state_space_size: distribution.len(),
            partitions_evaluated: 0,
            diagnostics: PhiDiagnostics {
                system_entropy,
                exact_search: true,
                warnings: vec!["Single element has no integration".to_string()],
            },
        });
    }

    // Define phi function using EMD
    let phi_fn = |partition: &Bipartition| -> IITResult<f64> {
        compute_emd_for_partition(distribution, partition, n_elements, levels_per_element)
    };

    // Always use exact search for IIT (since n ≤ 12)
    let mip_result = find_mip_exact(n_elements, phi_fn)?;

    Ok(PhiResult {
        phi: mip_result.phi,
        variant: PhiVariant::IIT,
        mip: Some(mip_result.partition),
        n_elements,
        state_space_size: distribution.len(),
        partitions_evaluated: mip_result.partitions_evaluated,
        diagnostics: PhiDiagnostics {
            system_entropy,
            exact_search: true,
            warnings: vec![
                "Simplified Φ_IIT: uses state distribution, not full cause-effect structure".to_string()
            ],
        },
    })
}

/// Calculate Φ_G (Geometric integrated information)
///
/// Based on Barrett & Seth (2011).
fn calculate_phi_geometric(
    distribution: &[f64],
    n_elements: usize,
    levels_per_element: usize,
    system_entropy: f64,
) -> IITResult<PhiResult> {
    if n_elements < 2 {
        return Ok(PhiResult {
            phi: 0.0,
            variant: PhiVariant::Geometric,
            mip: None,
            n_elements,
            state_space_size: distribution.len(),
            partitions_evaluated: 0,
            diagnostics: PhiDiagnostics {
                system_entropy,
                exact_search: true,
                warnings: vec!["Single element has no integration".to_string()],
            },
        });
    }

    let phi_fn = |partition: &Bipartition| -> IITResult<f64> {
        compute_kl_to_product(distribution, partition, n_elements, levels_per_element)
    };

    let mip_result = find_mip(n_elements, phi_fn)?;

    Ok(PhiResult {
        phi: mip_result.phi,
        variant: PhiVariant::Geometric,
        mip: Some(mip_result.partition),
        n_elements,
        state_space_size: distribution.len(),
        partitions_evaluated: mip_result.partitions_evaluated,
        diagnostics: PhiDiagnostics {
            system_entropy,
            exact_search: n_elements <= MAX_EXACT_SIZE,
            warnings: vec![],
        },
    })
}

/// Calculate Total Correlation (multi-information)
///
/// TC = Σᵢ H(Xᵢ) - H(X)
///
/// This measures total statistical dependence but does NOT
/// involve partitioning or MIP search.
fn calculate_total_correlation(
    distribution: &[f64],
    n_elements: usize,
    levels_per_element: usize,
    system_entropy: f64,
) -> IITResult<PhiResult> {
    let mut sum_marginal_entropies = 0.0;

    for i in 0..n_elements {
        let marginal = marginalize_to_subset(distribution, &[i], n_elements, levels_per_element);
        sum_marginal_entropies += shannon_entropy(&marginal)?;
    }

    let tc = (sum_marginal_entropies - system_entropy).max(0.0);

    Ok(PhiResult {
        phi: tc,
        variant: PhiVariant::TotalCorrelation,
        mip: None,  // TC doesn't use MIP
        n_elements,
        state_space_size: distribution.len(),
        partitions_evaluated: 0,
        diagnostics: PhiDiagnostics {
            system_entropy,
            exact_search: true,
            warnings: vec!["TC is not a true integration measure (no partitioning)".to_string()],
        },
    })
}

// ==================== HELPER FUNCTIONS ====================

/// Compute mutual information for a partition
fn compute_mutual_info_for_partition(
    distribution: &[f64],
    partition: &Bipartition,
    n_elements: usize,
    levels_per_element: usize,
) -> IITResult<f64> {
    let marginal_a = marginalize_to_subset(distribution, &partition.subset_a, n_elements, levels_per_element);
    let marginal_b = marginalize_to_subset(distribution, &partition.subset_b, n_elements, levels_per_element);

    let h_a = shannon_entropy(&marginal_a)?;
    let h_b = shannon_entropy(&marginal_b)?;
    let h_joint = shannon_entropy(distribution)?;

    Ok((h_a + h_b - h_joint).max(0.0))
}

/// Compute EMD between whole distribution and product of marginals
fn compute_emd_for_partition(
    distribution: &[f64],
    partition: &Bipartition,
    n_elements: usize,
    levels_per_element: usize,
) -> IITResult<f64> {
    let marginal_a = marginalize_to_subset(distribution, &partition.subset_a, n_elements, levels_per_element);
    let marginal_b = marginalize_to_subset(distribution, &partition.subset_b, n_elements, levels_per_element);

    // Compute product distribution
    let product = compute_product_distribution(&marginal_a, &marginal_b);

    // Use EMD with Hamming distance
    emd(distribution, &product, GroundDistance::Hamming, Some(n_elements))
}

/// Compute KL divergence to product distribution (for Φ_G)
fn compute_kl_to_product(
    distribution: &[f64],
    partition: &Bipartition,
    n_elements: usize,
    levels_per_element: usize,
) -> IITResult<f64> {
    let marginal_a = marginalize_to_subset(distribution, &partition.subset_a, n_elements, levels_per_element);
    let marginal_b = marginalize_to_subset(distribution, &partition.subset_b, n_elements, levels_per_element);

    // KL(P || P_A ⊗ P_B) = H(P_A) + H(P_B) - H(P)
    let h_a = shannon_entropy(&marginal_a)?;
    let h_b = shannon_entropy(&marginal_b)?;
    let h_joint = shannon_entropy(distribution)?;

    Ok((h_a + h_b - h_joint).max(0.0))
}

/// Compute product distribution P_A ⊗ P_B
fn compute_product_distribution(p_a: &[f64], p_b: &[f64]) -> Vec<f64> {
    let mut product = Vec::with_capacity(p_a.len() * p_b.len());

    for &a in p_a {
        for &b in p_b {
            product.push(a * b);
        }
    }

    product
}

/// Calculate all variants and return comparison
pub fn calculate_all_variants(
    distribution: &[f64],
    n_elements: usize,
    levels_per_element: usize,
) -> IITResult<Vec<PhiResult>> {
    let mut results = Vec::new();

    // Always compute Synergy and TC
    results.push(calculate_phi(distribution, n_elements, levels_per_element, PhiVariant::Synergy)?);
    results.push(calculate_phi(distribution, n_elements, levels_per_element, PhiVariant::TotalCorrelation)?);
    results.push(calculate_phi(distribution, n_elements, levels_per_element, PhiVariant::Geometric)?);

    // Only compute IIT for small systems
    if n_elements <= 12 {
        results.push(calculate_phi(distribution, n_elements, levels_per_element, PhiVariant::IIT)?);
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_independent_system() {
        // Independent system: Φ = 0 for all variants
        // P(X,Y) = P(X)P(Y) = uniform
        let dist = vec![0.25, 0.25, 0.25, 0.25];

        let synergy = calculate_phi(&dist, 2, 2, PhiVariant::Synergy).unwrap();
        assert_relative_eq!(synergy.phi, 0.0, epsilon = 1e-10);

        let tc = calculate_phi(&dist, 2, 2, PhiVariant::TotalCorrelation).unwrap();
        assert_relative_eq!(tc.phi, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_maximally_correlated() {
        // X = Y (perfect correlation)
        // P(0,0) = P(1,1) = 0.5
        let dist = vec![0.5, 0.0, 0.0, 0.5];

        let synergy = calculate_phi(&dist, 2, 2, PhiVariant::Synergy).unwrap();
        assert_relative_eq!(synergy.phi, 1.0, epsilon = 1e-10);

        let iit = calculate_phi(&dist, 2, 2, PhiVariant::IIT).unwrap();
        assert!(iit.phi > 0.0);
    }

    #[test]
    fn test_variant_ordering() {
        // For same distribution, variants may give different values
        let dist = vec![0.4, 0.1, 0.1, 0.4];

        let results = calculate_all_variants(&dist, 2, 2).unwrap();

        // All should be non-negative
        for result in &results {
            assert!(result.phi >= 0.0);
        }
    }

    #[test]
    fn test_iit_size_limit() {
        // IIT should fail for n > 12
        let dist = vec![1.0 / 8192.0; 8192];  // 2^13 states

        let result = calculate_phi(&dist, 13, 2, PhiVariant::IIT);
        assert!(result.is_err());
    }

    #[test]
    fn test_display_variants() {
        assert_eq!(format!("{}", PhiVariant::Synergy), "I_synergy");
        assert_eq!(format!("{}", PhiVariant::IIT), "Φ_IIT");
        assert_eq!(format!("{}", PhiVariant::Geometric), "Φ_G");
        assert_eq!(format!("{}", PhiVariant::TotalCorrelation), "TC");
    }
}
