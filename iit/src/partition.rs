//! System Partitioning for IIT
//!
//! Implements bipartition enumeration and MIP (Minimum Information Partition) search.
//!
//! ## Theory
//!
//! IIT defines Φ as the minimum information lost when the system is partitioned.
//! The Minimum Information Partition (MIP) is:
//!
//! ```text
//! MIP = argmin_{partitions} D(whole || partitioned)
//! ```
//!
//! Where D is Earth Mover's Distance between cause-effect repertoires.
//!
//! ## Computational Complexity
//!
//! - Bipartitions of n elements: 2^(n-1) - 1
//! - For n > 15, exhaustive search becomes impractical
//!
//! | n  | Bipartitions | Feasibility    |
//! |----|--------------|----------------|
//! | 5  | 15           | Easy           |
//! | 10 | 511          | Easy           |
//! | 15 | 16383        | Moderate       |
//! | 20 | 524287       | Difficult      |
//! | 25 | 16M          | Very difficult |
//!
//! ## References
//!
//! ### IIT Partitioning
//! - Oizumi, M., Albantakis, L., & Tononi, G. (2014).
//!   "From the Phenomenology to the Mechanisms of Consciousness."
//!   PLOS Computational Biology, 10(5), e1003588.
//!   DOI: 10.1371/journal.pcbi.1003588
//!
//! - Barrett, A. B., & Seth, A. K. (2011).
//!   "Practical Measures of Integrated Information."
//!   PLOS Computational Biology, 7(1), e1001052.
//!   DOI: 10.1371/journal.pcbi.1001052
//!
//! ### Combinatorics
//! - Tononi, G. (2004).
//!   "An information integration theory of consciousness."
//!   BMC Neuroscience, 5, 42.
//!   DOI: 10.1186/1471-2202-5-42
//!
//! ## Author
//!
//! Francisco Molina-Burgos
//! Avermex Research Division
//! fmolina@avermex.com

use crate::{IITError, IITResult};
use serde::{Deserialize, Serialize};
use rand::prelude::*;

/// Maximum system size for exact MIP search
pub const MAX_EXACT_SIZE: usize = 15;

/// Bipartition of a system
///
/// Represents dividing a system into two non-empty, disjoint subsets A and B.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Bipartition {
    /// Indices in subset A
    pub subset_a: Vec<usize>,
    /// Indices in subset B
    pub subset_b: Vec<usize>,
}

impl Bipartition {
    /// Create new bipartition
    pub fn new(subset_a: Vec<usize>, subset_b: Vec<usize>) -> Self {
        Self { subset_a, subset_b }
    }

    /// Check if partition is valid (non-empty, disjoint, complete)
    pub fn is_valid(&self, n_elements: usize) -> bool {
        // Non-empty
        if self.subset_a.is_empty() || self.subset_b.is_empty() {
            return false;
        }

        // Disjoint
        for &a in &self.subset_a {
            if self.subset_b.contains(&a) {
                return false;
            }
        }

        // Complete
        if self.subset_a.len() + self.subset_b.len() != n_elements {
            return false;
        }

        // Valid indices
        self.subset_a.iter().chain(&self.subset_b).all(|&i| i < n_elements)
    }

    /// Convert to bitmask representation
    pub fn to_bitmask(&self) -> usize {
        let mut mask = 0;
        for &i in &self.subset_a {
            mask |= 1 << i;
        }
        mask
    }

    /// Create from bitmask
    pub fn from_bitmask(mask: usize, n_elements: usize) -> Self {
        let mut subset_a = Vec::new();
        let mut subset_b = Vec::new();

        for i in 0..n_elements {
            if (mask & (1 << i)) != 0 {
                subset_a.push(i);
            } else {
                subset_b.push(i);
            }
        }

        Self::new(subset_a, subset_b)
    }

    /// Size of smaller subset (for balanced partition preference)
    pub fn balance(&self) -> usize {
        self.subset_a.len().min(self.subset_b.len())
    }
}

/// Information about MIP result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MIPResult {
    /// The minimum information partition
    pub partition: Bipartition,
    /// Φ value (information loss at MIP)
    pub phi: f64,
    /// Number of partitions evaluated
    pub partitions_evaluated: usize,
    /// Method used
    pub method: String,
}

/// Generate all non-trivial bipartitions
///
/// Returns 2^(n-1) - 1 partitions (excluding empty and full).
/// Only generates unique partitions (A,B and B,A are the same).
///
/// # Arguments
/// * `n_elements` - Number of elements to partition
pub fn all_bipartitions(n_elements: usize) -> Vec<Bipartition> {
    if n_elements < 2 {
        return Vec::new();
    }

    let mut partitions = Vec::new();

    // For n elements, we iterate masks from 1 to 2^(n-1) - 1
    // This gives us unique bipartitions (avoiding mirror images)
    let max_mask = 1 << (n_elements - 1);

    for mask in 1..max_mask {
        partitions.push(Bipartition::from_bitmask(mask, n_elements));
    }

    partitions
}

/// Count number of bipartitions for n elements
pub fn count_bipartitions(n_elements: usize) -> usize {
    if n_elements < 2 {
        0
    } else {
        (1 << (n_elements - 1)) - 1
    }
}

/// Iterator over bipartitions (memory efficient)
pub struct BipartitionIterator {
    n_elements: usize,
    current_mask: usize,
    max_mask: usize,
}

impl BipartitionIterator {
    /// Create new iterator
    pub fn new(n_elements: usize) -> Self {
        Self {
            n_elements,
            current_mask: 1,
            max_mask: 1 << (n_elements.saturating_sub(1)),
        }
    }
}

impl Iterator for BipartitionIterator {
    type Item = Bipartition;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_mask >= self.max_mask {
            return None;
        }

        let partition = Bipartition::from_bitmask(self.current_mask, self.n_elements);
        self.current_mask += 1;
        Some(partition)
    }
}

/// Find MIP using exhaustive search
///
/// Evaluates ALL bipartitions. Only use for n ≤ MAX_EXACT_SIZE.
///
/// # Arguments
/// * `n_elements` - Number of system elements
/// * `phi_fn` - Function computing Φ for a partition
///
/// # Returns
/// MIP and its Φ value
pub fn find_mip_exact<F>(n_elements: usize, phi_fn: F) -> IITResult<MIPResult>
where
    F: Fn(&Bipartition) -> IITResult<f64>,
{
    if n_elements > MAX_EXACT_SIZE {
        return Err(IITError::SystemTooLarge {
            n: n_elements,
            limit: MAX_EXACT_SIZE,
        });
    }

    if n_elements < 2 {
        return Err(IITError::PartitionError(
            "Cannot partition system with fewer than 2 elements".to_string()
        ));
    }

    let mut min_phi = f64::INFINITY;
    let mut mip = None;
    let mut count = 0;

    for partition in BipartitionIterator::new(n_elements) {
        let phi = phi_fn(&partition)?;
        count += 1;

        if phi < min_phi {
            min_phi = phi;
            mip = Some(partition);
        }
    }

    match mip {
        Some(partition) => Ok(MIPResult {
            partition,
            phi: min_phi,
            partitions_evaluated: count,
            method: "exact".to_string(),
        }),
        None => Err(IITError::PartitionError(
            "No valid partitions found".to_string()
        )),
    }
}

/// Find MIP using random sampling (for large systems)
///
/// # Arguments
/// * `n_elements` - Number of system elements
/// * `phi_fn` - Function computing Φ for a partition
/// * `n_samples` - Number of random partitions to try
/// * `seed` - Random seed for reproducibility
pub fn find_mip_sampled<F>(
    n_elements: usize,
    phi_fn: F,
    n_samples: usize,
    seed: u64,
) -> IITResult<MIPResult>
where
    F: Fn(&Bipartition) -> IITResult<f64>,
{
    if n_elements < 2 {
        return Err(IITError::PartitionError(
            "Cannot partition system with fewer than 2 elements".to_string()
        ));
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let mut min_phi = f64::INFINITY;
    let mut mip = None;

    for _ in 0..n_samples {
        let partition = random_bipartition(n_elements, &mut rng);
        let phi = phi_fn(&partition)?;

        if phi < min_phi {
            min_phi = phi;
            mip = Some(partition);
        }
    }

    match mip {
        Some(partition) => Ok(MIPResult {
            partition,
            phi: min_phi,
            partitions_evaluated: n_samples,
            method: format!("sampled_{}", n_samples),
        }),
        None => Err(IITError::PartitionError(
            "No valid partitions found in sampling".to_string()
        )),
    }
}

/// Generate random bipartition
fn random_bipartition<R: Rng>(n_elements: usize, rng: &mut R) -> Bipartition {
    let mut subset_a = Vec::new();
    let mut subset_b = Vec::new();

    for i in 0..n_elements {
        if rng.gen::<bool>() {
            subset_a.push(i);
        } else {
            subset_b.push(i);
        }
    }

    // Ensure non-empty
    if subset_a.is_empty() {
        let idx = rng.gen_range(0..subset_b.len());
        let elem = subset_b.remove(idx);
        subset_a.push(elem);
    } else if subset_b.is_empty() {
        let idx = rng.gen_range(0..subset_a.len());
        let elem = subset_a.remove(idx);
        subset_b.push(elem);
    }

    Bipartition::new(subset_a, subset_b)
}

/// Find MIP with automatic method selection
///
/// Uses exact search for n ≤ MAX_EXACT_SIZE, sampling otherwise.
pub fn find_mip<F>(n_elements: usize, phi_fn: F) -> IITResult<MIPResult>
where
    F: Fn(&Bipartition) -> IITResult<f64>,
{
    if n_elements <= MAX_EXACT_SIZE {
        find_mip_exact(n_elements, phi_fn)
    } else {
        // Sample proportional to problem size
        let n_samples = (1000 * n_elements).min(100_000);
        find_mip_sampled(n_elements, phi_fn, n_samples, 42)
    }
}

/// Marginalize distribution to subset of variables
///
/// For a joint distribution P(X₁,...,Xₙ), compute P(Xᵢ: i ∈ subset).
///
/// # Arguments
/// * `joint` - Joint distribution as flat array
/// * `subset` - Indices of variables to keep
/// * `n_elements` - Total number of variables
/// * `levels_per_element` - Number of states per variable
pub fn marginalize_to_subset(
    joint: &[f64],
    subset: &[usize],
    n_elements: usize,
    levels_per_element: usize,
) -> Vec<f64> {
    let subset_size = levels_per_element.pow(subset.len() as u32);
    let mut marginal = vec![0.0; subset_size];

    for (joint_idx, &prob) in joint.iter().enumerate() {
        // Decode joint index
        let config = decode_index(joint_idx, n_elements, levels_per_element);

        // Extract subset configuration
        let subset_config: Vec<usize> = subset.iter().map(|&i| config[i]).collect();
        let subset_idx = encode_config(&subset_config, levels_per_element);

        if subset_idx < marginal.len() {
            marginal[subset_idx] += prob;
        }
    }

    marginal
}

/// Decode flat index into configuration vector
fn decode_index(idx: usize, n_elements: usize, levels: usize) -> Vec<usize> {
    let mut config = vec![0; n_elements];
    let mut remaining = idx;

    for i in 0..n_elements {
        config[i] = remaining % levels;
        remaining /= levels;
    }

    config
}

/// Encode configuration into flat index
fn encode_config(config: &[usize], levels: usize) -> usize {
    let mut idx = 0;
    for (i, &val) in config.iter().enumerate() {
        idx += val * levels.pow(i as u32);
    }
    idx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bipartition_validity() {
        let p = Bipartition::new(vec![0, 1], vec![2, 3]);
        assert!(p.is_valid(4));

        let p = Bipartition::new(vec![0], vec![1, 2, 3]);
        assert!(p.is_valid(4));

        // Empty subset
        let p = Bipartition::new(vec![], vec![0, 1, 2]);
        assert!(!p.is_valid(3));

        // Overlapping
        let p = Bipartition::new(vec![0, 1], vec![1, 2]);
        assert!(!p.is_valid(3));
    }

    #[test]
    fn test_bipartition_count() {
        assert_eq!(count_bipartitions(2), 1);   // {0}|{1}
        assert_eq!(count_bipartitions(3), 3);   // 2^2 - 1
        assert_eq!(count_bipartitions(4), 7);   // 2^3 - 1
        assert_eq!(count_bipartitions(5), 15);  // 2^4 - 1
    }

    #[test]
    fn test_all_bipartitions() {
        let parts = all_bipartitions(3);
        assert_eq!(parts.len(), 3);

        for p in &parts {
            assert!(p.is_valid(3));
        }
    }

    #[test]
    fn test_bitmask_roundtrip() {
        for n in 2..6 {
            for mask in 1..(1 << n) / 2 {
                let p = Bipartition::from_bitmask(mask, n);
                assert!(p.is_valid(n));
                assert_eq!(p.to_bitmask(), mask);
            }
        }
    }

    #[test]
    fn test_find_mip_exact() {
        // Trivial phi function: prefer balanced partitions
        let phi_fn = |p: &Bipartition| -> IITResult<f64> {
            let diff = (p.subset_a.len() as f64 - p.subset_b.len() as f64).abs();
            Ok(diff)
        };

        let result = find_mip_exact(4, phi_fn).unwrap();

        // Should find a balanced partition (2,2)
        assert_eq!(result.phi, 0.0);
        assert_eq!(result.partition.balance(), 2);
    }

    #[test]
    fn test_marginalize() {
        // 2 binary variables, uniform distribution
        let joint = vec![0.25, 0.25, 0.25, 0.25];

        // Marginalize to first variable
        let m0 = marginalize_to_subset(&joint, &[0], 2, 2);
        assert_eq!(m0, vec![0.5, 0.5]);

        // Marginalize to second variable
        let m1 = marginalize_to_subset(&joint, &[1], 2, 2);
        assert_eq!(m1, vec![0.5, 0.5]);
    }
}
