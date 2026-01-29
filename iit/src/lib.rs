//! # Integrated Information Theory (IIT) Implementation
//!
//! Rigorous implementation of IIT measures for consciousness research,
//! demonstrating that **decoherence is necessary for Φ > 0** in quantum systems.
//!
//! ## Key Insight
//!
//! **Pure quantum states have Φ = 0.** Only mixed states arising from
//! decoherence (Lindblad dynamics) exhibit integrated information.
//!
//! ## Measures Implemented
//!
//! | Measure | Description | Tractable n |
//! |---------|-------------|-------------|
//! | **Φ_IIT** | Tononi's exact IIT 3.0 (EMD-based) | ≤ 12 |
//! | **I_synergy** | Mutual information proxy | Any |
//! | **Φ_G** | Geometric integrated information | ≤ 20 |
//! | **TC** | Total correlation (multi-information) | Any |
//!
//! ## References
//!
//! ### Integrated Information Theory
//! - Tononi, G., Boly, M., Massimini, M., & Koch, C. (2016).
//!   "Integrated information theory: from consciousness to its physical substrate."
//!   Nature Reviews Neuroscience, 17(7), 450-461.
//!   DOI: 10.1038/nrn.2016.44
//!
//! - Oizumi, M., Albantakis, L., & Tononi, G. (2014).
//!   "From the Phenomenology to the Mechanisms of Consciousness:
//!   Integrated Information Theory 3.0."
//!   PLOS Computational Biology, 10(5), e1003588.
//!   DOI: 10.1371/journal.pcbi.1003588
//!
//! ### IIT 4.0 and Quantum Extension
//! - Albantakis, L., Prentner, R., & Durham, I. (2023).
//!   "Computing the Integrated Information of a Quantum Mechanism."
//!   Entropy, 25(3), 449.
//!   DOI: 10.3390/e25030449
//!
//! ### Practical Measures
//! - Barrett, A. B., & Seth, A. K. (2011).
//!   "Practical Measures of Integrated Information for Time-Series Data."
//!   PLOS Computational Biology, 7(1), e1001052.
//!   DOI: 10.1371/journal.pcbi.1001052
//!
//! - Mediano, P. A., et al. (2019).
//!   "Measuring Integrated Information: Comparison of Candidate Measures
//!   in Theory and Simulation."
//!   Entropy, 21(1), 17.
//!   DOI: 10.3390/e21010017
//!
//! ### Quantum Consciousness
//! - Tegmark, M. (2015).
//!   "Consciousness as a State of Matter."
//!   Chaos, Solitons & Fractals, 76, 238-270.
//!   DOI: 10.1016/j.chaos.2015.03.014
//!
//! ## Author
//!
//! Francisco Molina-Burgos
//! Avermex Research Division
//! fmolina@avermex.com

#![deny(missing_docs)]

pub mod emd;
pub mod phi_variants;
pub mod partition;
pub mod entropy;

/// Convenient re-exports
pub mod prelude {
    pub use crate::emd::*;
    pub use crate::phi_variants::*;
    pub use crate::partition::*;
    pub use crate::entropy::*;
}

use thiserror::Error;

/// Errors in IIT calculations
#[derive(Error, Debug)]
pub enum IITError {
    /// Invalid probability distribution
    #[error("Invalid distribution: {0}")]
    InvalidDistribution(String),

    /// Dimension mismatch
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// Expected dimension
        expected: usize,
        /// Actual dimension
        actual: usize,
    },

    /// EMD computation failed
    #[error("EMD computation failed: {0}")]
    EMDError(String),

    /// Partition error
    #[error("Invalid partition: {0}")]
    PartitionError(String),

    /// System too large for exact computation
    #[error("System too large: {n} elements exceeds limit of {limit}")]
    SystemTooLarge {
        /// Actual system size
        n: usize,
        /// Maximum allowed size
        limit: usize,
    },
}

/// Result type for IIT operations
pub type IITResult<T> = Result<T, IITError>;
