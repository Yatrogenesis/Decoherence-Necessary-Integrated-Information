//! # Quantum Processor for Consciousness Research
//!
//! Rigorous quantum simulation with Lindblad master equation dynamics
//! for investigating integrated information in open quantum systems.
//!
//! ## Overview
//!
//! This crate implements coupled quantum harmonic oscillators with proper
//! open-system dynamics, enabling the study of how decoherence affects
//! integrated information (Φ).
//!
//! ## Key Features
//!
//! 1. **Lindblad dynamics** - Gorini-Kossakowski-Sudarshan-Lindblad equation
//! 2. **Density matrix** representation for mixed states
//! 3. **RK4 integration** with adaptive step control
//! 4. **Thermal bath coupling** with configurable temperature
//!
//! ## Physical Background
//!
//! The Lindblad master equation describes the evolution of open quantum systems:
//!
//! ```text
//! dρ/dt = -i/ℏ [H, ρ] + Σₖ γₖ D[Lₖ](ρ)
//! ```
//!
//! Where the dissipator is:
//! ```text
//! D[L](ρ) = L ρ L† - ½{L†L, ρ}
//! ```
//!
//! ## References
//!
//! ### Lindblad Dynamics
//! - Lindblad, G. (1976). "On the generators of quantum dynamical semigroups."
//!   Communications in Mathematical Physics, 48(2), 119-130.
//!   DOI: 10.1007/BF01608499
//!
//! - Gorini, V., Kossakowski, A., & Sudarshan, E. C. G. (1976).
//!   "Completely positive dynamical semigroups of N-level systems."
//!   Journal of Mathematical Physics, 17(5), 821-825.
//!
//! ### Open Quantum Systems
//! - Breuer, H.-P., & Petruccione, F. (2002).
//!   "The Theory of Open Quantum Systems." Oxford University Press.
//!   ISBN: 978-0199213900
//!
//! - Wiseman, H. M., & Milburn, G. J. (2009).
//!   "Quantum Measurement and Control." Cambridge University Press.
//!
//! ### Quantum Reservoir Computing
//! - Fujii, K., & Nakajima, K. (2017).
//!   "Harnessing Disordered-Ensemble Quantum Dynamics for Machine Learning."
//!   Physical Review Applied, 8(2), 024030.
//!
//! - Nakajima, K., et al. (2019).
//!   "Boosting Computational Power through Spatial Multiplexing in Quantum RC."
//!   Physical Review Applied, 11(3), 034021.
//!
//! ## Author
//!
//! Francisco Molina-Burgos
//! Avermex Research Division
//! fmolina@avermex.com

#![deny(missing_docs)]

pub mod lindblad;
pub mod quantum_reservoir;
pub mod density_matrix;
pub mod operators;

/// Convenient re-exports for common usage
pub mod prelude {
    pub use crate::lindblad::*;
    pub use crate::quantum_reservoir::*;
    pub use crate::density_matrix::*;
    pub use crate::operators::*;
}

use thiserror::Error;

/// Errors in quantum computation
#[derive(Error, Debug)]
pub enum QuantumError {
    /// Invalid density matrix (not positive semidefinite, trace != 1)
    #[error("Invalid density matrix: {0}")]
    InvalidDensityMatrix(String),

    /// Dimension mismatch in quantum operations
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// Expected dimension
        expected: usize,
        /// Actual dimension
        actual: usize,
    },

    /// Numerical instability detected
    #[error("Numerical instability: {0}")]
    NumericalInstability(String),

    /// Invalid Lindblad operator
    #[error("Invalid Lindblad operator: {0}")]
    InvalidLindbladOperator(String),
}

/// Result type for quantum operations
pub type QuantumResult<T> = Result<T, QuantumError>;
