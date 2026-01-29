//! Density Matrix representation for mixed quantum states
//!
//! The density matrix ρ is the fundamental object for describing open quantum systems.
//! Unlike pure states |ψ⟩, density matrices can represent:
//! - Pure states: ρ = |ψ⟩⟨ψ|
//! - Mixed states: ρ = Σᵢ pᵢ |ψᵢ⟩⟨ψᵢ|
//! - Subsystems of entangled states (via partial trace)
//!
//! ## Mathematical Properties
//!
//! A valid density matrix must satisfy:
//! 1. Hermitian: ρ = ρ†
//! 2. Positive semidefinite: ⟨ψ|ρ|ψ⟩ ≥ 0 for all |ψ⟩
//! 3. Normalized: Tr(ρ) = 1
//!
//! ## Key Insight for Integrated Information
//!
//! **Pure states have trivial integrated information (Φ = 0)** because they
//! can always be factored as product states in some basis. Mixed states
//! arising from decoherence create non-trivial correlations that generate
//! integrated information.
//!
//! ## References
//!
//! ### Density Matrices
//! - von Neumann, J. (1927). "Wahrscheinlichkeitstheoretischer Aufbau der
//!   Quantenmechanik." Göttinger Nachrichten, 1, 245-272.
//!
//! - Nielsen, M. A., & Chuang, I. L. (2010).
//!   "Quantum Computation and Quantum Information" (10th Anniversary ed.).
//!   Cambridge University Press. Chapter 2.4.
//!   ISBN: 978-1107002173
//!
//! ### Von Neumann Entropy
//! - von Neumann, J. (1932). "Mathematische Grundlagen der Quantenmechanik."
//!   Springer. English translation: Princeton University Press (1955).
//!
//! ### Partial Trace
//! - Nielsen & Chuang (2010), Section 2.4.3, Eq. 2.178
//!
//! ### Thermal States
//! - Pathria, R. K., & Beale, P. D. (2011).
//!   "Statistical Mechanics" (3rd ed.). Elsevier. Chapter 7.
//!   ISBN: 978-0123821881
//!
//! ## Author
//!
//! Francisco Molina-Burgos
//! Avermex Research Division
//! fmolina@avermex.com

use crate::{QuantumError, QuantumResult};
use nalgebra::DMatrix;
use num_complex::Complex64;
use serde::{Deserialize, Serialize};

/// Density matrix for quantum state representation
///
/// Stores the full ρ matrix of dimension d × d where d = (max_fock + 1)^n_oscillators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DensityMatrix {
    /// The density matrix data (Hermitian, positive semidefinite, Tr=1)
    data: DMatrix<Complex64>,
    /// Hilbert space dimension
    dimension: usize,
}

impl DensityMatrix {
    /// Create a new density matrix from raw data
    ///
    /// # Arguments
    /// * `data` - Square matrix of dimension d × d
    ///
    /// # Errors
    /// Returns error if matrix is not valid density matrix
    pub fn new(data: DMatrix<Complex64>) -> QuantumResult<Self> {
        let (rows, cols) = data.shape();

        if rows != cols {
            return Err(QuantumError::InvalidDensityMatrix(
                format!("Matrix must be square, got {}x{}", rows, cols)
            ));
        }

        let dm = Self {
            dimension: rows,
            data,
        };

        dm.validate()?;
        Ok(dm)
    }

    /// Create density matrix from pure state vector |ψ⟩
    ///
    /// ρ = |ψ⟩⟨ψ|
    ///
    /// # Arguments
    /// * `psi` - State vector (will be normalized)
    ///
    /// # Reference
    /// Nielsen & Chuang (2010), Eq. 2.138
    pub fn from_pure_state(psi: &[Complex64]) -> QuantumResult<Self> {
        let n = psi.len();
        if n == 0 {
            return Err(QuantumError::InvalidDensityMatrix(
                "Empty state vector".to_string()
            ));
        }

        // Normalize
        let norm: f64 = psi.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
        if norm < 1e-15 {
            return Err(QuantumError::InvalidDensityMatrix(
                "Zero norm state vector".to_string()
            ));
        }

        let normalized: Vec<Complex64> = psi.iter().map(|c| c / norm).collect();

        // ρ = |ψ⟩⟨ψ| = outer product
        let mut data = DMatrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                data[(i, j)] = normalized[i] * normalized[j].conj();
            }
        }

        Ok(Self {
            dimension: n,
            data,
        })
    }

    /// Create maximally mixed state ρ = I/d
    ///
    /// This represents complete ignorance about the quantum state.
    /// Has maximum von Neumann entropy S = log₂(d).
    ///
    /// # Reference
    /// Nielsen & Chuang (2010), Section 2.4.1
    pub fn maximally_mixed(dimension: usize) -> QuantumResult<Self> {
        if dimension == 0 {
            return Err(QuantumError::InvalidDensityMatrix(
                "Dimension must be > 0".to_string()
            ));
        }

        let mut data = DMatrix::zeros(dimension, dimension);
        let p = 1.0 / dimension as f64;
        for i in 0..dimension {
            data[(i, i)] = Complex64::new(p, 0.0);
        }

        Ok(Self { dimension, data })
    }

    /// Create ground state |0⟩⟨0|
    pub fn ground_state(dimension: usize) -> QuantumResult<Self> {
        if dimension == 0 {
            return Err(QuantumError::InvalidDensityMatrix(
                "Dimension must be > 0".to_string()
            ));
        }

        let mut data = DMatrix::zeros(dimension, dimension);
        data[(0, 0)] = Complex64::new(1.0, 0.0);

        Ok(Self { dimension, data })
    }

    /// Create thermal state at temperature T
    ///
    /// ρ = e^(-H/kT) / Z where Z = Tr(e^(-H/kT))
    ///
    /// For harmonic oscillator: H = ℏω(n + 1/2)
    ///
    /// # Arguments
    /// * `dimension` - Hilbert space dimension (max_fock + 1)
    /// * `omega` - Oscillator frequency in rad/s
    /// * `temperature` - Temperature in Kelvin
    ///
    /// # Reference
    /// Pathria & Beale (2011), Chapter 7
    pub fn thermal_state(dimension: usize, omega: f64, temperature: f64) -> QuantumResult<Self> {
        if dimension == 0 {
            return Err(QuantumError::InvalidDensityMatrix(
                "Dimension must be > 0".to_string()
            ));
        }

        const HBAR: f64 = 1.054571817e-34;  // J·s
        const KB: f64 = 1.380649e-23;        // J/K

        let mut data = DMatrix::zeros(dimension, dimension);

        if temperature < 1e-10 {
            // T → 0: ground state
            data[(0, 0)] = Complex64::new(1.0, 0.0);
        } else {
            let beta = HBAR * omega / (KB * temperature);
            let mut partition_fn = 0.0;

            // Calculate partition function and Boltzmann weights
            // p_n = e^(-βℏω(n+1/2)) / Z
            let mut weights = vec![0.0; dimension];
            for n in 0..dimension {
                weights[n] = (-beta * (n as f64 + 0.5)).exp();
                partition_fn += weights[n];
            }

            // Normalize and set diagonal
            for n in 0..dimension {
                data[(n, n)] = Complex64::new(weights[n] / partition_fn, 0.0);
            }
        }

        Ok(Self { dimension, data })
    }

    /// Validate density matrix properties
    fn validate(&self) -> QuantumResult<()> {
        // Check Hermitian: ρ = ρ†
        let adjoint = self.data.adjoint();
        let diff = (&self.data - &adjoint).norm();
        if diff > 1e-10 {
            return Err(QuantumError::InvalidDensityMatrix(
                format!("Not Hermitian: ||ρ - ρ†|| = {}", diff)
            ));
        }

        // Check trace = 1
        let trace = self.trace();
        if (trace - 1.0).abs() > 1e-10 {
            return Err(QuantumError::InvalidDensityMatrix(
                format!("Trace = {}, expected 1.0", trace)
            ));
        }

        // Check positive semidefinite (all eigenvalues ≥ 0)
        // For Hermitian matrix, use symmetric eigenvalue decomposition
        let eigenvalues = self.eigenvalues();
        for (i, &ev) in eigenvalues.iter().enumerate() {
            if ev < -1e-10 {
                return Err(QuantumError::InvalidDensityMatrix(
                    format!("Negative eigenvalue: λ_{} = {}", i, ev)
                ));
            }
        }

        Ok(())
    }

    /// Get eigenvalues of density matrix
    ///
    /// For a valid ρ, all eigenvalues are in [0, 1] and sum to 1.
    /// These eigenvalues are used to compute von Neumann entropy.
    pub fn eigenvalues(&self) -> Vec<f64> {
        // Extract real part for eigenvalue computation (ρ is Hermitian)
        let real_matrix: DMatrix<f64> = self.data.map(|c| c.re);
        let eigendecomp = real_matrix.symmetric_eigen();
        eigendecomp.eigenvalues.iter().copied().collect()
    }

    /// Trace of density matrix: Tr(ρ)
    ///
    /// Should always be 1.0 for valid density matrices.
    pub fn trace(&self) -> f64 {
        self.data.diagonal().iter().map(|c| c.re).sum()
    }

    /// Purity: Tr(ρ²)
    ///
    /// - Pure state: Tr(ρ²) = 1
    /// - Maximally mixed: Tr(ρ²) = 1/d
    /// - Range: 1/d ≤ Tr(ρ²) ≤ 1
    ///
    /// # Reference
    /// Nielsen & Chuang (2010), Section 2.4.1
    pub fn purity(&self) -> f64 {
        let rho_squared = &self.data * &self.data;
        rho_squared.diagonal().iter().map(|c| c.re).sum()
    }

    /// Von Neumann entropy: S(ρ) = -Tr(ρ log ρ)
    ///
    /// Computed via eigenvalues: S = -Σᵢ λᵢ log₂(λᵢ)
    ///
    /// - Pure state: S = 0
    /// - Maximally mixed: S = log₂(d)
    ///
    /// # Reference
    /// - von Neumann, J. (1932). "Mathematische Grundlagen der Quantenmechanik."
    /// - Nielsen & Chuang (2010), Eq. 11.53
    pub fn von_neumann_entropy(&self) -> f64 {
        let eigenvalues = self.eigenvalues();
        let mut entropy = 0.0;

        for &lambda in &eigenvalues {
            if lambda > 1e-15 {
                entropy -= lambda * lambda.log2();
            }
        }

        entropy.max(0.0)  // Numerical safety
    }

    /// Linear entropy: S_L(ρ) = 1 - Tr(ρ²)
    ///
    /// Simpler to compute than von Neumann entropy.
    /// - Pure state: S_L = 0
    /// - Maximally mixed: S_L = 1 - 1/d
    pub fn linear_entropy(&self) -> f64 {
        1.0 - self.purity()
    }

    /// Get reference to internal data
    pub fn data(&self) -> &DMatrix<Complex64> {
        &self.data
    }

    /// Get mutable reference to internal data
    ///
    /// WARNING: Caller must ensure matrix remains valid density matrix
    pub fn data_mut(&mut self) -> &mut DMatrix<Complex64> {
        &mut self.data
    }

    /// Get Hilbert space dimension
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get diagonal elements (populations)
    ///
    /// These are the probabilities of finding the system in each basis state.
    /// Used for IIT calculations via state probability distributions.
    pub fn populations(&self) -> Vec<f64> {
        self.data.diagonal().iter().map(|c| c.re).collect()
    }

    /// Get off-diagonal coherences
    ///
    /// Coherences represent quantum superpositions and are destroyed
    /// by dephasing (a form of decoherence).
    pub fn coherences(&self) -> DMatrix<Complex64> {
        let mut coh = self.data.clone();
        for i in 0..self.dimension {
            coh[(i, i)] = Complex64::new(0.0, 0.0);
        }
        coh
    }

    /// Partial trace over subsystem B
    ///
    /// For composite system ρ_AB of dimension d_A × d_B,
    /// computes ρ_A = Tr_B[ρ_AB]
    ///
    /// # Arguments
    /// * `dim_a` - Dimension of subsystem A to keep
    /// * `dim_b` - Dimension of subsystem B to trace out
    ///
    /// # Reference
    /// Nielsen & Chuang (2010), Eq. 2.178
    pub fn partial_trace_b(&self, dim_a: usize, dim_b: usize) -> QuantumResult<Self> {
        if dim_a * dim_b != self.dimension {
            return Err(QuantumError::DimensionMismatch {
                expected: self.dimension,
                actual: dim_a * dim_b,
            });
        }

        let mut rho_a = DMatrix::zeros(dim_a, dim_a);

        // ρ_A = Tr_B[ρ_AB] = Σⱼ ⟨j|_B ρ_AB |j⟩_B
        for j in 0..dim_b {
            for i1 in 0..dim_a {
                for i2 in 0..dim_a {
                    // Index in composite space: i_AB = i_A + j*dim_A (row-major for B)
                    let idx1 = i1 + j * dim_a;
                    let idx2 = i2 + j * dim_a;

                    rho_a[(i1, i2)] += self.data[(idx1, idx2)];
                }
            }
        }

        Self::new(rho_a)
    }

    /// Partial trace over subsystem A
    ///
    /// Computes ρ_B = Tr_A[ρ_AB]
    pub fn partial_trace_a(&self, dim_a: usize, dim_b: usize) -> QuantumResult<Self> {
        if dim_a * dim_b != self.dimension {
            return Err(QuantumError::DimensionMismatch {
                expected: self.dimension,
                actual: dim_a * dim_b,
            });
        }

        let mut rho_b = DMatrix::zeros(dim_b, dim_b);

        // ρ_B = Tr_A[ρ_AB] = Σᵢ ⟨i|_A ρ_AB |i⟩_A
        for i in 0..dim_a {
            for j1 in 0..dim_b {
                for j2 in 0..dim_b {
                    let idx1 = i + j1 * dim_a;
                    let idx2 = i + j2 * dim_a;

                    rho_b[(j1, j2)] += self.data[(idx1, idx2)];
                }
            }
        }

        Self::new(rho_b)
    }

    /// Check if state is pure (Tr(ρ²) ≈ 1)
    pub fn is_pure(&self, tolerance: f64) -> bool {
        (self.purity() - 1.0).abs() < tolerance
    }

    /// Renormalize trace to 1 (for numerical stability)
    pub fn renormalize(&mut self) {
        let trace = self.trace();
        if trace > 1e-15 {
            self.data /= Complex64::new(trace, 0.0);
        }
    }

    /// Expectation value of operator: ⟨O⟩ = Tr(ρO)
    ///
    /// # Reference
    /// Nielsen & Chuang (2010), Eq. 2.158
    pub fn expectation(&self, operator: &DMatrix<Complex64>) -> QuantumResult<Complex64> {
        if operator.shape() != (self.dimension, self.dimension) {
            return Err(QuantumError::DimensionMismatch {
                expected: self.dimension,
                actual: operator.nrows(),
            });
        }

        let product = &self.data * operator;
        Ok(product.diagonal().sum())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_pure_state_purity() {
        // |0⟩ should have purity 1
        let psi = vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];
        let rho = DensityMatrix::from_pure_state(&psi).unwrap();

        assert_relative_eq!(rho.purity(), 1.0, epsilon = 1e-10);
        assert_relative_eq!(rho.trace(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_maximally_mixed_purity() {
        // I/2 should have purity 1/2
        let rho = DensityMatrix::maximally_mixed(2).unwrap();

        assert_relative_eq!(rho.purity(), 0.5, epsilon = 1e-10);
        assert_relative_eq!(rho.trace(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_von_neumann_entropy_pure() {
        // Pure state has S = 0
        let rho = DensityMatrix::ground_state(4).unwrap();
        assert_relative_eq!(rho.von_neumann_entropy(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_von_neumann_entropy_mixed() {
        // Maximally mixed in dim 2 has S = log₂(2) = 1
        let rho = DensityMatrix::maximally_mixed(2).unwrap();
        assert_relative_eq!(rho.von_neumann_entropy(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_partial_trace() {
        // Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
        // Partial trace gives maximally mixed state
        let mut data = DMatrix::zeros(4, 4);
        data[(0, 0)] = Complex64::new(0.5, 0.0);
        data[(0, 3)] = Complex64::new(0.5, 0.0);
        data[(3, 0)] = Complex64::new(0.5, 0.0);
        data[(3, 3)] = Complex64::new(0.5, 0.0);

        let rho_ab = DensityMatrix::new(data).unwrap();
        let rho_a = rho_ab.partial_trace_b(2, 2).unwrap();

        // Should be maximally mixed
        assert_relative_eq!(rho_a.purity(), 0.5, epsilon = 1e-10);
    }
}
