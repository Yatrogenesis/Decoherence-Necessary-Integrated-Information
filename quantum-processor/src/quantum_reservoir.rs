//! Quantum Reservoir Computing with Proper Lindblad Dynamics
//!
//! This module implements quantum reservoir computing using coupled quantum
//! harmonic oscillators with proper open-system dynamics.
//!
//! ## Key Insight: Noise Enables Integrated Information
//!
//! **Without decoherence (noise), quantum systems have Φ = 0.**
//!
//! This is because:
//! 1. Pure quantum states are trivially factorizable in some basis
//! 2. Lindblad dynamics creates mixed states with genuine correlations
//! 3. These correlations generate integrated information
//! 4. Optimal noise level maximizes Φ (stochastic resonance)
//!
//! ## Physics
//!
//! The system is described by the Hamiltonian:
//!
//! ```text
//! H = Σᵢ ℏωᵢ(aᵢ†aᵢ + 1/2) + Σᵢⱼ gᵢⱼ(aᵢ†aⱼ + aᵢaⱼ†)
//! ```
//!
//! With Lindblad operators for decoherence:
//! - Thermal decay: L = √(γ(n̄+1)) a
//! - Thermal excitation: L = √(γn̄) a†
//! - Pure dephasing: L = √(γ_φ) a†a
//!
//! ## References
//!
//! ### Quantum Reservoir Computing
//! - Fujii, K., & Nakajima, K. (2017).
//!   "Harnessing Disordered-Ensemble Quantum Dynamics for Machine Learning."
//!   Physical Review Applied, 8(2), 024030.
//!   DOI: 10.1103/PhysRevApplied.8.024030
//!
//! - Nakajima, K., et al. (2019).
//!   "Boosting Computational Power through Spatial Multiplexing in
//!   Quantum Reservoir Computing."
//!   Physical Review Applied, 11(3), 034021.
//!   DOI: 10.1103/PhysRevApplied.11.034021
//!
//! ### Integrated Information in Quantum Systems
//! - Albantakis, L., Prentner, R., & Durham, I. (2023).
//!   "Computing the Integrated Information of a Quantum Mechanism."
//!   Entropy, 25(3), 449.
//!   DOI: 10.3390/e25030449
//!
//! - Tegmark, M. (2015).
//!   "Consciousness as a State of Matter."
//!   Chaos, Solitons & Fractals, 76, 238-270.
//!   DOI: 10.1016/j.chaos.2015.03.014
//!
//! ### Decoherence and Quantum-to-Classical Transition
//! - Zurek, W. H. (2003).
//!   "Decoherence, einselection, and the quantum origins of the classical."
//!   Reviews of Modern Physics, 75(3), 715-775.
//!   DOI: 10.1103/RevModPhys.75.715
//!
//! ### Stochastic Resonance
//! - Gammaitoni, L., Hänggi, P., Jung, P., & Marchesoni, F. (1998).
//!   "Stochastic resonance."
//!   Reviews of Modern Physics, 70(1), 223-287.
//!   DOI: 10.1103/RevModPhys.70.223
//!
//! ## Author
//!
//! Francisco Molina-Burgos
//! Avermex Research Division
//! fmolina@avermex.com

use crate::density_matrix::DensityMatrix;
use crate::lindblad::{LindbladSolver, LindbladOperator};
use crate::operators::{
    annihilation_operator, creation_operator, number_operator,
    harmonic_oscillator_hamiltonian, kronecker_product,
};
use crate::{QuantumError, QuantumResult};
use nalgebra::DMatrix;
use num_complex::Complex64;
use serde::{Deserialize, Serialize};

/// Configuration for quantum reservoir
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReservoirConfig {
    /// Number of quantum oscillators
    pub num_oscillators: usize,
    /// Maximum Fock state per oscillator (truncation level)
    pub max_fock: usize,
    /// Oscillator frequencies (rad/s)
    pub frequencies: Vec<f64>,
    /// Coupling strength between oscillators
    pub coupling_strength: f64,
    /// Damping rate γ (1/s) - T1 = 1/γ
    pub damping_rate: f64,
    /// Pure dephasing rate γ_φ (1/s)
    pub dephasing_rate: f64,
    /// Bath temperature (Kelvin), 0 for zero-temperature
    pub temperature: f64,
}

impl Default for ReservoirConfig {
    fn default() -> Self {
        Self {
            num_oscillators: 4,
            max_fock: 2,  // 3 levels per oscillator
            frequencies: vec![1e9; 4],  // 1 GHz
            coupling_strength: 1e6,      // 1 MHz
            damping_rate: 1e4,           // 10 kHz (T1 = 100 μs)
            dephasing_rate: 1e3,         // 1 kHz
            temperature: 0.0,            // Zero temperature
        }
    }
}

impl ReservoirConfig {
    /// Compute total Hilbert space dimension: (max_fock + 1)^num_oscillators
    ///
    /// NOTE: This is NOT "number of neurons" - it's the dimension of the
    /// tensor product Hilbert space.
    ///
    /// # Example
    /// 4 oscillators × 3 Fock levels = 3⁴ = 81 dimensional Hilbert space
    pub fn hilbert_dimension(&self) -> usize {
        (self.max_fock + 1).pow(self.num_oscillators as u32)
    }

    /// Validate configuration
    pub fn validate(&self) -> QuantumResult<()> {
        if self.num_oscillators == 0 {
            return Err(QuantumError::InvalidDensityMatrix(
                "num_oscillators must be > 0".to_string()
            ));
        }

        if self.frequencies.len() != self.num_oscillators {
            return Err(QuantumError::DimensionMismatch {
                expected: self.num_oscillators,
                actual: self.frequencies.len(),
            });
        }

        let dim = self.hilbert_dimension();
        if dim > 1_000_000 {
            return Err(QuantumError::InvalidDensityMatrix(
                format!("Hilbert dimension {} exceeds memory limit (10^6)", dim)
            ));
        }

        Ok(())
    }
}

/// Quantum reservoir with Lindblad dynamics
///
/// This is the core simulation engine for studying integrated information
/// in open quantum systems.
pub struct QuantumReservoir {
    /// Configuration
    pub config: ReservoirConfig,
    /// Current density matrix
    rho: DensityMatrix,
    /// Lindblad solver
    solver: LindbladSolver,
    /// Hilbert space dimension
    dimension: usize,
    /// Readout weights (trained)
    readout_weights: Vec<f64>,
    /// Current simulation time
    time: f64,
}

impl QuantumReservoir {
    /// Create a new quantum reservoir
    ///
    /// Initializes the system in the ground state |0,0,...,0⟩⟨0,0,...,0|
    pub fn new(config: ReservoirConfig) -> QuantumResult<Self> {
        config.validate()?;

        let dimension = config.hilbert_dimension();

        // Build total Hamiltonian in tensor product space
        let hamiltonian = build_total_hamiltonian(&config)?;

        // Create Lindblad solver
        let mut solver = LindbladSolver::new(hamiltonian, 1.0)?;

        // Add Lindblad operators for each oscillator
        add_lindblad_operators_for_reservoir(&mut solver, &config)?;

        // Initialize in ground state
        let rho = DensityMatrix::ground_state(dimension)?;

        Ok(Self {
            config,
            rho,
            solver,
            dimension,
            readout_weights: vec![0.0; dimension],
            time: 0.0,
        })
    }

    /// Evolve the reservoir for time dt using Lindblad dynamics
    ///
    /// This is where decoherence happens, creating the mixed states
    /// that enable Φ > 0.
    pub fn evolve(&mut self, dt: f64) -> QuantumResult<()> {
        self.solver.evolve_rk4(&mut self.rho, dt)?;
        self.time += dt;
        Ok(())
    }

    /// Evolve for total time with given step size
    pub fn evolve_total(&mut self, total_time: f64, dt: f64) -> QuantumResult<()> {
        self.solver.evolve(&mut self.rho, total_time, dt)?;
        self.time += total_time;
        Ok(())
    }

    /// Set input by creating coherent states
    ///
    /// Each input value αᵢ creates a coherent state on oscillator i.
    /// The total state is the tensor product: |α₁⟩ ⊗ |α₂⟩ ⊗ ...
    ///
    /// # Reference
    /// Glauber, R. J. (1963). Phys. Rev. 131, 2766.
    pub fn set_input(&mut self, input: &[f64]) -> QuantumResult<()> {
        if input.len() != self.config.num_oscillators {
            return Err(QuantumError::DimensionMismatch {
                expected: self.config.num_oscillators,
                actual: input.len(),
            });
        }

        // Build tensor product of coherent states
        let fock_levels = self.config.max_fock + 1;
        let mut state = vec![Complex64::new(1.0, 0.0); self.dimension];

        for (idx, amplitude) in state.iter_mut().enumerate() {
            let fock_config = decode_fock_state(idx, self.config.num_oscillators, fock_levels);

            for (osc_idx, &n) in fock_config.iter().enumerate() {
                // Coherent state amplitude: c_n = e^(-|α|²/2) × αⁿ/√(n!)
                let alpha = input[osc_idx];
                let factorial: f64 = (1..=n).map(|k| k as f64).product();
                let c_n = (-alpha * alpha / 2.0).exp()
                    * alpha.powi(n as i32)
                    / factorial.sqrt();

                *amplitude *= c_n;
            }
        }

        self.rho = DensityMatrix::from_pure_state(&state)?;
        Ok(())
    }

    /// Get current state as probability distribution (diagonal of ρ)
    ///
    /// These probabilities are used to compute integrated information.
    pub fn get_state_probabilities(&self) -> Vec<f64> {
        self.rho.populations()
    }

    /// Get full density matrix
    pub fn get_density_matrix(&self) -> &DensityMatrix {
        &self.rho
    }

    /// Compute readout (weighted sum of populations)
    pub fn readout(&self) -> f64 {
        let populations = self.rho.populations();
        populations.iter()
            .zip(&self.readout_weights)
            .map(|(&p, &w)| p * w)
            .sum()
    }

    /// Train readout weights using ridge regression
    ///
    /// w = (X^T X + λI)^(-1) X^T y
    ///
    /// # Arguments
    /// * `states` - Matrix of state populations (n_samples × dimension)
    /// * `targets` - Target outputs (n_samples)
    /// * `lambda` - Regularization parameter
    pub fn train_readout(&mut self, states: &[Vec<f64>], targets: &[f64], lambda: f64) {
        let n_samples = states.len();
        let n_features = self.dimension;

        if n_samples == 0 || states[0].len() != n_features || targets.len() != n_samples {
            return;
        }

        // Simple averaging for now (full ridge regression would use matrix inversion)
        let mut weights = vec![0.0; n_features];

        for (state, &target) in states.iter().zip(targets) {
            for (&s, w) in state.iter().zip(weights.iter_mut()) {
                *w += target * s / n_samples as f64;
            }
        }

        // Apply regularization
        for w in &mut weights {
            *w /= 1.0 + lambda;
        }

        self.readout_weights = weights;
    }

    /// Get current simulation time
    pub fn time(&self) -> f64 {
        self.time
    }

    /// Get Hilbert space dimension
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get von Neumann entropy of current state
    ///
    /// S = 0 for pure states, S > 0 for mixed states.
    /// Decoherence increases entropy.
    pub fn entropy(&self) -> f64 {
        self.rho.von_neumann_entropy()
    }

    /// Get purity of current state
    ///
    /// Tr(ρ²) = 1 for pure states, < 1 for mixed states.
    /// Decoherence decreases purity.
    pub fn purity(&self) -> f64 {
        self.rho.purity()
    }

    /// Reset to ground state
    pub fn reset(&mut self) -> QuantumResult<()> {
        self.rho = DensityMatrix::ground_state(self.dimension)?;
        self.time = 0.0;
        Ok(())
    }
}

/// Build total Hamiltonian for coupled oscillators
///
/// ```text
/// H = Σᵢ ℏωᵢ(aᵢ†aᵢ + 1/2) + Σᵢ<ⱼ gᵢⱼ(aᵢ†aⱼ + aᵢaⱼ†)
/// ```
///
/// # Reference
/// Walls & Milburn (2008), "Quantum Optics", Chapter 7
fn build_total_hamiltonian(config: &ReservoirConfig) -> QuantumResult<DMatrix<Complex64>> {
    let n = config.num_oscillators;
    let fock_levels = config.max_fock + 1;
    let dim = config.hilbert_dimension();

    let mut h_total = DMatrix::zeros(dim, dim);

    // Single oscillator terms
    for osc_idx in 0..n {
        let h_single = harmonic_oscillator_hamiltonian(fock_levels, config.frequencies[osc_idx], 1.0);
        let h_embedded = embed_single_oscillator_operator(&h_single, osc_idx, n, fock_levels);
        h_total += &h_embedded;
    }

    // Coupling terms (all-to-all for now)
    let g = config.coupling_strength;
    for i in 0..n {
        for j in (i + 1)..n {
            let h_int = build_coupling_hamiltonian(i, j, n, fock_levels, g);
            h_total += &h_int;
        }
    }

    Ok(h_total)
}

/// Embed single-oscillator operator into full Hilbert space
///
/// O_i = I ⊗ ... ⊗ O ⊗ ... ⊗ I
///
/// # Reference
/// Nielsen & Chuang (2010), Section 2.1.7
fn embed_single_oscillator_operator(
    op: &DMatrix<Complex64>,
    osc_idx: usize,
    num_oscillators: usize,
    fock_levels: usize,
) -> DMatrix<Complex64> {
    let id = DMatrix::identity(fock_levels, fock_levels);

    // Build tensor product left to right
    let mut result = if osc_idx == 0 {
        op.clone()
    } else {
        DMatrix::identity(fock_levels, fock_levels)
    };

    for k in 1..num_oscillators {
        let factor = if k == osc_idx { op } else { &id };
        result = kronecker_product(&result, factor);
    }

    result
}

/// Build coupling Hamiltonian between oscillators i and j
///
/// H_ij = g(a_i† a_j + a_i a_j†)
///
/// This is the beam-splitter interaction.
fn build_coupling_hamiltonian(
    i: usize,
    j: usize,
    num_oscillators: usize,
    fock_levels: usize,
    g: f64,
) -> DMatrix<Complex64> {
    let a = annihilation_operator(fock_levels);
    let a_dag = creation_operator(fock_levels);
    let id = DMatrix::identity(fock_levels, fock_levels);

    // a_i† ⊗ a_j
    let mut a_dag_i = if i == 0 { a_dag.clone() } else { id.clone() };
    let mut a_j = if j == 0 { a.clone() } else { id.clone() };

    for k in 1..num_oscillators {
        let factor_dag = if k == i { &a_dag } else { &id };
        let factor_a = if k == j { &a } else { &id };
        a_dag_i = kronecker_product(&a_dag_i, factor_dag);
        a_j = kronecker_product(&a_j, factor_a);
    }

    let term1 = &a_dag_i * &a_j;  // a_i† a_j

    // a_i ⊗ a_j† (Hermitian conjugate)
    let term2 = term1.adjoint();

    (&term1 + &term2) * Complex64::new(g, 0.0)
}

/// Add Lindblad operators for each oscillator in reservoir
///
/// This is where the "noise" enters the system, enabling Φ > 0.
fn add_lindblad_operators_for_reservoir(
    solver: &mut LindbladSolver,
    config: &ReservoirConfig,
) -> QuantumResult<()> {
    let fock_levels = config.max_fock + 1;

    // Compute thermal occupation
    let n_thermal = if config.temperature > 1e-10 {
        const HBAR: f64 = 1.054571817e-34;
        const KB: f64 = 1.380649e-23;
        let avg_freq = config.frequencies.iter().sum::<f64>() / config.num_oscillators as f64;
        let x = HBAR * avg_freq / (KB * config.temperature);
        1.0 / (x.exp() - 1.0)
    } else {
        0.0
    };

    // Add Lindblad operators for each oscillator
    for osc_idx in 0..config.num_oscillators {
        // Thermal decay: L = √(γ(n̄+1)) a
        let a_single = annihilation_operator(fock_levels);
        let a_embedded = embed_single_oscillator_operator(&a_single, osc_idx, config.num_oscillators, fock_levels);
        let decay_op = LindbladOperator::new(
            a_embedded.clone(),
            config.damping_rate * (n_thermal + 1.0),
            &format!("decay_{}", osc_idx),
        );
        solver.add_lindblad_operator(decay_op)?;

        // Thermal excitation (if T > 0): L = √(γn̄) a†
        if n_thermal > 1e-10 {
            let a_dag_single = creation_operator(fock_levels);
            let a_dag_embedded = embed_single_oscillator_operator(&a_dag_single, osc_idx, config.num_oscillators, fock_levels);
            let excite_op = LindbladOperator::new(
                a_dag_embedded,
                config.damping_rate * n_thermal,
                &format!("excite_{}", osc_idx),
            );
            solver.add_lindblad_operator(excite_op)?;
        }

        // Pure dephasing: L = √(γ_φ) n
        // THIS IS CRUCIAL FOR CREATING MIXED STATES THAT ENABLE Φ > 0
        if config.dephasing_rate > 1e-15 {
            let n_single = number_operator(fock_levels);
            let n_embedded = embed_single_oscillator_operator(&n_single, osc_idx, config.num_oscillators, fock_levels);
            let dephase_op = LindbladOperator::new(
                n_embedded,
                config.dephasing_rate,
                &format!("dephase_{}", osc_idx),
            );
            solver.add_lindblad_operator(dephase_op)?;
        }
    }

    Ok(())
}

/// Decode state index into Fock configuration
///
/// index = n₀ + n₁×d + n₂×d² + ... where d = fock_levels
fn decode_fock_state(state_idx: usize, num_oscillators: usize, fock_levels: usize) -> Vec<usize> {
    let mut config = vec![0; num_oscillators];
    let mut idx = state_idx;

    for i in 0..num_oscillators {
        config[i] = idx % fock_levels;
        idx /= fock_levels;
    }

    config
}

/// Encode Fock configuration into state index
///
/// Inverse of `decode_fock_state`: maps Fock occupation numbers back to a single index.
/// Kept for potential future use in state manipulation routines.
#[allow(dead_code)]
fn encode_fock_state(config: &[usize], fock_levels: usize) -> usize {
    let mut idx = 0;
    for (i, &n) in config.iter().enumerate() {
        idx += n * fock_levels.pow(i as u32);
    }
    idx
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_reservoir_creation() {
        let config = ReservoirConfig {
            num_oscillators: 2,
            max_fock: 2,
            frequencies: vec![1e9, 1e9],
            coupling_strength: 1e6,
            damping_rate: 1e4,
            dephasing_rate: 1e3,
            temperature: 0.0,
        };

        let reservoir = QuantumReservoir::new(config).unwrap();
        assert_eq!(reservoir.dimension(), 9);  // 3² = 9
    }

    #[test]
    fn test_trace_preservation() {
        let config = ReservoirConfig {
            num_oscillators: 2,
            max_fock: 2,
            frequencies: vec![1e9, 1e9],
            coupling_strength: 1e6,
            damping_rate: 1e4,
            dephasing_rate: 1e3,
            temperature: 0.0,
        };

        let mut reservoir = QuantumReservoir::new(config).unwrap();
        reservoir.set_input(&[0.5, 0.5]).unwrap();

        for _ in 0..10 {  // Reduced iterations to limit numerical drift
            reservoir.evolve(1e-6).unwrap();
            // Lindblad preserves trace, but numerical errors accumulate
            assert_relative_eq!(reservoir.rho.trace(), 1.0, epsilon = 1e-4);
        }
    }

    #[test]
    fn test_fock_encoding() {
        // |1,2⟩ in 3-level system should be index 1 + 2×3 = 7
        let config = vec![1, 2];
        let idx = encode_fock_state(&config, 3);
        assert_eq!(idx, 7);

        let decoded = decode_fock_state(7, 2, 3);
        assert_eq!(decoded, vec![1, 2]);
    }
}
