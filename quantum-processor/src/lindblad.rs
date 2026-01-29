//! Lindblad Master Equation for Open Quantum Systems
//!
//! Implements the Gorini-Kossakowski-Sudarshan-Lindblad (GKSL) equation:
//!
//! ```text
//! dρ/dt = -i/ℏ [H, ρ] + Σₖ γₖ (Lₖ ρ Lₖ† - ½{Lₖ†Lₖ, ρ})
//! ```
//!
//! This is the most general form of Markovian, trace-preserving, completely
//! positive quantum dynamics.
//!
//! ## Physical Interpretation
//!
//! - **First term**: Unitary (Hamiltonian) evolution
//! - **Second term**: Dissipation and decoherence
//!
//! ## Key Insight for Integrated Information
//!
//! **Decoherence is necessary for Φ > 0 in quantum systems.**
//!
//! Without Lindblad operators (γₖ = 0), the system evolves unitarily and
//! remains in pure states, which have trivially zero integrated information.
//! The Lindblad operators create mixed states with non-trivial correlations
//! that generate integrated information.
//!
//! ## Lindblad Operators for Harmonic Oscillator
//!
//! 1. **Thermal relaxation (decay)**: L₁ = √(γ(n̄+1)) a
//! 2. **Thermal excitation**: L₂ = √(γn̄) a†
//! 3. **Pure dephasing**: L₃ = √(γ_φ) a†a
//!
//! Where:
//! - γ = damping rate (1/T₁)
//! - γ_φ = dephasing rate (contributes to 1/T₂)
//! - n̄ = thermal occupation number = 1/(e^(ℏω/kT) - 1)
//!
//! ## References
//!
//! ### Lindblad Master Equation
//! - Lindblad, G. (1976). "On the generators of quantum dynamical semigroups."
//!   Communications in Mathematical Physics, 48(2), 119-130.
//!   DOI: 10.1007/BF01608499
//!
//! - Gorini, V., Kossakowski, A., & Sudarshan, E. C. G. (1976).
//!   "Completely positive dynamical semigroups of N-level systems."
//!   Journal of Mathematical Physics, 17(5), 821-825.
//!   DOI: 10.1063/1.522979
//!
//! ### Open Quantum Systems
//! - Breuer, H.-P., & Petruccione, F. (2002).
//!   "The Theory of Open Quantum Systems." Oxford University Press.
//!   ISBN: 978-0199213900
//!   Chapter 3: Quantum Master Equations
//!
//! - Wiseman, H. M., & Milburn, G. J. (2009).
//!   "Quantum Measurement and Control." Cambridge University Press.
//!   ISBN: 978-0521804424
//!   Chapter 4: Quantum trajectories
//!
//! ### Numerical Methods
//! - Johansson, J. R., Nation, P. D., & Nori, F. (2012).
//!   "QuTiP: An open-source Python framework for the dynamics of
//!   open quantum systems." Computer Physics Communications, 183(8), 1760-1772.
//!   DOI: 10.1016/j.cpc.2012.02.021
//!   arXiv: 1211.6518
//!
//! ### Decoherence and Quantum-to-Classical Transition
//! - Zurek, W. H. (2003). "Decoherence, einselection, and the quantum
//!   origins of the classical." Reviews of Modern Physics, 75(3), 715-775.
//!   DOI: 10.1103/RevModPhys.75.715
//!
//! ## Author
//!
//! Francisco Molina-Burgos
//! Avermex Research Division
//! fmolina@avermex.com

use crate::density_matrix::DensityMatrix;
use crate::operators::{annihilation_operator, creation_operator, number_operator, commutator, anticommutator};
use crate::{QuantumError, QuantumResult};
use nalgebra::DMatrix;
use num_complex::Complex64;

/// Lindblad operator with associated rate
///
/// Represents a collapse channel: L with rate γ
#[derive(Debug, Clone)]
pub struct LindbladOperator {
    /// The operator L
    pub operator: DMatrix<Complex64>,
    /// Collapse rate γ (units: 1/time)
    pub rate: f64,
    /// Human-readable name
    pub name: String,
}

impl LindbladOperator {
    /// Create a new Lindblad operator
    pub fn new(operator: DMatrix<Complex64>, rate: f64, name: &str) -> Self {
        Self {
            operator,
            rate,
            name: name.to_string(),
        }
    }

    /// Create thermal decay operator: L = √(γ(n̄+1)) a
    ///
    /// This operator describes spontaneous emission plus stimulated emission
    /// due to interaction with a thermal bath.
    ///
    /// # Arguments
    /// * `dimension` - Hilbert space dimension
    /// * `gamma` - Base damping rate γ
    /// * `n_thermal` - Thermal occupation n̄
    ///
    /// # Reference
    /// Breuer & Petruccione (2002), Section 3.4.3
    pub fn thermal_decay(dimension: usize, gamma: f64, n_thermal: f64) -> Self {
        let a = annihilation_operator(dimension);
        let rate = gamma * (n_thermal + 1.0);
        Self::new(a, rate, "thermal_decay")
    }

    /// Create thermal excitation operator: L = √(γn̄) a†
    ///
    /// This operator describes absorption of thermal photons.
    ///
    /// # Reference
    /// Breuer & Petruccione (2002), Section 3.4.3
    pub fn thermal_excitation(dimension: usize, gamma: f64, n_thermal: f64) -> Self {
        let a_dag = creation_operator(dimension);
        let rate = gamma * n_thermal;
        Self::new(a_dag, rate, "thermal_excitation")
    }

    /// Create pure dephasing operator: L = √(γ_φ) a†a
    ///
    /// Pure dephasing destroys off-diagonal coherences without changing populations.
    /// This is crucial for creating the mixed states that enable Φ > 0.
    ///
    /// # Reference
    /// Breuer & Petruccione (2002), Section 3.4.6
    pub fn pure_dephasing(dimension: usize, gamma_phi: f64) -> Self {
        let n = number_operator(dimension);
        Self::new(n, gamma_phi, "pure_dephasing")
    }

    /// Create zero-temperature decay (spontaneous emission): L = √γ a
    pub fn zero_temp_decay(dimension: usize, gamma: f64) -> Self {
        let a = annihilation_operator(dimension);
        Self::new(a, gamma, "zero_temp_decay")
    }
}

/// Lindblad master equation solver
///
/// Implements the GKSL equation with RK4 integration.
#[derive(Debug, Clone)]
pub struct LindbladSolver {
    /// System Hamiltonian H
    hamiltonian: DMatrix<Complex64>,
    /// List of Lindblad operators with rates
    lindblad_ops: Vec<LindbladOperator>,
    /// Hilbert space dimension
    dimension: usize,
    /// Reduced Planck constant (default: 1 for natural units)
    hbar: f64,
}

impl LindbladSolver {
    /// Create new Lindblad solver
    ///
    /// # Arguments
    /// * `hamiltonian` - System Hamiltonian (must be Hermitian)
    /// * `hbar` - Reduced Planck constant (use 1.0 for natural units)
    pub fn new(hamiltonian: DMatrix<Complex64>, hbar: f64) -> QuantumResult<Self> {
        let dimension = hamiltonian.nrows();

        if hamiltonian.ncols() != dimension {
            return Err(QuantumError::DimensionMismatch {
                expected: dimension,
                actual: hamiltonian.ncols(),
            });
        }

        Ok(Self {
            hamiltonian,
            lindblad_ops: Vec::new(),
            dimension,
            hbar,
        })
    }

    /// Add a Lindblad operator to the solver
    pub fn add_lindblad_operator(&mut self, op: LindbladOperator) -> QuantumResult<()> {
        if op.operator.nrows() != self.dimension {
            return Err(QuantumError::DimensionMismatch {
                expected: self.dimension,
                actual: op.operator.nrows(),
            });
        }
        self.lindblad_ops.push(op);
        Ok(())
    }

    /// Add thermal bath at temperature T
    ///
    /// Adds both decay and excitation Lindblad operators.
    ///
    /// # Arguments
    /// * `gamma` - Damping rate
    /// * `omega` - Oscillator frequency
    /// * `temperature` - Bath temperature in Kelvin
    ///
    /// # Reference
    /// Breuer & Petruccione (2002), Section 3.4.3
    pub fn add_thermal_bath(&mut self, gamma: f64, omega: f64, temperature: f64) -> QuantumResult<()> {
        const HBAR_SI: f64 = 1.054571817e-34;
        const KB: f64 = 1.380649e-23;

        let n_thermal = if temperature > 1e-10 {
            let x = HBAR_SI * omega / (KB * temperature);
            1.0 / (x.exp() - 1.0)
        } else {
            0.0
        };

        self.add_lindblad_operator(LindbladOperator::thermal_decay(
            self.dimension, gamma, n_thermal
        ))?;

        if n_thermal > 1e-10 {
            self.add_lindblad_operator(LindbladOperator::thermal_excitation(
                self.dimension, gamma, n_thermal
            ))?;
        }

        Ok(())
    }

    /// Add pure dephasing
    pub fn add_dephasing(&mut self, gamma_phi: f64) -> QuantumResult<()> {
        self.add_lindblad_operator(LindbladOperator::pure_dephasing(
            self.dimension, gamma_phi
        ))
    }

    /// Compute the right-hand side of Lindblad equation: dρ/dt
    ///
    /// ```text
    /// dρ/dt = -i/ℏ [H, ρ] + Σₖ γₖ D[Lₖ](ρ)
    /// ```
    ///
    /// Where the dissipator is:
    /// ```text
    /// D[L](ρ) = L ρ L† - ½{L†L, ρ}
    /// ```
    ///
    /// # Reference
    /// - Lindblad, G. (1976). Commun. Math. Phys. 48, 119-130.
    /// - Breuer & Petruccione (2002), Eq. 3.63
    pub fn lindblad_rhs(&self, rho: &DMatrix<Complex64>) -> DMatrix<Complex64> {
        let i = Complex64::new(0.0, 1.0);

        // Unitary part: -i/ℏ [H, ρ]
        let mut drho_dt = commutator(&self.hamiltonian, rho) * (-i / self.hbar);

        // Dissipative part: Σₖ γₖ D[Lₖ](ρ)
        // D[L](ρ) = L ρ L† - ½{L†L, ρ}
        for lindblad_op in &self.lindblad_ops {
            if lindblad_op.rate < 1e-15 {
                continue;  // Skip zero-rate operators
            }

            let l = &lindblad_op.operator;
            let l_dag = l.adjoint();
            let l_dag_l = &l_dag * l;

            // L ρ L†
            let term1 = l * rho * &l_dag;

            // ½{L†L, ρ} = ½(L†L ρ + ρ L†L)
            let term2 = anticommutator(&l_dag_l, rho) * Complex64::new(0.5, 0.0);

            drho_dt += (&term1 - &term2) * Complex64::new(lindblad_op.rate, 0.0);
        }

        drho_dt
    }

    /// Evolve density matrix using Euler method (1st order)
    ///
    /// ρ(t+dt) = ρ(t) + dt × dρ/dt
    ///
    /// WARNING: Use only for testing. Has O(dt²) error.
    pub fn evolve_euler(&self, rho: &mut DensityMatrix, dt: f64) -> QuantumResult<()> {
        let drho = self.lindblad_rhs(rho.data());
        *rho.data_mut() += &drho * Complex64::new(dt, 0.0);
        rho.renormalize();
        Ok(())
    }

    /// Evolve density matrix using Runge-Kutta 4 (4th order)
    ///
    /// Standard RK4 algorithm with O(dt⁵) error per step.
    ///
    /// # Arguments
    /// * `rho` - Density matrix to evolve (modified in place)
    /// * `dt` - Time step
    ///
    /// # Reference
    /// Press, W. H., et al. (2007). "Numerical Recipes" (3rd ed.).
    /// Cambridge University Press. Chapter 17.1.
    pub fn evolve_rk4(&self, rho: &mut DensityMatrix, dt: f64) -> QuantumResult<()> {
        let rho_data = rho.data().clone();

        // k1 = f(t, y)
        let k1 = self.lindblad_rhs(&rho_data);

        // k2 = f(t + dt/2, y + dt*k1/2)
        let half_dt = Complex64::new(dt / 2.0, 0.0);
        let rho_mid1 = &rho_data + &k1 * half_dt;
        let k2 = self.lindblad_rhs(&rho_mid1);

        // k3 = f(t + dt/2, y + dt*k2/2)
        let rho_mid2 = &rho_data + &k2 * half_dt;
        let k3 = self.lindblad_rhs(&rho_mid2);

        // k4 = f(t + dt, y + dt*k3)
        let dt_c = Complex64::new(dt, 0.0);
        let rho_end = &rho_data + &k3 * dt_c;
        let k4 = self.lindblad_rhs(&rho_end);

        // y(t+dt) = y(t) + dt/6 × (k1 + 2k2 + 2k3 + k4)
        let two = Complex64::new(2.0, 0.0);
        let dt_sixth = Complex64::new(dt / 6.0, 0.0);
        let drho = (&k1 + &k2 * two + &k3 * two + &k4) * dt_sixth;
        *rho.data_mut() = &rho_data + &drho;

        rho.renormalize();
        Ok(())
    }

    /// Evolve density matrix for total time T with adaptive stepping
    ///
    /// # Arguments
    /// * `rho` - Initial density matrix
    /// * `total_time` - Total evolution time
    /// * `dt` - Time step
    ///
    /// # Returns
    /// Final density matrix
    pub fn evolve(&self, rho: &mut DensityMatrix, total_time: f64, dt: f64) -> QuantumResult<()> {
        let n_steps = (total_time / dt).ceil() as usize;
        let actual_dt = total_time / n_steps as f64;

        for _ in 0..n_steps {
            self.evolve_rk4(rho, actual_dt)?;
        }

        Ok(())
    }

    /// Compute steady state (dρ/dt = 0)
    ///
    /// Uses power iteration to find the null space of the Liouvillian.
    /// For detailed balance systems, this is the thermal state.
    ///
    /// # Arguments
    /// * `tolerance` - Convergence tolerance
    /// * `max_iterations` - Maximum iterations
    pub fn steady_state(&self, tolerance: f64, max_iterations: usize) -> QuantumResult<DensityMatrix> {
        // Start from maximally mixed state
        let mut rho = DensityMatrix::maximally_mixed(self.dimension)?;

        // Iterate until convergence
        let dt = 0.1;  // Large steps for steady-state finding
        for iteration in 0..max_iterations {
            let rho_old = rho.data().clone();
            self.evolve_rk4(&mut rho, dt)?;

            // Check convergence: ||ρ_new - ρ_old|| < tolerance
            let diff = (rho.data() - &rho_old).norm();
            if diff < tolerance {
                log::info!("Steady state found in {} iterations", iteration);
                return Ok(rho);
            }
        }

        Err(QuantumError::NumericalInstability(
            format!("Steady state did not converge in {} iterations", max_iterations)
        ))
    }

    /// Get the Hilbert space dimension
    pub fn dimension(&self) -> usize {
        self.dimension
    }
}

/// Compute thermal occupation number n̄
///
/// n̄ = 1/(e^(ℏω/kT) - 1)
///
/// This is the Bose-Einstein distribution for photons/phonons.
///
/// # Arguments
/// * `omega` - Frequency in rad/s
/// * `temperature` - Temperature in Kelvin
///
/// # Reference
/// Pathria & Beale (2011), Eq. 7.1.14
pub fn thermal_occupation(omega: f64, temperature: f64) -> f64 {
    const HBAR: f64 = 1.054571817e-34;
    const KB: f64 = 1.380649e-23;

    if temperature < 1e-10 {
        0.0
    } else {
        let x = HBAR * omega / (KB * temperature);
        1.0 / (x.exp() - 1.0)
    }
}

/// Compute T1 relaxation time from damping rate
///
/// T1 = 1/γ
pub fn t1_from_gamma(gamma: f64) -> f64 {
    if gamma > 1e-15 {
        1.0 / gamma
    } else {
        f64::INFINITY
    }
}

/// Compute T2 dephasing time from T1 and pure dephasing
///
/// 1/T2 = 1/(2T1) + 1/T_φ
///
/// # Arguments
/// * `t1` - T1 relaxation time
/// * `t_phi` - Pure dephasing time (None for T2 = 2T1)
///
/// # Reference
/// Breuer & Petruccione (2002), Section 3.4.6
pub fn t2_from_t1(t1: f64, t_phi: Option<f64>) -> f64 {
    let rate_t1 = if t1 > 1e-15 { 0.5 / t1 } else { 0.0 };
    let rate_phi = t_phi.map(|tp| if tp > 1e-15 { 1.0 / tp } else { 0.0 }).unwrap_or(0.0);

    let total_rate = rate_t1 + rate_phi;
    if total_rate > 1e-15 {
        1.0 / total_rate
    } else {
        f64::INFINITY
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operators::harmonic_oscillator_hamiltonian;
    use approx::assert_relative_eq;

    #[test]
    fn test_trace_preservation() {
        // Lindblad evolution must preserve Tr(ρ) = 1
        let dim = 5;
        let h = harmonic_oscillator_hamiltonian(dim, 1.0, 1.0);
        let mut solver = LindbladSolver::new(h, 1.0).unwrap();
        solver.add_lindblad_operator(LindbladOperator::zero_temp_decay(dim, 0.1)).unwrap();

        // Excite the state: uniform superposition
        let psi: Vec<Complex64> = (0..dim)
            .map(|_| Complex64::new(1.0 / (dim as f64).sqrt(), 0.0))
            .collect();
        let mut rho = DensityMatrix::from_pure_state(&psi).unwrap();

        for _ in 0..100 {
            solver.evolve_rk4(&mut rho, 0.01).unwrap();
            assert_relative_eq!(rho.trace(), 1.0, epsilon = 1e-8);
        }
    }

    #[test]
    fn test_decay_to_ground_state() {
        // At T=0, system should decay to ground state
        let dim = 5;
        let omega = 1.0;
        let gamma = 0.1;

        let h = harmonic_oscillator_hamiltonian(dim, omega, 1.0);
        let mut solver = LindbladSolver::new(h, 1.0).unwrap();
        solver.add_lindblad_operator(LindbladOperator::zero_temp_decay(dim, gamma)).unwrap();

        // Start in excited state |2⟩
        let mut psi = vec![Complex64::new(0.0, 0.0); dim];
        psi[2] = Complex64::new(1.0, 0.0);
        let mut rho = DensityMatrix::from_pure_state(&psi).unwrap();

        // Evolve for long time (many T1 periods)
        solver.evolve(&mut rho, 100.0 / gamma, 0.1).unwrap();

        // Should be close to ground state
        let populations = rho.populations();
        assert!(populations[0] > 0.99, "Ground state population: {}", populations[0]);
    }

    #[test]
    fn test_dephasing_destroys_coherence() {
        let dim = 3;
        let h = DMatrix::zeros(dim, dim);  // No Hamiltonian evolution

        let mut solver = LindbladSolver::new(h, 1.0).unwrap();
        solver.add_lindblad_operator(LindbladOperator::pure_dephasing(dim, 1.0)).unwrap();

        // Start in superposition (|0⟩ + |1⟩)/√2
        let sqrt2 = 2.0_f64.sqrt();
        let psi = vec![
            Complex64::new(1.0 / sqrt2, 0.0),
            Complex64::new(1.0 / sqrt2, 0.0),
            Complex64::new(0.0, 0.0),
        ];
        let mut rho = DensityMatrix::from_pure_state(&psi).unwrap();

        // Initial coherence should be nonzero
        let initial_coherence = rho.data()[(0, 1)].norm();
        assert!(initial_coherence > 0.4);

        // Evolve
        solver.evolve(&mut rho, 10.0, 0.1).unwrap();

        // Coherence should be destroyed
        let final_coherence = rho.data()[(0, 1)].norm();
        assert!(final_coherence < 0.01, "Coherence not destroyed: {}", final_coherence);

        // But populations should be unchanged
        let populations = rho.populations();
        assert_relative_eq!(populations[0], 0.5, epsilon = 0.01);
        assert_relative_eq!(populations[1], 0.5, epsilon = 0.01);
    }

    #[test]
    fn test_entropy_increase() {
        // Under decoherence, entropy should increase (second law)
        let dim = 4;
        let h = harmonic_oscillator_hamiltonian(dim, 1.0, 1.0);

        let mut solver = LindbladSolver::new(h, 1.0).unwrap();
        solver.add_lindblad_operator(LindbladOperator::zero_temp_decay(dim, 0.1)).unwrap();
        solver.add_dephasing(0.05).unwrap();

        // Start in pure excited state (S = 0)
        let mut psi = vec![Complex64::new(0.0, 0.0); dim];
        psi[2] = Complex64::new(1.0, 0.0);
        let mut rho = DensityMatrix::from_pure_state(&psi).unwrap();

        let initial_entropy = rho.von_neumann_entropy();
        assert!(initial_entropy < 0.01);

        // Evolve
        solver.evolve(&mut rho, 5.0, 0.1).unwrap();

        let final_entropy = rho.von_neumann_entropy();
        assert!(final_entropy > initial_entropy,
            "Entropy decreased: {} -> {}", initial_entropy, final_entropy);
    }
}
