//! Quantum Operators for Harmonic Oscillators
//!
//! Implements the fundamental operators for quantum harmonic oscillators
//! in the Fock (number) basis:
//!
//! - Creation operator: a† |n⟩ = √(n+1) |n+1⟩
//! - Annihilation operator: a |n⟩ = √n |n-1⟩
//! - Number operator: n̂ = a†a
//! - Position: x̂ = (a + a†)/√2
//! - Momentum: p̂ = i(a† - a)/√2
//!
//! ## Commutation Relations
//!
//! ```text
//! [a, a†] = 1
//! [x̂, p̂] = iℏ
//! ```
//!
//! ## References
//!
//! ### Quantum Harmonic Oscillator
//! - Sakurai, J. J., & Napolitano, J. J. (2017).
//!   "Modern Quantum Mechanics" (2nd ed.). Cambridge University Press.
//!   Chapter 2.3: Simple Harmonic Oscillator.
//!   ISBN: 978-1108422413
//!
//! ### Creation and Annihilation Operators
//! - Dirac, P. A. M. (1927). "The Quantum Theory of the Emission and
//!   Absorption of Radiation." Proceedings of the Royal Society A, 114(767), 243-265.
//!   DOI: 10.1098/rspa.1927.0039
//!
//! ### Coherent States
//! - Glauber, R. J. (1963). "Coherent and Incoherent States of the Radiation Field."
//!   Physical Review, 131(6), 2766-2788.
//!   DOI: 10.1103/PhysRev.131.2766
//!
//! ### Tensor Products in Quantum Mechanics
//! - Nielsen, M. A., & Chuang, I. L. (2010).
//!   "Quantum Computation and Quantum Information" (10th Anniversary ed.).
//!   Cambridge University Press. Section 2.1.7.
//!   ISBN: 978-1107002173
//!
//! ## Author
//!
//! Francisco Molina-Burgos
//! Avermex Research Division
//! fmolina@avermex.com

use nalgebra::DMatrix;
use num_complex::Complex64;

/// Create annihilation operator a in Fock basis
///
/// ```text
/// a |n⟩ = √n |n-1⟩
/// ```
///
/// Matrix representation (dimension d):
/// ```text
/// a = | 0  √1  0   0  ... |
///     | 0   0  √2  0  ... |
///     | 0   0   0  √3 ... |
///     | .   .   .   .  .  |
/// ```
///
/// # Arguments
/// * `dimension` - Truncated Hilbert space dimension (max_fock + 1)
///
/// # Reference
/// Sakurai & Napolitano (2017), Eq. 2.3.1
pub fn annihilation_operator(dimension: usize) -> DMatrix<Complex64> {
    let mut a = DMatrix::zeros(dimension, dimension);

    for n in 1..dimension {
        a[(n - 1, n)] = Complex64::new((n as f64).sqrt(), 0.0);
    }

    a
}

/// Create creation operator a† in Fock basis
///
/// ```text
/// a† |n⟩ = √(n+1) |n+1⟩
/// ```
///
/// a† = (a)ᵀ for real basis (since a is real-valued)
///
/// # Arguments
/// * `dimension` - Truncated Hilbert space dimension
///
/// # Reference
/// Sakurai & Napolitano (2017), Eq. 2.3.3
pub fn creation_operator(dimension: usize) -> DMatrix<Complex64> {
    annihilation_operator(dimension).transpose()
}

/// Create number operator n̂ = a†a
///
/// ```text
/// n̂ |n⟩ = n |n⟩
/// ```
///
/// Diagonal matrix with eigenvalues 0, 1, 2, ...
///
/// # Reference
/// Sakurai & Napolitano (2017), Eq. 2.3.12
pub fn number_operator(dimension: usize) -> DMatrix<Complex64> {
    let mut n = DMatrix::zeros(dimension, dimension);

    for i in 0..dimension {
        n[(i, i)] = Complex64::new(i as f64, 0.0);
    }

    n
}

/// Create position operator x̂ = (a + a†)/√2
///
/// In units where ℏ = m = ω = 1
///
/// # Reference
/// Sakurai & Napolitano (2017), Eq. 2.3.15
pub fn position_operator(dimension: usize) -> DMatrix<Complex64> {
    let a = annihilation_operator(dimension);
    let a_dag = creation_operator(dimension);
    let sqrt2 = 2.0_f64.sqrt();

    (&a + &a_dag) / Complex64::new(sqrt2, 0.0)
}

/// Create momentum operator p̂ = i(a† - a)/√2
///
/// In units where ℏ = m = ω = 1
///
/// # Reference
/// Sakurai & Napolitano (2017), Eq. 2.3.16
pub fn momentum_operator(dimension: usize) -> DMatrix<Complex64> {
    let a = annihilation_operator(dimension);
    let a_dag = creation_operator(dimension);
    let sqrt2 = 2.0_f64.sqrt();
    let i = Complex64::new(0.0, 1.0);

    (&a_dag - &a) * i / Complex64::new(sqrt2, 0.0)
}

/// Create harmonic oscillator Hamiltonian H = ℏω(a†a + 1/2)
///
/// # Arguments
/// * `dimension` - Hilbert space dimension
/// * `omega` - Oscillator frequency (rad/s)
/// * `hbar` - Reduced Planck constant (default: 1 for natural units)
///
/// # Reference
/// Sakurai & Napolitano (2017), Eq. 2.3.10
pub fn harmonic_oscillator_hamiltonian(
    dimension: usize,
    omega: f64,
    hbar: f64,
) -> DMatrix<Complex64> {
    let n = number_operator(dimension);
    let half = DMatrix::from_diagonal_element(dimension, dimension, Complex64::new(0.5, 0.0));

    (&n + &half) * Complex64::new(hbar * omega, 0.0)
}

/// Create displacement operator D(α) = exp(αa† - α*a)
///
/// Creates coherent state: |α⟩ = D(α)|0⟩
///
/// For numerical implementation, we use the truncated series expansion.
///
/// # Arguments
/// * `alpha` - Displacement amplitude
/// * `dimension` - Hilbert space dimension
/// * `terms` - Number of terms in series expansion
///
/// # Reference
/// Glauber, R. J. (1963). Phys. Rev. 131, 2766.
/// DOI: 10.1103/PhysRev.131.2766
pub fn displacement_operator(
    alpha: Complex64,
    dimension: usize,
    terms: usize,
) -> DMatrix<Complex64> {
    let a = annihilation_operator(dimension);
    let a_dag = creation_operator(dimension);

    // X = αa† - α*a
    let x = &a_dag * alpha - &a * alpha.conj();

    // D(α) = exp(X) ≈ Σₖ X^k / k!
    matrix_exponential(&x, terms)
}

/// Compute matrix exponential via Taylor series
///
/// ```text
/// exp(A) = Σₖ Aᵏ/k!
/// ```
///
/// # Arguments
/// * `a` - Matrix to exponentiate
/// * `terms` - Number of terms in series
///
/// # Reference
/// Moler, C., & Van Loan, C. (2003). "Nineteen Dubious Ways to Compute
/// the Matrix Exponential, Twenty-Five Years Later."
/// SIAM Review, 45(1), 3-49. DOI: 10.1137/S00361445024180
fn matrix_exponential(a: &DMatrix<Complex64>, terms: usize) -> DMatrix<Complex64> {
    let n = a.nrows();
    let mut result = DMatrix::identity(n, n);
    let mut power = DMatrix::identity(n, n);
    let mut factorial = 1.0;

    for k in 1..terms {
        power = &power * a;
        factorial *= k as f64;
        result += &power / Complex64::new(factorial, 0.0);
    }

    result
}

/// Create beam-splitter interaction Hamiltonian
///
/// ```text
/// H_int = g(a†b + ab†)
/// ```
///
/// For two oscillators, this Hamiltonian in tensor product space.
///
/// # Arguments
/// * `dim_a` - Dimension of oscillator A
/// * `dim_b` - Dimension of oscillator B
/// * `g` - Coupling strength
///
/// # Reference
/// Walls, D. F., & Milburn, G. J. (2008).
/// "Quantum Optics" (2nd ed.). Springer. Section 7.3.
/// ISBN: 978-3540285731
pub fn beam_splitter_hamiltonian(
    dim_a: usize,
    dim_b: usize,
    g: f64,
) -> DMatrix<Complex64> {
    let a_single = annihilation_operator(dim_a);
    let a_dag_single = creation_operator(dim_a);
    let b_single = annihilation_operator(dim_b);
    let b_dag_single = creation_operator(dim_b);

    // Identity matrices
    let id_a = DMatrix::identity(dim_a, dim_a);
    let id_b = DMatrix::identity(dim_b, dim_b);

    // Tensor products: a ⊗ I_b, I_a ⊗ b
    let a = kronecker_product(&a_single, &id_b);
    let a_dag = kronecker_product(&a_dag_single, &id_b);
    let b = kronecker_product(&id_a, &b_single);
    let b_dag = kronecker_product(&id_a, &b_dag_single);

    // H = g(a†b + ab†)
    (&a_dag * &b + &a * &b_dag) * Complex64::new(g, 0.0)
}

/// Kronecker (tensor) product of two matrices
///
/// ```text
/// (A ⊗ B)_{(i₁,j₁),(i₂,j₂)} = A_{i₁,j₁} × B_{i₂,j₂}
/// ```
///
/// # Reference
/// Nielsen & Chuang (2010), Section 2.1.7, Eq. 2.25
pub fn kronecker_product(
    a: &DMatrix<Complex64>,
    b: &DMatrix<Complex64>,
) -> DMatrix<Complex64> {
    let (m_a, n_a) = a.shape();
    let (m_b, n_b) = b.shape();

    let mut result = DMatrix::zeros(m_a * m_b, n_a * n_b);

    for i_a in 0..m_a {
        for j_a in 0..n_a {
            for i_b in 0..m_b {
                for j_b in 0..n_b {
                    let row = i_a * m_b + i_b;
                    let col = j_a * n_b + j_b;
                    result[(row, col)] = a[(i_a, j_a)] * b[(i_b, j_b)];
                }
            }
        }
    }

    result
}

/// Commutator [A, B] = AB - BA
///
/// Fundamental operation in quantum mechanics for computing
/// time evolution and uncertainty relations.
///
/// # Reference
/// Sakurai & Napolitano (2017), Section 1.4
pub fn commutator(
    a: &DMatrix<Complex64>,
    b: &DMatrix<Complex64>,
) -> DMatrix<Complex64> {
    a * b - b * a
}

/// Anticommutator {A, B} = AB + BA
///
/// Used in Lindblad dissipator and fermionic systems.
///
/// # Reference
/// Breuer & Petruccione (2002), Eq. 3.63
pub fn anticommutator(
    a: &DMatrix<Complex64>,
    b: &DMatrix<Complex64>,
) -> DMatrix<Complex64> {
    a * b + b * a
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_commutation_relation() {
        // [a, a†] = 1
        let a = annihilation_operator(10);
        let a_dag = creation_operator(10);

        let comm = commutator(&a, &a_dag);

        // Should be approximately identity
        for i in 0..9 {  // Last element has truncation error
            assert_relative_eq!(comm[(i, i)].re, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_number_operator() {
        // n̂ = a†a
        let a = annihilation_operator(5);
        let a_dag = creation_operator(5);
        let n_computed = &a_dag * &a;
        let n_direct = number_operator(5);

        for i in 0..5 {
            for j in 0..5 {
                assert_relative_eq!(
                    n_computed[(i, j)].re,
                    n_direct[(i, j)].re,
                    epsilon = 1e-10
                );
            }
        }
    }

    #[test]
    fn test_annihilation_on_fock() {
        // a|n⟩ = √n |n-1⟩
        let a = annihilation_operator(5);

        // a|1⟩ = |0⟩
        let mut fock_1 = vec![Complex64::new(0.0, 0.0); 5];
        fock_1[1] = Complex64::new(1.0, 0.0);

        let result: Vec<Complex64> = (0..5)
            .map(|i| (0..5).map(|j| a[(i, j)] * fock_1[j]).sum())
            .collect();

        assert_relative_eq!(result[0].re, 1.0, epsilon = 1e-10);
        assert_relative_eq!(result[1].re, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_creation_on_fock() {
        // a†|0⟩ = |1⟩
        let a_dag = creation_operator(5);

        let mut fock_0 = vec![Complex64::new(0.0, 0.0); 5];
        fock_0[0] = Complex64::new(1.0, 0.0);

        let result: Vec<Complex64> = (0..5)
            .map(|i| (0..5).map(|j| a_dag[(i, j)] * fock_0[j]).sum())
            .collect();

        assert_relative_eq!(result[0].re, 0.0, epsilon = 1e-10);
        assert_relative_eq!(result[1].re, 1.0, epsilon = 1e-10);
    }
}
