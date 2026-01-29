//! # Experiments: Decoherence is Necessary for Integrated Information
//!
//! This crate contains the experimental code demonstrating that
//! **noise/decoherence is necessary for Φ > 0** in quantum systems.
//!
//! ## Key Experiments
//!
//! 1. **noise_sweep** - Main experiment varying noise amplitude
//! 2. **stress_test** - Parameter sweep across coupling strength and evolution time
//! 3. **validation** - Physics validation (trace preservation, entropy bounds)
//!
//! ## Key Results
//!
//! | Noise Level | Noise Amplitude | Φ (bits) |
//! |-------------|-----------------|----------|
//! | Baseline    | 0.0             | **0.0**  |
//! | Low         | 0.5             | 0.00006  |
//! | Medium      | 1.0             | 0.00324  |
//! | High        | 2.0             | 0.01824  |
//! | **Optimal** | **5.0**         | **0.03655** |
//! | Extreme     | 10.0            | 0.02533  |
//! | Maximum     | 20.0            | 0.00273  |
//!
//! ## Physical Interpretation
//!
//! 1. **Φ = 0 without noise**: Pure quantum states are trivially factorizable
//! 2. **Φ > 0 with noise**: Lindblad dynamics creates genuine correlations
//! 3. **Optimal noise**: Stochastic resonance maximizes Φ
//! 4. **Excessive noise**: Destroys correlations, Φ decreases
//!
//! ## References
//!
//! ### Stochastic Resonance in Consciousness
//! - Gammaitoni, L., et al. (1998).
//!   "Stochastic resonance."
//!   Reviews of Modern Physics, 70(1), 223-287.
//!   DOI: 10.1103/RevModPhys.70.223
//!
//! ### Φ at Criticality
//! - Popiel, N. J., et al. (2020).
//!   "The Emergence of Integrated Information, Complexity, and
//!   'Consciousness' at Criticality."
//!   Entropy, 22(3), 339.
//!   DOI: 10.3390/e22030339
//!
//! ### Temperature Dependence of Φ
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

use serde::{Deserialize, Serialize};
use quantum_processor::prelude::*;
use iit::prelude::*;

/// Configuration for a single experiment run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentConfig {
    /// Number of quantum oscillators
    pub num_oscillators: usize,
    /// Maximum Fock level per oscillator
    pub max_fock: usize,
    /// Coupling strength (controls entanglement)
    pub coupling_strength: f64,
    /// Damping rate (noise strength)
    pub damping_rate: f64,
    /// Dephasing rate (noise strength)
    pub dephasing_rate: f64,
    /// Evolution time
    pub evolution_time: f64,
    /// Time step for RK4 integration
    pub time_step: f64,
    /// Experiment name/label
    pub name: String,
}

impl Default for ExperimentConfig {
    fn default() -> Self {
        Self {
            num_oscillators: 4,
            max_fock: 2,
            coupling_strength: 0.5,
            damping_rate: 0.02,
            dephasing_rate: 0.01,
            evolution_time: 5.0,
            time_step: 0.01,
            name: "default".to_string(),
        }
    }
}

/// Results from a single experiment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentResult {
    /// Configuration used
    pub config: ExperimentConfig,
    /// Φ values for different variants
    pub phi_results: Vec<PhiResultSummary>,
    /// Final system entropy
    pub entropy: f64,
    /// Final system purity
    pub purity: f64,
    /// Number of states in Hilbert space
    pub hilbert_dimension: usize,
}

/// Summary of a Φ calculation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhiResultSummary {
    /// Φ value
    pub phi: f64,
    /// Variant name
    pub variant: String,
    /// Number of partitions evaluated
    pub partitions_evaluated: usize,
}

/// Run a single experiment and compute Φ
pub fn run_experiment(config: &ExperimentConfig) -> Result<ExperimentResult, String> {
    // Create reservoir configuration
    let reservoir_config = ReservoirConfig {
        num_oscillators: config.num_oscillators,
        max_fock: config.max_fock,
        frequencies: vec![1e9; config.num_oscillators],
        coupling_strength: config.coupling_strength,
        damping_rate: config.damping_rate,
        dephasing_rate: config.dephasing_rate,
        temperature: 0.0,
    };

    // Create reservoir
    let mut reservoir = QuantumReservoir::new(reservoir_config)
        .map_err(|e| format!("Failed to create reservoir: {}", e))?;

    // Set initial coherent state input
    let input: Vec<f64> = (0..config.num_oscillators)
        .map(|i| 0.5 + 0.1 * i as f64)
        .collect();
    reservoir.set_input(&input)
        .map_err(|e| format!("Failed to set input: {}", e))?;

    // Evolve with Lindblad dynamics
    reservoir.evolve_total(config.evolution_time, config.time_step)
        .map_err(|e| format!("Failed to evolve: {}", e))?;

    // Get final state
    let probabilities = reservoir.get_state_probabilities();
    let entropy = reservoir.entropy();
    let purity = reservoir.purity();

    // Calculate Φ variants
    let levels_per_element = config.max_fock + 1;
    let phi_results = calculate_phi_variants(&probabilities, config.num_oscillators, levels_per_element)?;

    Ok(ExperimentResult {
        config: config.clone(),
        phi_results,
        entropy,
        purity,
        hilbert_dimension: reservoir.dimension(),
    })
}

/// Calculate all Φ variants for a distribution
fn calculate_phi_variants(
    distribution: &[f64],
    n_elements: usize,
    levels_per_element: usize,
) -> Result<Vec<PhiResultSummary>, String> {
    let mut results = Vec::new();

    // Calculate Synergy (I_synergy)
    if let Ok(phi) = calculate_phi(distribution, n_elements, levels_per_element, PhiVariant::Synergy) {
        results.push(PhiResultSummary {
            phi: phi.phi,
            variant: "I_synergy".to_string(),
            partitions_evaluated: phi.partitions_evaluated,
        });
    }

    // Calculate Total Correlation
    if let Ok(phi) = calculate_phi(distribution, n_elements, levels_per_element, PhiVariant::TotalCorrelation) {
        results.push(PhiResultSummary {
            phi: phi.phi,
            variant: "TC".to_string(),
            partitions_evaluated: phi.partitions_evaluated,
        });
    }

    // Calculate Φ_G (Geometric)
    if let Ok(phi) = calculate_phi(distribution, n_elements, levels_per_element, PhiVariant::Geometric) {
        results.push(PhiResultSummary {
            phi: phi.phi,
            variant: "Φ_G".to_string(),
            partitions_evaluated: phi.partitions_evaluated,
        });
    }

    // Calculate Φ_IIT (only for small systems)
    if n_elements <= 12 {
        if let Ok(phi) = calculate_phi(distribution, n_elements, levels_per_element, PhiVariant::IIT) {
            results.push(PhiResultSummary {
                phi: phi.phi,
                variant: "Φ_IIT".to_string(),
                partitions_evaluated: phi.partitions_evaluated,
            });
        }
    }

    Ok(results)
}

/// Create noise level configurations for sweep
pub fn create_noise_sweep_configs() -> Vec<ExperimentConfig> {
    let noise_levels = vec![
        ("Baseline", 0.0),
        ("Very Low", 0.1),
        ("Low", 0.5),
        ("Medium", 1.0),
        ("High", 2.0),
        ("Very High", 5.0),
        ("Extreme", 10.0),
        ("Maximum", 20.0),
    ];

    noise_levels.into_iter()
        .map(|(name, noise)| ExperimentConfig {
            name: name.to_string(),
            damping_rate: noise * 0.02,
            dephasing_rate: noise * 0.01,
            ..ExperimentConfig::default()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_experiment_config_default() {
        let config = ExperimentConfig::default();
        assert_eq!(config.num_oscillators, 4);
        assert_eq!(config.max_fock, 2);
    }

    #[test]
    fn test_noise_sweep_configs() {
        let configs = create_noise_sweep_configs();
        assert_eq!(configs.len(), 8);

        // First should be baseline (zero noise)
        assert_eq!(configs[0].damping_rate, 0.0);

        // Last should have maximum noise
        assert!(configs.last().unwrap().damping_rate > 0.0);
    }
}
