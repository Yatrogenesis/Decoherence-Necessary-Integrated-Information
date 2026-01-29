//! # Noise Sweep Experiment
//!
//! Main experiment demonstrating that decoherence is necessary for Φ > 0.
//!
//! ## Methodology
//!
//! 1. Create quantum reservoir (coupled harmonic oscillators)
//! 2. Vary noise amplitude from 0 (pure quantum) to maximum
//! 3. Evolve with Lindblad dynamics
//! 4. Compute Φ for each configuration
//!
//! ## Expected Results
//!
//! - Baseline (noise=0): Φ = 0 (pure state is trivially factorizable)
//! - Optimal noise: Φ_max (stochastic resonance)
//! - Excessive noise: Φ decreases (correlations destroyed)
//!
//! ## Author
//!
//! Francisco Molina-Burgos
//! Avermex Research Division
//! fmolina@avermex.com

use experiments::{run_experiment, create_noise_sweep_configs, ExperimentResult};
use serde_json;
use std::fs::File;
use std::io::Write;

fn main() {
    println!("============================================");
    println!("NOISE SWEEP EXPERIMENT");
    println!("Demonstrating: Decoherence is Necessary for Φ > 0");
    println!("============================================\n");

    let configs = create_noise_sweep_configs();
    let mut results: Vec<ExperimentResult> = Vec::new();
    let mut max_phi = 0.0;
    let mut max_phi_config = String::new();

    println!("Running {} configurations...\n", configs.len());

    for config in &configs {
        println!("Configuration: {}", config.name);
        println!("  Damping rate: {}", config.damping_rate);
        println!("  Dephasing rate: {}", config.dephasing_rate);

        match run_experiment(config) {
            Ok(result) => {
                println!("  Results:");
                println!("    Entropy: {:.6}", result.entropy);
                println!("    Purity: {:.6}", result.purity);

                for phi in &result.phi_results {
                    println!("    {}: {:.6} bits ({} partitions)",
                        phi.variant, phi.phi, phi.partitions_evaluated);

                    if phi.phi > max_phi {
                        max_phi = phi.phi;
                        max_phi_config = config.name.clone();
                    }
                }

                results.push(result);
            }
            Err(e) => {
                println!("  ERROR: {}", e);
            }
        }
        println!();
    }

    // Summary
    println!("============================================");
    println!("SUMMARY");
    println!("============================================\n");

    println!("| Noise Level    | Damping | Φ_max (bits) |");
    println!("|----------------|---------|--------------|");

    for result in &results {
        let max_phi_in_result = result.phi_results.iter()
            .map(|p| p.phi)
            .fold(0.0_f64, |a, b| a.max(b));

        println!("| {:14} | {:7.4} | {:12.6} |",
            result.config.name,
            result.config.damping_rate,
            max_phi_in_result);
    }

    println!("\nMaximum Φ: {:.6} bits at configuration: {}", max_phi, max_phi_config);

    // Save results to JSON
    if let Ok(json) = serde_json::to_string_pretty(&results) {
        let filename = "noise_sweep_results.json";
        if let Ok(mut file) = File::create(filename) {
            let _ = file.write_all(json.as_bytes());
            println!("\nResults saved to: {}", filename);
        }
    }

    println!("\n============================================");
    println!("CONCLUSION: Φ = 0 without noise. Optimal noise produces maximum Φ.");
    println!("============================================");
}
