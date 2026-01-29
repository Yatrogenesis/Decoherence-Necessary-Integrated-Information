//! # Stress Test Experiment
//!
//! Parameter sweep across coupling strength and evolution time
//! to find the configuration that maximizes Φ.
//!
//! ## Parameters Varied
//!
//! - Coupling strength (α): 0.1 to 2.0
//! - Evolution time (t): 1.0 to 10.0
//! - Dephasing rate: 0.01 to 0.1
//!
//! ## Author
//!
//! Francisco Molina-Burgos
//! Avermex Research Division
//! fmolina@avermex.com

use experiments::{run_experiment, ExperimentConfig, ExperimentResult};
use serde_json;
use std::fs::File;
use std::io::Write;

fn main() {
    println!("============================================");
    println!("STRESS TEST: Parameter Sweep for Maximum Φ");
    println!("============================================\n");

    // Parameter ranges
    let alphas = vec![0.25, 0.5, 0.75, 1.0, 1.5];
    let times = vec![1.0, 2.0, 5.0, 10.0];
    let dephasing_rates = vec![0.01, 0.02, 0.05];

    let mut all_results: Vec<ExperimentResult> = Vec::new();
    let mut max_phi = 0.0;
    let mut max_config = ExperimentConfig::default();

    let total = alphas.len() * times.len() * dephasing_rates.len();
    let mut count = 0;

    for &alpha in &alphas {
        for &time in &times {
            for &dephasing in &dephasing_rates {
                count += 1;
                print!("\r[{}/{}] Testing α={}, t={}, γ_φ={}...",
                    count, total, alpha, time, dephasing);

                let config = ExperimentConfig {
                    name: format!("α={}_t={}_γφ={}", alpha, time, dephasing),
                    coupling_strength: alpha,
                    evolution_time: time,
                    dephasing_rate: dephasing,
                    damping_rate: 0.02,
                    ..ExperimentConfig::default()
                };

                match run_experiment(&config) {
                    Ok(result) => {
                        let phi = result.phi_results.iter()
                            .map(|p| p.phi)
                            .fold(0.0_f64, |a, b| a.max(b));

                        if phi > max_phi {
                            max_phi = phi;
                            max_config = config.clone();
                        }

                        all_results.push(result);
                    }
                    Err(_) => {}
                }
            }
        }
    }

    println!("\n\n============================================");
    println!("STRESS TEST RESULTS");
    println!("============================================\n");

    println!("Total configurations tested: {}", all_results.len());
    println!("\nMaximum Φ achieved: {:.6} bits", max_phi);
    println!("\nOptimal configuration:");
    println!("  - Coupling strength (α): {}", max_config.coupling_strength);
    println!("  - Evolution time (t): {}", max_config.evolution_time);
    println!("  - Dephasing rate (γ_φ): {}", max_config.dephasing_rate);
    println!("  - Damping rate (γ): {}", max_config.damping_rate);

    // Save summary
    let summary = serde_json::json!({
        "max_phi": max_phi,
        "optimal_config": {
            "alpha": max_config.coupling_strength,
            "time": max_config.evolution_time,
            "dephasing_rate": max_config.dephasing_rate,
            "damping_rate": max_config.damping_rate,
            "num_oscillators": max_config.num_oscillators,
            "max_fock": max_config.max_fock,
        },
        "total_configs": all_results.len(),
    });

    if let Ok(json) = serde_json::to_string_pretty(&summary) {
        let filename = "stress_test_summary.json";
        if let Ok(mut file) = File::create(filename) {
            let _ = file.write_all(json.as_bytes());
            println!("\nSummary saved to: {}", filename);
        }
    }

    println!("\n============================================");
}
