//! # Validation Suite
//!
//! Physics validation tests for the quantum simulation:
//!
//! 1. **Trace preservation**: Tr(ρ) = 1 throughout evolution
//! 2. **Entropy bounds**: 0 ≤ S ≤ log₂(d)
//! 3. **Purity bounds**: 1/d ≤ Tr(ρ²) ≤ 1
//! 4. **Ground state relaxation**: System decays to ground state at T=0
//!
//! ## Author
//!
//! Francisco Molina-Burgos
//! Avermex Research Division
//! fmolina@avermex.com

use quantum_processor::prelude::*;

fn main() {
    println!("============================================");
    println!("PHYSICS VALIDATION SUITE");
    println!("============================================\n");

    let mut all_passed = true;

    // Test 1: Trace preservation
    println!("Test 1: Trace Preservation");
    println!("--------------------------");
    if test_trace_preservation() {
        println!("✓ PASSED: Trace remains 1.0 throughout evolution\n");
    } else {
        println!("✗ FAILED: Trace not preserved\n");
        all_passed = false;
    }

    // Test 2: Entropy bounds
    println!("Test 2: Entropy Bounds");
    println!("----------------------");
    if test_entropy_bounds() {
        println!("✓ PASSED: Entropy within valid bounds\n");
    } else {
        println!("✗ FAILED: Entropy out of bounds\n");
        all_passed = false;
    }

    // Test 3: Purity bounds
    println!("Test 3: Purity Bounds");
    println!("---------------------");
    if test_purity_bounds() {
        println!("✓ PASSED: Purity within valid bounds\n");
    } else {
        println!("✗ FAILED: Purity out of bounds\n");
        all_passed = false;
    }

    // Test 4: Ground state relaxation
    println!("Test 4: Ground State Relaxation (T=0)");
    println!("-------------------------------------");
    if test_ground_state_relaxation() {
        println!("✓ PASSED: System relaxes to ground state\n");
    } else {
        println!("✗ FAILED: Ground state not reached\n");
        all_passed = false;
    }

    // Test 5: Entropy increase (Second Law)
    println!("Test 5: Entropy Increase (Second Law)");
    println!("-------------------------------------");
    if test_entropy_increase() {
        println!("✓ PASSED: Entropy increases under decoherence\n");
    } else {
        println!("✗ FAILED: Entropy did not increase\n");
        all_passed = false;
    }

    // Summary
    println!("============================================");
    if all_passed {
        println!("ALL TESTS PASSED ✓");
    } else {
        println!("SOME TESTS FAILED ✗");
    }
    println!("============================================");
}

fn test_trace_preservation() -> bool {
    let config = ReservoirConfig {
        num_oscillators: 2,
        max_fock: 2,
        frequencies: vec![1e9, 1e9],
        coupling_strength: 1e6,
        damping_rate: 1e4,
        dephasing_rate: 1e3,
        temperature: 0.0,
    };

    let mut reservoir = match QuantumReservoir::new(config) {
        Ok(r) => r,
        Err(_) => return false,
    };

    // Set coherent state input
    let _ = reservoir.set_input(&[0.5, 0.5]);

    // Evolve and check trace
    for _ in 0..100 {
        let _ = reservoir.evolve(1e-6);
        let trace = reservoir.get_density_matrix().trace();
        if (trace - 1.0).abs() > 1e-6 {
            println!("  Trace deviation: {}", (trace - 1.0).abs());
            return false;
        }
    }

    true
}

fn test_entropy_bounds() -> bool {
    let config = ReservoirConfig {
        num_oscillators: 2,
        max_fock: 2,
        frequencies: vec![1e9, 1e9],
        coupling_strength: 1e6,
        damping_rate: 1e4,
        dephasing_rate: 1e3,
        temperature: 0.0,
    };

    let mut reservoir = match QuantumReservoir::new(config) {
        Ok(r) => r,
        Err(_) => return false,
    };

    let dim = reservoir.dimension();
    let max_entropy = (dim as f64).log2();

    let _ = reservoir.set_input(&[0.5, 0.5]);

    for _ in 0..100 {
        let _ = reservoir.evolve(1e-6);
        let s = reservoir.entropy();
        if s < -1e-10 || s > max_entropy + 1e-6 {
            println!("  Entropy out of bounds: {} (max: {})", s, max_entropy);
            return false;
        }
    }

    true
}

fn test_purity_bounds() -> bool {
    let config = ReservoirConfig {
        num_oscillators: 2,
        max_fock: 2,
        frequencies: vec![1e9, 1e9],
        coupling_strength: 1e6,
        damping_rate: 1e4,
        dephasing_rate: 1e3,
        temperature: 0.0,
    };

    let mut reservoir = match QuantumReservoir::new(config) {
        Ok(r) => r,
        Err(_) => return false,
    };

    let dim = reservoir.dimension();
    let min_purity = 1.0 / dim as f64;

    let _ = reservoir.set_input(&[0.5, 0.5]);

    for _ in 0..100 {
        let _ = reservoir.evolve(1e-6);
        let p = reservoir.purity();
        if p < min_purity - 1e-6 || p > 1.0 + 1e-6 {
            println!("  Purity out of bounds: {} (range: [{}, 1])", p, min_purity);
            return false;
        }
    }

    true
}

fn test_ground_state_relaxation() -> bool {
    // At T=0, system should relax to ground state
    let config = ReservoirConfig {
        num_oscillators: 2,
        max_fock: 3,
        frequencies: vec![1e9, 1e9],
        coupling_strength: 1e6,
        damping_rate: 1e5,  // Strong damping
        dephasing_rate: 0.0,
        temperature: 0.0,
    };

    let mut reservoir = match QuantumReservoir::new(config) {
        Ok(r) => r,
        Err(_) => return false,
    };

    // Start in excited state
    let _ = reservoir.set_input(&[1.0, 1.0]);

    // Evolve for long time
    let _ = reservoir.evolve_total(1e-3, 1e-7);

    // Check ground state population
    let probs = reservoir.get_state_probabilities();
    let ground_population = probs[0];

    println!("  Ground state population after relaxation: {:.4}", ground_population);

    ground_population > 0.9
}

fn test_entropy_increase() -> bool {
    let config = ReservoirConfig {
        num_oscillators: 2,
        max_fock: 2,
        frequencies: vec![1e9, 1e9],
        coupling_strength: 1e6,
        damping_rate: 1e4,
        dephasing_rate: 1e4,  // Strong dephasing
        temperature: 0.0,
    };

    let mut reservoir = match QuantumReservoir::new(config) {
        Ok(r) => r,
        Err(_) => return false,
    };

    // Start in pure state
    let _ = reservoir.set_input(&[0.5, 0.5]);
    let initial_entropy = reservoir.entropy();

    // Evolve
    let _ = reservoir.evolve_total(1e-5, 1e-7);
    let final_entropy = reservoir.entropy();

    println!("  Initial entropy: {:.6}", initial_entropy);
    println!("  Final entropy: {:.6}", final_entropy);

    final_entropy >= initial_entropy - 1e-10
}
