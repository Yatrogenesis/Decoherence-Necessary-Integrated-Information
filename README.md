# Decoherence is Necessary for Integrated Information

**Evidence from Quantum Reservoir Computing**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17932389.svg)](https://doi.org/10.5281/zenodo.17932389)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Rust](https://img.shields.io/badge/Rust-1.75+-orange.svg)](https://www.rust-lang.org/)

**Repository**: `Decoherence-Necessary-Integrated-Information-AVRD_2601`

## Abstract

This repository contains the code and experimental data supporting the discovery that **decoherence (noise) is necessary for integrated information (Φ > 0)** in quantum systems. Using rigorous Lindblad master equation dynamics combined with Integrated Information Theory (IIT) 4.0, we demonstrate that:

- **Pure quantum states have Φ = 0** (no integrated information)
- **Systems coupled to thermal baths exhibit Φ > 0**
- **Optimal noise level maximizes Φ** (stochastic resonance)
- **Maximum Φ achieved: 0.0365 bits** at intermediate noise

## Key Results

| Noise Level | Noise Amplitude (ε) | Φ_max (bits) |
|-------------|---------------------|--------------|
| Baseline (none) | 0.0 | **0.0000** (exactly zero) |
| Low | 0.5 | 0.00008 |
| Medium | 1.0 | 0.00324 |
| High | 2.0 | 0.01824 |
| **Very High** | **5.0** | **0.03655** |
| Extreme | 10.0 | 0.02533 |
| Maximum | 20.0 | 0.00273 |

**Conclusion**: Φ = 0 without noise. Optimal noise produces maximum integrated information.

## Figures

### Figure 1: Integrated Information vs Noise Amplitude
![Φ vs Noise](figures/fig1_phi_vs_noise.png)

### Figure 2: Stochastic Resonance Fit
![Stochastic Resonance](figures/fig2_stochastic_resonance.png)

### Figure 3: System Size Scaling
![System Scaling](figures/fig3_system_scaling.png)

### Figure 4: Pure vs Mixed States Comparison
![Baseline Comparison](figures/fig4_baseline_comparison.png)

## Physical Interpretation

The Lindblad master equation governs open quantum system dynamics:

```
dρ/dt = -i/ℏ [H, ρ] + Σₖ γₖ D[Lₖ](ρ)
```

Where the dissipator is:
```
D[L](ρ) = L ρ L† - ½{L†L, ρ}
```

**Jump operators implemented**:
- `L_decay = √(γ(n̄+1)) a` — Thermal decay
- `L_excite = √(γn̄) a†` — Thermal excitation
- `L_dephase = √(γ_φ) n` — Pure dephasing

**Why noise is necessary**:
1. Pure quantum states are trivially factorizable → Φ = 0
2. Decoherence creates mixed states with non-trivial correlations
3. These correlations generate integrated information
4. Too much noise destroys correlations → Φ decreases

## Installation

### Requirements
- Rust 1.75+ (with Cargo)
- Python 3.9+ (for figure generation)

### Build
```bash
git clone https://github.com/Yatrogenesis/Decoherence-Necessary-Integrated-Information-AVRD_2601.git
cd Decoherence-Necessary-Integrated-Information-AVRD_2601
cargo build --release
```

### Run experiments
```bash
# Main noise sweep experiment
cargo run --release --bin noise_sweep

# Stress test with all Φ variants
cargo run --release --bin stress_test

# Validation suite
cargo run --release --bin validation
```

### Generate figures
```bash
cd figures
pip install -r requirements.txt
python generate_figures.py
```

## Repository Structure

```
Decoherence-Necessary-Integrated-Information-AVRD_2601/
├── quantum-processor/     # Lindblad dynamics & density matrices
│   └── src/
│       ├── lindblad.rs           # Master equation solver
│       ├── density_matrix.rs     # ρ operations
│       ├── quantum_reservoir.rs  # Coupled oscillators
│       └── operators.rs          # a, a†, n operators
│
├── iit/                   # Integrated Information Theory
│   └── src/
│       ├── phi_variants.rs       # Φ_IIT, Φ_G, I_synergy, TC
│       ├── emd.rs                # Earth Mover's Distance (LP)
│       ├── entropy.rs            # Shannon, von Neumann
│       └── partition.rs          # MIP search
│
├── experiments/           # Experimental code
│   └── src/
│       ├── noise_sweep.rs        # Main experiment
│       ├── stress_test.rs        # Parameter sweep
│       └── validation.rs         # Physics checks
│
├── paper/                 # LaTeX paper
│   ├── main.tex
│   ├── references.bib
│   └── figures/
│
├── figures/               # Publication figures (600 DPI)
│   ├── generate_figures.py
│   ├── fig1_phi_vs_noise.{pdf,png,eps}
│   ├── fig2_stochastic_resonance.{pdf,png,eps}
│   ├── fig3_system_scaling.{pdf,png,eps}
│   └── fig4_baseline_comparison.{pdf,png,eps}
│
└── results/               # Pre-computed results
    └── consciousness_maximum_entanglement_results.json
```

## Theory

### Integrated Information Theory (IIT) 4.0

IIT proposes that consciousness corresponds to integrated information Φ, which measures a system's irreducibility. A system has Φ > 0 if and only if it cannot be reduced to independent parts without losing information.

**References**:
- Tononi et al. (2016) "Integrated information theory: from consciousness to its physical substrate"
- Albantakis et al. (2023) "IIT 4.0: Formulating the properties of phenomenal existence"
- Zanardi et al. (2018) "Towards quantum integrated information theory"
- Kleiner & Tull (2023) "Computing the integrated information of a quantum mechanism"

### Quantum Extension

For quantum systems, we compute Φ from the density matrix ρ by:
1. Computing state probabilities from diagonals
2. Partitioning over oscillators (not Hilbert space states)
3. Finding the Minimum Information Partition (MIP)
4. Computing EMD between whole and partitioned distributions

**Key insight**: Without decoherence, quantum systems remain in pure states that are trivially reducible, yielding Φ = 0.

### Related Work

| Author | Year | Key Finding |
|--------|------|-------------|
| Zanardi et al. | 2018 | Quantum IIT formulation with "dis-integrated" phase |
| Tegmark | 2015 | Φ maximizes at intermediate temperature |
| Popiel et al. | 2020 | Φ undergoes phase transition at criticality |
| Kleiner & Tull | 2023 | Formal extension of IIT to quantum gates |

## Citation

If you use this code or data, please cite:

```bibtex
@software{molina2026decoherence,
  author = {Molina-Burgos, Francisco},
  title = {Decoherence is Necessary for Integrated Information:
           Evidence from Quantum Reservoir Computing},
  year = {2026},
  doi = {10.5281/zenodo.17932389},
  url = {https://doi.org/10.5281/zenodo.17932389},
  license = {CC-BY-4.0}
}
```

**DOI**: [10.5281/zenodo.17932389](https://doi.org/10.5281/zenodo.17932389)

## License

This work is licensed under **Creative Commons Attribution 4.0 International (CC BY 4.0)**.

You are free to:
- **Share** — copy and redistribute the material in any medium or format
- **Adapt** — remix, transform, and build upon the material for any purpose

Under the following terms:
- **Attribution** — You must give appropriate credit to the author

Full license: https://creativecommons.org/licenses/by/4.0/

## Author

**Francisco Molina-Burgos**
Avermex Research Division
Mérida, Yucatán, México
fmolina@avermex.com

---

*"Consciousness requires noise."*

© 2026 Francisco Molina-Burgos, Avermex Research Division
