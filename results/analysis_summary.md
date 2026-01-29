# Analysis Summary: Decoherence is Necessary for Integrated Information

## Key Finding

**Pure quantum states (noise=0) have exactly Φ = 0.**
**Mixed states from Lindblad dynamics exhibit Φ > 0 with an optimal noise level.**

## Experimental Results

### Noise Sweep (28 configurations, 4 system sizes × 7 noise levels)

| System Size | Effective Neurons | Baseline Φ | Optimal Noise | Maximum Φ |
|-------------|-------------------|------------|---------------|-----------|
| Small       | 81                | 0.0        | 10.0          | 0.0327    |
| Medium      | 243               | 0.0        | 10.0          | 0.0312    |
| Large       | 64                | 0.0        | 10.0          | 0.0286    |
| XLarge      | 729               | 0.0        | 5.0           | **0.0365** |

### Global Maximum

- **Φ_max = 0.0365 bits**
- Configuration: XLarge system (6 oscillators, max_fock=2, 729 effective states)
- Noise amplitude: 5.0 (Very High)
- Runtime: ~2 hours on Apple M1

## Physical Interpretation

1. **Φ = 0 without noise**: Pure quantum states are trivially factorizable across any bipartition (Schmidt decomposition)

2. **Φ > 0 with noise**: Lindblad dynamics creates genuine statistical correlations that resist factorization

3. **Optimal noise (stochastic resonance)**: Too little noise → insufficient mixing; too much noise → correlations destroyed

4. **Scaling**: Larger systems achieve higher Φ with moderate noise

## Methodology

- **Quantum system**: Coupled harmonic oscillators (quantum reservoir computing)
- **Dynamics**: Lindblad master equation with thermal and dephasing channels
- **Φ calculation**: IIT 4.0 with exact MIP search for small systems

## References

1. Tononi et al. (2016) - IIT 3.0
2. Albantakis et al. (2023) - IIT 4.0
3. Tegmark (2015) - Consciousness as a state of matter
4. Lindblad (1976) - Master equation

---

*Author: Francisco Molina-Burgos, Avermex Research Division*
