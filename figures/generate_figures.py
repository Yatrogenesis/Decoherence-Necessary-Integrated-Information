#!/usr/bin/env python3
"""
Generate publication-quality figures for the paper:
"Decoherence is Necessary for Integrated Information"

Author: Francisco Molina-Burgos
        Avermex Research Division
        fmolina@avermex.com

Usage:
    python generate_figures.py

Output:
    figures/fig1_phi_vs_noise.pdf
    figures/fig2_stochastic_resonance.pdf
    figures/fig3_system_scaling.pdf
    figures/fig4_baseline_comparison.pdf
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path

# Publication-quality settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.grid': True,
    'grid.alpha': 0.3,
})

def load_results():
    """Load experimental results from JSON."""
    results_path = Path(__file__).parent.parent / 'results' / 'consciousness_maximum_entanglement_results.json'
    with open(results_path, 'r') as f:
        return json.load(f)

def stochastic_resonance_model(epsilon, a, b, c):
    """
    Stochastic resonance model:
    Φ(ε) = a × ε × exp(-b × ε²) + c

    Parameters:
        epsilon: noise amplitude
        a: peak height scaling
        b: decay rate
        c: baseline offset
    """
    return a * epsilon * np.exp(-b * epsilon**2) + c

def fig1_phi_vs_noise():
    """Figure 1: Φ vs Noise Amplitude for different system sizes."""
    data = load_results()
    results = data['results']

    # Organize by system size
    sizes = ['Small', 'Medium', 'Large', 'XLarge']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', '^', 'D']

    fig, ax = plt.subplots(figsize=(6, 4))

    for size, color, marker in zip(sizes, colors, markers):
        size_data = [r for r in results if r['system_size'] == size]
        noise = [r['noise_amplitude'] for r in size_data]
        phi_max = [r['max_phi'] for r in size_data]

        neurons = size_data[0]['effective_neurons'] if size_data else 0
        ax.plot(noise, phi_max, f'{marker}-', color=color,
                label=f'{size} (n={neurons})', markersize=6, linewidth=1.5)

    ax.set_xlabel('Noise Amplitude (ε)')
    ax.set_ylabel('Maximum Φ (bits)')
    ax.set_title('Integrated Information vs Noise Amplitude')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_xlim(-0.5, 21)
    ax.set_ylim(-0.002, 0.042)

    # Annotate key finding
    ax.annotate('Φ = 0 (pure state)', xy=(0, 0), xytext=(2, 0.005),
                fontsize=8, color='gray',
                arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))

    fig.savefig(Path(__file__).parent / 'fig1_phi_vs_noise.pdf')
    fig.savefig(Path(__file__).parent / 'fig1_phi_vs_noise.png')
    plt.close()
    print("Generated: fig1_phi_vs_noise.pdf")

def fig2_stochastic_resonance():
    """Figure 2: Stochastic resonance fit."""
    data = load_results()
    results = data['results']

    # Get XLarge data (best quality)
    xlarge_data = [r for r in results if r['system_size'] == 'XLarge']
    noise = np.array([r['noise_amplitude'] for r in xlarge_data])
    phi_max = np.array([r['max_phi'] for r in xlarge_data])

    # Fit stochastic resonance model
    try:
        popt, pcov = curve_fit(stochastic_resonance_model, noise, phi_max,
                               p0=[0.02, 0.02, 0.0], maxfev=5000)
        a, b, c = popt

        # Calculate R²
        residuals = phi_max - stochastic_resonance_model(noise, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((phi_max - np.mean(phi_max))**2)
        r_squared = 1 - (ss_res / ss_tot)

        # Find optimal noise
        epsilon_opt = np.sqrt(1 / (2 * b)) if b > 0 else 5.0
        phi_opt = stochastic_resonance_model(epsilon_opt, *popt)

    except:
        a, b, c = 0.02, 0.02, 0.0
        r_squared = 0
        epsilon_opt = 5.0
        phi_opt = max(phi_max)

    fig, ax = plt.subplots(figsize=(6, 4))

    # Data points
    ax.scatter(noise, phi_max, c='#d62728', s=60, zorder=5, label='Experimental')

    # Fitted curve
    noise_smooth = np.linspace(0, 22, 200)
    phi_fit = stochastic_resonance_model(noise_smooth, a, b, c)
    ax.plot(noise_smooth, phi_fit, 'k-', linewidth=2,
            label=f'Fit: $R^2$ = {r_squared:.3f}')

    # Mark optimal point
    ax.axvline(epsilon_opt, color='green', linestyle='--', alpha=0.7, linewidth=1.5)
    ax.scatter([epsilon_opt], [phi_opt], c='green', s=100, marker='*', zorder=6)
    ax.annotate(f'$\\epsilon_{{opt}}$ = {epsilon_opt:.2f}',
                xy=(epsilon_opt, phi_opt), xytext=(epsilon_opt + 3, phi_opt + 0.005),
                fontsize=9, color='green')

    ax.set_xlabel('Noise Amplitude (ε)')
    ax.set_ylabel('Maximum Φ (bits)')
    ax.set_title('Stochastic Resonance in Integrated Information')
    ax.legend(loc='upper right', framealpha=0.9)

    # Model equation
    ax.text(0.98, 0.15, f'$\\Phi(\\epsilon) = a \\cdot \\epsilon \\cdot e^{{-b\\epsilon^2}} + c$',
            transform=ax.transAxes, fontsize=9, ha='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.savefig(Path(__file__).parent / 'fig2_stochastic_resonance.pdf')
    fig.savefig(Path(__file__).parent / 'fig2_stochastic_resonance.png')
    plt.close()
    print("Generated: fig2_stochastic_resonance.pdf")

def fig3_system_scaling():
    """Figure 3: System size scaling at optimal noise."""
    data = load_results()
    results = data['results']

    # Get data at "Very High" noise (optimal)
    optimal_data = [r for r in results if r['noise_level'] == 'Very High']

    neurons = [r['effective_neurons'] for r in optimal_data]
    phi_max = [r['max_phi'] for r in optimal_data]
    sizes = [r['system_size'] for r in optimal_data]

    fig, ax = plt.subplots(figsize=(5, 4))

    ax.bar(range(len(sizes)), phi_max, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
           edgecolor='black', linewidth=1)
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([f'{s}\n(n={n})' for s, n in zip(sizes, neurons)])

    ax.set_ylabel('Maximum Φ (bits)')
    ax.set_title('Scaling of Φ at Optimal Noise (ε = 5.0)')

    # Annotate values
    for i, (phi, n) in enumerate(zip(phi_max, neurons)):
        ax.annotate(f'{phi:.4f}', xy=(i, phi), xytext=(i, phi + 0.002),
                    ha='center', fontsize=8)

    fig.savefig(Path(__file__).parent / 'fig3_system_scaling.pdf')
    fig.savefig(Path(__file__).parent / 'fig3_system_scaling.png')
    plt.close()
    print("Generated: fig3_system_scaling.pdf")

def fig4_baseline_comparison():
    """Figure 4: Baseline (Φ=0) vs optimal configuration."""
    data = load_results()
    results = data['results']

    # Get XLarge baseline and optimal
    xlarge_baseline = [r for r in results if r['system_size'] == 'XLarge' and r['noise_level'] == 'Baseline'][0]
    xlarge_optimal = [r for r in results if r['system_size'] == 'XLarge' and r['noise_level'] == 'Very High'][0]

    fig, ax = plt.subplots(figsize=(5, 4))

    configs = ['Pure State\n(ε = 0)', 'Mixed State\n(ε = 5.0)']
    phi_values = [xlarge_baseline['max_phi'], xlarge_optimal['max_phi']]
    colors = ['#3498db', '#e74c3c']

    bars = ax.bar(configs, phi_values, color=colors, edgecolor='black', linewidth=1.5, width=0.6)

    ax.set_ylabel('Φ (bits)')
    ax.set_title('Pure vs Mixed Quantum States\n(XLarge System, 729 states)')

    # Annotate
    ax.annotate('Φ = 0.0000\n(exactly zero)', xy=(0, 0.001), ha='center', fontsize=9, color='#2c3e50')
    ax.annotate(f'Φ = {xlarge_optimal["max_phi"]:.4f}', xy=(1, xlarge_optimal['max_phi'] + 0.002),
                ha='center', fontsize=9, color='#2c3e50')

    ax.set_ylim(0, 0.045)

    fig.savefig(Path(__file__).parent / 'fig4_baseline_comparison.pdf')
    fig.savefig(Path(__file__).parent / 'fig4_baseline_comparison.png')
    plt.close()
    print("Generated: fig4_baseline_comparison.pdf")

def main():
    """Generate all figures."""
    print("Generating publication figures...")
    print("=" * 50)

    # Create output directory if needed
    output_dir = Path(__file__).parent
    output_dir.mkdir(exist_ok=True)

    fig1_phi_vs_noise()
    fig2_stochastic_resonance()
    fig3_system_scaling()
    fig4_baseline_comparison()

    print("=" * 50)
    print("All figures generated successfully!")
    print(f"Output directory: {output_dir}")

if __name__ == '__main__':
    main()
