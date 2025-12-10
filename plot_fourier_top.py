"""Plot learned Fourier curves for top layers with strongest non-linearities."""
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from peft import AutoPeftModelForCausalLM

def plot_fourier_curves(adapter_path, output_dir, top_n=15):
    """
    Plot learned Fourier activation curves for layers with strongest non-linearities.

    Args:
        adapter_path: Path to saved Fourier LoRA adapter
        output_dir: Directory to save plots
        top_n: Number of top layers to plot
    """
    print(f"Loading adapter from {adapter_path}")
    print("Loading model...")

    model = AutoPeftModelForCausalLM.from_pretrained(
        adapter_path,
        device_map="auto",
    )

    # First pass: collect all modules and their non-linearity strengths
    print("\nAnalyzing learned non-linearities...")
    modules_data = []

    for name, module in model.named_modules():
        if hasattr(module, "fourier_params") and len(module.fourier_params) > 0:
            for adapter_name, params in module.fourier_params.items():
                coeffs = params.weight.detach().cpu()
                a0, a1, a2, a3 = coeffs.numpy()
                sigma = params.sigma.detach().cpu().item()

                # Calculate non-linearity strength
                nonlinearity = abs(a2) + abs(a3)

                modules_data.append({
                    'name': name,
                    'adapter': adapter_name,
                    'a0': a0, 'a1': a1, 'a2': a2, 'a3': a3,
                    'sigma': sigma,
                    'nonlinearity': nonlinearity
                })

    # Sort by non-linearity strength (descending)
    modules_data.sort(key=lambda x: x['nonlinearity'], reverse=True)

    print(f"\nTotal modules: {len(modules_data)}")
    print(f"Plotting top {min(top_n, len(modules_data))} modules with strongest non-linearities\n")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Plot top N
    for i, data in enumerate(modules_data[:top_n]):
        name = data['name']
        a0, a1, a2, a3 = data['a0'], data['a1'], data['a2'], data['a3']
        sigma = data['sigma']
        nonlinearity = data['nonlinearity']

        print(f"[{i+1}/{top_n}] {name}")
        print(f"  Coefficients: a₀={a0:+.4f}, a₁={a1:+.4f}, a₂={a2:+.4f}, a₃={a3:+.4f}")
        print(f"  σ={sigma:.4f}, non-linearity={nonlinearity:.4f}")

        # Create input range for LoRA output z
        z = np.linspace(-5, 5, 500)

        # Apply normalization
        x = np.tanh(z / sigma)

        # Apply Fourier series
        phi = a0 + a1 * x + a2 * np.sin(np.pi * x) + a3 * np.cos(np.pi * x)

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Left plot: Full activation function φ(z)
        ax1.plot(z, phi, label='Learned φ(z)', linewidth=2, color='blue')
        ax1.plot(z, z, '--', color='gray', label='Identity (y=z)', alpha=0.7)
        ax1.set_xlabel('LoRA output (z)')
        ax1.set_ylabel('Activation output φ(z)')
        ax1.set_title(f'{name}\nNon-linearity: {nonlinearity:.4f}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='k', linewidth=0.5)
        ax1.axvline(x=0, color='k', linewidth=0.5)

        # Right plot: Individual Fourier components
        linear = a0 + a1 * x
        sine = a2 * np.sin(np.pi * x)
        cosine = a3 * np.cos(np.pi * x)

        ax2.plot(x, linear, label=f'Linear: {a0:.3f}+{a1:.3f}x', alpha=0.7)
        ax2.plot(x, sine, label=f'Sine: {a2:.3f}sin(πx)', alpha=0.7)
        ax2.plot(x, cosine, label=f'Cosine: {a3:.3f}cos(πx)', alpha=0.7)
        ax2.plot(x, linear + sine + cosine, 'k--', label='Sum', linewidth=2, alpha=0.5)
        ax2.set_xlabel('Normalized input x = tanh(z/σ)')
        ax2.set_ylabel('Component value')
        ax2.set_title(f'Fourier Components (σ={sigma:.3f})')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='k', linewidth=0.5)
        ax2.axvline(x=0, color='k', linewidth=0.5)

        plt.tight_layout()

        # Save plot
        safe_name = name.replace(".", "_").replace("/", "_")
        output_path = os.path.join(output_dir, f"rank{i+1:02d}_{safe_name}.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    print(f"\n{'='*80}")
    print(f"Generated {min(top_n, len(modules_data))} plots in {output_dir}")
    print(f"{'='*80}")

    # Print summary of all modules
    print("\nSummary Statistics:")
    print(f"  Mean non-linearity: {np.mean([d['nonlinearity'] for d in modules_data]):.4f}")
    print(f"  Max non-linearity: {modules_data[0]['nonlinearity']:.4f} ({modules_data[0]['name']})")
    print(f"  Min non-linearity: {modules_data[-1]['nonlinearity']:.4f} ({modules_data[-1]['name']})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize top learned Fourier activation curves")
    parser.add_argument("adapter_path", type=str, help="Path to saved adapter")
    parser.add_argument("--output_dir", type=str, default="fourier_plots_top", help="Output directory")
    parser.add_argument("--top_n", type=int, default=15, help="Number of top layers to plot")
    args = parser.parse_args()

    plot_fourier_curves(args.adapter_path, args.output_dir, args.top_n)
