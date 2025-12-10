"""Plot specific layers to show learned non-linearities clearly."""
import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
from peft import AutoPeftModelForCausalLM

def plot_layer_comparison(adapter_path, output_file="layer_comparison.png"):
    """
    Plot early, middle, and late layers to show learning progression.
    Also show deviation from identity more clearly.
    """
    print(f"Loading model from: {adapter_path}")
    model = AutoPeftModelForCausalLM.from_pretrained(adapter_path, device_map="auto")

    # Target layers to visualize
    target_layers = {
        'Layer 1 (Early)': 'base_model.model.model.layers.1.mlp.gate_up_proj',
        'Layer 16 (Middle)': 'base_model.model.model.layers.16.mlp.gate_up_proj',
        'Layer 31 (Late)': 'base_model.model.model.layers.31.mlp.gate_up_proj',
        'Layer 26 (Strongest)': 'base_model.model.model.layers.26.mlp.gate_up_proj',
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, (label, module_name) in enumerate(target_layers.items()):
        ax = axes[idx]

        # Find the module
        found = False
        for name, module in model.named_modules():
            if name == module_name and hasattr(module, 'fourier_params'):
                for adapter_name, params in module.fourier_params.items():
                    # Get parameters
                    coeffs = params.weight.detach().cpu()
                    a0, a1, a2, a3 = coeffs.numpy()
                    sigma = params.sigma.detach().cpu().item()

                    # Calculate non-linearity
                    nonlin = abs(a2) + abs(a3)

                    print(f"\n{label}:")
                    print(f"  Coefficients: a₀={a0:+.4f}, a₁={a1:+.4f}, a₂={a2:+.4f}, a₃={a3:+.4f}")
                    print(f"  σ={sigma:.4f}, Non-linearity={nonlin:.4f}")

                    # Create input range
                    z = np.linspace(-3, 3, 1000)

                    # Apply normalization
                    x = np.tanh(z / sigma)

                    # Apply Fourier series
                    phi = a0 + a1 * x + a2 * np.sin(np.pi * x) + a3 * np.cos(np.pi * x)

                    # Plot activation function
                    ax.plot(z, phi, label=f'Learned φ(z)', linewidth=3, color='blue')
                    ax.plot(z, z, '--', color='red', label='Identity (y=z)', linewidth=2, alpha=0.7)

                    # Calculate and show deviation
                    deviation = phi - z

                    ax.set_xlabel('LoRA output (z)', fontsize=12)
                    ax.set_ylabel('Activation φ(z)', fontsize=12)
                    ax.set_title(f'{label}\nφ(z) = {a0:.3f} + {a1:.3f}x + {a2:.3f}sin(πx) + {a3:.3f}cos(πx)\n'
                                f'σ={sigma:.3f}, Non-linearity={nonlin:.4f}',
                                fontsize=11, pad=10)
                    ax.legend(fontsize=10)
                    ax.grid(True, alpha=0.3)
                    ax.axhline(y=0, color='k', linewidth=0.5)
                    ax.axvline(x=0, color='k', linewidth=0.5)

                    found = True
                    break

        if not found:
            ax.text(0.5, 0.5, f'Module not found:\n{module_name}',
                   ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"\n✓ Saved comparison plot to: {output_file}")
    plt.close()

    # Now create a SECOND plot showing deviations more clearly
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, (label, module_name) in enumerate(target_layers.items()):
        ax = axes[idx]

        for name, module in model.named_modules():
            if name == module_name and hasattr(module, 'fourier_params'):
                for adapter_name, params in module.fourier_params.items():
                    coeffs = params.weight.detach().cpu()
                    a0, a1, a2, a3 = coeffs.numpy()
                    sigma = params.sigma.detach().cpu().item()
                    nonlin = abs(a2) + abs(a3)

                    z = np.linspace(-3, 3, 1000)
                    x = np.tanh(z / sigma)
                    phi = a0 + a1 * x + a2 * np.sin(np.pi * x) + a3 * np.cos(np.pi * x)

                    # Plot DEVIATION from identity (this shows the non-linearity!)
                    deviation = phi - z

                    ax.plot(z, deviation, linewidth=3, color='purple', label='Deviation: φ(z) - z')
                    ax.axhline(y=0, color='red', linewidth=2, linestyle='--', alpha=0.7, label='Zero (Identity)')

                    # Show individual Fourier component contributions
                    linear_dev = a0 + (a1 - 1.0) * x  # Linear deviation from identity
                    sine_comp = a2 * np.sin(np.pi * x)
                    cosine_comp = a3 * np.cos(np.pi * x)

                    ax.plot(z, linear_dev, '--', linewidth=1.5, alpha=0.6, label=f'Linear: {a0:.3f}+{a1-1:.3f}x')
                    ax.plot(z, sine_comp, '--', linewidth=1.5, alpha=0.6, label=f'Sin: {a2:.3f}sin(πx)')
                    ax.plot(z, cosine_comp, '--', linewidth=1.5, alpha=0.6, label=f'Cos: {a3:.3f}cos(πx)')

                    ax.set_xlabel('LoRA output (z)', fontsize=12)
                    ax.set_ylabel('Deviation from Identity: φ(z) - z', fontsize=12)
                    ax.set_title(f'{label} - DEVIATION FROM IDENTITY\n'
                                f'Non-linearity Strength: {nonlin:.4f}',
                                fontsize=11, pad=10)
                    ax.legend(fontsize=9)
                    ax.grid(True, alpha=0.3)
                    ax.axvline(x=0, color='k', linewidth=0.5)
                    break

    plt.tight_layout()
    deviation_file = output_file.replace('.png', '_deviation.png')
    plt.savefig(deviation_file, dpi=200, bbox_inches='tight')
    print(f"✓ Saved deviation plot to: {deviation_file}")
    plt.close()

    # Create a THIRD plot: zoomed in on a specific range to see the curve shape
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, (label, module_name) in enumerate(target_layers.items()):
        ax = axes[idx]

        for name, module in model.named_modules():
            if name == module_name and hasattr(module, 'fourier_params'):
                for adapter_name, params in module.fourier_params.items():
                    coeffs = params.weight.detach().cpu()
                    a0, a1, a2, a3 = coeffs.numpy()
                    sigma = params.sigma.detach().cpu().item()
                    nonlin = abs(a2) + abs(a3)

                    # Zoom in on [-1, 1] range
                    z = np.linspace(-1, 1, 1000)
                    x = np.tanh(z / sigma)
                    phi = a0 + a1 * x + a2 * np.sin(np.pi * x) + a3 * np.cos(np.pi * x)

                    ax.plot(z, phi, linewidth=3, color='blue', label='Learned φ(z)')
                    ax.plot(z, z, '--', color='red', linewidth=2, alpha=0.7, label='Identity')

                    # Highlight the difference region
                    ax.fill_between(z, z, phi, alpha=0.3, color='green', label='Non-linear region')

                    ax.set_xlabel('LoRA output (z)', fontsize=12)
                    ax.set_ylabel('Activation φ(z)', fontsize=12)
                    ax.set_title(f'{label} - ZOOMED IN [-1, 1]\n'
                                f'Max deviation: {np.max(np.abs(phi - z)):.4f}',
                                fontsize=11, pad=10)
                    ax.legend(fontsize=10)
                    ax.grid(True, alpha=0.3)
                    ax.set_xlim(-1, 1)
                    break

    plt.tight_layout()
    zoom_file = output_file.replace('.png', '_zoom.png')
    plt.savefig(zoom_file, dpi=200, bbox_inches='tight')
    print(f"✓ Saved zoomed plot to: {zoom_file}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("adapter_path", type=str, help="Path to trained adapter")
    parser.add_argument("--output", type=str, default="layer_comparison.png")
    args = parser.parse_args()

    plot_layer_comparison(args.adapter_path, args.output)
    print("\n✓ All plots generated successfully!")
