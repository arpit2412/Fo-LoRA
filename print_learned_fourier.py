"""Load trained model and print learned Fourier coefficients."""
import argparse
import torch
from peft import AutoPeftModelForCausalLM

def print_fourier_coefficients(adapter_path):
    """Load model and print learned non-linearities."""
    print(f"Loading model from: {adapter_path}")
    print("="*80)

    try:
        model = AutoPeftModelForCausalLM.from_pretrained(adapter_path)
        print("✅ Model loaded successfully\n")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    identity = torch.tensor([0.0, 1.0, 0.0, 0.0])

    print("="*80)
    print("LEARNED FOURIER NON-LINEARITIES")
    print("="*80)
    print("\nFormat: φ(x) = a₀ + a₁·x + a₂·sin(πx) + a₃·cos(πx), where x = tanh(z/σ)\n")

    # Group by layer for better readability
    layers_data = {}

    for name, module in model.named_modules():
        if hasattr(module, 'fourier_params'):
            for adapter_name, params in module.fourier_params.items():
                # Extract layer number and module type
                parts = name.split('.')
                layer_num = None
                module_type = parts[-1] if parts else name

                for i, part in enumerate(parts):
                    if part == 'layers' and i + 1 < len(parts):
                        try:
                            layer_num = int(parts[i + 1])
                            break
                        except:
                            pass

                if layer_num is not None:
                    if layer_num not in layers_data:
                        layers_data[layer_num] = {}

                    coeffs = params.weight
                    a0, a1, a2, a3 = coeffs.detach().cpu().tolist()
                    sigma = params.sigma.detach().cpu().item()

                    # Calculate metrics
                    dev = (coeffs - identity.to(coeffs.device)).abs().mean().item()
                    nonlin = abs(a2) + abs(a3)  # Strength of non-linearity

                    layers_data[layer_num][module_type] = {
                        'a0': a0, 'a1': a1, 'a2': a2, 'a3': a3,
                        'sigma': sigma,
                        'deviation': dev,
                        'nonlinearity': nonlin
                    }

    # Print sorted by layer
    for layer_num in sorted(layers_data.keys()):
        print(f"\n{'='*80}")
        print(f"LAYER {layer_num}")
        print(f"{'='*80}")

        for module_type, data in sorted(layers_data[layer_num].items()):
            a0, a1, a2, a3 = data['a0'], data['a1'], data['a2'], data['a3']
            sigma = data['sigma']
            dev = data['deviation']
            nonlin = data['nonlinearity']

            print(f"\n  {module_type}:")
            print(f"    Coefficients: a₀={a0:+.4f}, a₁={a1:+.4f}, a₂={a2:+.4f}, a₃={a3:+.4f}")
            print(f"    σ = {sigma:.4f}")
            print(f"    Deviation from identity: {dev:.4f}")
            print(f"    Non-linearity strength: {nonlin:.4f}")

            # Interpretation
            if nonlin < 0.01:
                status = "⚪ Nearly linear (identity)"
            elif nonlin < 0.1:
                status = "🟡 Weak non-linearity"
            elif nonlin < 0.3:
                status = "🟠 Moderate non-linearity"
            else:
                status = "🔴 Strong non-linearity"
            print(f"    Status: {status}")

    # Summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")

    all_deviations = []
    all_nonlinearities = []

    for layer_data in layers_data.values():
        for module_data in layer_data.values():
            all_deviations.append(module_data['deviation'])
            all_nonlinearities.append(module_data['nonlinearity'])

    if all_deviations:
        print(f"\nTotal layers analyzed: {len(layers_data)}")
        print(f"Total modules: {len(all_deviations)}")
        print(f"\nDeviation from identity:")
        print(f"  Mean: {sum(all_deviations)/len(all_deviations):.4f}")
        print(f"  Min: {min(all_deviations):.4f}")
        print(f"  Max: {max(all_deviations):.4f}")
        print(f"\nNon-linearity strength:")
        print(f"  Mean: {sum(all_nonlinearities)/len(all_nonlinearities):.4f}")
        print(f"  Min: {min(all_nonlinearities):.4f}")
        print(f"  Max: {max(all_nonlinearities):.4f}")

        # Count by strength
        nearly_linear = sum(1 for n in all_nonlinearities if n < 0.01)
        weak = sum(1 for n in all_nonlinearities if 0.01 <= n < 0.1)
        moderate = sum(1 for n in all_nonlinearities if 0.1 <= n < 0.3)
        strong = sum(1 for n in all_nonlinearities if n >= 0.3)

        print(f"\nDistribution by strength:")
        print(f"  Nearly linear: {nearly_linear}/{len(all_nonlinearities)}")
        print(f"  Weak: {weak}/{len(all_nonlinearities)}")
        print(f"  Moderate: {moderate}/{len(all_nonlinearities)}")
        print(f"  Strong: {strong}/{len(all_nonlinearities)}")

    print(f"\n{'='*80}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print learned Fourier coefficients")
    parser.add_argument("adapter_path", type=str, help="Path to saved adapter")
    args = parser.parse_args()

    print_fourier_coefficients(args.adapter_path)
