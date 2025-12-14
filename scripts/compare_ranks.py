#!/usr/bin/env python3
"""
Compare Stable Ranks: SincLoRA vs Standard LoRA
"""

import argparse
import json
import torch
import numpy as np
from peft import AutoPeftModelForCausalLM
from tabulate import tabulate

def compute_stable_rank_lora(A, B):
    """
    Compute stable rank of delta W = B @ A
    SR = ||W||_F^2 / ||W||_2^2
    """
    # W shape: [out, in]
    # B: [out, r], A: [r, in]
    # For efficiency, we can compute SVD of B and A separately if r is small,
    # but computing W directly is fine for small r.
    
    # Cast to float32 for stability
    A = A.float()
    B = B.float()
    
    W = B @ A
    
    fro_norm_sq = torch.sum(W ** 2).item()
    
    # Spectral norm = largest singular value
    # Use torch.linalg.svdvals for speed (only need singular values)
    try:
        s = torch.linalg.svdvals(W)
        spectral_norm_sq = (s[0].item()) ** 2
    except:
        # Fallback if SVD fails
        return float('nan')
        
    if spectral_norm_sq == 0:
        return 0.0
        
    return fro_norm_sq / spectral_norm_sq

def compute_stable_rank_sinc(alpha, r_basis):
    """
    Compute effective rank of SincLoRA
    SR = (sum(alpha^2) / max(alpha^2)) * r_basis
    """
    alpha = alpha.float()
    alpha_sq = alpha ** 2
    numerator = torch.sum(alpha_sq)
    denominator = torch.max(alpha_sq)
    
    if denominator == 0:
        return 0.0
        
    sr_alpha = (numerator / denominator).item()
    return sr_alpha * r_basis

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sinclora_path", type=str, required=True)
    parser.add_argument("--lora_path", type=str, required=True)
    args = parser.parse_args()
    
    print(f"Loading SincLoRA from {args.sinclora_path}...")
    sinc_model = AutoPeftModelForCausalLM.from_pretrained(args.sinclora_path)
    
    print(f"Loading Standard LoRA from {args.lora_path}...")
    lora_model = AutoPeftModelForCausalLM.from_pretrained(args.lora_path)
    
    results = []
    
    # Iterate through SincLoRA layers
    print("\nAnalyzing layers...")
    
    # Map layer names to modules
    sinc_modules = {n: m for n, m in sinc_model.named_modules() if hasattr(m, "sinc_params")}
    lora_modules = {n: m for n, m in lora_model.named_modules() if "lora_A" in dir(m)}
    
    # Common layers
    common_layers = sorted(list(set(sinc_modules.keys()) & set(lora_modules.keys())))
    
    sinc_ranks = []
    lora_ranks = []
    
    for name in common_layers:
        # SincLoRA Rank
        sinc_mod = sinc_modules[name]
        # Assuming 'default' adapter
        if "default" in sinc_mod.sinc_params:
            params = sinc_mod.sinc_params["default"]
            K = sinc_mod.K["default"]
            # r is not directly stored in module, but we know r_basis = r/K
            # Let's infer r_basis from lora_A_basis shape
            # lora_A_basis is a ModuleList
            A_sample = sinc_mod.lora_A_basis["default"][0]
            r_basis = A_sample.out_features
            
            sr_sinc = compute_stable_rank_sinc(params.alpha.detach().cpu(), r_basis)
        else:
            sr_sinc = float('nan')
            
        # LoRA Rank
        lora_mod = lora_modules[name]
        if "default" in lora_mod.lora_A:
            A = lora_mod.lora_A["default"].weight.detach().cpu()
            B = lora_mod.lora_B["default"].weight.detach().cpu()
            sr_lora = compute_stable_rank_lora(A, B)
        else:
            sr_lora = float('nan')
            
        results.append({
            "Layer": name,
            "SincLoRA": sr_sinc,
            "LoRA": sr_lora,
            "Type": "Attn" if "attn" in name else "MLP"
        })
        
        if not np.isnan(sr_sinc): sinc_ranks.append(sr_sinc)
        if not np.isnan(sr_lora): lora_ranks.append(sr_lora)

    # Print Table
    print("\n" + "="*80)
    print("STABLE RANK COMPARISON")
    print("="*80)
    
    # Sort by difference
    results.sort(key=lambda x: abs(x["SincLoRA"] - x["LoRA"]), reverse=True)
    
    # Top 10 differences
    print("\nTop 10 Layers with Largest Difference:")
    headers = ["Layer", "Type", "SincLoRA (Adaptive)", "LoRA (Fixed)"]
    table_data = [[r["Layer"][-40:], r["Type"], f"{r['SincLoRA']:.2f}", f"{r['LoRA']:.2f}"] for r in results[:10]]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # Statistics
    print("\n" + "="*80)
    print("STATISTICS")
    print("="*80)
    
    print(f"{'Metric':<20} | {'SincLoRA':<15} | {'Standard LoRA':<15}")
    print("-" * 56)
    print(f"{'Mean Rank':<20} | {np.mean(sinc_ranks):<15.2f} | {np.mean(lora_ranks):<15.2f}")
    print(f"{'Std Dev':<20} | {np.std(sinc_ranks):<15.2f} | {np.std(lora_ranks):<15.2f}")
    print(f"{'Min Rank':<20} | {np.min(sinc_ranks):<15.2f} | {np.min(lora_ranks):<15.2f}")
    print(f"{'Max Rank':<20} | {np.max(sinc_ranks):<15.2f} | {np.max(lora_ranks):<15.2f}")
    
    # Save results
    with open("comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nFull results saved to comparison_results.json")

if __name__ == "__main__":
    main()
