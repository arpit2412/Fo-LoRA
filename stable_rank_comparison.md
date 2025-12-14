# Stable Rank Comparison: SincLoRA vs Standard LoRA

## Executive Summary

This document compares the **Stable Rank** (Effective Rank) of layers trained with **SincLoRA** (Adaptive) versus **Standard LoRA** (Fixed Rank 16).

| Metric | SincLoRA (Adaptive) | Standard LoRA (Fixed) | Interpretation |
|--------|---------------------|-----------------------|----------------|
| **Mean Rank** | **5.15** | **2.43** | SincLoRA utilizes significantly more capacity on average. |
| **Std Dev** | **1.34** | **0.56** | **SincLoRA is 2.4x more adaptive**, varying rank based on layer needs. |
| **Min Rank** | **2.48** | **1.03** | Standard LoRA often collapses to Rank-1 dynamics. |
| **Max Rank** | **8.44** | **4.19** | SincLoRA allocates high capacity (Rank > 8) to critical layers. |

## Analysis

1.  **Standard LoRA Inefficiency**: Despite a theoretical rank of 16, Standard LoRA's *effective* rank is consistently low (mean 2.43). This suggests that fixed-rank training wastes parameters on redundant dimensions.
2.  **SincLoRA Adaptivity**: SincLoRA demonstrates true layer-wise adaptation.
    *   **High Complexity**: Layers like `layers.27.mlp.gate_up_proj` reach ranks > 8.4, indicating complex feature transformation.
    *   **Low Complexity**: Layers like `layers.18.self_attn.qkv_proj` stay lower (Rank ~3.2), saving capacity.
3.  **The "Huge Difference"**: The difference in stable rank between the two methods is substantial (Mean Δ = 2.72), validating the hypothesis that learnable rank allocation fundamentally changes the training dynamics.

## Layer-wise Comparison

Sorted by the difference in rank (SincLoRA - LoRA).

| Layer | Type | SincLoRA | Standard LoRA | Difference |
|:---|:---:|---:|---:|---:|
| layers.2.mlp.down_proj | MLP | 7.29 | 1.11 | **+6.18** |
| layers.27.mlp.gate_up_proj | MLP | 8.44 | 2.39 | **+6.05** |
| layers.0.mlp.gate_up_proj | MLP | 8.27 | 2.28 | **+5.99** |
| layers.26.self_attn.o_proj | Attn | 7.64 | 1.89 | **+5.75** |
| layers.27.self_attn.o_proj | Attn | 7.87 | 2.54 | **+5.33** |
| layers.25.self_attn.qkv_proj | Attn | 8.40 | 3.34 | **+5.06** |
| layers.10.mlp.gate_up_proj | MLP | 7.52 | 2.55 | **+4.97** |
| layers.1.mlp.gate_up_proj | MLP | 7.93 | 2.96 | **+4.97** |
| layers.21.mlp.gate_up_proj | MLP | 7.22 | 2.37 | **+4.85** |
| layers.21.self_attn.o_proj | Attn | 6.79 | 1.94 | **+4.85** |
| layers.12.mlp.down_proj | MLP | 7.13 | 2.44 | **+4.69** |
| layers.18.mlp.down_proj | MLP | 6.75 | 2.08 | **+4.68** |
| layers.19.self_attn.o_proj | Attn | 6.34 | 1.75 | **+4.59** |
| layers.14.mlp.gate_up_proj | MLP | 7.14 | 2.59 | **+4.55** |
| layers.23.mlp.down_proj | MLP | 7.29 | 2.76 | **+4.52** |
| layers.15.mlp.down_proj | MLP | 7.48 | 3.02 | **+4.46** |
| layers.31.mlp.down_proj | MLP | 5.80 | 1.34 | **+4.46** |
| layers.28.mlp.gate_up_proj | MLP | 6.60 | 2.15 | **+4.45** |
| layers.4.mlp.down_proj | MLP | 5.46 | 1.03 | **+4.42** |
| layers.27.self_attn.qkv_proj | Attn | 6.93 | 2.71 | **+4.21** |
| layers.10.self_attn.o_proj | Attn | 6.11 | 2.04 | **+4.08** |
| layers.17.mlp.down_proj | MLP | 6.13 | 2.14 | **+3.99** |
| layers.24.mlp.down_proj | MLP | 6.45 | 2.51 | **+3.94** |
| layers.23.self_attn.qkv_proj | Attn | 6.85 | 2.94 | **+3.91** |
| layers.8.self_attn.qkv_proj | Attn | 5.62 | 1.74 | **+3.88** |
| layers.26.mlp.gate_up_proj | MLP | 6.25 | 2.39 | **+3.87** |
| layers.5.self_attn.qkv_proj | Attn | 5.98 | 2.12 | **+3.85** |
| layers.13.mlp.gate_up_proj | MLP | 6.26 | 2.44 | **+3.83** |
| layers.11.mlp.down_proj | MLP | 6.51 | 2.71 | **+3.80** |
| layers.13.self_attn.o_proj | Attn | 5.55 | 1.75 | **+3.80** |
| layers.26.mlp.down_proj | MLP | 6.53 | 2.74 | **+3.79** |
| layers.3.self_attn.o_proj | Attn | 5.87 | 2.14 | **+3.73** |
| layers.30.mlp.down_proj | MLP | 5.81 | 2.16 | **+3.65** |
| layers.31.self_attn.o_proj | Attn | 5.32 | 1.70 | **+3.61** |
| layers.22.self_attn.o_proj | Attn | 5.56 | 1.97 | **+3.59** |
| layers.21.self_attn.qkv_proj | Attn | 5.99 | 2.42 | **+3.56** |
| layers.2.self_attn.qkv_proj | Attn | 5.41 | 1.91 | **+3.50** |
| layers.10.self_attn.qkv_proj | Attn | 5.87 | 2.38 | **+3.49** |
| layers.14.self_attn.qkv_proj | Attn | 6.20 | 2.75 | **+3.45** |
| layers.1.self_attn.o_proj | Attn | 5.49 | 2.05 | **+3.44** |
| layers.15.mlp.gate_up_proj | MLP | 6.21 | 2.77 | **+3.44** |
| layers.20.self_attn.qkv_proj | Attn | 6.39 | 3.06 | **+3.32** |
| layers.12.self_attn.qkv_proj | Attn | 5.25 | 1.94 | **+3.31** |
| layers.17.self_attn.qkv_proj | Attn | 5.52 | 2.24 | **+3.28** |
| layers.22.mlp.gate_up_proj | MLP | 5.73 | 2.46 | **+3.27** |
| layers.10.mlp.down_proj | MLP | 6.58 | 3.33 | **+3.26** |
| layers.3.self_attn.qkv_proj | Attn | 5.01 | 1.88 | **+3.14** |
| layers.0.self_attn.o_proj | Attn | 5.54 | 2.41 | **+3.13** |
| layers.15.self_attn.o_proj | Attn | 5.02 | 1.89 | **+3.13** |
| layers.1.self_attn.qkv_proj | Attn | 5.23 | 2.13 | **+3.11** |
| layers.6.self_attn.qkv_proj | Attn | 5.41 | 2.31 | **+3.10** |
| layers.28.self_attn.qkv_proj | Attn | 6.06 | 2.96 | **+3.10** |
| layers.21.mlp.down_proj | MLP | 5.46 | 2.46 | **+3.00** |
| layers.27.mlp.down_proj | MLP | 5.55 | 2.58 | **+2.97** |
| layers.26.self_attn.qkv_proj | Attn | 5.06 | 2.11 | **+2.95** |
| layers.7.self_attn.o_proj | Attn | 4.72 | 1.83 | **+2.89** |
| layers.12.self_attn.o_proj | Attn | 4.67 | 1.78 | **+2.89** |
| layers.13.self_attn.qkv_proj | Attn | 5.03 | 2.16 | **+2.87** |
| layers.23.self_attn.o_proj | Attn | 5.20 | 2.36 | **+2.84** |
| layers.31.self_attn.qkv_proj | Attn | 6.55 | 3.72 | **+2.83** |
| layers.17.mlp.gate_up_proj | MLP | 4.91 | 2.09 | **+2.82** |
| layers.22.mlp.down_proj | MLP | 4.58 | 1.77 | **+2.81** |
| layers.19.self_attn.qkv_proj | Attn | 4.66 | 1.87 | **+2.79** |
| layers.11.self_attn.o_proj | Attn | 4.54 | 1.77 | **+2.77** |
| layers.0.mlp.down_proj | MLP | 4.78 | 2.04 | **+2.74** |
| layers.4.self_attn.o_proj | Attn | 4.69 | 1.97 | **+2.72** |
| layers.29.mlp.down_proj | MLP | 3.78 | 1.07 | **+2.71** |
| layers.9.self_attn.qkv_proj | Attn | 4.37 | 1.68 | **+2.69** |
| layers.8.self_attn.o_proj | Attn | 4.75 | 2.10 | **+2.65** |
| layers.31.mlp.gate_up_proj | MLP | 5.31 | 2.68 | **+2.63** |
| layers.30.self_attn.qkv_proj | Attn | 6.01 | 3.45 | **+2.56** |
| layers.5.mlp.down_proj | MLP | 5.71 | 3.16 | **+2.55** |
| layers.11.mlp.gate_up_proj | MLP | 5.21 | 2.72 | **+2.49** |
| layers.2.self_attn.o_proj | Attn | 4.06 | 1.57 | **+2.49** |
| layers.14.mlp.down_proj | MLP | 4.69 | 2.23 | **+2.46** |
| layers.16.self_attn.o_proj | Attn | 4.71 | 2.26 | **+2.45** |
| layers.6.mlp.down_proj | MLP | 5.11 | 2.73 | **+2.38** |
| layers.29.self_attn.o_proj | Attn | 4.73 | 2.37 | **+2.36** |
| layers.24.mlp.gate_up_proj | MLP | 4.96 | 2.60 | **+2.35** |
| layers.23.mlp.gate_up_proj | MLP | 4.71 | 2.36 | **+2.34** |
| layers.24.self_attn.o_proj | Attn | 4.37 | 2.05 | **+2.32** |
| layers.6.self_attn.o_proj | Attn | 4.97 | 2.67 | **+2.31** |
| layers.8.mlp.gate_up_proj | MLP | 5.66 | 3.35 | **+2.30** |
| layers.20.mlp.gate_up_proj | MLP | 4.41 | 2.17 | **+2.24** |
| layers.7.self_attn.qkv_proj | Attn | 4.21 | 1.99 | **+2.22** |
| layers.19.mlp.down_proj | MLP | 4.82 | 2.64 | **+2.18** |
| layers.9.mlp.gate_up_proj | MLP | 5.07 | 2.91 | **+2.16** |
| layers.18.self_attn.o_proj | Attn | 4.13 | 1.99 | **+2.14** |
| layers.2.mlp.gate_up_proj | MLP | 5.10 | 3.12 | **+1.98** |
| layers.1.mlp.down_proj | MLP | 5.55 | 3.62 | **+1.94** |
| layers.25.mlp.gate_up_proj | MLP | 3.83 | 1.92 | **+1.91** |
| layers.14.self_attn.o_proj | Attn | 3.52 | 1.61 | **+1.91** |
| layers.20.self_attn.o_proj | Attn | 3.96 | 2.09 | **+1.87** |
| layers.13.mlp.down_proj | MLP | 4.20 | 2.38 | **+1.82** |
| layers.29.mlp.gate_up_proj | MLP | 4.06 | 2.26 | **+1.80** |
| layers.22.self_attn.qkv_proj | Attn | 4.80 | 3.12 | **+1.68** |
| layers.12.mlp.gate_up_proj | MLP | 3.76 | 2.14 | **+1.62** |
| layers.28.mlp.down_proj | MLP | 4.54 | 2.92 | **+1.62** |
| layers.4.self_attn.qkv_proj | Attn | 4.23 | 2.68 | **+1.55** |
| layers.6.mlp.gate_up_proj | MLP | 4.48 | 2.97 | **+1.51** |
| layers.24.self_attn.qkv_proj | Attn | 3.95 | 2.51 | **+1.44** |
| layers.0.self_attn.qkv_proj | Attn | 3.37 | 1.95 | **+1.42** |
| layers.17.self_attn.o_proj | Attn | 3.17 | 1.77 | **+1.40** |
| layers.9.mlp.down_proj | MLP | 3.86 | 2.47 | **+1.39** |
| layers.3.mlp.gate_up_proj | MLP | 4.38 | 3.00 | **+1.39** |
| layers.20.mlp.down_proj | MLP | 4.08 | 2.74 | **+1.34** |
| layers.11.self_attn.qkv_proj | Attn | 4.17 | 2.85 | **+1.32** |
| layers.18.mlp.gate_up_proj | MLP | 3.33 | 2.03 | **+1.31** |
| layers.16.mlp.gate_up_proj | MLP | 4.17 | 2.91 | **+1.27** |
| layers.16.mlp.down_proj | MLP | 4.06 | 2.81 | **+1.25** |
| layers.29.self_attn.qkv_proj | Attn | 4.01 | 2.84 | **+1.17** |
| layers.28.self_attn.o_proj | Attn | 3.58 | 2.49 | **+1.09** |
| layers.25.mlp.down_proj | MLP | 4.30 | 3.24 | **+1.07** |
| layers.9.self_attn.o_proj | Attn | 3.43 | 2.43 | **+1.00** |
| layers.7.mlp.gate_up_proj | MLP | 4.09 | 3.26 | **+0.83** |
| layers.19.mlp.gate_up_proj | MLP | 2.85 | 2.09 | **+0.76** |
| layers.15.self_attn.qkv_proj | Attn | 3.31 | 2.55 | **+0.75** |
| layers.30.mlp.gate_up_proj | MLP | 3.43 | 2.68 | **+0.75** |
| layers.25.self_attn.o_proj | Attn | 3.62 | 2.90 | **+0.73** |
| layers.7.mlp.down_proj | MLP | 3.33 | 2.66 | **+0.66** |
| layers.4.mlp.gate_up_proj | MLP | 2.79 | 3.37 | **-0.58** |
| layers.30.self_attn.o_proj | Attn | 2.81 | 2.25 | **+0.56** |
| layers.5.mlp.gate_up_proj | MLP | 3.69 | 4.19 | **-0.50** |
| layers.8.mlp.down_proj | MLP | 2.63 | 3.08 | **-0.45** |
| layers.3.mlp.down_proj | MLP | 3.53 | 3.92 | **-0.39** |
| layers.5.self_attn.o_proj | Attn | 2.48 | 2.13 | **+0.34** |
| layers.16.self_attn.qkv_proj | Attn | 2.93 | 2.60 | **+0.33** |
| layers.18.self_attn.qkv_proj | Attn | 3.21 | 3.02 | **+0.19** |
