# Fourier Series-Based LS-LoRA: Mathematical Foundations

## Table of Contents
1. [Problem Statement](#problem-statement)
2. [Standard LoRA Recap](#standard-lora-recap)
3. [Fourier LS-LoRA Architecture](#fourier-ls-lora-architecture)
4. [Mathematical Formulation](#mathematical-formulation)
5. [Initialization Strategy](#initialization-strategy)
6. [Regularization](#regularization)
7. [Gradient Flow](#gradient-flow)
8. [Why Fourier Series?](#why-fourier-series)
9. [Parameter Efficiency](#parameter-efficiency)
10. [Computational Complexity](#computational-complexity)
11. [Implementation Details](#implementation-details)

---

## Problem Statement

**Goal**: Extend Low-Rank Adaptation (LoRA) with learnable non-linear transformations that:
- Start as identity (standard LoRA)
- Adapt per-layer when beneficial for the task
- Remain parameter-efficient
- Have smooth, differentiable gradients
- Are interpretable

**Challenge**: Previous attempts (piecewise linear splines) had gradient issues due to hard boundaries and indexing operations.

---

## Standard LoRA Recap

### Linear Transformation

For a pre-trained weight matrix **W₀ ∈ ℝ^(d×k)**, LoRA introduces a low-rank update:

```
h = W₀x + ΔWx
```

where the update is factorized:

```
ΔW = BA
```

with:
- **A ∈ ℝ^(r×k)**: down-projection (reduces dimensionality)
- **B ∈ ℝ^(d×r)**: up-projection (restores dimensionality)
- **r ≪ min(d,k)**: rank (typically 4-64)

### Forward Pass

```
h = W₀x + (α/r) · B(Ax)
```

where:
- **α**: scaling factor (typically 2r to 4r)
- **α/r**: effective scaling to normalize by rank

### Key Properties
- **Linear**: Output is a linear function of input
- **Low-rank**: Only **r(d+k)** parameters instead of **dk**
- **Frozen base**: W₀ remains unchanged

---

## Fourier LS-LoRA Architecture

### High-Level Flow

```
Input x
   ↓
Base Layer: W₀x
   ↓
LoRA Path: z = (α/r) · B(Ax)  ← Standard LoRA computation
   ↓
Normalization: x̂ = tanh(z/σ)  ← Map to [-1, 1], σ learnable per layer
   ↓
Fourier Transform: φ(x̂)       ← Apply learnable Fourier series
   ↓
Output: W₀x + φ(tanh(z/σ))    ← Add to base
```

### Why This Design?

1. **Normalization (tanh)**: LoRA outputs can have varying magnitudes across layers. Normalizing to [-1,1] creates a consistent domain for the Fourier series.

2. **Learnable σ**: Each layer learns its own normalization scale, adapting to its activation magnitude.

3. **Fourier series**: Universal approximator on [-1,1] with smooth, periodic basis functions.

### Visual Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Fourier LS-LoRA Flow                      │
└─────────────────────────────────────────────────────────────┘

Input (x)
   │
   ├──────────────────────┐
   │                      │
   v                      v
Base Layer         LoRA Path (per layer)
   │                      │
   │                  ┌───┴────┐
   │                  │   A    │  Low-rank down-projection
   │                  │ (r×d)  │
   │                  └───┬────┘
   │                      │
   │                  ┌───┴────┐
   │                  │   B    │  Low-rank up-projection
   │                  │ (d×r)  │
   │                  └───┬────┘
   │                      │
   │                  z = B(A(x)) * scaling
   │                      │
   │              ┌───────┴────────┐
   │              │  Fourier NL    │  ← **NEW: Learnable Non-Linearity**
   │              │                │
   │              │  x = tanh(z/σ) │  σ learned per layer
   │              │                │
   │              │  φ(x) = a₀ +   │  4 coefficients learned
   │              │         a₁·x + │  per layer
   │              │         a₂·sin(πx) +
   │              │         a₃·cos(πx)
   │              └───────┬────────┘
   │                      │
   └──────────(+)─────────┘
              │
           Output

┌─────────────────────────────────────────────────────────────┐
│                    Loss Computation                         │
└─────────────────────────────────────────────────────────────┘

Total Loss = Language Model Loss + 0.01 × Regularization Loss
                    ↓                           ↓
            Cross-Entropy Loss      Σ ||[a₀,a₁,a₂,a₃] - [0,1,0,0]||²
            (helps task)            (encourages identity)
```

---

## Mathematical Formulation

### Complete Forward Pass

For input **x ∈ ℝ^n**:

**Step 1: Base computation**
```
h_base = W₀x
```

**Step 2: LoRA low-rank computation**
```
z = (α/r) · B(A(x))
```
where:
- z ∈ ℝ^d is the LoRA output (before non-linearity)

**Step 3: Normalization**
```
x̂ = tanh(z/σ)
```
where:
- **σ > 0** is a learnable scalar per layer
- **x̂ ∈ [-1, 1]^d** element-wise

**Step 4: Fourier transformation (element-wise)**
```
φ(x̂ᵢ) = a₀ + a₁·x̂ᵢ + a₂·sin(πx̂ᵢ) + a₃·cos(πx̂ᵢ)
```
for each element i = 1, ..., d

**Step 5: Final output**
```
h = h_base + φ(x̂)
  = W₀x + [a₀ + a₁·tanh(z/σ) + a₂·sin(π·tanh(z/σ)) + a₃·cos(π·tanh(z/σ))]
```

### Fourier Series Details

**Definition**: A Fourier series approximates a function using sines and cosines:

```
f(x) = a₀/2 + Σ[aₙ·cos(nπx) + bₙ·sin(nπx)]
              n=1
```

**Our truncation (4 terms)**:
```
φ(x) = a₀ + a₁·x + a₂·sin(πx) + a₃·cos(πx)
```

This includes:
- **Constant term**: a₀ (DC offset)
- **Linear term**: a₁·x (first-order)
- **First harmonic**: a₂·sin(πx) + a₃·cos(πx)

**Why 4 terms?**
- **a₀**: Bias/offset
- **a₁**: Controls linearity strength
- **a₂, a₃**: Single frequency non-linearity (sufficient for most tasks)
- More terms → more parameters, diminishing returns

**Domain**: x ∈ [-1, 1]
- sin(πx) ranges from sin(-π) to sin(π) ≈ [-0, 0] (odd function)
- cos(πx) ranges from cos(-π) to cos(π) = [-1, 1] (even function)

**Properties**:
1. **Smooth**: Infinitely differentiable
2. **Bounded**: Output is bounded (important for training stability)
3. **Periodic**: Natural for modeling cyclic patterns
4. **Universal**: Can approximate any continuous function on [-1, 1] (Stone-Weierstrass theorem)

### Layer-wise Shared Parameters

**Key insight**: All elements in a layer's output share the same Fourier coefficients.

For layer ℓ with output dimension d:
- **Shared**: [a₀, a₁, a₂, a₃, σ] ← 5 scalars
- **Applied**: To all d elements using the same transformation

This means:
```
φℓ(z₁, z₂, ..., zd) = [φℓ(z₁), φℓ(z₂), ..., φℓ(zd)]
```
where each element uses the **same** φℓ function but with different inputs.

**Benefits**:
- **Efficiency**: 5 parameters per layer instead of 5d
- **Regularization**: Reduces overfitting
- **Interpretability**: Single non-linearity per layer

---

## Initialization Strategy

### Identity Initialization

**Goal**: Start as standard LoRA, learn non-linearity only if beneficial.

**Initial values**:
```
a₀ = 0    (no offset)
a₁ = 1    (identity mapping)
a₂ = 0    (no sine component)
a₃ = 0    (no cosine component)
σ = 1     (neutral normalization)
```

**At initialization**:
```
φ(x̂) = 0 + 1·x̂ + 0·sin(πx̂) + 0·cos(πx̂) = x̂ = tanh(z/1) ≈ z (for small z)
```

So initially:
```
h ≈ W₀x + z = W₀x + (α/r)·B(Ax)
```
which is exactly standard LoRA!

**During training**:
- If non-linearity helps → a₂, a₃ grow
- If linear is sufficient → a₂, a₃ stay near 0 (encouraged by regularization)

### Why tanh(z/σ) ≈ z for small z?

Taylor expansion of tanh:
```
tanh(u) = u - u³/3 + 2u⁵/15 - ...
```

For small u = z/σ:
```
tanh(z/σ) ≈ z/σ - (z/σ)³/3 + ... ≈ z/σ
```

With σ = 1 and small z:
```
tanh(z) ≈ z
```

Combined with identity coefficients:
```
φ(tanh(z)) ≈ φ(z) ≈ 1·z = z
```

---

## Regularization

### L2 Penalty on Deviation from Identity

**Objective**: Encourage staying close to identity unless non-linearity improves task performance.

**Regularization loss per layer**:
```
ℒreg(θℓ) = ||θℓ - θidentity||²₂
         = (a₀ - 0)² + (a₁ - 1)² + (a₂ - 0)² + (a₃ - 0)²
         = a₀² + (a₁ - 1)² + a₂² + a₃²
```

Note: We don't penalize σ (it adapts to layer scale).

**Total regularization loss**:
```
ℒreg = (1/L) Σ ℒreg(θℓ)
            ℓ=1
```
where L is the number of LoRA layers.

**Final training objective**:
```
ℒtotal = ℒtask + λ·ℒreg
```
where:
- **ℒtask**: Language modeling loss (cross-entropy)
- **λ**: Regularization weight (we use λ = 0.01)

### Regularization Strength

**λ = 0.01** means:
- Task loss dominates (weight = 1.0)
- But there's a small penalty for non-linearity
- Model learns non-linearity only if it reduces task loss by more than 0.01 times the deviation

**Example**: If learning a₂ = 0.5 reduces task loss by 0.003:
- Benefit: -0.003 (task loss decrease)
- Cost: +0.01 × 0.5² = +0.0025 (regularization increase)
- Net: -0.003 + 0.0025 = -0.0005 (small net benefit → will learn)

---

## Gradient Flow

### Backpropagation Through Fourier Layer

**Forward pass**:
```
x̂ = tanh(z/σ)
φ = a₀ + a₁·x̂ + a₂·sin(πx̂) + a₃·cos(πx̂)
```

**Gradients w.r.t. Fourier coefficients** (element-wise):

```
∂φ/∂a₀ = 1

∂φ/∂a₁ = x̂ = tanh(z/σ)

∂φ/∂a₂ = sin(πx̂) = sin(π·tanh(z/σ))

∂φ/∂a₃ = cos(πx̂) = cos(π·tanh(z/σ))
```

**Gradient w.r.t. σ**:

Using chain rule:
```
∂φ/∂σ = (∂φ/∂x̂)·(∂x̂/∂σ)
```

where:
```
∂φ/∂x̂ = a₁ + πa₂·cos(πx̂) - πa₃·sin(πx̂)

∂x̂/∂σ = ∂[tanh(z/σ)]/∂σ
       = sech²(z/σ)·(-z/σ²)
       = -(z/σ²)·sech²(z/σ)
```

Combined:
```
∂φ/∂σ = [a₁ + πa₂·cos(πx̂) - πa₃·sin(πx̂)]·[-(z/σ²)·sech²(z/σ)]
```

**Key properties**:
1. **All gradients continuous**: No jumps or discontinuities
2. **No hard indexing**: Unlike splines, no discrete bin selection
3. **Bounded gradients**: tanh derivative is bounded (sech² ≤ 1)
4. **Smooth optimization landscape**: Easy for gradient descent

### Comparison with Splines

**Piecewise Linear Splines**:
```
φspline(x) = wᵢ·(x - bᵢ) + cᵢ    if bᵢ ≤ x < bᵢ₊₁
```

**Problems**:
- **Gradient discontinuities** at bin boundaries
- **Hard indexing** (argmax/argmin operations)
- **Non-differentiable** bin selection
- **Numerical instability** during backprop

**Fourier Series**:
- ✅ **Smooth everywhere**
- ✅ **No discrete choices**
- ✅ **Stable gradients**
- ✅ **Efficient computation** (just sin/cos)

---

## Why Fourier Series?

### Theoretical Justification

**1. Universal Approximation**

By the **Stone-Weierstrass theorem**, Fourier series can approximate any continuous function f: [-1,1] → ℝ arbitrarily well.

For any ε > 0 and continuous f, there exist coefficients {aₙ, bₙ} such that:
```
|f(x) - Σ[aₙ·cos(nπx) + bₙ·sin(nπx)]| < ε
     n=0
```

**2. Orthogonal Basis**

Fourier basis functions are orthogonal:
```
∫₋₁¹ sin(nπx)·cos(mπx) dx = 0

∫₋₁¹ sin(nπx)·sin(mπx) dx = δₙₘ (Kronecker delta)

∫₋₁¹ cos(nπx)·cos(mπx) dx = δₙₘ
```

This means each coefficient independently contributes to the approximation.

**3. Spectral Interpretation**

Fourier coefficients represent frequency components:
- **a₀**: DC component (mean)
- **a₁**: Linear trend
- **a₂, a₃**: First harmonic (fundamental frequency)

Different layers may need different frequency components, naturally learned during training.

### Practical Advantages

**1. Smooth and Differentiable**
- sin, cos, tanh are infinitely differentiable
- Gradients well-behaved (no explosions or vanishing)

**2. Bounded**
- tanh maps to [-1, 1]
- Fourier output bounded by |a₀| + |a₁| + |a₂| + |a₃|
- No risk of unbounded activations

**3. Interpretable**
- Visualize φ(x) as a curve
- Understand layer behavior
- a₂, a₃ magnitude → "non-linearity strength"

**4. Efficient**
- Only 4 multiplications + 2 transcendental functions per element
- Vectorized easily (PyTorch sin/cos are fast)

---

## Parameter Efficiency

### Parameter Count Comparison

**Standard LoRA**:
For layer with input dim k, output dim d, rank r:
- A: r × k parameters
- B: d × r parameters
- Total: **r(k + d)** parameters

**Fourier LS-LoRA**:
- A: r × k parameters (same)
- B: d × r parameters (same)
- Fourier: 5 parameters (a₀, a₁, a₂, a₃, σ)
- Total: **r(k + d) + 5** parameters

**Overhead**: Only **+5 parameters** per layer!

### Example: Phi-3-mini

Architecture:
- 32 transformer layers
- Hidden dim: 3072
- 5 attention modules per layer: q_proj, k_proj, v_proj, o_proj, gate_up_proj

**Standard LoRA (r=16)**:
- Parameters per module: ~16 × (3072 + 3072) = ~100K
- Total: 32 × 5 × 100K = **16M parameters**

**Fourier LS-LoRA (r=16)**:
- LoRA parameters: 16M (same)
- Fourier parameters: 32 × 5 × 5 = 800
- Total: **16M + 800 parameters**

**Overhead**: 800 / 16M = **0.005%** ← Negligible!

---

## Computational Complexity

### Forward Pass Analysis

**Per element computation**:

**Step 1: Normalization**
```
x̂ = tanh(z/σ)
```
- 1 division
- 1 tanh (exponential-based)
- Complexity: O(1) per element

**Step 2: Fourier evaluation**
```
φ = a₀ + a₁·x̂ + a₂·sin(πx̂) + a₃·cos(πx̂)
```
- 4 multiplications
- 2 additions
- 1 sine, 1 cosine (CORDIC or lookup table)
- Complexity: O(1) per element

**Total per layer**:
- LoRA computation: O(rk + rd) for matrix multiplies
- Fourier: O(d) for element-wise operations
- Total: O(rk + rd + d) = O(rk + rd) since d is absorbed

**Comparison with standard LoRA**:
- Standard LoRA: O(rk + rd)
- Fourier LS-LoRA: O(rk + rd + d)
- Overhead: O(d) element-wise operations
- Typically d ≪ rk, so overhead is **~1-5%**

### Memory Overhead

**Additional memory per layer**:
- Fourier parameters: 5 scalars × 4 bytes = **20 bytes**
- Intermediate x̂: d × 4 bytes (same size as z, reused)
- Total: **~20 bytes per layer** (negligible)

---

## Implementation Details

### Numerical Stability

**1. Safe σ**
```python
σ_safe = |σ| + ε
```
where ε = 1e-6 prevents division by zero.

**2. Bounded gradients**
- tanh derivative: sech²(x) ∈ (0, 1]
- sin/cos derivatives: ∈ [-1, 1]
- No gradient explosions

**3. Float precision**
- Use bfloat16 or float32 for training
- Fourier coefficients are small, no precision issues

### PyTorch Implementation

```python
def apply_fourier(z, a0, a1, a2, a3, sigma):
    """
    Apply Fourier transformation element-wise.

    Args:
        z: LoRA output, shape [batch, seq_len, d]
        a0, a1, a2, a3: Fourier coefficients, scalars
        sigma: Normalization scale, scalar

    Returns:
        Transformed output, same shape as z
    """
    # Normalize
    sigma_safe = torch.abs(sigma) + 1e-6
    x = torch.tanh(z / sigma_safe)

    # Fourier series
    phi = a0 + a1 * x + a2 * torch.sin(math.pi * x) + a3 * torch.cos(math.pi * x)

    return phi
```

**Vectorization**:
- All operations are element-wise
- Fully parallelizable on GPU
- No loops, no conditionals

### Training Tips

**1. Learning rate**
- Use standard LoRA learning rate (2e-4)
- Fourier parameters learn at same rate
- No special scheduling needed

**2. Regularization weight**
- Start with λ = 0.01
- Increase (0.05-0.1) for more regularization (stays more linear)
- Decrease (0.001-0.005) for less regularization (learns more non-linearity)

**3. Rank selection**
- Same guidelines as standard LoRA
- Typical: r = 8-32
- Higher rank → more capacity (may need less non-linearity)

**4. Initialization**
- Always use identity initialization
- Random initialization breaks the "start as LoRA" property

---

## Summary

### What We Implemented

**Fourier LS-LoRA** = Standard LoRA + Element-wise Fourier transformation

**Mathematical formula**:
```
h = W₀x + φ(tanh((α/r)·B(Ax)/σ))

where:
φ(x̂) = a₀ + a₁·x̂ + a₂·sin(πx̂) + a₃·cos(πx̂)
```

### Key Properties

1. ✅ **Starts as LoRA**: Identity initialization
2. ✅ **Learns non-linearity**: When beneficial for task
3. ✅ **Per-layer adaptive**: Different layers learn different transformations
4. ✅ **Efficient**: +5 parameters per layer (<0.01% overhead)
5. ✅ **Stable**: Smooth gradients, bounded activations
6. ✅ **Interpretable**: Can visualize learned curves
7. ✅ **Regularized**: L2 penalty encourages simplicity

### Why It Works

1. **Universal approximation**: Fourier series can represent any continuous function
2. **Smooth optimization**: No discontinuities, well-behaved gradients
3. **Proper initialization**: Starts from known good solution (standard LoRA)
4. **Regularization**: Only learns complexity when needed
5. **Adaptive normalization**: σ adapts to each layer's scale

---

## References

### Mathematical Background
- **Fourier Series**: Approximation theory, harmonic analysis
- **Stone-Weierstrass Theorem**: Universal approximation with polynomials/trig functions
- **Low-Rank Adaptation**: LoRA paper (Hu et al., 2021)

### Related Work
- **LoRA**: Low-Rank Adaptation of Large Language Models
- **Adaptive Activations**: Learned activation functions
- **Spline Activations**: Piecewise polynomial activations (we improved on this!)

---

## Appendix: Visualization

### Example Learned Curves

**Identity (Linear)**:
```
a₀=0, a₁=1, a₂=0, a₃=0
φ(x) = x
```
Straight line through origin, slope 1.

**Weak Non-linearity**:
```
a₀=0.01, a₁=0.98, a₂=0.15, a₃=-0.08
φ(x) ≈ x + 0.15·sin(πx) - 0.08·cos(πx)
```
Slightly wavy curve, mostly linear.

**Strong Non-linearity**:
```
a₀=0.1, a₁=0.7, a₂=0.5, a₃=0.4
φ(x) = 0.1 + 0.7x + 0.5·sin(πx) + 0.4·cos(πx)
```
S-shaped curve with multiple inflection points.

### Metrics

**Non-linearity strength**:
```
NL = |a₂| + |a₃|
```

Interpretation:
- NL < 0.01: Nearly linear
- 0.01 ≤ NL < 0.1: Weak non-linearity
- 0.1 ≤ NL < 0.3: Moderate non-linearity
- NL ≥ 0.3: Strong non-linearity

**Deviation from identity**:
```
Dev = √[a₀² + (a₁-1)² + a₂² + a₃²]
```

---

**Document Status**: Complete mathematical treatment of Fourier LS-LoRA implementation.
**Last Updated**: December 10, 2024
**Implementation**: `/home/arpit/peft/src/peft/tuners/lslora.py`
