# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
SincLoRA: Learnable Rank Adaptation via Sinc Basis Functions

This module implements SincLoRA, a novel parameter-efficient fine-tuning method
that uses sinc basis functions to enable learnable rank allocation per layer.

Key innovation: Instead of fixed rank for all layers, each layer learns optimal
effective rank through weighted sinc basis functions: ΔW = Σ αᵢ·sinc(ωᵢ·(z-aᵢ))·BᵢAᵢ

Reference: Based on the principle that different layers need different ranks
for optimal adaptation (attention layers often need higher rank than MLPs).
"""

import math
import warnings
from dataclasses import dataclass, field
from typing import Optional, Union

import torch
import torch.nn as nn

from peft.tuners.lora import LoraConfig, LoraLayer, LoraModel
from peft.utils import PeftType


# =====================================================================
# Configuration
# =====================================================================

@dataclass
class SincLoraConfig(LoraConfig):
    """
    Configuration class for SincLoRA.

    SincLoRA extends LoRA by using K sinc basis functions instead of a single
    rank-r decomposition. Each layer learns optimal weights for the basis functions,
    enabling adaptive rank allocation.

    Args:
        K (int): Number of sinc basis functions (default: 8)
            Each basis function has its own (Aᵢ, Bᵢ) pair with rank r/K
        init_sigma (float): Initial temperature for tanh normalization (default: 1.0)
            Controls the input range to sinc functions
        omega_init (float): Initial frequency for sinc functions (default: 1.0)
            Starting value for learnable frequency parameters
        anchor_spacing (str): How to initialize anchor points (default: "uniform")
            Options: "uniform" (spread across [-0.8, 0.8]) or "random"
        sinc_reg_weight (float): Regularization weight for sinc parameters (default: 0.01)
            Encourages sparsity in alpha coefficients

    Inherited from LoraConfig:
        r (int): Total rank budget (will be split across K basis functions)
        lora_alpha (int): Scaling parameter
        lora_dropout (float): Dropout probability
        target_modules: Which modules to apply SincLoRA to
        ... (see LoraConfig for full list)

    Example:
        ```python
        config = SincLoraConfig(
            r=16,              # Total rank budget
            K=8,               # 8 sinc basis functions
            lora_alpha=32,
            target_modules="all-linear",
        )
        model = get_peft_model(base_model, config)
        ```
    """

    K: int = field(default=8, metadata={"help": "Number of sinc basis functions"})
    init_sigma: float = field(default=1.0, metadata={"help": "Initial temperature for tanh normalization"})
    omega_init: float = field(default=1.0, metadata={"help": "Initial frequency for sinc functions"})
    anchor_spacing: str = field(default="uniform", metadata={"help": "Anchor initialization: 'uniform' or 'random'"})
    alpha_init: str = field(default="uniform", metadata={"help": "Alpha initialization: 'uniform', 'gaussian', or 'dirichlet'"})
    sinc_reg_weight: float = field(default=0.01, metadata={"help": "Regularization weight for sinc parameters"})

    def __post_init__(self):
        super().__post_init__()
        # Set peft_type AFTER calling super().__post_init__() to avoid being overridden
        self.peft_type = PeftType.SINCLORA


# =====================================================================
# Sinc Parameter Container
# =====================================================================

class SincParams(nn.Module):
    """
    Container for per-layer sinc parameters.

    Stores learnable parameters for K sinc basis functions:
    - alpha: Amplitude coefficients (which basis functions are important)
    - omega: Frequency parameters (how oscillatory each basis is)
    - anchors: Shift parameters (where each basis is centered)
    - sigma: Temperature for input normalization

    After training, the learned alpha coefficients determine the effective rank
    of the layer via stable rank: (Σ αᵢ²) / (max αⱼ²)
    """

    def __init__(self, K: int, init_sigma: float, omega_init: float, anchor_spacing: str, alpha_init: str = "uniform"):
        super().__init__()

        # Amplitude coefficients (K basis functions)
        if alpha_init == "uniform":
            # Initialize uniformly to give all bases equal weight initially
            self.alpha = nn.Parameter(torch.ones(K) / K)
        elif alpha_init == "gaussian":
            # Random initialization centered at 0
            self.alpha = nn.Parameter(torch.randn(K) * 0.1)
        elif alpha_init == "dirichlet":
            # Sample from Dirichlet distribution (sums to 1, but random)
            # Use alpha=1.0 for uniform prior over simplex
            dist = torch.distributions.Dirichlet(torch.ones(K))
            self.alpha = nn.Parameter(dist.sample())
        else:
            raise ValueError(f"Unknown alpha_init: {alpha_init}. Use 'uniform', 'gaussian', or 'dirichlet'.")

        # Frequency parameters (learnable)
        self.omega = nn.Parameter(torch.ones(K) * omega_init)

        # Anchor points (learnable)
        if anchor_spacing == "uniform":
            # Spread uniformly across [-0.8, 0.8] (avoid edges for numerical stability)
            anchors = torch.linspace(-0.8, 0.8, K)
        elif anchor_spacing == "random":
            # Random initialization with small variance
            anchors = torch.randn(K) * 0.5
        else:
            raise ValueError(f"Unknown anchor_spacing: {anchor_spacing}. Use 'uniform' or 'random'.")
        self.anchors = nn.Parameter(anchors)

        # Temperature for tanh normalization
        self.sigma = nn.Parameter(torch.tensor(init_sigma, dtype=torch.float32))


# =====================================================================
# SincLoRA Layer Base Class
# =====================================================================

class SincLoraLayer(LoraLayer):
    """
    SincLoRA layer with learnable rank via sinc basis functions.

    Key differences from standard LoRA:
    1. K pairs of (Aᵢ, Bᵢ) matrices instead of single (A, B)
    2. Each pair has rank r/K (total parameters same as LoRA!)
    3. Sinc basis functions φᵢ(z) = sinc(ωᵢ·(z - aᵢ)) weight the contributions
    4. Learnable per-layer parameters: α, ω, anchors, σ

    Forward computation:
        h = W₀(x) + Σᵢ αᵢ · Bᵢ(φᵢ(Aᵢ(x)))

    where φᵢ applies sinc transformation element-wise.
    """

    # Register sinc parameters for saving/loading
    adapter_layer_names = (
        "lora_A", "lora_B",
        "lora_embedding_A", "lora_embedding_B",
        "lora_A_basis", "lora_B_basis",  # Our K pairs of matrices
        "sinc_params"  # Our custom sinc parameters
    )
    other_param_names = (
        "r", "lora_alpha", "scaling", "lora_dropout",
        "K", "init_sigma", "omega_init", "sinc_reg_weight"
    )

    def __init__(self, base_layer: nn.Module, **kwargs):
        super().__init__(base_layer, **kwargs)
        # Container for sinc parameters per adapter
        self.sinc_params = nn.ModuleDict({})
        # Store K per adapter
        self.K = {}
        # Containers for K pairs of LoRA matrices
        self.lora_A_basis = nn.ModuleDict({})
        self.lora_B_basis = nn.ModuleDict({})

    def update_layer(
        self,
        adapter_name: str,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        init_lora_weights: Union[bool, str],
        **kwargs
    ):
        """
        Initialize SincLoRA parameters for this adapter.

        Creates K pairs of (Aᵢ, Bᵢ) matrices with rank r/K each, and
        initializes sinc parameters (α, ω, anchors, σ).
        """

        # Get SincLoRA-specific config
        K = kwargs.get("K", 8)
        init_sigma = kwargs.get("init_sigma", 1.0)
        omega_init = kwargs.get("omega_init", 1.0)
        anchor_spacing = kwargs.get("anchor_spacing", "uniform")
        alpha_init = kwargs.get("alpha_init", "uniform")

        # Store K for this adapter
        self.K[adapter_name] = K

        # Compute rank per basis: r_basis = ceil(r / K)
        r_basis = math.ceil(r / K)

        # Get device from base layer (critical for DDP - base layer already on correct device)
        device = self.get_base_layer().weight.device

        # Create K pairs of LoRA matrices
        self.lora_A_basis[adapter_name] = nn.ModuleList()
        self.lora_B_basis[adapter_name] = nn.ModuleList()

        for i in range(K):
            # Create A_i and B_i with rank r/K
            A_i = nn.Linear(self.in_features, r_basis, bias=False)
            B_i = nn.Linear(r_basis, self.out_features, bias=False)

            # Initialize with standard LoRA init
            if init_lora_weights == "gaussian":
                nn.init.normal_(A_i.weight, std=1/r_basis)
                nn.init.zeros_(B_i.weight)
            elif init_lora_weights is True or init_lora_weights == "kaiming":
                # Default: kaiming for A, zeros for B
                nn.init.kaiming_uniform_(A_i.weight, a=math.sqrt(5))
                nn.init.zeros_(B_i.weight)
            else:
                # Custom initialization (e.g., "pissa", "olora") not supported yet
                # Fall back to kaiming
                nn.init.kaiming_uniform_(A_i.weight, a=math.sqrt(5))
                nn.init.zeros_(B_i.weight)

            # Move to same device as base layer BEFORE appending (critical for DDP)
            A_i = A_i.to(device)
            B_i = B_i.to(device)

            self.lora_A_basis[adapter_name].append(A_i)
            self.lora_B_basis[adapter_name].append(B_i)

        # Initialize sinc parameters and move to same device as base layer
        sinc_params = SincParams(K, init_sigma, omega_init, anchor_spacing, alpha_init).to(device)
        self.sinc_params[adapter_name] = sinc_params

        # Scaling (use standard LoRA scaling)
        if kwargs.get("use_rslora", False):
            self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = lora_alpha / r

        # Dropout
        if lora_dropout > 0.0:
            self.lora_dropout[adapter_name] = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout[adapter_name] = nn.Identity()

    def apply_sinc_basis(self, z: torch.Tensor, adapter_name: str) -> tuple:
        """
        Apply sinc basis transformation element-wise.

        Args:
            z: Input tensor [K, batch, seq, r_basis] (stacked from K basis outputs)
            adapter_name: Name of adapter

        Returns:
            sinc_tensor: Transformed tensor [K, batch, seq, r_basis]
            alpha: Amplitude coefficients [K]
        """
        params = self.sinc_params[adapter_name]

        # Normalize input to [-1, 1] using tanh
        # This keeps sinc arguments in a reasonable range
        sigma_safe = torch.abs(params.sigma) + 1e-6
        z_norm = torch.tanh(z / sigma_safe)

        # Apply sinc basis functions
        # z_norm: [K, batch, seq, r_basis]
        K = self.K[adapter_name]
        sinc_outputs = []

        for i in range(K):
            omega_i = params.omega[i]
            anchor_i = params.anchors[i]

            # Compute sinc(ωᵢ · (z - aᵢ))
            # Extract i-th basis output: [batch, seq, r_basis]
            z_i = z_norm[i]

            arg = math.pi * omega_i * (z_i - anchor_i)

            # Sinc function: sin(x) / x, with special case for x=0
            # torch.sinc(x) = sin(πx) / (πx), so we need sinc(arg/π)
            sinc_val = torch.sinc(arg / math.pi)

            sinc_outputs.append(sinc_val)

        # Stack: [K, batch, seq, r_basis]
        sinc_tensor = torch.stack(sinc_outputs, dim=0)

        return sinc_tensor, params.alpha

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass with sinc basis weighting.

        Computes: h = W₀(x) + Σᵢ αᵢ · Bᵢ(sinc(ωᵢ·(Aᵢ(x) - aᵢ)))
        """
        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            # Cannot merge sinc LoRA due to non-linearity
            warnings.warn("SincLoRA cannot be merged due to non-linear sinc transformation. Using unmerged mode.")
            self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        else:
            # Standard forward: base layer + adapter contributions
            result = self.base_layer(x, *args, **kwargs)

            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A_basis:
                    continue

                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                K = self.K[active_adapter]

                # Apply dropout to input
                x_dropped = dropout(x)

                # Compute each basis contribution: z_i = A_i(x)
                basis_outputs = []
                for i in range(K):
                    A_i = self.lora_A_basis[active_adapter][i]
                    z_i = A_i(x_dropped.to(A_i.weight.dtype))
                    basis_outputs.append(z_i)

                # Stack: [K, batch, seq, r_basis]
                z_stacked = torch.stack(basis_outputs, dim=0)

                # Apply sinc basis transformation
                # Returns: [K, batch, seq, r_basis] and alpha [K]
                sinc_vals, alpha = self.apply_sinc_basis(z_stacked, active_adapter)

                # Weight by alpha: α_i · φ_i(z_i)
                # alpha: [K] -> [K, 1, 1, 1]
                alpha_expanded = alpha.view(K, 1, 1, 1)
                weighted = sinc_vals * alpha_expanded  # [K, batch, seq, r_basis]

                # Apply B_i to each weighted output and sum
                delta = 0
                for i in range(K):
                    B_i = self.lora_B_basis[active_adapter][i]
                    delta = delta + B_i(weighted[i])

                # Apply scaling
                delta = delta * scaling

                result = result + delta.to(previous_dtype)

        return result


# =====================================================================
# Concrete Layer Implementations
# =====================================================================

class Linear(nn.Module, SincLoraLayer):
    """SincLoRA applied to nn.Linear layer."""

    def __init__(
        self,
        base_layer,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        K: int = 8,
        **kwargs,
    ):
        super().__init__()
        SincLoraLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
            K=K,
            **kwargs,
        )

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return SincLoraLayer.forward(self, x, *args, **kwargs)


# =====================================================================
# SincLoRA Model
# =====================================================================

class SincLoraModel(LoraModel):
    """
    SincLoRA model with learnable rank allocation.

    Extends LoraModel to:
    1. Mark sinc parameters as trainable
    2. Compute sinc regularization loss (encourages sparsity in alpha)
    3. Dispatch to SincLoRA layer implementations
    """

    prefix: str = "lora_"

    def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
        """Mark both LoRA matrices and sinc parameters as trainable."""
        for n, p in model.named_parameters():
            if self.prefix not in n and "sinc_params" not in n and "lora_A_basis" not in n and "lora_B_basis" not in n:
                p.requires_grad = False

        # Handle bias based on config
        for active_adapter in self.active_adapters:
            bias = getattr(self.peft_config[active_adapter], "bias", "none")
            if bias == "none":
                continue
            if bias == "all":
                for n, p in model.named_parameters():
                    if "bias" in n:
                        p.requires_grad = True
            elif bias == "lora_only":
                for n, p in model.named_parameters():
                    if "bias" in n and self.prefix in n:
                        p.requires_grad = True

    @staticmethod
    def _create_new_module(lora_config, adapter_name, target, **kwargs):
        """Create new SincLoRA module."""
        if isinstance(lora_config, SincLoraConfig):
            kwargs["K"] = lora_config.K
            kwargs["init_sigma"] = lora_config.init_sigma
            kwargs["omega_init"] = lora_config.omega_init
            kwargs["anchor_spacing"] = lora_config.anchor_spacing
            kwargs["alpha_init"] = lora_config.alpha_init

        if isinstance(target, nn.Linear):
            new_module = Linear(target, adapter_name, **kwargs)
        else:
            # Fallback to parent LoRA for other layer types
            # (could extend to Conv2d, Embedding, etc. in future)
            new_module = LoraModel._create_new_module(
                lora_config, adapter_name, target, **kwargs
            )

        return new_module

    def _get_peft_specific_state_dict(self, *args, **kwargs):
        """
        Override to include sinc parameters in saved state.

        The base implementation filters by prefix, which excludes sinc_params.
        We need to explicitly add them back.
        """
        # Get base state dict (includes lora_A_basis, lora_B_basis)
        state_dict = super()._get_peft_specific_state_dict(*args, **kwargs)

        # Add all sinc parameters from full state dict
        full_state = self.model.state_dict()
        for key in full_state:
            if "sinc_params" in key:
                state_dict[key] = full_state[key]

        return state_dict

    def get_sinc_loss(self):
        """
        Compute regularization loss for sinc parameters.

        Encourages:
        1. Sparsity in alpha (L1 loss) - use only necessary basis functions
        2. Reasonable omega values (L2 loss) - prevent extreme frequencies

        Returns:
            Scalar tensor with regularization loss
        """
        losses = []
        target_device = next(self.parameters()).device

        for module in self.modules():
            if isinstance(module, SincLoraLayer):
                for adapter_name, params in module.sinc_params.items():
                    # L1 loss on alpha (encourage sparsity)
                    # Layers that need low rank will have few large alphas
                    alpha_loss = torch.mean(torch.abs(params.alpha))

                    # L2 loss on omega (prevent extreme frequencies)
                    # Keep omegas close to init value (1.0)
                    omega_loss = torch.mean((params.omega - 1.0) ** 2)

                    # Combined loss (omega loss weighted lower)
                    loss = alpha_loss + 0.1 * omega_loss
                    losses.append(loss.to(target_device))

        if len(losses) == 0:
            return torch.tensor(0.0, device=target_device)

        return torch.stack(losses).mean()


# =====================================================================
# Registration
# =====================================================================

from peft.utils import register_peft_method

# Register SincLoRA with PEFT
register_peft_method(
    name="sinclora",
    config_cls=SincLoraConfig,
    model_cls=SincLoraModel,
    prefix="lora_"
)
