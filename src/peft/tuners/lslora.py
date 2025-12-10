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
import math
import warnings
from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from peft.tuners.lora import LoraLayer, LoraConfig, LoraModel
from peft.tuners.tuners_utils import BaseTunerLayer
from peft.utils.integrations import gather_params_ctx
from peft.utils import PeftType, register_peft_method

class LSLoraConfig(LoraConfig):
    """
    Configuration class for Fourier series-based LS-LoRA.

    Applies learnable Fourier activation: φ(x) = a₀ + a₁·x + a₂·sin(πx) + a₃·cos(πx)
    where x = tanh(z/σ) normalizes LoRA output to [-1, 1].
    """
    def __init__(
        self,
        fourier_reg_weight: float = 0.01,  # L2 penalty for deviation from identity
        init_sigma: float = 1.0,           # Initial tanh normalization scale
        **kwargs
    ):
        super().__init__(**kwargs)
        self.fourier_reg_weight = fourier_reg_weight
        self.init_sigma = init_sigma
        self.peft_type = PeftType.LSLORA

class FourierParams(nn.Module):
    """Wrapper module to hold Fourier parameters for proper saving/loading."""
    def __init__(self, init_sigma: float = 1.0):
        super().__init__()
        # Identity initialization: [a₀=0, a₁=1, a₂=0, a₃=0]
        identity_coeffs = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float32)
        self.weight = nn.Parameter(identity_coeffs)  # Use 'weight' for PEFT compatibility
        self.sigma = nn.Parameter(torch.tensor(init_sigma, dtype=torch.float32))

class LSLoraLayer(LoraLayer):
    """
    Fourier series-based LS-LoRA layer implementation.

    Applies φ(z) = a₀ + a₁·x + a₂·sin(πx) + a₃·cos(πx) where x = tanh(z/σ)
    Layer-wise shared: all elements use same [a₀, a₁, a₂, a₃, σ] coefficients.
    """
    # Register Fourier parameters for saving/loading
    adapter_layer_names = ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B", "fourier_params")
    other_param_names = ("r", "lora_alpha", "scaling", "lora_dropout", "fourier_reg_weight", "init_sigma")

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        super().__init__(base_layer, **kwargs)
        # Fourier parameters module per adapter
        self.fourier_params = nn.ModuleDict({})

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, **kwargs):
        super().update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, **kwargs)

        # Initialize Fourier parameters for this adapter
        if adapter_name not in self.fourier_params:
            init_sigma = kwargs.get("init_sigma", 1.0)
            self.fourier_params[adapter_name] = FourierParams(init_sigma=init_sigma)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            # Fourier LoRA cannot be merged due to non-linearity
            warnings.warn("FourierLoRA cannot be merged due to non-linearity. Using unmerged mode.")
            self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)

            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue

                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]

                # Standard LoRA computation: z = B(A(x)) * scaling
                x_lora = x.to(lora_A.weight.dtype)
                z = lora_B(lora_A(dropout(x_lora))) * scaling

                # Apply Fourier activation
                if active_adapter in self.fourier_params:
                    z = self.apply_fourier(z, active_adapter)

                result += z.to(previous_dtype)

        return result

    def apply_fourier(self, z: torch.Tensor, adapter_name: str) -> torch.Tensor:
        """
        Apply Fourier series activation element-wise.

        Math:
            x = tanh(z / σ)                                    # Normalize to [-1, 1]
            φ(x) = a₀ + a₁·x + a₂·sin(πx) + a₃·cos(πx)       # Fourier series

        Args:
            z: LoRA output tensor of any shape [..., features]
            adapter_name: Name of the adapter

        Returns:
            Transformed tensor with same shape as z

        Gradient flow: All operations are differentiable:
            - ∂φ/∂a₀ = 1
            - ∂φ/∂a₁ = x
            - ∂φ/∂a₂ = sin(πx)
            - ∂φ/∂a₃ = cos(πx)
            - ∂φ/∂σ = -z/σ² · sech²(z/σ) · [a₁ + πa₂cos(πx) - πa₃sin(πx)]
        """
        # Get parameters and move to device
        params = self.fourier_params[adapter_name]
        coeffs = params.weight.to(z.device)
        sigma = params.sigma.to(z.device)

        # Safe sigma: prevent division by zero, ensure sigma > 0
        sigma_safe = torch.abs(sigma) + 1e-6

        # Normalize to [-1, 1] using tanh
        # σ is learnable and adapts to activation magnitudes per layer
        x = torch.tanh(z / sigma_safe)

        # Extract coefficients
        a0, a1, a2, a3 = coeffs[0], coeffs[1], coeffs[2], coeffs[3]

        # Compute Fourier series (all element-wise, fully differentiable)
        # No clamping, no hard indexing, no exponentials
        phi = a0 + a1 * x + a2 * torch.sin(math.pi * x) + a3 * torch.cos(math.pi * x)

        return phi


class Linear(nn.Module, LSLoraLayer):
    """
    Fourier LS-LoRA implemented in a Linear layer.

    This is the concrete implementation that combines nn.Module with LSLoraLayer.
    Required by PEFT framework for proper layer creation and parameter management.
    """
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        is_target_conv_1d_layer: bool = False,
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        use_dora: bool = False,
        init_sigma: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__()
        LSLoraLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
            use_dora=use_dora,
            init_sigma=init_sigma,
        )

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        # Use the LSLoraLayer forward implementation
        return LSLoraLayer.forward(self, x, *args, **kwargs)

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "fourier_lora." + rep


class LSLoraModel(LoraModel):
    """
    Fourier series-based LS-LoRA model.
    """
    prefix: str = "lora_"

    def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
        """
        Override to mark both LoRA and Fourier parameters as trainable.

        The base implementation only checks for self.prefix ("lora_"), but we also
        have Fourier parameters ("fourier_coeffs", "fourier_sigma") that need to be trainable.
        """
        for n, p in model.named_parameters():
            # Keep trainable if parameter name contains "lora_" OR "fourier_params"
            if self.prefix not in n and "fourier_params" not in n:
                p.requires_grad = False

        # Handle bias parameters (from base class)
        for active_adapter in self.active_adapters:
            bias = getattr(self.peft_config[active_adapter], "bias", "none")
            if bias == "none":
                continue

            if bias == "all":
                for n, p in model.named_parameters():
                    if "bias" in n:
                        p.requires_grad = True
            elif bias.endswith("_only"):
                for m in model.modules():
                    if isinstance(m, LSLoraLayer) and hasattr(m, "bias") and m.bias is not None:
                        m.bias.requires_grad = True

    @staticmethod
    def _create_new_module(lora_config, adapter_name, target, **kwargs):
        """
        Create a new Fourier LoRA module.

        This method dispatches to the appropriate layer type (Linear, Conv2d, etc.)
        based on the target module type.
        """
        if isinstance(lora_config, LSLoraConfig):
            kwargs["init_sigma"] = lora_config.init_sigma

        # Dispatch to correct layer type based on target module
        if isinstance(target, nn.Linear):
            new_module = Linear(target, adapter_name, **kwargs)
        else:
            # Fallback to parent implementation for other layer types
            new_module = LoraModel._create_new_module(lora_config, adapter_name, target, **kwargs)

        return new_module

    def _get_peft_specific_state_dict(self, *args, **kwargs):
        """
        Override to include Fourier parameters in saved state.

        The base implementation filters by prefix, which excludes fourier_params.
        We need to explicitly add them back.
        """
        # Get base state dict (includes lora_A, lora_B)
        state_dict = super()._get_peft_specific_state_dict(*args, **kwargs)

        # Add all Fourier parameters from full state dict
        full_state = self.model.state_dict()
        for key in full_state:
            if "fourier_params" in key:
                state_dict[key] = full_state[key]

        return state_dict

    def get_fourier_loss(self):
        """
        Compute regularization loss for Fourier coefficients.

        Penalizes deviation from identity [a₀=0, a₁=1, a₂=0, a₃=0]:
        L_reg = mean over layers of MSE((coeffs - identity)²)

        Returns:
            Scalar tensor with regularization loss
        """
        total_loss = 0.0
        count = 0
        identity = torch.tensor([0.0, 1.0, 0.0, 0.0])

        for module in self.modules():
            if isinstance(module, LSLoraLayer):
                for adapter_name, params in module.fourier_params.items():
                    coeffs = params.weight
                    target = identity.to(coeffs.device)
                    # MSE loss: mean((coeffs - identity)²)
                    loss = torch.mean((coeffs - target) ** 2)
                    total_loss += loss
                    count += 1

        if count == 0:
            return torch.tensor(0.0)

        # Average over all layers
        return total_loss / count

# Register the method (keep name "lslora" for backward compatibility)
# Use prefix="lora_" to match both lora_* and fourier_* parameters
register_peft_method(name="lslora", config_cls=LSLoraConfig, model_cls=LSLoraModel, prefix="lora_")
