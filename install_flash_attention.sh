#!/bin/bash
# Flash Attention 2 Installation Script
# Provides 50-100% speedup on attention operations

echo "=============================================================================="
echo "FLASH ATTENTION 2 INSTALLATION"
echo "=============================================================================="
echo ""
echo "This will install Flash Attention 2 for ~2x faster training."
echo "Expected installation time: 5-10 minutes (needs to compile CUDA kernels)"
echo ""

# Check CUDA availability
if ! command -v nvcc &> /dev/null; then
    echo "❌ ERROR: CUDA not found. Flash Attention requires CUDA."
    echo "   Please ensure CUDA toolkit is installed."
    exit 1
fi

echo "✓ CUDA found: $(nvcc --version | grep release)"
echo ""

# Check PyTorch
python -c "import torch; print(f'✓ PyTorch: {torch.__version__}'); print(f'✓ CUDA available: {torch.cuda.is_available()}')" || {
    echo "❌ ERROR: PyTorch not found or CUDA not available"
    exit 1
}

echo ""
echo "Installing Flash Attention 2..."
echo "(This may take 5-10 minutes - compiling CUDA kernels)"
echo ""

# Install with no build isolation (recommended for compatibility)
pip install flash-attn --no-build-isolation

# Verify installation
echo ""
echo "Verifying installation..."
python -c "
import flash_attn
print(f'✓ Flash Attention {flash_attn.__version__} installed successfully')
print('')
print('Flash Attention 2 is now available!')
print('Your training script will automatically use it.')
" && {
    echo ""
    echo "=============================================================================="
    echo "✅ INSTALLATION SUCCESSFUL"
    echo "=============================================================================="
    echo ""
    echo "Flash Attention 2 is now active. Expected benefits:"
    echo "  • 50-100% faster attention operations"
    echo "  • 40-60% overall training speedup"
    echo "  • Lower memory usage for attention"
    echo ""
    echo "Your next training run will automatically use Flash Attention 2."
    echo "=============================================================================="
    exit 0
} || {
    echo ""
    echo "=============================================================================="
    echo "⚠ INSTALLATION FAILED"
    echo "=============================================================================="
    echo ""
    echo "Flash Attention 2 could not be installed. Training will continue"
    echo "with standard attention (slower but still works)."
    echo ""
    echo "Common issues:"
    echo "  1. CUDA version mismatch"
    echo "  2. Insufficient compilation resources"
    echo "  3. Missing build dependencies (gcc, g++, ninja)"
    echo ""
    echo "Your training will still work, just without Flash Attention speedup."
    echo "=============================================================================="
    exit 1
}
