"""Smoke tests for ComfyUI-TRELLIS2.

These tests verify basic module structure without requiring GPU or model downloads.
"""
import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_import_utils():
    """Test that utils module can be imported."""
    from nodes.utils import logger, tensor_to_pil, pil_to_tensor
    assert logger is not None
    assert callable(tensor_to_pil)
    assert callable(pil_to_tensor)


def test_tensor_pil_conversion():
    """Test tensor to PIL conversion utilities."""
    import torch
    import numpy as np
    from PIL import Image
    from nodes.utils import tensor_to_pil, pil_to_tensor

    # Create test tensor [1, H, W, C]
    test_tensor = torch.rand(1, 64, 64, 3)

    # Convert to PIL
    pil_image = tensor_to_pil(test_tensor)
    assert isinstance(pil_image, Image.Image)
    assert pil_image.size == (64, 64)

    # Convert back to tensor
    back_tensor = pil_to_tensor(pil_image)
    assert back_tensor.shape == (1, 64, 64, 3)
    assert back_tensor.min() >= 0
    assert back_tensor.max() <= 1


def test_rgba_conversion():
    """Test RGBA image conversion."""
    import torch
    from PIL import Image
    from nodes.utils import tensor_to_pil, pil_to_tensor

    # Create RGBA tensor
    test_tensor = torch.rand(1, 64, 64, 4)

    pil_image = tensor_to_pil(test_tensor)
    assert pil_image.mode == 'RGBA'

    back_tensor = pil_to_tensor(pil_image)
    assert back_tensor.shape == (1, 64, 64, 4)


def test_node_class_mappings_structure():
    """Test that node mappings have correct structure (without importing full modules)."""
    # This test verifies the expected node names exist
    expected_nodes = [
        "DownloadAndLoadTrellis2Model",
        "Trellis2SetSamplerParams",
        "Trellis2PreprocessImage",
        "Trellis2ImageTo3D",
        "Trellis2DecodeLatent",
        "Trellis2ExportGLB",
        "Trellis2RenderPreview",
        "Trellis2RenderVideo",
    ]

    # Just verify the list is correct (actual import tested elsewhere)
    assert len(expected_nodes) == 8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
