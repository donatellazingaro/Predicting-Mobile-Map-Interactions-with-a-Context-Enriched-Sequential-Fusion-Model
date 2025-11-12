"""
Basic environment and import tests for the torch-environment.
Verifies that core packages are installed and that project scripts
(Fusion Context and LSTM) can be imported successfully without errors.
"""

import importlib
import sys
from pathlib import Path


def test_core_packages():
    """Check that critical packages are installed and importable."""
    required_packages = ["torch", "pandas", "numpy", "geopandas", "osmnx"]
    for pkg in required_packages:
        assert importlib.util.find_spec(pkg) is not None, f"Package '{pkg}' not found."


def test_fusion_scripts_import():
    """Ensure Fusion Context scripts import cleanly (no syntax errors)."""
    fusion_path = Path("Fusion Context").resolve()
    sys.path.append(str(fusion_path))
    try:
        import fusion_train_autoencoder
    except Exception as e:
        raise AssertionError(f"Fusion Context import failed: {e}")
    finally:
        if str(fusion_path) in sys.path:
            sys.path.remove(str(fusion_path))


def test_lstm_scripts_import():
    """Ensure LSTM scripts import cleanly (no syntax errors)."""
    lstm_path = Path("LSTM").resolve()
    sys.path.append(str(lstm_path))
    try:
        import augmentation_sweep
    except Exception as e:
        raise AssertionError(f"LSTM import failed: {e}")
    finally:
        if str(lstm_path) in sys.path:
            sys.path.remove(str(lstm_path))
