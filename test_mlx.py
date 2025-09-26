#!/usr/bin/env python3
"""
Test MLX integration.
"""
import platform

import script as tx


def test_mlx_availability() -> bool:
    """Check that MLX core can be imported."""
    try:
        import mlx.core as mx  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        print(f"✗ MLX core import failed: {exc}")
        return False
    else:
        print("✓ MLX core imported successfully")
        return True


def test_mlx_backend() -> bool:
    """Confirm that the transcription backend wrapper is ready."""
    backend = getattr(tx, "MLX_BACKEND", None)
    if not getattr(tx, "LightningWhisperMLX", None):
        print("✗ No MLX backend wired up in script.py")
        return False
    print(f"✓ MLX backend available: {backend or 'lightning-whisper-mlx'}")
    try:
        tx.LightningWhisperMLX(model="tiny")
    except Exception as exc:  # noqa: BLE001
        print(f"✗ LightningWhisperMLX instantiation failed: {exc}")
        return False
    else:
        print("✓ LightningWhisperMLX wrapper instantiates")
        return True


if __name__ == "__main__":
    print(f"Platform: {platform.system()}, Machine: {platform.machine()}")
    print(f"Is Apple Silicon: {tx.is_apple_silicon()}")

    mlx_available = test_mlx_availability()
    backend_ready = test_mlx_backend()

    print(f"Default mode would be: {tx.get_default_mode()}")

    if mlx_available and backend_ready:
        print("\n🎉 MLX integration should work!")
    else:
        print("\n⚠️  MLX integration may have issues")
