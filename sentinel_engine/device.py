"""
Device Detection — auto-selects cuda → mps → cpu.
No hardcoded device strings anywhere in the codebase.
"""

import logging

logger = logging.getLogger(__name__)


def detect_device() -> str:
    """
    Return the best available compute device.

    Returns:
        "cuda" if NVIDIA GPU available
        "mps"  if Apple Silicon GPU available
        "cpu"  fallback
    """
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            logger.info(f"Using CUDA device: {name}")
            return "cuda"
        if torch.backends.mps.is_available():
            logger.info("Using Apple MPS device")
            return "mps"
    except ImportError:
        pass

    logger.info("Using CPU device")
    return "cpu"
