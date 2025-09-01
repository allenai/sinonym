"""
Sinonym: Chinese Name Detection and Normalization Library

A sophisticated library for detecting and normalizing Chinese names from various
romanization systems with robust filtering capabilities.
"""

__version__ = "0.1.0"

__all__ = ["ChineseNameDetector"]

def __getattr__(name):
    """Lazy import to avoid eager loading of heavy dependencies."""
    if name == "ChineseNameDetector":
        from .detector import ChineseNameDetector
        return ChineseNameDetector
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
