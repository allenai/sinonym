"""
Text processing module for Chinese name processing.

This module contains specialized text processing components that handle
the core string manipulation operations for Chinese name normalization.
"""

from .compound_detector import CompoundDetector
from .text_normalizer import TextNormalizer
from .text_preprocessor import TextPreprocessor

__all__ = [
    "CompoundDetector",
    "TextNormalizer",
    "TextPreprocessor",
]
