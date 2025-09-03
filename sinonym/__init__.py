"""
Sinonym: Chinese Name Detection and Normalization Library

A sophisticated library for detecting and normalizing Chinese names from various
romanization systems with robust filtering capabilities.
"""

import importlib.metadata

__version__ = importlib.metadata.version("sinonym")
__all__ = ["ChineseNameDetector"]

# Avoid accidental shadowing of stdlib modules when editors set CWD to package dir
def _warn_if_cwd_is_package_dir():
    try:
        import os, sys
        pkg_dir = os.path.dirname(__file__)
        # If current working directory equals package directory, importing stdlib modules
        # like `types` may resolve to our subpackages. This is a common Jupyter misconfig.
        if os.path.abspath(os.getcwd()) == os.path.abspath(pkg_dir):
            sys.stderr.write(
                "[sinonym] Warning: Working directory is the package folder; this may shadow stdlib modules.\n"
            )
    except Exception:
        pass

_warn_if_cwd_is_package_dir()


def __getattr__(name):
    """Lazy import to avoid eager loading of heavy dependencies."""
    if name == "ChineseNameDetector":
        from .detector import ChineseNameDetector

        return ChineseNameDetector
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
