"""
Resource loading helpers for package data files.

This module provides utilities to access data files included with the sinonym package
using importlib.resources, ensuring compatibility across all installation methods.

Includes helpers to load ML artifacts persisted with joblib or skops.
"""

from __future__ import annotations

import atexit
import csv
import json
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any

_DATA_PKG = "sinonym.data"
_RESOURCE_PATH_STACK = ExitStack()
atexit.register(_RESOURCE_PATH_STACK.close)


def read_text(name: str, encoding: str = "utf-8") -> str:
    """Read a text file from package resources."""
    return (files(_DATA_PKG) / name).read_text(encoding=encoding)


def read_bytes(name: str) -> bytes:
    """Read a binary file from package resources."""
    return (files(_DATA_PKG) / name).read_bytes()


def read_json(name: str, encoding: str = "utf-8") -> Any:
    """Read and parse a JSON file from package resources."""
    return json.loads(read_text(name, encoding=encoding))


def load_joblib(name: str):
    """Load a joblib model from package resources."""
    import joblib

    with open_resource_path(name) as path:
        return joblib.load(path)


def load_skops(name: str, trusted: list[str] | None = None):
    """Load a skops-serialized model from package resources.

    skops>=0.10 requires an explicit list of trusted types instead of a boolean.
    We derive the list via get_untrusted_types() and pass it back to load(),
    effectively trusting our own artifact. Callers can override by passing a
    specific ``trusted`` list.
    """
    from skops.io import get_untrusted_types, load

    with open_resource_path(name) as path:
        if trusted is None:
            # Support different skops versions: keyword-only, positional, or no-arg.
            try:
                trusted = get_untrusted_types(file=path)  # skops>=0.10 keyword-only
            except TypeError:
                try:
                    trusted = get_untrusted_types(path)  # older positional signature
                except TypeError:
                    trusted = get_untrusted_types()  # no-arg fallback
        return load(path, trusted=trusted)


@contextmanager
def open_resource_path(name: str) -> Iterator[Path]:
    """
    Context manager that provides a filesystem path to a resource.

    This handles extraction to temporary files when the package is inside a zip.
    Use this when you need a real filesystem path (e.g., for libraries that require file paths).
    """
    with as_file(files(_DATA_PKG) / name) as path:
        yield path


def open_csv_reader(name: str, encoding: str = "utf-8", **kwargs):
    """
    Open a CSV file from package resources and return a csv.DictReader.

    This is a convenience function for the common pattern of reading CSV files.
    """
    text_content = read_text(name, encoding=encoding)
    import io

    # Create a StringIO object from the text content
    csv_file = io.StringIO(text_content)
    return csv.DictReader(csv_file, **kwargs)


def resource_path(name: str) -> Path:
    """
    Returns a real filesystem Path to the resource.

    This compatibility helper keeps the extraction context alive for process
    lifetime via a module-level ExitStack.
    """
    ref = files(_DATA_PKG) / name
    return _RESOURCE_PATH_STACK.enter_context(as_file(ref))
