"""
Resource loading helpers for package data files.

This module provides utilities to access data files included with the sinonym package
using importlib.resources, ensuring compatibility across all installation methods.
"""
from __future__ import annotations

import csv
import json
from collections.abc import Iterator
from contextlib import contextmanager
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any

_DATA_PKG = "sinonym.data"


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
    
    WARNING: This uses as_file().__enter__() which opens a temp file handle.
    Prefer using the open_resource_path() context manager or other read_* functions.
    This is provided for compatibility but should be used carefully.
    """
    ref = files(_DATA_PKG) / name
    return as_file(ref).__enter__()
