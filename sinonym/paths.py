import logging
import os
from pathlib import Path

logger = logging.getLogger("sinonym")

try:
    PROJECT_ROOT_PATH = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
except NameError:
    PROJECT_ROOT_PATH = os.path.abspath(os.path.join(os.getcwd()))

DATA_PATH = Path(PROJECT_ROOT_PATH) / "data"
