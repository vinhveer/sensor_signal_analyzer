"""Default constants used as fallbacks when not provided by YAML config."""
from __future__ import annotations

DEFAULT_CSV_SEP_CANDIDATES = [",", ";", None]
DEFAULT_SOURCE_SUBDIR_CANDIDATES = [".", "Processed"]
DEFAULT_ACC_COLUMNS = [
    "Acceleration - x (m/s\u00b2)",
    "Acceleration - x (m/s\u00b2).1",
]
DEFAULT_EXPECTED_CHANNELS = 2
