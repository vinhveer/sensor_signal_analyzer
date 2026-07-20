"""CSV reading and conversion of raw sensor sources into .npy arrays."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def normalize_col_name(name: str) -> str:
    return " ".join(str(name).strip().lower().split())


def read_csv_flexible(file_path: Path, csv_sep_candidates: list) -> pd.DataFrame:
    last_error = None
    for sep in csv_sep_candidates:
        try:
            if sep is None:
                frame = pd.read_csv(file_path, sep=None, engine="python")
            else:
                frame = pd.read_csv(file_path, sep=sep)
            if frame.shape[1] > 1:
                return frame
        except Exception as exc:
            last_error = exc
    raise ValueError(f"Failed to read CSV {file_path}: {last_error}")


def select_acc_columns(frame: pd.DataFrame, explicit_cols: list[str], expected_channels: int) -> list[str]:
    normalized = {normalize_col_name(col): col for col in frame.columns}
    picked = [normalized[normalize_col_name(col)] for col in explicit_cols if normalize_col_name(col) in normalized]
    if len(picked) == expected_channels:
        return picked
    fallback = [col for col in frame.columns if "acceleration - x" in normalize_col_name(col)]
    if len(fallback) < expected_channels:
        raise ValueError(f"Expected at least {expected_channels} acceleration-x columns, found {len(fallback)}. Columns={list(frame.columns)}")
    return fallback[:expected_channels]


def load_source_array(file_path: Path, acc_columns: list[str], expected_channels: int, csv_sep_candidates: list) -> np.ndarray:
    frame = read_csv_flexible(file_path, csv_sep_candidates)
    cols = select_acc_columns(frame, acc_columns, expected_channels)
    numeric = frame.loc[:, cols].apply(pd.to_numeric, errors="coerce")
    array = numeric.to_numpy(dtype=np.float32)
    if array.ndim != 2 or int(array.shape[1]) != expected_channels:
        raise ValueError(f"Invalid array shape from {file_path}: {array.shape}")
    if not np.all(np.isfinite(array)):
        bad_rows = np.where(np.any(~np.isfinite(array), axis=1))[0]
        if bad_rows.size:
            first_bad = int(bad_rows[0])
            trailing_bad = np.arange(first_bad, array.shape[0])
            if np.array_equal(bad_rows, trailing_bad):
                trimmed = array[:first_bad, :]
                if trimmed.shape[0] > 0:
                    print(f"[TRIM] {file_path}: trimmed trailing bad rows from index {first_bad} to {array.shape[0] - 1}")
                    return np.asarray(trimmed, dtype=np.float32)
        raise ValueError(f"Non-finite values detected in selected columns of {file_path}")
    return array


def convert_source_to_npy(
    input_root: str,
    output_root: str,
    output_subdir: str,
    channels: int,
    end_t: Optional[int],
    overwrite: bool,
    acc_columns: list[str],
    source_subdir_candidates: list[str],
    csv_sep_candidates: list,
) -> dict:
    from .discovery import class_dirs, find_source_dir_for_class

    in_root = Path(input_root)
    out_root = Path(output_root)
    converted = 0
    skipped = 0
    errors = 0
    for class_dir in class_dirs(in_root):
        src_dir = find_source_dir_for_class(class_dir, source_subdir_candidates)
        if src_dir is None:
            continue
        out_dir = out_root / class_dir.name / output_subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        for source_path in tqdm(sorted(path for path in src_dir.iterdir() if path.is_file() and path.suffix.lower() == ".csv"), desc=f"class {class_dir.name}", leave=False):
            npy_path = out_dir / f"{source_path.stem}.npy"
            if npy_path.exists() and not overwrite:
                skipped += 1
                continue
            try:
                array = load_source_array(source_path, acc_columns, channels, csv_sep_candidates)
                np.save(npy_path, np.asarray(array if end_t is None else array[: int(end_t), :], dtype=np.float32))
                converted += 1
            except Exception as exc:
                errors += 1
                print(f"[ERROR] {source_path}: {exc}")
    return {"converted": converted, "skipped": skipped, "errors": errors}
