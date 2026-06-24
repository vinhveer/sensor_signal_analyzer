"""Dataset root discovery for both pre-converted .npy and raw CSV layouts."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .conversion import convert_source_to_npy


def class_dirs(base: Path) -> list[Path]:
    if not base.is_dir():
        return []
    return sorted(path for path in base.iterdir() if path.is_dir())


def find_source_dir_for_class(class_dir: Path, subdir_candidates: list[str]) -> Optional[Path]:
    for subdir in subdir_candidates:
        candidate = class_dir if subdir == "." else class_dir / subdir
        if candidate.is_dir() and any(path.suffix.lower() == ".csv" for path in candidate.iterdir() if path.is_file()):
            return candidate
    return None


def detect_npy_dataset_root(candidates: list[str], subdir: str) -> str:
    subdir_path = Path(subdir)
    for candidate in candidates:
        root = Path(candidate)
        dirs = class_dirs(root)
        if dirs and any((class_dir / subdir_path).is_dir() and any((class_dir / subdir_path).glob("*.npy")) for class_dir in dirs):
            return str(root)
    raise FileNotFoundError("Cannot find dataset root containing .npy files.")


def detect_source_root(candidates: list[str], subdir_candidates: list[str]) -> str:
    for candidate in candidates:
        root = Path(candidate)
        dirs = class_dirs(root)
        if dirs and any(find_source_dir_for_class(class_dir, subdir_candidates) is not None for class_dir in dirs):
            return str(root)
    raise FileNotFoundError("Cannot find dataset root containing .csv files.")


def prepare_dataset_root(data_config: dict) -> str:
    working_root = data_config["kaggle_working_dataset_root"]
    input_candidates = list(data_config["kaggle_input_root_candidates"])
    subdir = data_config["kaggle_dataset_subdir"]
    source_subdir_candidates = data_config["source_subdir_candidates"]
    candidates = [working_root] + input_candidates
    try:
        dataset_root = detect_npy_dataset_root(candidates, subdir)
        print(f"Using existing npy dataset: {dataset_root}")
        return dataset_root
    except FileNotFoundError:
        source_root = detect_source_root(input_candidates, source_subdir_candidates)
        print(f"Found raw source dataset: {source_root}")
        summary = convert_source_to_npy(
            input_root=source_root,
            output_root=working_root,
            output_subdir=subdir,
            channels=int(data_config["expected_channels"]),
            end_t=data_config.get("end_t"),
            overwrite=bool(data_config.get("convert_overwrite", False)),
            acc_columns=data_config["acc_columns"],
            source_subdir_candidates=source_subdir_candidates,
            csv_sep_candidates=data_config["csv_sep_candidates"],
        )
        print("Conversion summary:", summary)
        dataset_root = detect_npy_dataset_root([working_root], subdir)
        print(f"Converted dataset ready at: {dataset_root}")
        return dataset_root
