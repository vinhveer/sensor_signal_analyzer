"""Zip archiving of per-seed run outputs and stats file."""
from __future__ import annotations

import zipfile
from pathlib import Path


def zip_run_outputs(output_root: str, dataset_name: str, model_name: str, seeds: list[int], stats_path: str) -> str:
    seed_label = "_".join(str(seed) for seed in seeds)
    zip_path = str(Path(output_root) / f"{dataset_name}_{model_name}_seeds{seed_label}.zip")
    root = Path(output_root)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(stats_path, Path(stats_path).relative_to(root))
        for seed in seeds:
            seed_dir = root / f"seed{seed}"
            for path in seed_dir.rglob("*"):
                if path.is_file():
                    archive.write(path, path.relative_to(root))
    return zip_path
