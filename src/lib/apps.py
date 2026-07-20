"""Abstract command-line apps shared by model packages."""
from __future__ import annotations

import argparse
import json
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch

from .config.loader import default_config, finalize_config
from .data.preprocessing import preprocess_window, starts_for_length
from .engine.train import run_training
from .visualization.inference import plot_inference


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


class TrainApp(ABC):
    model_name: str

    @abstractmethod
    def add_model_args(self, parser: argparse.ArgumentParser) -> None:
        pass

    @abstractmethod
    def build_model(self, in_channels: int, num_classes: int, window: int, config: dict):
        pass

    @abstractmethod
    def model_config(self, args: argparse.Namespace) -> dict:
        pass

    def parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description=f"Train {self.model_name}.")
        parser.add_argument("--dataset-root", required=True)
        parser.add_argument("--output-root", default="History")
        parser.add_argument("--window", type=int, default=1024)
        parser.add_argument("--overlap", type=float, default=0.75)
        parser.add_argument("--train-run-ids", default="1,2,3,4")
        parser.add_argument("--val-run-ids", default="5")
        parser.add_argument("--test-run-ids", default="6")
        parser.add_argument("--expected-files-per-class", type=int, default=18)
        parser.add_argument("--seeds", default="42")
        parser.add_argument("--batch-size", type=int, default=32)
        parser.add_argument("--epochs", type=int, default=100)
        parser.add_argument("--learning-rate", type=float, default=1e-3)
        parser.add_argument("--weight-decay", type=float, default=1e-3)
        parser.add_argument("--num-workers", type=int, default=0)
        parser.add_argument("--grad-clip-norm", type=float, default=1.0)
        parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
        parser.add_argument("--no-save-zip", action="store_true")
        self.add_model_args(parser)
        return parser

    def run(self, argv: list[str] | None = None) -> dict:
        args = self.parser().parse_args(argv)
        config = default_config()
        config["model_name"] = self.model_name
        config["model"] = {"name": self.model_name, **self.model_config(args)}
        config["data"]["kaggle_working_dataset_root"] = args.dataset_root
        config["data"]["kaggle_input_root_candidates"] = []
        config["outputs"] = {"root": args.output_root, "save_zip": not args.no_save_zip}
        config["windowing"].update(window=args.window, overlap=args.overlap, step_size=None)
        config["split"].update(
            train_run_ids=parse_int_list(args.train_run_ids),
            val_run_ids=parse_int_list(args.val_run_ids),
            test_run_ids=parse_int_list(args.test_run_ids),
            expected_files_per_class=args.expected_files_per_class,
        )
        config["training"].update(
            seeds=parse_int_list(args.seeds),
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            num_workers=args.num_workers,
            grad_clip_norm=args.grad_clip_norm,
            use_amp=args.amp,
        )
        finalize_config(config)
        return run_training(config, self.build_model)


class InferenceApp(ABC):
    @abstractmethod
    def build_model(self, in_channels: int, num_classes: int, window: int, config: dict):
        pass

    def parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="Run inference on a sensor signal.")
        parser.add_argument("--checkpoint", required=True)
        parser.add_argument("--input", required=True, help="A two-channel .npy signal.")
        parser.add_argument("--batch-size", type=int, default=32)
        return parser

    def predict(self, checkpoint: str, input_path: str, batch_size: int = 32) -> dict:
        payload = torch.load(checkpoint, map_location="cpu")
        data_config = payload["config"]["data"]
        model = self.build_model(
            int(data_config["channels"]),
            len(payload["classes"]),
            int(data_config["window"]),
            payload["config"]["model"],
        )
        model.load_state_dict(payload["model_state_dict"])
        model.eval()

        signal = np.load(input_path)
        window = int(data_config["window"])
        starts = starts_for_length(len(signal), window, int(data_config["step"]))
        if not starts:
            raise ValueError(f"Input must contain at least {window} samples")
        batches = np.stack([preprocess_window(signal[start:start + window], 1e-8).T for start in starts])
        probabilities = []
        with torch.no_grad():
            for start in range(0, len(batches), batch_size):
                logits = model(torch.from_numpy(batches[start:start + batch_size]).float())
                probabilities.append(torch.softmax(logits, dim=1))
        window_probabilities = torch.cat(probabilities)
        mean_probabilities = window_probabilities.mean(dim=0)
        index = int(mean_probabilities.argmax())
        return {
            "class": payload["classes"][index],
            "probabilities": dict(zip(payload["classes"], mean_probabilities.tolist())),
            "windows": len(starts),
            "classes": payload["classes"],
            "window_starts": starts,
            "window_probabilities": window_probabilities.tolist(),
            "window_predictions": [payload["classes"][int(value)] for value in window_probabilities.argmax(dim=1)],
        }

    def run(self, argv: list[str] | None = None) -> dict:
        args = self.parser().parse_args(argv)
        result = self.predict(args.checkpoint, args.input, args.batch_size)
        public_result = {key: value for key, value in result.items() if not key.startswith("window_") and key != "classes"}
        print(json.dumps(public_result, indent=2))
        return public_result


class VisualizationApp(InferenceApp):
    def parser(self) -> argparse.ArgumentParser:
        parser = super().parser()
        parser.description = "Visualize inference on a sensor signal."
        parser.add_argument("--output-dir", required=True)
        return parser

    def run(self, argv: list[str] | None = None) -> dict:
        args = self.parser().parse_args(argv)
        result = self.predict(args.checkpoint, args.input, args.batch_size)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(args.input).stem
        json_path = output_dir / f"inference_{stem}.json"
        figure_path = output_dir / f"inference_{stem}.png"
        signal = np.load(args.input)
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        plot_inference(signal, result, str(figure_path))
        output = {
            "class": result["class"],
            "probabilities": result["probabilities"],
            "windows": result["windows"],
            "json_path": str(json_path),
            "figure_path": str(figure_path),
        }
        print(json.dumps(output, indent=2))
        return output
