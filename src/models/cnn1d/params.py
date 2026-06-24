from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass(frozen=True)
class Conv1DParams:
    fc_dim: int = 128
    dropout: float = 0.0

    @classmethod
    def from_config(cls, model_config: dict) -> "Conv1DParams":
        names = {field.name for field in fields(cls)}
        unknown = sorted(set(model_config) - names - {"name"})
        if unknown:
            raise ValueError(f"Unknown cnn1d params: {unknown}")
        params = cls(**{name: model_config[name] for name in names if name in model_config})
        return cls(fc_dim=int(params.fc_dim), dropout=float(params.dropout))
