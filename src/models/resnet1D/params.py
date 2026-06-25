from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass(frozen=True)
class ResNet1DParams:
    n_feature_maps: int = 64

    @classmethod
    def from_config(cls, model_config: dict) -> "ResNet1DParams":
        names = {field.name for field in fields(cls)}
        unknown = sorted(set(model_config) - names - {"name"})
        if unknown:
            raise ValueError(f"Unknown resnet1D params: {unknown}")
        params = cls(**{name: model_config[name] for name in names if name in model_config})
        return cls(n_feature_maps=int(params.n_feature_maps))
