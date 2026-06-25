from __future__ import annotations


def add_cli_args(parser) -> None:
    group = parser.add_argument_group("resnet1D model options")
    group.add_argument("--n-feature-maps", type=int, help="resnet1D base feature map count.")


def apply_cli_overrides(model_config: dict, args) -> None:
    if args.n_feature_maps is not None:
        model_config["n_feature_maps"] = args.n_feature_maps
