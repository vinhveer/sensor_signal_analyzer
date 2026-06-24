from __future__ import annotations


def add_cli_args(parser) -> None:
    group = parser.add_argument_group("cnn1d model options")
    group.add_argument("--fc-dim", type=int, help="cnn1d fully-connected hidden dimension.")
    group.add_argument("--dropout", "--drop-out", dest="dropout", type=float, help="cnn1d dropout rate.")


def apply_cli_overrides(model_config: dict, args) -> None:
    if args.fc_dim is not None:
        model_config["fc_dim"] = args.fc_dim
    if args.dropout is not None:
        model_config["dropout"] = args.dropout
