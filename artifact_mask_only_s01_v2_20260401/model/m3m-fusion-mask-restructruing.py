import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Config yaml must be a mapping/dictionary")
    return data


def _build_cmd(base_script: Path, cfg: dict, cli: argparse.Namespace) -> list[str]:
    # Fixed architecture: fusion-style mask-only reconstruction pipeline.
    cmd = [
        sys.executable,
        str(base_script),
        "--prediction_mode",
        "encoder_mask_only",
    ]

    def add_opt(name: str, value):
        if value is None:
            return
        cmd.extend([f"--{name}", str(value)])

    # Priority: CLI > yaml
    data_path = cli.data if cli.data else cfg.get("data")
    outdir = cli.outdir if cli.outdir else cfg.get("outdir")

    add_opt("data", data_path)
    add_opt("outdir", outdir)
    add_opt("selected_channels_1based", cfg.get("selected_channels_1based"))
    add_opt("seed", cli.seed if cli.seed is not None else cfg.get("seed"))
    add_opt("device", cfg.get("device", "auto"))

    add_opt("epochs", cli.epochs if cli.epochs is not None else cfg.get("epochs"))
    add_opt("batch_size", cli.batch_size if cli.batch_size is not None else cfg.get("batch_size"))
    add_opt("lr", cli.lr if cli.lr is not None else cfg.get("lr"))
    add_opt("weight_decay", cfg.get("weight_decay"))
    add_opt("patience", cfg.get("patience"))

    add_opt("d_model", cfg.get("d_model"))
    add_opt("d_state", cfg.get("d_state"))
    add_opt("headdim", cfg.get("headdim"))
    add_opt("n_bi_layers", cfg.get("n_bi_layers"))
    add_opt("chunk_size", cfg.get("chunk_size"))
    add_opt("patch_size", cfg.get("patch_size"))
    add_opt("dropout", cfg.get("dropout"))
    add_opt("preconv_kernel", cfg.get("preconv_kernel"))
    add_opt("disable_preconv", cfg.get("disable_preconv", "true"))

    add_opt("encoder_random_mask_ratio", cfg.get("encoder_random_mask_ratio", 0.15))
    add_opt("encoder_eval_mask_ratio", cfg.get("encoder_eval_mask_ratio", 0.15))
    add_opt("mask_observed_residual", cfg.get("mask_observed_residual", "true"))
    add_opt("input_ratio", cfg.get("input_ratio", 0.85))

    return cmd


def main():
    parser = argparse.ArgumentParser(
        description="Minimal runner for m3m-fusion-mask-restructruing (single main model, no feature switches)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="artifact_mask_only_s01_v2_20260401/config/m3m-fusion-mask-restructruing.yaml",
    )
    parser.add_argument("--data", type=str, default="")
    parser.add_argument("--outdir", type=str, default="")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    base_script = repo_root / "artifact_mask_only_s01_v2_20260401/model/deap_mamba3_multimodal_decoder.py"

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = repo_root / cfg_path
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    cfg = _load_yaml(cfg_path)
    cmd = _build_cmd(base_script, cfg, args)

    print("[m3m-fusion-mask-restructruing] launching:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
