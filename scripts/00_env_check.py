# 00_env_check.py
#
# Diffusers probe 用 dev venv (~/.venvs/dfs2026-dev) の環境チェック。
# Python / 主要 package / device (MPS/CUDA) / workspace のパスを表示する。
# import に失敗した optional package は "not installed" と表示するだけで、script は止まらない。

from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path

from common import (
    load_config,
    project_root,
    resolve_logs_dir,
    resolve_outputs_dir,
)


def _pkg_version(name: str) -> str:
    try:
        mod = importlib.import_module(name)
    except Exception as e:
        return f"not installed ({type(e).__name__}: {e})"
    return getattr(mod, "__version__", "unknown")


def main() -> None:
    print("=== diffusers_probe / env check ===")
    print()

    print("[python]")
    print(f"  version  : {sys.version.split()[0]}")
    print(f"  platform : {platform.platform()}")
    print(f"  machine  : {platform.machine()}")
    print(f"  executable: {sys.executable}")
    print()

    print("[packages]")
    for name in [
        "torch",
        "diffusers",
        "transformers",
        "accelerate",
        "huggingface_hub",
        "safetensors",
        "PIL",
        "matplotlib",
        "pandas",
    ]:
        print(f"  {name:16s}: {_pkg_version(name)}")
    print()

    print("[torch device]")
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        mps_built = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_built())
        mps_available = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())

        print(f"  cuda available: {cuda_available}")
        if cuda_available:
            print(f"  cuda version  : {torch.version.cuda}")
            print(f"  cuda device  : {torch.cuda.get_device_name(0)}")
        print(f"  mps built     : {mps_built}")
        print(f"  mps available : {mps_available}")
    except Exception as e:
        print(f"  (torch error: {type(e).__name__}: {e})")
    print()

    print("[workspace]")
    root = project_root()
    print(f"  workspace path: {root}")
    print(f"  outputs_dir   : {resolve_outputs_dir()}")
    print(f"  logs_dir      : {resolve_logs_dir()}")
    print()

    print("[config]")
    cfg = load_config()
    print(f"  model_id     : {cfg.get('default_model_id')}")
    print(f"  prompt       : {cfg.get('default_prompt')}")
    print(f"  size         : {cfg.get('width')} x {cfg.get('height')}")
    print(f"  steps        : {cfg.get('num_inference_steps')}")
    print(f"  guidance_scale: {cfg.get('guidance_scale')}")
    print(f"  seed         : {cfg.get('seed')}")
    print()

    print("[venv]")
    venv_path = Path(sys.executable).resolve()
    print(f"  python exec : {venv_path}")
    print(f"  venv root   : {venv_path.parent.parent}")
    print()

    print("=== done ===")


if __name__ == "__main__":
    main()
