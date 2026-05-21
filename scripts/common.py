# diffusers_probe の script 間で共有するユーティリティ。
# project_root()        : workspace ルートを返す
# load_config()         : scripts/diffusers_probe.json を読み込む
# get_common(cfg)       : common セクションを返す
# get_model_config(cfg, key) : models.{key} の per-model 設定を返す
# ensure_dir(path)      : 指定パスのディレクトリを作成して返す
# resolve_outputs_dir() : <workspace>/outputs を返す（なければ作成）
# resolve_logs_dir()    : <workspace>/logs を返す（なければ作成）

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config() -> dict[str, Any]:
    cfg_path = Path(__file__).parent / "diffusers_probe.json"
    with cfg_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_common(cfg: dict[str, Any]) -> dict[str, Any]:
    """共通設定 (prompt / negative_prompt / seed 等) を返す。"""
    return cfg.get("common", {})


def get_model_config(cfg: dict[str, Any], model_key: str | None = None) -> dict[str, Any]:
    """models.{key} の per-model 設定を返す。key 省略時は cfg["default_model"] を使う。"""
    if model_key is None:
        model_key = cfg.get("default_model")
        if not model_key:
            raise KeyError("default_model not set in config")
    try:
        return cfg["models"][model_key]
    except KeyError as e:
        raise KeyError(f"models.{model_key} not found in config") from e


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_outputs_dir() -> Path:
    return ensure_dir(project_root() / "outputs")


def resolve_logs_dir() -> Path:
    return ensure_dir(project_root() / "logs")
