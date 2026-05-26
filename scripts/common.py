# diffusers_probe の script 間で共有するユーティリティ。
#
# - project_root(), load_config(), get_common(), get_model_config()
# - load_prompt_sets(), get_prompt_set()
# - load_model_sets(), get_model_set()
# - ensure_dir(), resolve_outputs_dir(), resolve_legacy_outputs_dir(),
#   resolve_quickgen_outputs_dir(), resolve_logs_dir()
# - parse_dtype(), pick_device_and_dtype()
# - apply_vae_fp32_override()
# - build_summary_base() : 03-07 で共通の summary JSON 形式
# - write_outputs() : png / summary json / txt を統一フォーマットで保存

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from typing import Any

import torch


# ---------- パス / config ----------

def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config() -> dict[str, Any]:
    cfg_path = Path(__file__).parent / "diffusers_probe.json"
    with cfg_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_common(cfg: dict[str, Any]) -> dict[str, Any]:
    return cfg.get("common", {})


def get_model_config(cfg: dict[str, Any], model_key: str | None = None) -> dict[str, Any]:
    if model_key is None:
        model_key = cfg.get("default_model")
        if not model_key:
            raise KeyError("default_model not set in config")
    try:
        return cfg["models"][model_key]
    except KeyError as e:
        raise KeyError(f"models.{model_key} not found in config") from e


def load_model_sets() -> dict[str, Any]:
    # 10_quickgen.py が使う model entry 集。スキーマは model_sets.json 冒頭の _doc 参照。
    path = Path(__file__).parent / "model_sets.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_model_set(model_sets_cfg: dict[str, Any], key: str | None = None) -> tuple[str, dict[str, Any]]:
    sets = model_sets_cfg.get("model_sets", {})
    if key is None:
        key = model_sets_cfg.get("default_model")
        if not key:
            raise KeyError("default_model not set in model_sets.json")
    try:
        return key, sets[key]
    except KeyError as e:
        available = ", ".join(sorted(sets.keys())) or "(none)"
        raise KeyError(f"model_sets.{key} not found. available: {available}") from e


def load_prompt_sets() -> dict[str, Any]:
    # 10_quickgen.py が使う prompt set 一覧。スキーマは prompt_sets.json 冒頭の _doc 参照。
    path = Path(__file__).parent / "prompt_sets.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_prompt_set(prompt_sets_cfg: dict[str, Any], key: str | None = None) -> tuple[str, dict[str, Any]]:
    sets = prompt_sets_cfg.get("prompt_sets", {})
    if key is None:
        key = prompt_sets_cfg.get("default_prompt_set")
        if not key:
            raise KeyError("default_prompt_set not set in prompt_sets.json")
    try:
        return key, sets[key]
    except KeyError as e:
        available = ", ".join(sorted(sets.keys())) or "(none)"
        raise KeyError(f"prompt_sets.{key} not found. available: {available}") from e


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_outputs_dir() -> Path:
    return ensure_dir(project_root() / "outputs")


def resolve_legacy_outputs_dir() -> Path:
    # 01-07 (single-shot 流) は outputs/00-07_legacy/ 配下にまとめる。
    return ensure_dir(project_root() / "outputs" / "00-07_legacy")


def resolve_quickgen_outputs_dir(run_label: str) -> Path:
    # 10_quickgen の run ごとの出力先。outputs/10_quickgen/<run_label>/。
    return ensure_dir(project_root() / "outputs" / "10_quickgen" / run_label)


def resolve_logs_dir() -> Path:
    return ensure_dir(project_root() / "logs")


# ---------- device / dtype ----------

_DTYPE_MAP: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "float32": torch.float32,
    "fp32": torch.float32,
    "full": torch.float32,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def parse_dtype(name: str | None, default: torch.dtype) -> torch.dtype:
    if name is None:
        return default
    return _DTYPE_MAP.get(str(name).lower(), default)


def pick_device_and_dtype(model_cfg: dict[str, Any]) -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        return "cuda", parse_dtype(model_cfg.get("cuda_dtype"), torch.float16)
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps", parse_dtype(model_cfg.get("mps_dtype"), torch.float16)
    return "cpu", parse_dtype(model_cfg.get("cpu_dtype"), torch.float32)


def apply_vae_fp32_override(pipe) -> None:
    """VAE のみ fp32 化し、decode 直前で latent を fp32 にキャストする hook を入れる。
    AUTOMATIC1111 の --no-half-vae 相当。SDXL の fp16 で VAE が黒画像になる対策。
    """
    pipe.vae = pipe.vae.to(dtype=torch.float32)
    original_decode = pipe.vae.decode

    def patched_decode(z, *args, **kwargs):
        z = z.to(dtype=torch.float32)
        return original_decode(z, *args, **kwargs)

    pipe.vae.decode = patched_decode  # type: ignore[method-assign]


# ---------- summary 共通フォーマット ----------

def _versions() -> dict[str, str]:
    out: dict[str, str] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
    }
    try:
        import diffusers  # type: ignore

        out["diffusers"] = diffusers.__version__
    except Exception:
        out["diffusers"] = "unknown"
    try:
        import transformers  # type: ignore

        out["transformers"] = transformers.__version__
    except Exception:
        out["transformers"] = "unknown"
    return out


def build_summary_base(
    *,
    script: str,
    cfg: dict[str, Any],
    model_key: str,
    model_id: str,
    device: str,
    dtype: torch.dtype,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int,
    load_time_sec: float,
    generation_time_sec: float,
    total_time_sec: float,
    image_relpath: str,
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "script": script,
        "workspace_name": cfg.get("workspace_name", "diffusers_probe"),
        "model_key": model_key,
        "model_id": model_id,
        "device": device,
        "dtype": str(dtype),
        **_versions(),
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "seed": seed,
        "load_time_sec": round(load_time_sec, 3),
        "generation_time_sec": round(generation_time_sec, 3),
        "total_time_sec": round(total_time_sec, 3),
        "image_path": image_relpath,
    }
    if extras:
        summary.update(extras)
    return summary


def write_outputs(
    *,
    image,
    summary: dict[str, Any],
    output_basename: str,
    prompt: str,
    negative_prompt: str,
) -> tuple[Path, Path, Path]:
    """outputs/00-07_legacy/<basename>.{png,_summary.json,.txt} を一括保存して 3 つのパスを返す。"""
    outputs_dir = resolve_legacy_outputs_dir()
    png_path = outputs_dir / f"{output_basename}.png"
    json_path = outputs_dir / f"{output_basename}_summary.json"
    txt_path = outputs_dir / f"{output_basename}.txt"

    image.save(png_path)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    txt_lines = [
        f"script: {summary.get('script', '')}",
        f"model_id: {summary.get('model_id', '')}",
        f"prompt: {prompt}",
        f"negative_prompt: {negative_prompt}",
        f"image_path: {png_path}",
        f"summary_path: {json_path}",
    ]
    with txt_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(txt_lines) + "\n")

    return png_path, json_path, txt_path


# ---------- エラーメッセージ補助 ----------

def hint_for_load_error(exc: BaseException, model_id: str) -> list[str]:
    """model load の例外から、ユーザーに見せる短い hint を組み立てる。token は表示しない。"""
    name = type(exc).__name__
    msg = str(exc)
    hints: list[str] = []
    lower = msg.lower()
    if (
        "401" in msg
        or "403" in msg
        or "gated" in lower
        or "access" in lower
        or "unauthorized" in lower
        or "restricted" in lower
        or "you are not authorized" in lower
    ):
        hints.append(f"[hint] {model_id} が gated repo / access denied の可能性があります。")
        hints.append(f"       1) https://huggingface.co/{model_id} で利用条件を承認してください")
        hints.append( "       2) 端末で `hf auth login` を実行してください (token は表示・保存しないでください)")
    if "ConnectionError" in name or "Timeout" in name or "Resolve" in msg or "Could not reach" in msg:
        hints.append("[hint] ネットワーク接続に問題がある可能性があります。")
    if "out of memory" in lower or "oom" in lower:
        hints.append("[hint] memory 不足の可能性があります。fallback (低解像度・少ステップ) を試してください。")
    return hints
