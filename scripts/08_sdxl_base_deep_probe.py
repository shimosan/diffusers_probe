# 08_sdxl_base_deep_probe.py
#
# SDXL Base 1.0 の深掘り調査 probe script。講義 notebook 化の前段階。
# 第 3 ラウンド改修: 3 年前 SD1.5 講義スライドと同等の cross-attention 可視化を
# SDXL Base で試作する。
#
# 3 本柱:
#   Phase 1: prompt embedding geometry (CLIP-L + OpenCLIP-G, PCA + t-SNE)
#   Phase 2: denoising trajectory (手動 scheduler ループ)
#   Phase 3: cross-attention probe — U-Net 構造順 inventory / token x module grid
# plus:
#   Phase 0: preflight / module inventory
#   Phase 4: guidance / negative prompt grid
#   Phase 5: summary.md / summary.json (画像埋め込み + 各 subdir index.md)
#
# 出力: outputs/08_sdxl_base_deep_probe/<timestamp>/
# runner: tmp/run_08_sdxl_base_deep_probe.sh
#
# 注意:
#   - MPS では fp32 固定。fp16/bf16 高速化は試さない。
#   - SDXL は SD1.5 と U-Net 構造が異なる。空間解像度は module 名でなく query_len
#     (sqrt が整数) から推定する。SDXL Base の cross-attention 解像度は基本 2 種類:
#     1024 生成時は {64, 32}、512 生成時は {32, 16}。
#   - step index 方向: i=0 が最ノイズ、i=N-1 が最終 (DDPM 論文の t=T → t=0 と逆順)。
#   - attention capture は失敗しても他 phase を続行。summary.md は必ず生成。

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from common import (
    apply_vae_fp32_override,
    ensure_dir,
    get_common,
    get_model_config,
    hint_for_load_error,
    load_config,
    pick_device_and_dtype,
    project_root,
)

MODEL_KEY = "sdxl_base"
SCRIPT_NAME = "08_sdxl_base_deep_probe.py"
OUTPUT_SUBDIR = "08_sdxl_base_deep_probe"

DEFAULT_PROMPT_BANK = [
    "a small robot in a classroom",
    "a small robot in a laboratory",
    "a robot reading a book",
    "a robot writing on a blackboard",
    "a cat reading a book",
    "a dog reading a book",
    "a university classroom",
    "a futuristic city",
    "a watercolor painting of a robot",
    "an anime style robot",
    "a realistic photo of a robot",
    "a simple illustration of a robot",
]

# 3 年前スライド対応 target tokens (default prompt 中の主要単語)
ATTN_TARGET_TOKENS = [
    "robot", "studying", "artificial", "intelligence",
    "university", "classroom", "illustration",
]
ATTN_TARGET_PHRASES = [
    ("artificial_intelligence", ["artificial", "intelligence"]),
    ("university_classroom", ["university", "classroom"]),
]
ATTN_TARGET_STEP_FRACTIONS = [0.17, 0.5, 0.83]

GUIDANCE_VALUES = [0.0, 3.0, 7.5, 12.0]

# stage 並び順 (legacy grid の列順)
STAGE_ORDER = {"down": 0, "mid": 1, "up": 2}


# ---------------------------------------------------------------------------
# 全体状態
# ---------------------------------------------------------------------------

@dataclass
class PhaseStatus:
    name: str
    ok: bool = False
    skipped: bool = False
    error: str | None = None
    notes: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "ok": self.ok, "skipped": self.skipped,
            "error": self.error, "notes": self.notes, "files": self.files,
            "extras": self.extras,
        }


@dataclass
class RunContext:
    args: argparse.Namespace
    run_dir: Path
    cfg: dict[str, Any]
    model_cfg: dict[str, Any]
    device: str
    dtype: torch.dtype
    model_id: str
    width: int
    height: int
    num_inference_steps: int
    guidance_scale: float
    prompt: str
    negative_prompt: str
    seed: int
    started_at: str
    phases: list[PhaseStatus] = field(default_factory=list)

    def phase(self, name: str) -> PhaseStatus:
        st = PhaseStatus(name=name)
        self.phases.append(st)
        return st

    def get_phase(self, name: str) -> PhaseStatus | None:
        for p in self.phases:
            if p.name == name:
                return p
        return None


# ---------------------------------------------------------------------------
# 補助
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    print(msg, flush=True)


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.dtype):
        return str(obj)
    if isinstance(obj, Path):
        return str(obj)
    return repr(obj)


def write_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=_json_default)


def write_text(path: Path, content: str) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        f.write(content)


def tensor_stats(t: torch.Tensor) -> dict[str, float]:
    t_f = t.detach().to(torch.float32).cpu()
    return {
        "mean": float(t_f.mean().item()),
        "std": float(t_f.std().item()),
        "min": float(t_f.min().item()),
        "max": float(t_f.max().item()),
        "norm": float(t_f.norm().item()),
    }


@contextmanager
def phase_guard(status: PhaseStatus, *, label: str | None = None):
    label = label or status.name
    log(f"\n=== [{label}] start ===")
    t0 = time.perf_counter()
    try:
        yield
        status.ok = True
        log(f"=== [{label}] ok ({time.perf_counter() - t0:.2f} s) ===")
    except Exception as e:
        status.ok = False
        status.error = f"{type(e).__name__}: {e}"
        tb = traceback.format_exc()
        log(f"=== [{label}] FAILED: {status.error} ===")
        log(tb)
        status.notes.append("traceback saved to FAILED.txt under phase dir")
    finally:
        status.extras["elapsed_sec"] = round(time.perf_counter() - t0, 3)


def save_failure(status: PhaseStatus, phase_dir: Path) -> None:
    if status.ok or status.skipped:
        return
    ensure_dir(phase_dir)
    fpath = phase_dir / "FAILED.txt"
    with fpath.open("w", encoding="utf-8") as f:
        f.write(f"phase: {status.name}\n")
        f.write(f"error: {status.error}\n\n")
        f.write(traceback.format_exc())


def _get_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def md_img(rel: str, alt: str = "") -> str:
    """画像埋め込み markdown。"""
    return f"![{alt or rel}]({rel})"


# ---------------------------------------------------------------------------
# module info parser
# ---------------------------------------------------------------------------

def parse_module_info(name: str) -> dict[str, Any]:
    """attention module 名から stage / 各 index を抽出。

    例:
      "down_blocks.1.attentions.0.transformer_blocks.0.attn2"
        → stage=down, stage_index=1, attn_block_index=0,
          transformer_index=0, attention_index="attn2"
    """
    info: dict[str, Any] = {
        "name": name, "stage": "unknown", "stage_index": None,
        "attn_block_index": None, "transformer_index": None,
        "attention_index": None,
    }
    parts = name.split(".")
    if not parts:
        return info
    if parts[0] == "down_blocks" and len(parts) >= 2:
        info["stage"] = "down"
        try:
            info["stage_index"] = int(parts[1])
        except ValueError:
            pass
    elif parts[0] == "mid_block":
        info["stage"] = "mid"
        info["stage_index"] = 0
    elif parts[0] == "up_blocks" and len(parts) >= 2:
        info["stage"] = "up"
        try:
            info["stage_index"] = int(parts[1])
        except ValueError:
            pass
    if "attentions" in parts:
        i = parts.index("attentions")
        if i + 1 < len(parts):
            try:
                info["attn_block_index"] = int(parts[i + 1])
            except ValueError:
                pass
    if "transformer_blocks" in parts:
        i = parts.index("transformer_blocks")
        if i + 1 < len(parts):
            try:
                info["transformer_index"] = int(parts[i + 1])
            except ValueError:
                pass
    for p in parts:
        if p in ("attn1", "attn2"):
            info["attention_index"] = p
    return info


def module_sort_key(info: dict[str, Any]) -> tuple:
    """U-Net traversal order (down → mid → up, 同 stage 内は index 昇順)。"""
    return (
        STAGE_ORDER.get(info.get("stage", "unknown"), 99),
        info.get("stage_index") if info.get("stage_index") is not None else 99,
        info.get("attn_block_index") if info.get("attn_block_index") is not None else 99,
        info.get("transformer_index") if info.get("transformer_index") is not None else 99,
    )


def module_short_name(info: dict[str, Any]) -> str:
    """グラフラベル用の短い module 名。"""
    stage = info["stage"]
    if stage == "mid":
        return f"mid.t{info['transformer_index']}"
    return f"{stage}{info['stage_index']}.a{info['attn_block_index']}.t{info['transformer_index']}"


# ---------------------------------------------------------------------------
# SDXL 手動デノイズヘルパ
# ---------------------------------------------------------------------------

def _encode_for_manual(pipe, ctx: RunContext, prompt: str, negative_prompt: str,
                       do_cfg: bool) -> dict[str, Any]:
    with torch.no_grad():
        pe, npe, pooled, npooled = pipe.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            device=ctx.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=do_cfg,
        )
    add_time_ids = torch.tensor(
        [[ctx.height, ctx.width, 0, 0, ctx.height, ctx.width]],
        dtype=pe.dtype, device=ctx.device,
    )
    if do_cfg:
        pe_in = torch.cat([npe, pe], dim=0)
        pooled_in = torch.cat([npooled, pooled], dim=0)
        time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)
    else:
        pe_in, pooled_in, time_ids = pe, pooled, add_time_ids
    return {
        "prompt_embeds": pe_in,
        "added_cond_kwargs": {"text_embeds": pooled_in, "time_ids": time_ids},
    }


def _decode_latents_to_pil(pipe, latents: torch.Tensor) -> Image.Image:
    vae = pipe.vae
    with torch.no_grad():
        z = latents.to(dtype=next(vae.parameters()).dtype) / vae.config.scaling_factor
        img = vae.decode(z, return_dict=False)[0]
    img = (img / 2 + 0.5).clamp(0, 1)
    arr = (img[0].permute(1, 2, 0).float().cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def _select_trajectory_steps(num_steps: int) -> list[int]:
    if num_steps >= 30:
        cand = [0, 1, 3, 5, 10, 15, 20, 25, num_steps - 1]
    elif num_steps >= 10:
        cand = [0, 1, num_steps // 4, num_steps // 2,
                (3 * num_steps) // 4, num_steps - 1]
    else:
        cand = list(range(num_steps))
    return sorted(set(c for c in cand if 0 <= c < num_steps))


# ---------------------------------------------------------------------------
# Phase 0: inventory
# ---------------------------------------------------------------------------

def phase0_inventory(ctx: RunContext, pipe) -> PhaseStatus:
    st = ctx.phase("phase0_inventory")
    phase_dir = ctx.run_dir
    with phase_guard(st, label="phase 0: inventory"):
        snap: dict[str, Any] = {
            "started_at": ctx.started_at,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "device": ctx.device,
            "dtype": str(ctx.dtype),
            "model_id": ctx.model_id,
            "prompt": ctx.prompt,
            "negative_prompt": ctx.negative_prompt,
            "seed": ctx.seed,
            "width": ctx.width,
            "height": ctx.height,
            "num_inference_steps": ctx.num_inference_steps,
            "guidance_scale": ctx.guidance_scale,
            "scheduler_class": type(pipe.scheduler).__name__,
            "args": vars(ctx.args),
        }
        for pkg in ("diffusers", "transformers", "accelerate", "safetensors", "scikit-learn"):
            mod_name = pkg.replace("-", "_")
            try:
                mod = __import__(mod_name)
                snap[pkg] = getattr(mod, "__version__", "unknown")
            except Exception:
                snap[pkg] = "not installed"

        def _cfg_summary(obj) -> dict[str, Any]:
            c = getattr(obj, "config", None)
            if c is None:
                return {}
            try:
                return {k: c[k] for k in list(c.keys())[:50]}
            except Exception:
                return {k: getattr(c, k) for k in dir(c) if not k.startswith("_")}[:50]

        snap["unet_config"] = _cfg_summary(pipe.unet)
        snap["vae_config"] = _cfg_summary(pipe.vae)
        snap["tokenizer"] = {
            "class": type(pipe.tokenizer).__name__,
            "vocab_size": getattr(pipe.tokenizer, "vocab_size", None),
            "model_max_length": getattr(pipe.tokenizer, "model_max_length", None),
        }
        snap["tokenizer_2"] = {
            "class": type(pipe.tokenizer_2).__name__,
            "vocab_size": getattr(pipe.tokenizer_2, "vocab_size", None),
            "model_max_length": getattr(pipe.tokenizer_2, "model_max_length", None),
        }
        snap["text_encoder"] = {
            "class": type(pipe.text_encoder).__name__,
            "hidden_size": getattr(pipe.text_encoder.config, "hidden_size", None),
            "num_hidden_layers": getattr(pipe.text_encoder.config, "num_hidden_layers", None),
        }
        snap["text_encoder_2"] = {
            "class": type(pipe.text_encoder_2).__name__,
            "hidden_size": getattr(pipe.text_encoder_2.config, "hidden_size", None),
            "num_hidden_layers": getattr(pipe.text_encoder_2.config, "num_hidden_layers", None),
        }
        write_json(phase_dir / "config_snapshot.json", snap)

        from diffusers.models.attention_processor import Attention
        attn_records: list[dict[str, Any]] = []
        for name, module in pipe.unet.named_modules():
            if isinstance(module, Attention):
                proc = getattr(module, "processor", None)
                is_cross = bool(getattr(module, "is_cross_attention", False))
                info = parse_module_info(name)
                rec = {
                    **info,
                    "class": type(module).__name__,
                    "processor": type(proc).__name__ if proc is not None else None,
                    "is_cross_attention": is_cross,
                    "heads": int(getattr(module, "heads", 0) or 0),
                }
                attn_records.append(rec)
        n_cross = sum(1 for r in attn_records if r["is_cross_attention"])
        n_self = sum(1 for r in attn_records if not r["is_cross_attention"])
        write_json(phase_dir / "module_inventory.json", {
            "total_attention_modules": len(attn_records),
            "cross_attention_modules": n_cross,
            "self_attention_modules": n_self,
            "modules": attn_records,
        })
        lines = [
            "# module inventory (Phase 0)",
            "",
            f"- total Attention modules: {len(attn_records)}",
            f"- cross-attention: {n_cross}",
            f"- self-attention: {n_self}",
            "",
            "詳細な structure / 代表選択は [attention/unet_attention_structure.md](attention/unet_attention_structure.md)",
            "と [attention/selected_attention_modules.md](attention/selected_attention_modules.md) を参照。",
            "",
            "## cross-attention modules (一部抜粋)",
            "",
            "| name | stage | heads | processor |",
            "|---|---|---:|---|",
        ]
        cross_list = [r for r in attn_records if r["is_cross_attention"]]
        cross_list.sort(key=module_sort_key)
        for r in cross_list[:30]:
            lines.append(f"| `{r['name']}` | {r['stage']} | {r['heads']} | {r['processor']} |")
        if len(cross_list) > 30:
            lines.append(f"| ... ({len(cross_list) - 30} more) | | | |")
        write_text(phase_dir / "module_inventory.md", "\n".join(lines) + "\n")

        st.extras["n_cross_attention"] = n_cross
        st.extras["n_self_attention"] = n_self
        st.extras["scheduler_class"] = type(pipe.scheduler).__name__
    return st


# ---------------------------------------------------------------------------
# Phase 1: embeddings
# ---------------------------------------------------------------------------

def phase1_embeddings(ctx: RunContext, pipe) -> PhaseStatus:
    st = ctx.phase("phase1_embeddings")
    phase_dir = ctx.run_dir / "prompt_embeddings"
    ensure_dir(phase_dir)
    with phase_guard(st, label="phase 1: embeddings"):
        prompts = list(DEFAULT_PROMPT_BANK)

        token_rows: list[dict[str, Any]] = []
        for p in prompts:
            ids1 = pipe.tokenizer(p, return_tensors="pt").input_ids[0].tolist()
            toks1 = pipe.tokenizer.convert_ids_to_tokens(ids1)
            ids2 = pipe.tokenizer_2(p, return_tensors="pt").input_ids[0].tolist()
            toks2 = pipe.tokenizer_2.convert_ids_to_tokens(ids2)
            token_rows.append({
                "prompt": p, "tok1_ids": ids1, "tok1_tokens": toks1,
                "tok2_ids": ids2, "tok2_tokens": toks2,
            })
        lines = ["# tokenization (CLIP-L = tokenizer, OpenCLIP-G = tokenizer_2)", ""]
        for r in token_rows:
            lines.append(f"## `{r['prompt']}`")
            lines.append("")
            lines.append(f"- tokenizer (CLIP-L) len={len(r['tok1_ids'])}: `{' '.join(r['tok1_tokens'])}`")
            lines.append(f"- tokenizer_2 (OpenCLIP-G) len={len(r['tok2_ids'])}: `{' '.join(r['tok2_tokens'])}`")
            lines.append("")
        write_text(phase_dir / "prompt_tokens.md", "\n".join(lines))

        pooled_list: list[np.ndarray] = []
        full_mean_list: list[np.ndarray] = []
        shapes_info: list[dict[str, Any]] = []
        for p in prompts:
            with torch.no_grad():
                emb, _, pooled, _ = pipe.encode_prompt(
                    prompt=p, device=ctx.device, num_images_per_prompt=1,
                    do_classifier_free_guidance=False,
                )
            emb_np = emb.detach().to(torch.float32).cpu().numpy()
            pooled_np = pooled.detach().to(torch.float32).cpu().numpy()
            pooled_list.append(pooled_np[0])
            full_mean_list.append(emb_np[0].mean(axis=0))
            shapes_info.append({
                "prompt": p,
                "prompt_embeds_shape": list(emb_np.shape),
                "pooled_prompt_embeds_shape": list(pooled_np.shape),
            })
        write_json(phase_dir / "prompt_embedding_shapes.json", {
            "n_prompts": len(prompts),
            "shapes": shapes_info,
            "_note": "prompt_embeds (B,77,2048) = concat[CLIP-L 768, OpenCLIP-G 1280]. pooled は OpenCLIP-G の pooled (1280).",
        })

        pooled_arr = np.stack(pooled_list, axis=0)
        full_mean_arr = np.stack(full_mean_list, axis=0)

        csv_lines = ["prompt,||pooled||,||token_mean||,cos(pooled,prompt0)"]
        for i, p in enumerate(prompts):
            v = pooled_arr[i]
            v0 = pooled_arr[0]
            cos = float(np.dot(v, v0) / (np.linalg.norm(v) * np.linalg.norm(v0) + 1e-9))
            csv_lines.append(f"\"{p}\",{np.linalg.norm(v):.4f},{np.linalg.norm(full_mean_arr[i]):.4f},{cos:.4f}")
        write_text(phase_dir / "prompt_embeddings.csv", "\n".join(csv_lines) + "\n")

        def _pca_2d(X: np.ndarray) -> np.ndarray:
            Xc = X - X.mean(axis=0, keepdims=True)
            _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
            return Xc @ Vt[:2].T

        plt = _get_plt()

        def _scatter(coords: np.ndarray, title: str, out_path: Path) -> None:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(coords[:, 0], coords[:, 1], s=40)
            for i, p in enumerate(prompts):
                ax.annotate(p, (coords[i, 0], coords[i, 1]), fontsize=7,
                            xytext=(4, 2), textcoords="offset points")
            ax.set_title(title)
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(out_path, dpi=140)
            plt.close(fig)

        _scatter(_pca_2d(pooled_arr),
                 "PCA (pooled embeds, OpenCLIP-G 1280-d)",
                 phase_dir / "prompt_pca.png")
        _scatter(_pca_2d(full_mean_arr),
                 "PCA (token-mean of full prompt_embeds 2048-d)",
                 phase_dir / "prompt_pca_tokenmean.png")

        tsne_ok = False
        tsne_err = None
        try:
            from sklearn.manifold import TSNE
            n = pooled_arr.shape[0]
            perplexity = max(2.0, min(5.0, (n - 1) / 2.0))
            tsne = TSNE(n_components=2, perplexity=perplexity,
                        init="pca", random_state=ctx.seed, max_iter=1000)
            tsne_coords = tsne.fit_transform(pooled_arr)
            _scatter(tsne_coords, f"t-SNE (pooled embeds, perplexity={perplexity:.1f})",
                     phase_dir / "prompt_tsne.png")
            tsne_ok = True
        except Exception as e:
            tsne_err = f"{type(e).__name__}: {e}"
            st.notes.append(f"t-SNE 失敗: {tsne_err}")
            log(f"[phase1] t-SNE failed: {tsne_err}")

        # index.md
        idx_lines = [
            "# prompt_embeddings/",
            "",
            "SDXL の prompt encoding (CLIP-L + OpenCLIP-G) から取り出した埋め込みを",
            f"{len(prompts)} 件の prompt について並べた図と表。",
            "",
            "## PCA (numpy SVD)",
            "",
            md_img("prompt_pca.png", "PCA pooled"),
            "",
            md_img("prompt_pca_tokenmean.png", "PCA token mean"),
            "",
        ]
        if tsne_ok:
            idx_lines += ["## t-SNE (sklearn)", "", md_img("prompt_tsne.png", "t-SNE pooled"), ""]
        else:
            idx_lines += ["## t-SNE", "", f"(失敗または skip: {tsne_err})", ""]
        idx_lines += [
            "## 表", "",
            "- [token table](prompt_tokens.md)",
            "- [embeddings CSV](prompt_embeddings.csv)",
            "- [shapes JSON](prompt_embedding_shapes.json)",
            "",
            "## メモ", "",
            "- pooled = OpenCLIP-G の最終 pooled (1280-d)",
            "- token_mean = 77 token × 2048-d を token 軸で平均",
            "- t-SNE は perplexity を `min(5, (n-1)/2)` で自動調整",
        ]
        write_text(phase_dir / "index.md", "\n".join(idx_lines) + "\n")

        st.extras.update({
            "n_prompts": len(prompts),
            "pooled_dim": int(pooled_arr.shape[1]),
            "prompt_embeds_dim": int(full_mean_arr.shape[1]),
            "tsne_ok": tsne_ok,
        })
    return st


# ---------------------------------------------------------------------------
# Phase 2: trajectory
# ---------------------------------------------------------------------------

def phase2_trajectory(ctx: RunContext, pipe) -> PhaseStatus:
    st = ctx.phase("phase2_trajectory")
    phase_dir = ctx.run_dir / "trajectory"
    ensure_dir(phase_dir)
    with phase_guard(st, label="phase 2: trajectory"):
        do_cfg = ctx.guidance_scale > 1.0
        enc = _encode_for_manual(pipe, ctx, ctx.prompt, ctx.negative_prompt, do_cfg)
        prompt_embeds = enc["prompt_embeds"]
        added_cond = enc["added_cond_kwargs"]

        pipe.scheduler.set_timesteps(ctx.num_inference_steps, device=ctx.device)
        timesteps = pipe.scheduler.timesteps

        gen = torch.Generator(device="cpu" if ctx.device == "mps" else ctx.device).manual_seed(ctx.seed)
        unet = pipe.unet
        latent_dtype = next(unet.parameters()).dtype
        shape = (1, unet.config.in_channels, ctx.height // 8, ctx.width // 8)
        latents = torch.randn(shape, generator=gen, dtype=latent_dtype).to(ctx.device)
        latents = latents * pipe.scheduler.init_noise_sigma

        selected = _select_trajectory_steps(ctx.num_inference_steps)
        st.extras["selected_steps"] = selected
        st.extras["timesteps_first5"] = [float(t) for t in timesteps[:5].tolist()]

        latent_stats_rows: list[dict[str, Any]] = []
        decoded_imgs: dict[int, Image.Image] = {}

        for i, t in enumerate(timesteps):
            latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
            with torch.no_grad():
                noise_pred = unet(
                    latent_model_input, t,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs=added_cond,
                    return_dict=False,
                )[0]
            if do_cfg:
                np_u, np_t = noise_pred.chunk(2)
                noise_pred = np_u + ctx.guidance_scale * (np_t - np_u)
            latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            stats = tensor_stats(latents)
            stats.update({"step": i, "timestep": float(t)})
            latent_stats_rows.append(stats)

            if i in selected:
                img = _decode_latents_to_pil(pipe, latents)
                img.save(phase_dir / f"step_{i:03d}.png")
                decoded_imgs[i] = img
                log(f"  [trajectory] step {i:03d} t={float(t):.1f} decoded (mean {stats['mean']:+.3f}±{stats['std']:.3f})")

        final_img = _decode_latents_to_pil(pipe, latents)
        final_img.save(phase_dir / "final.png")
        decoded_imgs["final"] = final_img

        # grid
        try:
            ordered = sorted([k for k in decoded_imgs.keys() if k != "final"]) + ["final"]
            imgs = [decoded_imgs[k] for k in ordered]
            labels = [(f"step {k} (i={k})" if k != "final" else "final") for k in ordered]
            n = len(imgs)
            cols = min(5, n)
            rows = math.ceil(n / cols)
            plt = _get_plt()
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
            axes = np.atleast_1d(axes).flatten()
            for ax in axes:
                ax.axis("off")
            for ax, img, lbl in zip(axes, imgs, labels):
                ax.imshow(img)
                ax.set_title(lbl, fontsize=9)
            fig.suptitle(f"trajectory ({ctx.num_inference_steps} steps, i=0 most noisy -> i={ctx.num_inference_steps - 1} final)",
                         fontsize=10)
            fig.tight_layout()
            fig.savefig(phase_dir / "trajectory_grid.png", dpi=110)
            plt.close(fig)
        except Exception as e:
            st.notes.append(f"grid 作成失敗: {type(e).__name__}: {e}")

        write_json(phase_dir / "trajectory_metadata.json", {
            "num_inference_steps": ctx.num_inference_steps,
            "selected_steps": selected,
            "scheduler": type(pipe.scheduler).__name__,
            "do_cfg": do_cfg,
            "guidance_scale": ctx.guidance_scale,
            "latent_shape": list(shape),
            "init_noise_sigma": float(pipe.scheduler.init_noise_sigma),
            "step_index_note": "i=0 = 最ノイズ, i=N-1 = 最終 (DDPM t=T→t=0 と逆順)",
        })

        cols = ["step", "timestep", "mean", "std", "min", "max", "norm"]
        csv_lines = [",".join(cols)]
        for r in latent_stats_rows:
            csv_lines.append(",".join(f"{r[c]:.6g}" if isinstance(r[c], float) else str(r[c]) for c in cols))
        write_text(phase_dir / "latents_stats.csv", "\n".join(csv_lines) + "\n")

        # index.md
        idx_lines = [
            "# trajectory/",
            "",
            "SDXL Base の手動 scheduler ループによる denoising 過程。",
            f"step 数: {ctx.num_inference_steps} (i=0 が最ノイズ、i={ctx.num_inference_steps - 1} が最終)",
            "",
            "## 全体 grid", "",
            md_img("trajectory_grid.png", "trajectory grid"),
            "",
            "## 最終画像", "",
            md_img("final.png", "final"),
            "",
            "## 各 step",
            "",
        ]
        for k in sorted([k for k in decoded_imgs.keys() if k != "final"]):
            idx_lines.append(f"### step {k}")
            idx_lines.append("")
            idx_lines.append(md_img(f"step_{k:03d}.png", f"step {k}"))
            idx_lines.append("")
        idx_lines += [
            "## メモ", "",
            "- step index `i=0` が最もノイズが多い初期状態",
            f"- step index `i={ctx.num_inference_steps - 1}` が最終 (DDPM 論文の t=T → t=0 と逆順)",
            "- 3 年前 SD1.5 スライドの「i=0,1,2,...,50」表記もこれと同じ向き",
            "- [latents_stats.csv](latents_stats.csv) に各 step の mean/std/min/max/norm",
        ]
        write_text(phase_dir / "index.md", "\n".join(idx_lines) + "\n")

        st.extras["final_image"] = "trajectory/final.png"
        st.extras["decoded_steps"] = sorted([k for k in decoded_imgs.keys() if k != "final"])
    if not st.ok:
        save_failure(st, phase_dir)
    return st


# ---------------------------------------------------------------------------
# Recording attention processor
# ---------------------------------------------------------------------------

class RecordingAttnProcessor:
    """diffusers の classic AttnProcessor を写経。cross-attention の attention_probs
    を capture_now_flag が ON のときだけ保存する。"""

    def __init__(self, name: str, info: dict[str, Any],
                 capture_now_flag: dict[str, bool],
                 captured: list[dict[str, Any]]):
        self.name = name
        self.info = info
        self.capture_now_flag = capture_now_flag
        self.captured = captured

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, temb=None, *args, **kwargs):
        residual = hidden_states
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        query = attn.to_q(hidden_states)
        is_cross = encoder_hidden_states is not None
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        if is_cross and self.capture_now_flag.get("on", False):
            self.captured.append({
                "name": self.name,
                "info": self.info,
                "shape": list(attention_probs.shape),
                "probs": attention_probs.detach().to(torch.float32).cpu().clone(),
                "heads": int(attn.heads),
                "batch": int(batch_size),
                "query_len": int(attention_probs.shape[1]),
                "key_len": int(attention_probs.shape[2]),
            })

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


# ---------------------------------------------------------------------------
# Smart module selection
# ---------------------------------------------------------------------------

def _enumerate_cross_attention(unet) -> list[dict[str, Any]]:
    from diffusers.models.attention_processor import Attention
    out: list[dict[str, Any]] = []
    for name, mod in unet.named_modules():
        if isinstance(mod, Attention) and getattr(mod, "is_cross_attention", False):
            info = parse_module_info(name)
            info["heads"] = int(getattr(mod, "heads", 0) or 0)
            out.append(info)
    return out


def _select_representatives(all_cross: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """各 (stage, stage_index, attn_block_index) について transformer_index=0 の module を
    1 つ代表として選ぶ。これにより各 cross-attention 群から 1 つずつ拾える。"""
    by_key: dict[tuple, list[dict[str, Any]]] = {}
    for r in all_cross:
        k = (r["stage"], r["stage_index"], r["attn_block_index"])
        by_key.setdefault(k, []).append(r)
    selected: list[dict[str, Any]] = []
    for k, group in by_key.items():
        group.sort(key=lambda r: r["transformer_index"] or 0)
        selected.append(group[0])
    selected.sort(key=module_sort_key)
    return selected


# ---------------------------------------------------------------------------
# Token table
# ---------------------------------------------------------------------------

def _build_token_table(tokenizer, prompt: str, target_words: list[str],
                       phrases: list[tuple[str, list[str]]]) -> dict[str, Any]:
    """CLIP-L tokenizer に基づき prompt の token sequence と target word の対応を取る。"""
    ids = tokenizer(prompt, padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True, return_tensors="pt").input_ids[0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(ids)

    def _find_indices(word: str) -> list[int]:
        # " word" として encode (CLIP BPE は先頭 space を意識する)
        wid_with_space = tokenizer(" " + word, add_special_tokens=False).input_ids
        wid_no_space = tokenizer(word, add_special_tokens=False).input_ids
        hits: list[int] = []
        for cand in (wid_with_space, wid_no_space):
            if not cand:
                continue
            target_id = cand[0]
            for i_t, tid in enumerate(ids):
                if tid == target_id and i_t not in hits:
                    hits.append(i_t)
            if hits:
                break
        return hits

    token_index: dict[str, list[int]] = {}
    for w in target_words:
        token_index[w] = _find_indices(w)

    phrase_index: dict[str, list[int]] = {}
    for name, parts in phrases:
        idxs: list[int] = []
        for p in parts:
            idxs += token_index.get(p, [])
        # 重複削除 + sort
        phrase_index[name] = sorted(set(idxs))

    return {
        "ids": ids,
        "tokens": tokens,
        "token_index": token_index,
        "phrase_index": phrase_index,
    }


# ---------------------------------------------------------------------------
# Phase 3: cross-attention probe (refactored for SD1.5 slide parity)
# ---------------------------------------------------------------------------

def phase3_attention(ctx: RunContext, pipe) -> PhaseStatus:
    st = ctx.phase("phase3_attention")
    phase_dir = ctx.run_dir / "attention"
    ensure_dir(phase_dir)
    if ctx.args.skip_attn:
        st.skipped = True
        st.notes.append("--skip-attn 指定によりスキップ")
        log("[phase3] skipped by --skip-attn")
        return st

    with phase_guard(st, label="phase 3: cross-attention probe"):
        # 3-A: enumerate + structure
        all_cross = _enumerate_cross_attention(pipe.unet)
        # write attention_inventory (full)
        write_json(phase_dir / "attention_inventory.json", {
            "total_cross_attention": len(all_cross),
            "modules": all_cross,
        })
        inv_md = [
            "# attention_inventory",
            "",
            f"- total cross-attention modules: {len(all_cross)}",
            "",
            "| name | stage | sIdx | aIdx | tIdx | heads |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for r in sorted(all_cross, key=module_sort_key):
            inv_md.append(f"| `{r['name']}` | {r['stage']} | {r['stage_index']} "
                          f"| {r['attn_block_index']} | {r['transformer_index']} | {r['heads']} |")
        write_text(phase_dir / "attention_inventory.md", "\n".join(inv_md) + "\n")

        # group by (stage, stage_index, attn_block_index)
        groups: dict[tuple, list[dict[str, Any]]] = {}
        for r in all_cross:
            k = (r["stage"], r["stage_index"], r["attn_block_index"])
            groups.setdefault(k, []).append(r)
        structure_lines = [
            "# U-Net cross-attention structure",
            "",
            f"- cross-attention は {len(groups)} 個の (stage, stage_index, attn_block_index) グループに分かれる",
            "- 各グループ内に複数の transformer_block があり、それぞれに attn2 (cross-attention) がある",
            "",
            "## グループ別 transformer_block 数",
            "",
            "| stage | stage_idx | attn_idx | n_transformer_blocks | heads |",
            "|---|---:|---:|---:|---:|",
        ]
        sorted_groups = sorted(groups.items(), key=lambda kv: (
            STAGE_ORDER.get(kv[0][0], 99),
            kv[0][1] if kv[0][1] is not None else 99,
            kv[0][2] if kv[0][2] is not None else 99,
        ))
        for (stage, sidx, aidx), members in sorted_groups:
            structure_lines.append(
                f"| {stage} | {sidx} | {aidx} | {len(members)} | {members[0]['heads']} |"
            )
        write_text(phase_dir / "unet_attention_structure.md",
                   "\n".join(structure_lines) + "\n")
        write_json(phase_dir / "unet_attention_structure.json", {
            "n_groups": len(groups),
            "groups": [{"stage": k[0], "stage_index": k[1], "attn_block_index": k[2],
                        "n_transformer_blocks": len(v), "heads": v[0]["heads"]}
                       for k, v in sorted_groups],
            "_note": "transformer_index = 0 を各グループの代表として selected_attention_modules に採用",
        })

        # 3-B: representatives selection
        selected = _select_representatives(all_cross)
        st.extras["n_selected_representatives"] = len(selected)
        write_json(phase_dir / "selected_attention_modules.json", {
            "n_selected": len(selected),
            "rationale": "各 (stage, stage_index, attn_block_index) グループから "
                         "transformer_index=0 を代表として 1 つ選択",
            "modules": selected,
        })
        sel_md = [
            "# selected_attention_modules",
            "",
            f"- 選択数: {len(selected)}",
            "- 各 (stage, stage_index, attn_block_index) グループから transformer_index=0 を代表として 1 つ選択",
            "- 空間解像度 (H × W) は capture 後に query_len の sqrt から推定 (後述)",
            "",
            "| traversal | short | name | stage | sIdx | aIdx | tIdx |",
            "|---:|---|---|---|---:|---:|---:|",
        ]
        for i, r in enumerate(selected):
            sel_md.append(f"| {i} | `{module_short_name(r)}` | `{r['name']}` | "
                          f"{r['stage']} | {r['stage_index']} | {r['attn_block_index']} | "
                          f"{r['transformer_index']} |")
        write_text(phase_dir / "selected_attention_modules.md", "\n".join(sel_md) + "\n")

        # 3-C: token table
        ttab = _build_token_table(pipe.tokenizer, ctx.prompt,
                                  ATTN_TARGET_TOKENS, ATTN_TARGET_PHRASES)
        write_json(phase_dir / "token_table.json", {
            "prompt": ctx.prompt,
            "tokens_first30": ttab["tokens"][:30],
            "token_index": ttab["token_index"],
            "phrase_index": ttab["phrase_index"],
        })
        tt_md = [
            "# token_table",
            "",
            f"- prompt: `{ctx.prompt}`",
            f"- tokens[:20]: `{' '.join(ttab['tokens'][:20])}`",
            "",
            "## target tokens",
            "",
            "| word | token indices |",
            "|---|---|",
        ]
        for w in ATTN_TARGET_TOKENS:
            idxs = ttab["token_index"].get(w, [])
            tt_md.append(f"| {w} | {idxs if idxs else '✗ not found'} |")
        tt_md += ["", "## target phrases (aggregated columns)", "",
                  "| phrase | token indices |", "|---|---|"]
        for pname, idxs in ttab["phrase_index"].items():
            tt_md.append(f"| {pname} | {idxs if idxs else '✗ empty'} |")
        write_text(phase_dir / "token_table.md", "\n".join(tt_md) + "\n")

        # 3-D: capture
        from diffusers.models.attention_processor import AttnProcessor, Attention

        capture_now_flag = {"on": False}
        captured: list[dict[str, Any]] = []
        original_procs: dict[str, Any] = {}
        selected_names = {r["name"] for r in selected}

        for name, mod in pipe.unet.named_modules():
            if isinstance(mod, Attention):
                original_procs[name] = mod.processor
                if name in selected_names:
                    mod.processor = RecordingAttnProcessor(
                        name, parse_module_info(name), capture_now_flag, captured)
                else:
                    mod.processor = AttnProcessor()

        try:
            do_cfg = ctx.guidance_scale > 1.0
            enc = _encode_for_manual(pipe, ctx, ctx.prompt, ctx.negative_prompt, do_cfg)
            prompt_embeds = enc["prompt_embeds"]
            added_cond = enc["added_cond_kwargs"]

            pipe.scheduler.set_timesteps(ctx.num_inference_steps, device=ctx.device)
            timesteps = pipe.scheduler.timesteps
            unet = pipe.unet
            latent_dtype = next(unet.parameters()).dtype
            shape = (1, unet.config.in_channels, ctx.height // 8, ctx.width // 8)
            gen = torch.Generator(device="cpu" if ctx.device == "mps" else ctx.device).manual_seed(ctx.seed)
            latents = torch.randn(shape, generator=gen, dtype=latent_dtype).to(ctx.device)
            latents = latents * pipe.scheduler.init_noise_sigma

            n_steps = ctx.num_inference_steps
            capture_steps = sorted({max(0, min(n_steps - 1, int(round(f * (n_steps - 1)))))
                                    for f in ATTN_TARGET_STEP_FRACTIONS})
            st.extras["capture_steps"] = capture_steps
            log(f"  [attn] capture_steps = {capture_steps}, selected = {len(selected)} modules")

            # step → list of captures
            step_captured: dict[int, list[dict[str, Any]]] = {}
            decoded_at_capture: dict[int, Image.Image] = {}
            for i, t in enumerate(timesteps):
                latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
                latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
                capture_now_flag["on"] = i in capture_steps
                captured.clear()
                with torch.no_grad():
                    noise_pred = unet(
                        latent_model_input, t,
                        encoder_hidden_states=prompt_embeds,
                        added_cond_kwargs=added_cond,
                        return_dict=False,
                    )[0]
                if do_cfg:
                    np_u, np_t = noise_pred.chunk(2)
                    noise_pred = np_u + ctx.guidance_scale * (np_t - np_u)
                latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                if i in capture_steps and captured:
                    step_captured[i] = [dict(c) for c in captured]
                    # decode at capture step for overlay
                    try:
                        decoded_at_capture[i] = _decode_latents_to_pil(pipe, latents)
                    except Exception as ee:
                        log(f"  [attn] decode failed at step {i}: {ee}")

            # final image (再生成 = trajectory が既に書いた final.png を使う)
            final_img_path = ctx.run_dir / "trajectory" / "final.png"
            if final_img_path.exists():
                final_img = Image.open(final_img_path).convert("RGB")
            else:
                final_img = _decode_latents_to_pil(pipe, latents)

            # raw_attention_shapes
            shapes = []
            for step_idx, recs in step_captured.items():
                for r in recs:
                    shapes.append({
                        "step": step_idx, "name": r["name"],
                        "stage": r["info"]["stage"],
                        "stage_index": r["info"]["stage_index"],
                        "attn_block_index": r["info"]["attn_block_index"],
                        "transformer_index": r["info"]["transformer_index"],
                        "shape": r["shape"], "query_len": r["query_len"],
                        "key_len": r["key_len"], "heads": r["heads"],
                    })
            # spatial 推定
            for s in shapes:
                Q = s["query_len"]
                side = int(round(math.sqrt(Q)))
                s["inferred_spatial_height"] = side if side * side == Q else None
                s["inferred_spatial_width"] = side if side * side == Q else None
            write_json(phase_dir / "raw_attention_shapes.json", {
                "n_steps_captured": len(step_captured),
                "n_captures_total": sum(len(v) for v in step_captured.values()),
                "captures": shapes,
            })

            # ----- 3-E: heatmap data 構築 -----
            # heatmap[(step, name, token_or_phrase)] -> 2D numpy array (H, W)
            heatmap_db: dict[tuple, np.ndarray] = {}
            module_res_db: dict[str, tuple[int, int]] = {}
            for step_idx, recs in step_captured.items():
                for rec in recs:
                    name = rec["name"]
                    probs = rec["probs"]  # (B*heads, Q, K)
                    heads = rec["heads"]
                    bhi = probs.shape[0]
                    n_batch = bhi // heads
                    p = probs.view(n_batch, heads, probs.shape[1], probs.shape[2])
                    # CFG text side
                    if n_batch == 2:
                        p = p[1:2]
                    p_avg = p.mean(dim=1)[0]  # (Q, K)
                    Q = p_avg.shape[0]
                    side = int(round(math.sqrt(Q)))
                    if side * side != Q:
                        continue
                    module_res_db[name] = (side, side)
                    # per-token
                    for word, idxs in ttab["token_index"].items():
                        if not idxs:
                            continue
                        cols = p_avg[:, idxs].sum(dim=1)  # 複数 hit があれば加算
                        heatmap_db[(step_idx, name, word)] = cols.view(side, side).numpy()
                    # per-phrase
                    for pname, idxs in ttab["phrase_index"].items():
                        if not idxs:
                            continue
                        cols = p_avg[:, idxs].sum(dim=1)
                        heatmap_db[(step_idx, name, pname)] = cols.view(side, side).numpy()

            # ---- raw_heatmaps/ : 個別 heatmap ----
            raw_dir = phase_dir / "raw_heatmaps"
            ensure_dir(raw_dir)
            plt = _get_plt()
            for (step_idx, name, word), arr in heatmap_db.items():
                fig, ax = plt.subplots(figsize=(2.6, 2.6))
                a = arr.copy()
                if a.max() > a.min():
                    a = (a - a.min()) / (a.max() - a.min())
                ax.imshow(a, cmap="inferno")
                ax.set_title(f"step {step_idx} / {word}\n{module_short_name(parse_module_info(name))}",
                             fontsize=7)
                ax.axis("off")
                fig.tight_layout()
                short = module_short_name(parse_module_info(name)).replace(".", "_")
                fig.savefig(raw_dir / f"step_{step_idx:03d}_{short}_token_{word}.png", dpi=110)
                plt.close(fig)

            # ---- legacy/ : token x module grid (3 年前スライド対応) ----
            legacy_dir = phase_dir / "legacy"
            ensure_dir(legacy_dir)
            # 表示用 token order (phrases 含む)
            row_names: list[tuple[str, str]] = []  # (display_label, key_in_db)
            for w in ATTN_TARGET_TOKENS:
                if ttab["token_index"].get(w):
                    row_names.append((w, w))
            for pname, idxs in ttab["phrase_index"].items():
                if idxs:
                    row_names.append((pname.replace("_", " "), pname))
            # 列 (selected modules, traversal order)
            col_modules = [r for r in selected if r["name"] in module_res_db]

            def _legacy_grid(step_idx: int, normalize_globally: bool,
                             out_path: Path) -> None:
                cols = len(col_modules) + 1  # +1 = decoded image at this step
                rows = len(row_names)
                if cols == 1 or rows == 0:
                    return
                fig, axes = plt.subplots(rows, cols,
                                         figsize=(cols * 1.7, rows * 1.7))
                axes = np.atleast_2d(axes)
                # 共通スケール用 vmax/vmin
                vmin, vmax = None, None
                if normalize_globally:
                    vals = []
                    for ri, (_, key) in enumerate(row_names):
                        for cj, mr in enumerate(col_modules):
                            arr = heatmap_db.get((step_idx, mr["name"], key))
                            if arr is not None:
                                vals.append(arr)
                    if vals:
                        all_v = np.concatenate([v.ravel() for v in vals])
                        vmin, vmax = float(all_v.min()), float(all_v.max())

                for ri, (label, key) in enumerate(row_names):
                    for cj, mr in enumerate(col_modules):
                        ax = axes[ri, cj]
                        ax.axis("off")
                        arr = heatmap_db.get((step_idx, mr["name"], key))
                        if arr is None:
                            ax.set_facecolor("#222")
                            continue
                        if normalize_globally:
                            ax.imshow(arr, cmap="inferno", vmin=vmin, vmax=vmax)
                        else:
                            a = arr.copy()
                            if a.max() > a.min():
                                a = (a - a.min()) / (a.max() - a.min())
                            ax.imshow(a, cmap="inferno")
                        if ri == 0:
                            res = module_res_db.get(mr["name"], (0, 0))
                            ax.set_title(
                                f"{mr['stage']}{mr['stage_index']}\n"
                                f"{res[0]}x{res[1]}\n"
                                f"{module_short_name(mr)}",
                                fontsize=7)
                        if cj == 0:
                            ax.text(-0.18, 0.5, label, transform=ax.transAxes,
                                    fontsize=9, ha="right", va="center",
                                    rotation=0)
                    # last column: decoded image at this step
                    ax_im = axes[ri, -1]
                    ax_im.axis("off")
                    if ri == 0:
                        ax_im.set_title(f"decoded\nstep {step_idx}", fontsize=7)
                    img_pick = decoded_at_capture.get(step_idx, final_img)
                    ax_im.imshow(img_pick)

                ttl = f"legacy-style cross-attention grid - step {step_idx} " \
                      f"({'global' if normalize_globally else 'per-cell'} norm)"
                fig.suptitle(ttl, fontsize=10)
                fig.tight_layout()
                fig.savefig(out_path, dpi=130)
                plt.close(fig)

            for step_idx in sorted(step_captured.keys()):
                _legacy_grid(step_idx, normalize_globally=False,
                             out_path=legacy_dir / f"legacy_style_attention_grid_step_{step_idx:03d}.png")
            # global norm 版は最後の step だけ (情報密度確保)
            last_step = max(step_captured.keys()) if step_captured else None
            if last_step is not None:
                _legacy_grid(last_step, normalize_globally=True,
                             out_path=legacy_dir / f"legacy_style_attention_grid_step_{last_step:03d}_globalnorm.png")

            # ---- per_token/ : 単語ごと U-Net 通過順 (列=step×module) ----
            per_token_dir = phase_dir / "per_token"
            ensure_dir(per_token_dir)

            def _per_token_grid(label: str, key: str, out_path: Path) -> None:
                # rows = steps, cols = modules + decoded image
                steps_avail = sorted(step_captured.keys())
                cols = len(col_modules) + 1
                rows = len(steps_avail)
                if cols == 1 or rows == 0:
                    return
                fig, axes = plt.subplots(rows, cols,
                                         figsize=(cols * 1.8, rows * 1.8))
                axes = np.atleast_2d(axes)
                for ri, step_idx in enumerate(steps_avail):
                    for cj, mr in enumerate(col_modules):
                        ax = axes[ri, cj]
                        ax.axis("off")
                        arr = heatmap_db.get((step_idx, mr["name"], key))
                        if arr is None:
                            continue
                        a = arr.copy()
                        if a.max() > a.min():
                            a = (a - a.min()) / (a.max() - a.min())
                        ax.imshow(a, cmap="inferno")
                        if ri == 0:
                            res = module_res_db.get(mr["name"], (0, 0))
                            ax.set_title(
                                f"{mr['stage']}{mr['stage_index']}\n"
                                f"{res[0]}x{res[1]}\n"
                                f"{module_short_name(mr)}",
                                fontsize=7)
                        if cj == 0:
                            ax.text(-0.18, 0.5, f"step {step_idx}",
                                    transform=ax.transAxes,
                                    fontsize=8, ha="right", va="center")
                    ax_im = axes[ri, -1]
                    ax_im.axis("off")
                    img_pick = decoded_at_capture.get(step_idx, final_img)
                    ax_im.imshow(img_pick)
                    if ri == 0:
                        ax_im.set_title("decoded\nat step", fontsize=7)

                fig.suptitle(f"token = '{label}' across U-Net levels x capture steps",
                             fontsize=10)
                fig.tight_layout()
                fig.savefig(out_path, dpi=130)
                plt.close(fig)

            for label, key in row_names:
                fname_key = key.replace(" ", "_")
                _per_token_grid(label, key,
                                per_token_dir / f"token_{fname_key}_unet_levels.png")

            # ---- per_resolution/ : 解像度ごとに transformer block traversal ----
            # 同じ resolution に複数 module が並ぶ SDXL の特徴を見せる
            per_res_dir = phase_dir / "per_resolution"
            ensure_dir(per_res_dir)
            res_to_modules: dict[tuple[int, int], list[dict[str, Any]]] = {}
            for mr in col_modules:
                res = module_res_db.get(mr["name"])
                if res is None:
                    continue
                res_to_modules.setdefault(res, []).append(mr)
            # 各解像度: rows=tokens, cols=modules at this res
            for res, modules in sorted(res_to_modules.items()):
                modules_sorted = sorted(modules, key=module_sort_key)
                if not modules_sorted:
                    continue
                # 1 capture step を 1 枚にする (混在を避ける)。中央 step を使う。
                if not step_captured:
                    continue
                steps_avail = sorted(step_captured.keys())
                step_pick = steps_avail[len(steps_avail) // 2]
                cols = len(modules_sorted) + 1
                rows = len(row_names)
                if rows == 0:
                    continue
                fig, axes = plt.subplots(rows, cols,
                                         figsize=(cols * 1.7, rows * 1.7))
                axes = np.atleast_2d(axes)
                for ri, (label, key) in enumerate(row_names):
                    for cj, mr in enumerate(modules_sorted):
                        ax = axes[ri, cj]
                        ax.axis("off")
                        arr = heatmap_db.get((step_pick, mr["name"], key))
                        if arr is None:
                            continue
                        a = arr.copy()
                        if a.max() > a.min():
                            a = (a - a.min()) / (a.max() - a.min())
                        ax.imshow(a, cmap="inferno")
                        if ri == 0:
                            ax.set_title(f"{mr['stage']}{mr['stage_index']}\n{module_short_name(mr)}",
                                         fontsize=7)
                        if cj == 0:
                            ax.text(-0.18, 0.5, label, transform=ax.transAxes,
                                    fontsize=9, ha="right", va="center")
                    ax_im = axes[ri, -1]
                    ax_im.axis("off")
                    if ri == 0:
                        ax_im.set_title("decoded\n(at step)", fontsize=7)
                    img_pick = decoded_at_capture.get(step_pick, final_img)
                    ax_im.imshow(img_pick)
                fig.suptitle(f"resolution {res[0]}x{res[1]} - transformer block traversal "
                             f"(step {step_pick}, SDXL has multiple blocks at same resolution)",
                             fontsize=10)
                fig.tight_layout()
                out = per_res_dir / f"resolution_{res[0]}x{res[1]}_blocks_grid.png"
                fig.savefig(out, dpi=130)
                plt.close(fig)

            # ---- overlays/ : showpiece overlay on final image ----
            overlays_dir = phase_dir / "overlays"
            ensure_dir(overlays_dir)
            # 軽く: 最後の capture step の各 module × 主要 3 token
            showpiece_tokens = []
            for w in ["robot", "classroom"]:
                if ttab["token_index"].get(w):
                    showpiece_tokens.append(w)
            if ttab["phrase_index"].get("artificial_intelligence"):
                showpiece_tokens.append("artificial_intelligence")

            if step_captured and showpiece_tokens:
                step_for_overlay = max(step_captured.keys())
                W_img, H_img = final_img.size
                for mr in col_modules:
                    res = module_res_db.get(mr["name"])
                    if res is None:
                        continue
                    short = module_short_name(mr).replace(".", "_")
                    for token_key in showpiece_tokens:
                        arr = heatmap_db.get((step_for_overlay, mr["name"], token_key))
                        if arr is None:
                            continue
                        a = arr.copy()
                        if a.max() > a.min():
                            a = (a - a.min()) / (a.max() - a.min())
                        # bilinear upsample by PIL
                        heat_img = Image.fromarray((a * 255).astype(np.uint8)).resize(
                            (W_img, H_img), Image.BILINEAR)
                        fig, ax = plt.subplots(figsize=(4, 4))
                        ax.imshow(final_img)
                        ax.imshow(np.asarray(heat_img), cmap="inferno", alpha=0.45)
                        ax.set_title(f"step {step_for_overlay} / {token_key}\n"
                                     f"{module_short_name(mr)} ({res[0]}x{res[1]}) "
                                     f"overlay on final",
                                     fontsize=8)
                        ax.axis("off")
                        fig.tight_layout()
                        out = overlays_dir / (
                            f"step_{step_for_overlay:03d}_{token_key}_{short}_overlay_final.png")
                        fig.savefig(out, dpi=120)
                        plt.close(fig)

            # ---- structured_attention_summary.md (= attention/index.md と統合) ----
            sas = []
            sas.append("# attention/ — structured summary")
            sas.append("")
            sas.append("3 年前 SD1.5 講義スライドに相当する cross-attention 可視化を SDXL Base で試作。")
            sas.append("")
            sas.append("## 取得できた spatial resolution 一覧")
            sas.append("")
            res_seen: dict[tuple[int, int], list[str]] = {}
            for name, res in module_res_db.items():
                res_seen.setdefault(res, []).append(name)
            if res_seen:
                sas.append("| resolution (H×W) | n_modules | example |")
                sas.append("|---|---:|---|")
                for res in sorted(res_seen.keys()):
                    sas.append(f"| {res[0]}×{res[1]} | {len(res_seen[res])} | `{res_seen[res][0]}` |")
            else:
                sas.append("(none captured)")
            sas.append("")
            sas.append("## SDXL Base と SD1.x の構造差")
            sas.append("")
            sas.append("- 3 年前の SD1.x 版は 64→32→16→8→16→32 の 4 種の空間解像度で attention があった")
            sas.append("- SDXL Base の cross-attention は基本 2 種類だけ (1024 生成時 = {64×64, 32×32})")
            sas.append("- 一方 SDXL は同じ解像度に複数 transformer block が並ぶ (down2/mid/up0 はそれぞれ 10 個)")
            sas.append("- そのため legacy grid の列順 (down1 → down2 → mid → up0 → up1) は")
            sas.append("  解像度では {64 → 32 → 32 → 32 → 64} となり、SD1.5 スライドの 6 列展開には届かない")
            sas.append("")
            sas.append("## down / mid / up 集計")
            sas.append("")
            sas.append("| stage | total modules | representatives selected |")
            sas.append("|---|---:|---:|")
            stage_total = {"down": 0, "mid": 0, "up": 0}
            stage_sel = {"down": 0, "mid": 0, "up": 0}
            for r in all_cross:
                stage_total[r["stage"]] = stage_total.get(r["stage"], 0) + 1
            for r in selected:
                stage_sel[r["stage"]] = stage_sel.get(r["stage"], 0) + 1
            for s in ["down", "mid", "up"]:
                sas.append(f"| {s} | {stage_total[s]} | {stage_sel[s]} |")
            sas.append("")
            sas.append("## 主要図 (3 年前スライド対応)")
            sas.append("")
            sas.append("### legacy-style grid (token × U-Net 通過順)")
            sas.append("")
            legacy_files = sorted((phase_dir / "legacy").glob("*.png"))
            if legacy_files:
                for lf in legacy_files:
                    rel = lf.relative_to(phase_dir)
                    sas.append(md_img(str(rel), lf.name))
                    sas.append("")
            else:
                sas.append("(none)")
                sas.append("")

            sas.append("### per-token grid (単語ごと U-Net 通過順 × capture step)")
            sas.append("")
            for lf in sorted((phase_dir / "per_token").glob("*.png")):
                rel = lf.relative_to(phase_dir)
                sas.append(f"#### {lf.stem}")
                sas.append("")
                sas.append(md_img(str(rel), lf.name))
                sas.append("")

            sas.append("### per-resolution grid (同一解像度の transformer block traversal)")
            sas.append("")
            for lf in sorted((phase_dir / "per_resolution").glob("*.png")):
                rel = lf.relative_to(phase_dir)
                sas.append(f"#### {lf.stem}")
                sas.append("")
                sas.append(md_img(str(rel), lf.name))
                sas.append("")

            sas.append("### overlay on final image (主要 token × 全 module)")
            sas.append("")
            overlay_files = sorted((phase_dir / "overlays").glob("*.png"))
            if overlay_files:
                sas.append(f"全 {len(overlay_files)} 枚 (overlays/ subdir 参照)。代表 3 枚を埋め込み:")
                sas.append("")
                for lf in overlay_files[:3]:
                    rel = lf.relative_to(phase_dir)
                    sas.append(md_img(str(rel), lf.name))
                    sas.append("")
            else:
                sas.append("(none)")
                sas.append("")
            sas.append("> attention は capture step の値、overlay は final image の上に重ねたもの。")
            sas.append("> step と overlay 対象画像は厳密には別タイミングであることに注意。")
            sas.append("")

            sas.append("## ファイル別ナビ")
            sas.append("")
            sas.append("- [attention_inventory.md](attention_inventory.md) — 全 cross-attention module 一覧")
            sas.append("- [unet_attention_structure.md](unet_attention_structure.md) — グループ別 transformer block 数")
            sas.append("- [selected_attention_modules.md](selected_attention_modules.md) — capture 対象代表 module")
            sas.append("- [token_table.md](token_table.md) — token / phrase の index 対応")
            sas.append("- [raw_attention_shapes.json](raw_attention_shapes.json) — capture shape 詳細")
            sas.append("- raw_heatmaps/ — 個別 (step, module, token) heatmap")
            sas.append("- legacy/ — 3 年前スライド対応 grid")
            sas.append("- per_token/ — 単語ごと grid")
            sas.append("- per_resolution/ — 解像度ごと grid")
            sas.append("- overlays/ — final image overlay")
            sas.append("")
            sas.append("## メモ")
            sas.append("")
            sas.append("- step index `i=0` が最ノイズ、`i=N-1` が最終 (DDPM 論文の t=T → t=0 と逆順)")
            sas.append(f"- capture step: {capture_steps}")
            sas.append(f"- num_inference_steps: {ctx.num_inference_steps}")
            sas.append("- attention_probs は head 平均、CFG batch は text 側のみ使用")
            sas.append("- phrase aggregate (artificial_intelligence など) は構成 token の attention 列を加算")
            write_text(phase_dir / "structured_attention_summary.md", "\n".join(sas) + "\n")
            # index.md = structured_attention_summary.md と同じ内容 (簡略コピー)
            write_text(phase_dir / "index.md", "\n".join(sas) + "\n")

            st.extras["resolutions_captured"] = sorted(res_seen.keys())
            st.extras["selected_modules"] = [r["name"] for r in selected]
            st.extras["tokens_found"] = [w for w in ATTN_TARGET_TOKENS
                                          if ttab["token_index"].get(w)]
            st.extras["phrases_found"] = [k for k, v in ttab["phrase_index"].items() if v]

        finally:
            # processor を元に戻す
            for name, mod in pipe.unet.named_modules():
                if isinstance(mod, Attention) and name in original_procs:
                    mod.processor = original_procs[name]

    if not st.ok:
        save_failure(st, phase_dir)
    return st


# ---------------------------------------------------------------------------
# Phase 4: guidance / negative prompt
# ---------------------------------------------------------------------------

def phase4_guidance(ctx: RunContext, pipe) -> PhaseStatus:
    st = ctx.phase("phase4_guidance")
    if ctx.args.skip_guidance:
        st.skipped = True
        st.notes.append("--skip-guidance によりスキップ")
        log("[phase4] skipped by --skip-guidance")
        return st

    g_dir = ctx.run_dir / "guidance"
    n_dir = ctx.run_dir / "negative_prompt"
    ensure_dir(g_dir)
    ensure_dir(n_dir)

    with phase_guard(st, label="phase 4: guidance / negative prompt"):
        if ctx.args.quick:
            gw, gh = 512, 512
            steps = max(8, ctx.num_inference_steps)
        else:
            gw, gh = ctx.width, ctx.height
            steps = ctx.num_inference_steps

        g_imgs: list[Image.Image] = []
        g_meta: list[dict[str, Any]] = []
        for g in GUIDANCE_VALUES:
            log(f"  [guidance] g={g} ...")
            gen = torch.Generator(device="cpu" if ctx.device == "mps" else ctx.device).manual_seed(ctx.seed)
            t0 = time.perf_counter()
            with torch.no_grad():
                out = pipe(prompt=ctx.prompt,
                           negative_prompt=ctx.negative_prompt or None,
                           width=gw, height=gh,
                           num_inference_steps=steps,
                           guidance_scale=float(g),
                           generator=gen)
            elapsed = time.perf_counter() - t0
            img = out.images[0]
            p = g_dir / f"guidance_{g:.1f}.png"
            img.save(p)
            g_imgs.append(img)
            g_meta.append({"guidance": float(g), "elapsed_sec": round(elapsed, 2),
                           "image": f"guidance/{p.name}"})

        plt = _get_plt()
        fig, axes = plt.subplots(1, len(GUIDANCE_VALUES), figsize=(len(GUIDANCE_VALUES) * 3, 3.3))
        axes = np.atleast_1d(axes)
        for ax, im, g in zip(axes, g_imgs, GUIDANCE_VALUES):
            ax.imshow(im); ax.set_title(f"g = {g}", fontsize=10); ax.axis("off")
        fig.suptitle(f"guidance scale sweep ({gw}x{gh}, {steps} steps)", fontsize=10)
        fig.tight_layout()
        fig.savefig(g_dir / "guidance_grid.png", dpi=120)
        plt.close(fig)
        write_json(g_dir / "guidance_metadata.json", {
            "guidance_values": GUIDANCE_VALUES, "size": [gw, gh],
            "steps": steps, "items": g_meta,
        })

        neg_imgs: list[Image.Image] = []
        neg_meta: list[dict[str, Any]] = []
        for label, neg in [("no_negative", ""),
                           ("with_negative", ctx.negative_prompt or "blurry, low quality, distorted")]:
            log(f"  [negative] {label} ...")
            gen = torch.Generator(device="cpu" if ctx.device == "mps" else ctx.device).manual_seed(ctx.seed)
            t0 = time.perf_counter()
            with torch.no_grad():
                out = pipe(prompt=ctx.prompt, negative_prompt=neg or None,
                           width=gw, height=gh, num_inference_steps=steps,
                           guidance_scale=ctx.guidance_scale, generator=gen)
            elapsed = time.perf_counter() - t0
            img = out.images[0]
            p = n_dir / f"{label}.png"
            img.save(p)
            neg_imgs.append(img)
            neg_meta.append({"label": label, "negative_prompt": neg,
                             "elapsed_sec": round(elapsed, 2),
                             "image": f"negative_prompt/{p.name}"})

        fig, axes = plt.subplots(1, 2, figsize=(7, 3.6))
        for ax, im, meta in zip(axes, neg_imgs, neg_meta):
            ax.imshow(im); ax.set_title(meta["label"], fontsize=10); ax.axis("off")
        fig.suptitle("negative prompt: off vs on", fontsize=10)
        fig.tight_layout()
        fig.savefig(n_dir / "negative_prompt_comparison.png", dpi=120)
        plt.close(fig)
        write_json(n_dir / "negative_prompt_metadata.json", {
            "size": [gw, gh], "steps": steps,
            "guidance_scale": ctx.guidance_scale, "items": neg_meta,
        })

        # index.md
        g_idx = ["# guidance/", "",
                 f"guidance_scale = {GUIDANCE_VALUES} の sweep ({gw}×{gh}, {steps} steps)。",
                 "", "## grid", "", md_img("guidance_grid.png", "guidance grid"), "",
                 "## 個別", ""]
        for g in GUIDANCE_VALUES:
            g_idx.append(f"### g = {g}")
            g_idx.append("")
            g_idx.append(md_img(f"guidance_{g:.1f}.png", f"g={g}"))
            g_idx.append("")
        g_idx.append("[metadata](guidance_metadata.json)")
        write_text(g_dir / "index.md", "\n".join(g_idx) + "\n")

        n_idx = ["# negative_prompt/", "",
                 "negative prompt あり/なしの比較 (他条件は同一)。",
                 "", "## 比較", "",
                 md_img("negative_prompt_comparison.png", "comparison"), "",
                 "## 個別", "",
                 "### no_negative", "", md_img("no_negative.png", "no_negative"), "",
                 "### with_negative", "", md_img("with_negative.png", "with_negative"), "",
                 "[metadata](negative_prompt_metadata.json)"]
        write_text(n_dir / "index.md", "\n".join(n_idx) + "\n")

        st.extras.update({"guidance_values": GUIDANCE_VALUES,
                          "size": [gw, gh], "steps": steps})
    return st


# ---------------------------------------------------------------------------
# Phase 5: summary
# ---------------------------------------------------------------------------

def _find_external_phase4(ctx: RunContext) -> dict[str, str] | None:
    """guidance を skip した場合、同じ outputs/08_sdxl_base_deep_probe/ 内の最近の
    full run から guidance/ と negative_prompt/ 出力があれば、それを summary.md
    から参照する。"""
    parent = ctx.run_dir.parent
    if not parent.exists():
        return None
    candidates = sorted([d for d in parent.iterdir()
                         if d.is_dir() and d != ctx.run_dir], reverse=True)
    for d in candidates:
        gg = d / "guidance" / "guidance_grid.png"
        ng = d / "negative_prompt" / "negative_prompt_comparison.png"
        if gg.exists() and ng.exists():
            return {
                "run_name": d.name,
                "guidance_grid": str(gg.relative_to(parent)),
                "neg_comparison": str(ng.relative_to(parent)),
            }
    return None


def phase5_summary(ctx: RunContext, total_elapsed: float) -> PhaseStatus:
    st = ctx.phase("phase5_summary")
    with phase_guard(st, label="phase 5: summary"):
        sj_path = ctx.run_dir / "summary.json"
        sm_path = ctx.run_dir / "summary.md"

        summary_json = {
            "script": SCRIPT_NAME, "model_key": MODEL_KEY,
            "model_id": ctx.model_id, "device": ctx.device, "dtype": str(ctx.dtype),
            "started_at": ctx.started_at, "finished_at": now_iso(),
            "total_elapsed_sec": round(total_elapsed, 2),
            "args": vars(ctx.args),
            "prompt": ctx.prompt, "negative_prompt": ctx.negative_prompt,
            "seed": ctx.seed, "width": ctx.width, "height": ctx.height,
            "num_inference_steps": ctx.num_inference_steps,
            "guidance_scale": ctx.guidance_scale,
            "phases": [p.to_dict() for p in ctx.phases],
        }
        write_json(sj_path, summary_json)

        def pstat(name: str) -> PhaseStatus | None:
            return ctx.get_phase(name)

        def status_str(p: PhaseStatus | None) -> str:
            if p is None:
                return "n/a"
            if p.skipped:
                return "skipped"
            return "ok" if p.ok else "FAILED"

        attn_st = pstat("phase3_attention")
        traj_st = pstat("phase2_trajectory")
        emb_st = pstat("phase1_embeddings")
        gst = pstat("phase4_guidance")

        # 重要 figure のパス
        traj_final = "trajectory/final.png"
        traj_grid = "trajectory/trajectory_grid.png"
        pca = "prompt_embeddings/prompt_pca.png"
        tsne = "prompt_embeddings/prompt_tsne.png"
        attn_struct = "attention/structured_attention_summary.md"
        legacy_dir = ctx.run_dir / "attention" / "legacy"
        per_token_dir = ctx.run_dir / "attention" / "per_token"

        def _exists(rel: str) -> bool:
            return (ctx.run_dir / rel).exists()

        def _embed_or_skip(rel: str, alt: str) -> str:
            if _exists(rel):
                return md_img(rel, alt)
            return f"(`{rel}` 未生成)"

        lines: list[str] = []
        lines.append(f"# 08 SDXL Base deep probe — {ctx.run_dir.name}")
        lines.append("")
        lines.append("> SDXL Base 1.0 (MPS / fp32) で 3 年前 SD1.5 講義スライド相当の")
        lines.append("> cross-attention 可視化を試作したもの。")
        lines.append("")
        lines.append("## 最初に見るべきファイル")
        lines.append("")
        lines.append("- **入口**: この summary.md")
        lines.append(f"- **attention の主役**: [{attn_struct}]({attn_struct})")
        for sub in ["trajectory", "attention", "prompt_embeddings", "guidance", "negative_prompt"]:
            p = ctx.run_dir / sub / "index.md"
            if p.exists():
                lines.append(f"- [{sub}/index.md]({sub}/index.md)")
        lines.append("")

        lines.append("## 主要画像 (preview)")
        lines.append("")
        lines.append("### 最終生成画像 (1 枚)")
        lines.append("")
        lines.append(_embed_or_skip(traj_final, "final image"))
        lines.append("")
        lines.append("### denoising trajectory grid")
        lines.append("")
        lines.append(_embed_or_skip(traj_grid, "trajectory grid"))
        lines.append("")

        lines.append("## 3 年前スライド対応の主要図")
        lines.append("")
        lines.append("3 年前 SD1.x 講義スライドでは U-Net の cross-attention weight を")
        lines.append("64→32→16→8→16→32 と 6 段の解像度に沿って token ごとに横並びにしていた。")
        lines.append("今回の SDXL Base 版では、capture した attention tensor の query_len から")
        lines.append("空間解像度を推定し、U-Net の traversal 順 (down → mid → up) に並べている。")
        lines.append("**SDXL Base の cross-attention 解像度は基本 2 種類 (1024 生成時 = {64×64, 32×32})** で、")
        lines.append("SD1.5 のような 4 段展開にはならない (詳細は structured_attention_summary 参照)。")
        lines.append("")
        lines.append("### legacy-style grid (token × U-Net 通過順)")
        lines.append("")
        if legacy_dir.exists():
            for lf in sorted(legacy_dir.glob("*.png")):
                rel = lf.relative_to(ctx.run_dir)
                lines.append(f"#### {lf.stem}")
                lines.append("")
                lines.append(md_img(str(rel), lf.name))
                lines.append("")
        else:
            lines.append("(none generated)")
            lines.append("")

        lines.append("### per-token grid (単語ごと U-Net 通過順 × step)")
        lines.append("")
        if per_token_dir.exists():
            # showpiece tokens 優先表示
            showpiece = ["token_robot_unet_levels.png",
                         "token_classroom_unet_levels.png",
                         "token_artificial_intelligence_unet_levels.png"]
            shown: set[str] = set()
            for sp in showpiece:
                lf = per_token_dir / sp
                if lf.exists():
                    rel = lf.relative_to(ctx.run_dir)
                    lines.append(f"#### {lf.stem}")
                    lines.append("")
                    lines.append(md_img(str(rel), lf.name))
                    lines.append("")
                    shown.add(sp)
            # 残りはリンクだけ
            others = [lf for lf in sorted(per_token_dir.glob("*.png")) if lf.name not in shown]
            if others:
                lines.append("他 token (リンク):")
                lines.append("")
                for lf in others:
                    rel = lf.relative_to(ctx.run_dir)
                    lines.append(f"- [{lf.stem}]({rel})")
                lines.append("")
        else:
            lines.append("(none generated)")
            lines.append("")

        # per_resolution
        per_res_dir = ctx.run_dir / "attention" / "per_resolution"
        if per_res_dir.exists():
            lines.append("### per-resolution grid (同一解像度の transformer block traversal)")
            lines.append("")
            for lf in sorted(per_res_dir.glob("*.png")):
                rel = lf.relative_to(ctx.run_dir)
                lines.append(f"#### {lf.stem}")
                lines.append("")
                lines.append(md_img(str(rel), lf.name))
                lines.append("")

        lines.append("## prompt embedding geometry")
        lines.append("")
        lines.append("### PCA")
        lines.append("")
        lines.append(_embed_or_skip(pca, "PCA pooled"))
        lines.append("")
        lines.append("### t-SNE")
        lines.append("")
        lines.append(_embed_or_skip(tsne, "t-SNE pooled"))
        lines.append("")

        # guidance / negative
        lines.append("## guidance / negative prompt")
        lines.append("")
        if gst and gst.ok:
            lines.append(_embed_or_skip("guidance/guidance_grid.png", "guidance grid"))
            lines.append("")
            lines.append(_embed_or_skip("negative_prompt/negative_prompt_comparison.png", "neg prompt comparison"))
            lines.append("")
        elif gst and gst.skipped:
            # 前回 run の参照
            ext = _find_external_phase4(ctx)
            lines.append("この run では `--skip-guidance` で省略。")
            lines.append("")
            if ext:
                lines.append(f"代わりに前回 run [`{ext['run_name']}/`](../{ext['run_name']}/) の図を参照:")
                lines.append("")
                lines.append(md_img(f"../{ext['guidance_grid']}", "guidance grid (previous run)"))
                lines.append("")
                lines.append(md_img(f"../{ext['neg_comparison']}", "neg comparison (previous run)"))
                lines.append("")
            else:
                lines.append("(参照可能な前回 run が見つかりません)")
                lines.append("")
        else:
            lines.append(f"- 状態: {status_str(gst)}")
            if gst and gst.error:
                lines.append(f"- error: `{gst.error}`")
            lines.append("")

        lines.append("## 実行条件")
        lines.append("")
        lines.append(f"- model_id: `{ctx.model_id}`")
        lines.append(f"- device / dtype: `{ctx.device}` / `{ctx.dtype}`")
        lines.append(f"- size: {ctx.width} × {ctx.height}")
        lines.append(f"- num_inference_steps: {ctx.num_inference_steps}")
        lines.append(f"  - step index `i=0` が最ノイズ、`i={ctx.num_inference_steps - 1}` が最終 (DDPM 論文の t=T→t=0 と逆順)")
        lines.append(f"- guidance_scale: {ctx.guidance_scale}")
        lines.append(f"- seed: {ctx.seed}")
        lines.append(f"- prompt: `{ctx.prompt}`")
        lines.append(f"- negative_prompt: `{ctx.negative_prompt}`")
        lines.append(f"- mode: {'quick' if ctx.args.quick else 'full'} "
                     f"/ skip_attn={ctx.args.skip_attn} / skip_guidance={ctx.args.skip_guidance}")
        lines.append(f"- started_at: {ctx.started_at}")
        lines.append(f"- total elapsed: {total_elapsed:.1f} s")
        lines.append("")

        lines.append("## phase 状況")
        lines.append("")
        lines.append("| phase | status | elapsed (s) | notes |")
        lines.append("|---|---|---:|---|")
        for p in ctx.phases:
            if p.name == "phase5_summary":
                continue
            note_short = "; ".join(p.notes) if p.notes else (p.error or "")
            note_short = note_short.replace("\n", " ")
            if len(note_short) > 90:
                note_short = note_short[:87] + "..."
            lines.append(f"| {p.name} | {status_str(p)} | {p.extras.get('elapsed_sec', '')} | {note_short} |")
        lines.append("")

        # 失敗 / 警告
        lines.append("## 失敗・警告・未解決点")
        lines.append("")
        any_issue = False
        for p in ctx.phases:
            if p.name == "phase5_summary":
                continue
            if not p.ok and not p.skipped:
                any_issue = True
                lines.append(f"- **{p.name}**: {p.error}")
            for n in p.notes:
                any_issue = True
                lines.append(f"- {p.name} note: {n}")
        if not any_issue:
            lines.append("- なし")
        lines.append("")

        lines.append("## 次に notebook 化すべき候補")
        lines.append("")
        if traj_st and traj_st.ok:
            lines.append("- **trajectory notebook**: 手動 scheduler ループは安定。latent stats CSV あり。")
        if attn_st and attn_st.ok:
            lines.append("- **cross-attention notebook**: legacy grid / per-token / overlay が一連で動いた。"
                         "SDXL は解像度 2 種だけなので、SD1.5 と並べた比較スライドが教育的。")
        if emb_st and emb_st.ok:
            lines.append("- **embedding geometry notebook**: PCA + t-SNE 両方使える。"
                         "中間 embedding (50% blend) の生成は今回未実装、将来追加候補。")
        if gst and gst.ok:
            lines.append("- **guidance / negative prompt notebook**: 同 seed で副作用少なく比較できる。")
        lines.append("")

        write_text(sm_path, "\n".join(lines))
        log(f"[summary] wrote {sj_path}")
        log(f"[summary] wrote {sm_path}")
    return st


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--full", action="store_true", default=True)
    mode.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-attn", action="store_true")
    ap.add_argument("--skip-guidance", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--run-dir", type=str, default=None)
    return ap.parse_args()


def build_run_dir(args: argparse.Namespace) -> Path:
    root = project_root() / "outputs" / OUTPUT_SUBDIR
    if args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.is_absolute():
            run_dir = root / args.run_dir
    else:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = root / ts
    if run_dir.exists() and not args.force:
        for suffix in range(1, 100):
            cand = run_dir.with_name(run_dir.name + f"_{suffix:02d}")
            if not cand.exists():
                run_dir = cand
                break
    ensure_dir(run_dir)
    return run_dir


def main() -> int:
    args = parse_args()
    if args.quick:
        args.full = False

    cfg = load_config()
    common = get_common(cfg)
    model_cfg = get_model_config(cfg, MODEL_KEY)

    model_id: str = model_cfg["model_id"]
    width = int(common.get("width", model_cfg.get("width", 1024)))
    height = int(common.get("height", model_cfg.get("height", 1024)))
    num_steps = int(model_cfg["num_inference_steps"])
    guidance = float(model_cfg["guidance_scale"])

    if args.quick:
        width, height = 512, 512
        num_steps = 8

    prompt = common["prompt"]
    negative_prompt = common.get("negative_prompt", "")
    seed = int(common["seed"])

    device, dtype = pick_device_and_dtype(model_cfg)
    run_dir = build_run_dir(args)

    ctx = RunContext(
        args=args, run_dir=run_dir, cfg=cfg, model_cfg=model_cfg,
        device=device, dtype=dtype, model_id=model_id,
        width=width, height=height,
        num_inference_steps=num_steps, guidance_scale=guidance,
        prompt=prompt, negative_prompt=negative_prompt, seed=seed,
        started_at=now_iso(),
    )

    log("=== diffusers_probe / 08 SDXL Base deep probe (round 3) ===")
    log(f"  run_dir : {run_dir}")
    log(f"  mode    : {'quick' if args.quick else 'full'}")
    log(f"  device  : {device}")
    log(f"  dtype   : {dtype}")
    log(f"  model   : {model_id}")
    log(f"  size    : {width} x {height}")
    log(f"  steps   : {num_steps}")
    log(f"  guidance: {guidance}")
    log(f"  prompt  : {prompt}")
    log(f"  skip_attn={args.skip_attn} skip_guidance={args.skip_guidance}")
    log("")

    t_overall = time.perf_counter()

    pipe = None
    load_st = ctx.phase("phase_load_pipeline")
    with phase_guard(load_st, label="pipeline load"):
        try:
            from diffusers import StableDiffusionXLPipeline
        except Exception as e:
            raise RuntimeError(f"diffusers import 失敗: {e}") from e
        t0 = time.perf_counter()
        try:
            pipe = StableDiffusionXLPipeline.from_pretrained(
                model_id, torch_dtype=dtype, use_safetensors=True,
                variant="fp16" if dtype in (torch.float16, torch.bfloat16) else None,
            )
        except Exception as e:
            for line in hint_for_load_error(e, model_id):
                log(line)
            raise
        pipe = pipe.to(device)
        if model_cfg.get("vae_fp32_override", False) and dtype != torch.float32:
            apply_vae_fp32_override(pipe)
            log("[vae] vae_fp32_override 適用")
        if model_cfg.get("enable_attention_slicing_on_mps", True) and device == "mps":
            pipe.enable_attention_slicing()
            log("[mps] enable_attention_slicing() 有効化")
        load_st.extras["load_time_sec"] = round(time.perf_counter() - t0, 2)
        log(f"[load] done in {load_st.extras['load_time_sec']} s")

    if not load_st.ok or pipe is None:
        phase5_summary(ctx, time.perf_counter() - t_overall)
        log("[fatal] pipeline load 失敗。summary を書いて終了。")
        return 1

    phase0_inventory(ctx, pipe)
    phase1_embeddings(ctx, pipe)
    phase2_trajectory(ctx, pipe)
    phase3_attention(ctx, pipe)
    phase4_guidance(ctx, pipe)
    phase5_summary(ctx, time.perf_counter() - t_overall)

    total = time.perf_counter() - t_overall
    log("")
    log("=== done ===")
    log(f"  total elapsed: {total:.1f} s")
    log(f"  summary.md   : {ctx.run_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
