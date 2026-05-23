# 09_prompt_explore.py
#
# SDXL Base 1.0 用 prompt 探索 script (Phase A: worker)。
# 1 config file = 1 prompt 実験 = 1 出力 dir。
#
# 目的: 任意の positive/negative prompt について、3 年前 SD1.5 講義 notebook と
# 同様の処理 (生成 + denoising trajectory + 主要単語の cross-attention) を
# 素早く出力する。target tokens / capture_steps / seeds / blend などはすべて
# config JSON で事前指定する (CLI で項目が多くなりすぎるため)。
#
# 使い方:
#   python scripts/09_prompt_explore.py --config configs/09_explore/<slug>.json
#   bash tmp/run_09_explore.sh configs/09_explore/<slug>.json
#
# 出力構造 (--config が configs/09_explore/foo.json なら):
#   outputs/09_explore/foo/
#   ├─ summary.md            ← 入口 (画像埋め込み)
#   ├─ summary.json
#   ├─ config.json           ← 入力 config の copy
#   ├─ seed_0123/
#   │   ├─ final.png
#   │   ├─ trajectory_grid.png
#   │   ├─ index.md
#   │   └─ attention/
#   │       ├─ legacy_grid_step_*.png
#   │       ├─ per_token/
#   │       └─ overlays/
#   ├─ seed_0234/
#   │   └─ ... (同構造)
#   └─ blend/                 (blend_with 指定時のみ)
#       ├─ final.png
#       └─ blend_metadata.json
#
# 設計方針:
#   - UNet pass は 1 seed あたり 1 回 (trajectory decode + attention capture を同時)
#   - 非選択 attention module は SDPA (AttnProcessor2_0) のまま (08 round 3 からの改善)
#   - MPS では fp32 固定
#   - 08 と同じ helper を多数持つが、09 は独立 script として動作する (08 import なし)

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
    get_model_config,
    hint_for_load_error,
    load_config as load_diffusers_probe_config,
    pick_device_and_dtype,
    project_root,
)

MODEL_KEY = "sdxl_base"
SCRIPT_NAME = "09_prompt_explore.py"

STAGE_ORDER = {"down": 0, "mid": 1, "up": 2}


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


def md_img(rel: str, alt: str = "") -> str:
    return f"![{alt or rel}]({rel})"


def _get_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

@dataclass
class ExploreConfig:
    slug: str
    prompt: str
    negative_prompt: str
    seeds: list[int]
    size: tuple[int, int]
    num_inference_steps: int
    guidance_scale: float
    target_tokens: list[str]
    capture_step_fractions: list[float]
    blend_with: dict | None
    raw_path: Path

    @classmethod
    def from_file(cls, path: Path) -> "ExploreConfig":
        with path.open("r", encoding="utf-8") as f:
            d = json.load(f)
        # 必須キー
        for k in ("prompt",):
            if k not in d:
                raise ValueError(f"config missing key: {k}")
        slug = d.get("slug") or path.stem
        size = tuple(d.get("size", [1024, 1024]))
        if len(size) != 2:
            raise ValueError("size must be [width, height]")
        return cls(
            slug=slug,
            prompt=d["prompt"],
            negative_prompt=d.get("negative_prompt", ""),
            seeds=list(d.get("seeds", [42])),
            size=(int(size[0]), int(size[1])),
            num_inference_steps=int(d.get("num_inference_steps", 20)),
            guidance_scale=float(d.get("guidance_scale", 7.5)),
            target_tokens=list(d.get("target_tokens", [])),
            capture_step_fractions=list(d.get("capture_step_fractions", [0.17, 0.5, 0.83])),
            blend_with=d.get("blend_with"),
            raw_path=path,
        )

    def as_dict(self) -> dict:
        return {
            "slug": self.slug, "prompt": self.prompt,
            "negative_prompt": self.negative_prompt, "seeds": self.seeds,
            "size": list(self.size),
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "target_tokens": self.target_tokens,
            "capture_step_fractions": self.capture_step_fractions,
            "blend_with": self.blend_with,
            "source_config_path": str(self.raw_path),
        }


# ---------------------------------------------------------------------------
# module info / selection
# ---------------------------------------------------------------------------

def parse_module_info(name: str) -> dict[str, Any]:
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
    return (
        STAGE_ORDER.get(info.get("stage", "unknown"), 99),
        info.get("stage_index") if info.get("stage_index") is not None else 99,
        info.get("attn_block_index") if info.get("attn_block_index") is not None else 99,
        info.get("transformer_index") if info.get("transformer_index") is not None else 99,
    )


def module_short_name(info: dict[str, Any]) -> str:
    stage = info["stage"]
    if stage == "mid":
        return f"mid.t{info['transformer_index']}"
    return f"{stage}{info['stage_index']}.a{info['attn_block_index']}.t{info['transformer_index']}"


def select_representatives(unet) -> list[dict[str, Any]]:
    """各 (stage, stage_index, attn_block_index) グループから transformer_index=0 を代表選択。"""
    from diffusers.models.attention_processor import Attention
    all_cross: list[dict[str, Any]] = []
    for name, mod in unet.named_modules():
        if isinstance(mod, Attention) and getattr(mod, "is_cross_attention", False):
            info = parse_module_info(name)
            info["heads"] = int(getattr(mod, "heads", 0) or 0)
            all_cross.append(info)
    by_key: dict[tuple, list[dict]] = {}
    for r in all_cross:
        k = (r["stage"], r["stage_index"], r["attn_block_index"])
        by_key.setdefault(k, []).append(r)
    selected: list[dict] = []
    for k, group in by_key.items():
        group.sort(key=lambda r: r["transformer_index"] or 0)
        selected.append(group[0])
    selected.sort(key=module_sort_key)
    return selected


# ---------------------------------------------------------------------------
# Recording attention processor
# ---------------------------------------------------------------------------

class RecordingAttnProcessor:
    """classic AttnProcessor 写経 + capture_flag が ON のとき attention_probs を保存。"""

    def __init__(self, name: str, info: dict[str, Any],
                 capture_flag: dict[str, bool],
                 captured: list[dict[str, Any]]):
        self.name = name
        self.info = info
        self.capture_flag = capture_flag
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

        if is_cross and self.capture_flag.get("on", False):
            self.captured.append({
                "name": self.name, "info": self.info,
                "shape": list(attention_probs.shape),
                "probs": attention_probs.detach().to(torch.float32).cpu().clone(),
                "heads": int(attn.heads), "batch": int(batch_size),
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


def install_recording_processors(unet, selected: list[dict[str, Any]],
                                 capture_flag: dict[str, bool],
                                 captured: list[dict[str, Any]]) -> dict[str, Any]:
    """選択 module だけ RecordingAttnProcessor に差し替え。残りは SDPA のまま温存。"""
    from diffusers.models.attention_processor import Attention
    selected_names = {r["name"] for r in selected}
    originals: dict[str, Any] = {}
    for name, mod in unet.named_modules():
        if isinstance(mod, Attention) and name in selected_names:
            originals[name] = mod.processor
            mod.processor = RecordingAttnProcessor(
                name, parse_module_info(name), capture_flag, captured)
    return originals


def restore_processors(unet, originals: dict[str, Any]) -> None:
    from diffusers.models.attention_processor import Attention
    for name, mod in unet.named_modules():
        if isinstance(mod, Attention) and name in originals:
            mod.processor = originals[name]


# ---------------------------------------------------------------------------
# SDXL ヘルパ
# ---------------------------------------------------------------------------

def encode_for_manual(pipe, device: str, width: int, height: int,
                      prompt: str, negative_prompt: str,
                      do_cfg: bool) -> dict[str, Any]:
    with torch.no_grad():
        pe, npe, pooled, npooled = pipe.encode_prompt(
            prompt=prompt, negative_prompt=negative_prompt or None,
            device=device, num_images_per_prompt=1,
            do_classifier_free_guidance=do_cfg,
        )
    add_time_ids = torch.tensor(
        [[height, width, 0, 0, height, width]],
        dtype=pe.dtype, device=device,
    )
    if do_cfg:
        pe_in = torch.cat([npe, pe], dim=0)
        pooled_in = torch.cat([npooled, pooled], dim=0)
        time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)
    else:
        pe_in, pooled_in, time_ids = pe, pooled, add_time_ids
    return {
        "prompt_embeds_raw": pe, "negative_prompt_embeds_raw": npe,
        "pooled_raw": pooled, "negative_pooled_raw": npooled,
        "prompt_embeds": pe_in,
        "added_cond_kwargs": {"text_embeds": pooled_in, "time_ids": time_ids},
    }


def decode_latents_to_pil(pipe, latents: torch.Tensor) -> Image.Image:
    vae = pipe.vae
    with torch.no_grad():
        z = latents.to(dtype=next(vae.parameters()).dtype) / vae.config.scaling_factor
        img = vae.decode(z, return_dict=False)[0]
    img = (img / 2 + 0.5).clamp(0, 1)
    arr = (img[0].permute(1, 2, 0).float().cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def select_trajectory_decode_steps(num_steps: int) -> list[int]:
    """trajectory grid 用に decode する step を選ぶ (探索なので 5-6 個程度)。"""
    if num_steps >= 20:
        cand = [0, 1, num_steps // 4, num_steps // 2, (3 * num_steps) // 4, num_steps - 1]
    elif num_steps >= 8:
        cand = [0, 1, num_steps // 2, (3 * num_steps) // 4, num_steps - 1]
    else:
        cand = list(range(num_steps))
    return sorted(set(c for c in cand if 0 <= c < num_steps))


def build_token_table(tokenizer, prompt: str, target_words: list[str]) -> dict[str, Any]:
    """CLIP-L tokenizer に基づき prompt の token sequence と target word を対応。"""
    ids = tokenizer(prompt, padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True, return_tensors="pt").input_ids[0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(ids)

    def _find_indices(word: str) -> list[int]:
        wid_with_space = tokenizer(" " + word, add_special_tokens=False).input_ids
        wid_no_space = tokenizer(word, add_special_tokens=False).input_ids
        for cand in (wid_with_space, wid_no_space):
            if not cand:
                continue
            target_id = cand[0]
            hits = [i for i, tid in enumerate(ids) if tid == target_id]
            if hits:
                return hits
        return []

    token_index: dict[str, list[int]] = {w: _find_indices(w) for w in target_words}
    return {"ids": ids, "tokens": tokens, "token_index": token_index}


# ---------------------------------------------------------------------------
# Core: 1 seed run (UNet pass + decode + capture)
# ---------------------------------------------------------------------------

def run_one_seed(pipe, device: str, dtype: torch.dtype,
                 cfg: ExploreConfig, seed: int,
                 selected: list[dict[str, Any]],
                 out_dir: Path,
                 prompt_override_embeds: dict | None = None,
                 label: str | None = None) -> dict[str, Any]:
    """1 seed (or blend) について UNet pass を回し、trajectory decode + attention capture を同時実行。"""
    label = label or f"seed_{seed:04d}"
    ensure_dir(out_dir)
    width, height = cfg.size
    n_steps = cfg.num_inference_steps
    do_cfg = cfg.guidance_scale > 1.0

    # encode (blend が override してる場合は使う)
    if prompt_override_embeds is None:
        enc = encode_for_manual(pipe, device, width, height,
                                cfg.prompt, cfg.negative_prompt, do_cfg)
    else:
        enc = prompt_override_embeds
    prompt_embeds = enc["prompt_embeds"]
    added_cond = enc["added_cond_kwargs"]

    # scheduler / latents
    pipe.scheduler.set_timesteps(n_steps, device=device)
    timesteps = pipe.scheduler.timesteps
    unet = pipe.unet
    latent_dtype = next(unet.parameters()).dtype
    shape = (1, unet.config.in_channels, height // 8, width // 8)
    gen = torch.Generator(device="cpu" if device == "mps" else device).manual_seed(seed)
    latents = torch.randn(shape, generator=gen, dtype=latent_dtype).to(device)
    latents = latents * pipe.scheduler.init_noise_sigma

    # capture flag + container
    capture_flag = {"on": False}
    captured: list[dict[str, Any]] = []
    originals = install_recording_processors(unet, selected, capture_flag, captured)

    capture_steps_set = {max(0, min(n_steps - 1, int(round(f * (n_steps - 1)))))
                         for f in cfg.capture_step_fractions}
    capture_steps = sorted(capture_steps_set)
    decode_steps_set = set(select_trajectory_decode_steps(n_steps)) | capture_steps_set

    log(f"  [{label}] seed={seed} steps={n_steps} capture={capture_steps} "
        f"decode={sorted(decode_steps_set)}")

    step_captured: dict[int, list[dict[str, Any]]] = {}
    decoded_imgs: dict[int, Image.Image] = {}

    t0 = time.perf_counter()
    try:
        for i, t in enumerate(timesteps):
            latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
            capture_flag["on"] = i in capture_steps_set
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
                noise_pred = np_u + cfg.guidance_scale * (np_t - np_u)
            latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            if i in capture_steps_set and captured:
                step_captured[i] = [dict(c) for c in captured]

            if i in decode_steps_set:
                try:
                    decoded_imgs[i] = decode_latents_to_pil(pipe, latents)
                except Exception as e:
                    log(f"    decode failed at step {i}: {e}")

        final_img = decode_latents_to_pil(pipe, latents)
    finally:
        restore_processors(unet, originals)

    elapsed = time.perf_counter() - t0
    final_img.save(out_dir / "final.png")
    log(f"  [{label}] generation done in {elapsed:.1f} s")

    # トラジェクトリ grid
    plt = _get_plt()
    try:
        ordered = sorted(decoded_imgs.keys()) + ["final"]
        imgs: list[Image.Image] = []
        labels: list[str] = []
        for k in ordered:
            if k == "final":
                imgs.append(final_img)
                labels.append("final")
            else:
                imgs.append(decoded_imgs[k])
                labels.append(f"i={k}")
        n = len(imgs)
        cols = min(6, n)
        rows = math.ceil(n / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
        axes = np.atleast_1d(axes).flatten()
        for ax in axes:
            ax.axis("off")
        for ax, img, lbl in zip(axes, imgs, labels):
            ax.imshow(img)
            ax.set_title(lbl, fontsize=8)
        fig.suptitle(f"{label}: trajectory ({n_steps} steps, i=0 noisiest -> i={n_steps - 1} final)",
                     fontsize=9)
        fig.tight_layout()
        fig.savefig(out_dir / "trajectory_grid.png", dpi=110)
        plt.close(fig)
    except Exception as e:
        log(f"  [{label}] trajectory grid failed: {e}")

    # attention figure generation
    ttab = build_token_table(pipe.tokenizer, cfg.prompt, cfg.target_tokens)
    attn_dir = out_dir / "attention"
    ensure_dir(attn_dir)
    write_json(attn_dir / "token_table.json", {
        "prompt": cfg.prompt, "tokens_first30": ttab["tokens"][:30],
        "token_index": ttab["token_index"],
    })

    # heatmap_db: (step, name, token) -> 2D ndarray
    heatmap_db: dict[tuple, np.ndarray] = {}
    module_res_db: dict[str, tuple[int, int]] = {}
    for step_idx, recs in step_captured.items():
        for rec in recs:
            name = rec["name"]
            probs = rec["probs"]
            heads = rec["heads"]
            bhi = probs.shape[0]
            n_batch = bhi // heads
            p = probs.view(n_batch, heads, probs.shape[1], probs.shape[2])
            if n_batch == 2:
                p = p[1:2]
            p_avg = p.mean(dim=1)[0]
            Q = p_avg.shape[0]
            side = int(round(math.sqrt(Q)))
            if side * side != Q:
                continue
            module_res_db[name] = (side, side)
            for word, idxs in ttab["token_index"].items():
                if not idxs:
                    continue
                cols = p_avg[:, idxs].sum(dim=1)
                heatmap_db[(step_idx, name, word)] = cols.view(side, side).numpy()

    col_modules = [r for r in selected if r["name"] in module_res_db]
    row_tokens = [w for w in cfg.target_tokens if ttab["token_index"].get(w)]
    missing_tokens = [w for w in cfg.target_tokens if not ttab["token_index"].get(w)]
    if missing_tokens:
        log(f"  [{label}] target_tokens not found in prompt: {missing_tokens}")

    # 1) legacy_grid_step_*.png (token rows × module cols + decoded image right)
    legacy_dir = attn_dir / "legacy"
    ensure_dir(legacy_dir)
    for step_idx in sorted(step_captured.keys()):
        cols_n = len(col_modules) + 1
        rows_n = len(row_tokens)
        if cols_n <= 1 or rows_n == 0:
            continue
        fig, axes = plt.subplots(rows_n, cols_n, figsize=(cols_n * 1.7, rows_n * 1.7))
        axes = np.atleast_2d(axes)
        for ri, w in enumerate(row_tokens):
            for cj, mr in enumerate(col_modules):
                ax = axes[ri, cj]
                ax.axis("off")
                arr = heatmap_db.get((step_idx, mr["name"], w))
                if arr is None:
                    continue
                a = arr.copy()
                if a.max() > a.min():
                    a = (a - a.min()) / (a.max() - a.min())
                ax.imshow(a, cmap="inferno")
                if ri == 0:
                    res = module_res_db.get(mr["name"], (0, 0))
                    ax.set_title(f"{mr['stage']}{mr['stage_index']}\n"
                                 f"{res[0]}x{res[1]}\n{module_short_name(mr)}",
                                 fontsize=7)
                if cj == 0:
                    ax.text(-0.18, 0.5, w, transform=ax.transAxes,
                            fontsize=9, ha="right", va="center")
            ax_im = axes[ri, -1]
            ax_im.axis("off")
            if ri == 0:
                ax_im.set_title(f"decoded\nstep {step_idx}", fontsize=7)
            img_pick = decoded_imgs.get(step_idx, final_img)
            ax_im.imshow(img_pick)
        fig.suptitle(f"{label}: legacy-style cross-attention grid - step {step_idx} (per-cell norm)",
                     fontsize=9)
        fig.tight_layout()
        fig.savefig(legacy_dir / f"legacy_grid_step_{step_idx:03d}.png", dpi=130)
        plt.close(fig)

    # 2) per_token/token_<word>_unet_levels.png (per-token, rows = steps × cols = modules)
    per_token_dir = attn_dir / "per_token"
    ensure_dir(per_token_dir)
    for w in row_tokens:
        steps_avail = sorted(step_captured.keys())
        cols_n = len(col_modules) + 1
        rows_n = len(steps_avail)
        if cols_n <= 1 or rows_n == 0:
            continue
        fig, axes = plt.subplots(rows_n, cols_n, figsize=(cols_n * 1.8, rows_n * 1.8))
        axes = np.atleast_2d(axes)
        for ri, step_idx in enumerate(steps_avail):
            for cj, mr in enumerate(col_modules):
                ax = axes[ri, cj]
                ax.axis("off")
                arr = heatmap_db.get((step_idx, mr["name"], w))
                if arr is None:
                    continue
                a = arr.copy()
                if a.max() > a.min():
                    a = (a - a.min()) / (a.max() - a.min())
                ax.imshow(a, cmap="inferno")
                if ri == 0:
                    res = module_res_db.get(mr["name"], (0, 0))
                    ax.set_title(f"{mr['stage']}{mr['stage_index']}\n"
                                 f"{res[0]}x{res[1]}\n{module_short_name(mr)}",
                                 fontsize=7)
                if cj == 0:
                    ax.text(-0.18, 0.5, f"step {step_idx}",
                            transform=ax.transAxes,
                            fontsize=8, ha="right", va="center")
            ax_im = axes[ri, -1]
            ax_im.axis("off")
            img_pick = decoded_imgs.get(step_idx, final_img)
            ax_im.imshow(img_pick)
            if ri == 0:
                ax_im.set_title("decoded\nat step", fontsize=7)
        fig.suptitle(f"{label}: token = '{w}' across U-Net levels x steps", fontsize=9)
        fig.tight_layout()
        fig.savefig(per_token_dir / f"token_{w}_unet_levels.png", dpi=130)
        plt.close(fig)

    # 3) overlays/ : showpiece (final step, all modules, top-3 tokens) overlay on final image
    overlays_dir = attn_dir / "overlays"
    ensure_dir(overlays_dir)
    showpiece_tokens = row_tokens[:3]
    if step_captured and showpiece_tokens:
        step_for_overlay = max(step_captured.keys())
        W_img, H_img = final_img.size
        for mr in col_modules:
            res = module_res_db.get(mr["name"])
            if res is None:
                continue
            short = module_short_name(mr).replace(".", "_")
            for token in showpiece_tokens:
                arr = heatmap_db.get((step_for_overlay, mr["name"], token))
                if arr is None:
                    continue
                a = arr.copy()
                if a.max() > a.min():
                    a = (a - a.min()) / (a.max() - a.min())
                heat_img = Image.fromarray((a * 255).astype(np.uint8)).resize(
                    (W_img, H_img), Image.BILINEAR)
                fig, ax = plt.subplots(figsize=(3.6, 3.6))
                ax.imshow(final_img)
                ax.imshow(np.asarray(heat_img), cmap="inferno", alpha=0.45)
                ax.set_title(f"{label} step {step_for_overlay} / {token}\n"
                             f"{module_short_name(mr)} ({res[0]}x{res[1]})",
                             fontsize=7)
                ax.axis("off")
                fig.tight_layout()
                fig.savefig(overlays_dir / f"step_{step_for_overlay:03d}_{token}_{short}.png",
                            dpi=110)
                plt.close(fig)

    # per-seed index.md
    idx = [f"# {label}", "",
           f"- seed: {seed}", f"- generation: {elapsed:.1f} s",
           "", "## final", "", md_img("final.png", "final"), "",
           "## trajectory", "", md_img("trajectory_grid.png", "trajectory"), "",
           "## legacy-style attention grids (token x U-Net traversal order)", ""]
    for lf in sorted(legacy_dir.glob("*.png")):
        idx.append(f"### {lf.stem}")
        idx.append("")
        idx.append(md_img(f"attention/legacy/{lf.name}", lf.name))
        idx.append("")
    idx += ["## per-token grids (token x step x U-Net level)", ""]
    for lf in sorted(per_token_dir.glob("*.png")):
        idx.append(f"### {lf.stem}")
        idx.append("")
        idx.append(md_img(f"attention/per_token/{lf.name}", lf.name))
        idx.append("")
    idx += ["## overlays on final image", "",
            f"{len(list(overlays_dir.glob('*.png')))} overlays (subdir attention/overlays/)", ""]
    if missing_tokens:
        idx += ["## メモ", "",
                f"- prompt に含まれず heatmap が出ない target token: {missing_tokens}", ""]
    write_text(out_dir / "index.md", "\n".join(idx) + "\n")

    return {
        "seed": seed,
        "label": label,
        "elapsed_sec": round(elapsed, 2),
        "n_capture_steps": len(step_captured),
        "capture_steps": capture_steps,
        "resolutions_captured": sorted({tuple(v) for v in module_res_db.values()}),
        "tokens_found": row_tokens,
        "tokens_missing": missing_tokens,
        "n_legacy_figs": len(list(legacy_dir.glob("*.png"))),
        "n_per_token_figs": len(list(per_token_dir.glob("*.png"))),
        "n_overlay_figs": len(list(overlays_dir.glob("*.png"))),
    }


# ---------------------------------------------------------------------------
# blend (optional)
# ---------------------------------------------------------------------------

def run_blend(pipe, device: str, dtype: torch.dtype,
              cfg: ExploreConfig, selected: list[dict[str, Any]],
              out_dir: Path) -> dict[str, Any]:
    """blend_with の prompt と現在 prompt の embedding を線形和して 1 枚生成。"""
    bw = cfg.blend_with
    if not bw or "prompt" not in bw:
        return {"skipped": True, "reason": "blend_with not specified"}
    other_prompt = bw["prompt"]
    ratio = float(bw.get("ratio", 0.5))
    neg_same = bool(bw.get("negative_same", True))
    other_neg = cfg.negative_prompt if neg_same else bw.get("negative_prompt", "")
    seed = int(bw.get("seed", cfg.seeds[0]))

    log(f"  [blend] ratio={ratio} other_prompt='{other_prompt[:60]}...'")
    width, height = cfg.size
    do_cfg = cfg.guidance_scale > 1.0
    enc_a = encode_for_manual(pipe, device, width, height,
                              cfg.prompt, cfg.negative_prompt, do_cfg)
    enc_b = encode_for_manual(pipe, device, width, height,
                              other_prompt, other_neg, do_cfg)
    # 線形和
    pe_a = enc_a["prompt_embeds_raw"]
    pe_b = enc_b["prompt_embeds_raw"]
    npe_a = enc_a["negative_prompt_embeds_raw"]
    npe_b = enc_b["negative_prompt_embeds_raw"]
    pooled_a = enc_a["pooled_raw"]
    pooled_b = enc_b["pooled_raw"]
    npooled_a = enc_a["negative_pooled_raw"]
    npooled_b = enc_b["negative_pooled_raw"]

    pe_mix = ratio * pe_a + (1.0 - ratio) * pe_b
    npe_mix = ratio * npe_a + (1.0 - ratio) * npe_b
    pooled_mix = ratio * pooled_a + (1.0 - ratio) * pooled_b
    npooled_mix = ratio * npooled_a + (1.0 - ratio) * npooled_b

    add_time_ids = torch.tensor(
        [[height, width, 0, 0, height, width]],
        dtype=pe_mix.dtype, device=device,
    )
    if do_cfg:
        prompt_embeds = torch.cat([npe_mix, pe_mix], dim=0)
        pooled_in = torch.cat([npooled_mix, pooled_mix], dim=0)
        time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)
    else:
        prompt_embeds = pe_mix
        pooled_in = pooled_mix
        time_ids = add_time_ids
    override = {
        "prompt_embeds_raw": pe_mix, "negative_prompt_embeds_raw": npe_mix,
        "pooled_raw": pooled_mix, "negative_pooled_raw": npooled_mix,
        "prompt_embeds": prompt_embeds,
        "added_cond_kwargs": {"text_embeds": pooled_in, "time_ids": time_ids},
    }

    label = f"blend_{int(ratio * 100):d}_{int((1 - ratio) * 100):d}"
    blend_dir = out_dir / "blend"
    ensure_dir(blend_dir)
    result = run_one_seed(pipe, device, dtype, cfg, seed, selected,
                          blend_dir, prompt_override_embeds=override, label=label)

    write_json(blend_dir / "blend_metadata.json", {
        "ratio": ratio,
        "prompt_a": cfg.prompt,
        "prompt_b": other_prompt,
        "negative_a": cfg.negative_prompt,
        "negative_b": other_neg,
        "seed": seed,
        "_note": "embedding は ratio * A + (1-ratio) * B の単純線形和。pooled / negative も同様。",
    })
    return {"skipped": False, "ratio": ratio, "other_prompt": other_prompt,
            "result": result}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True,
                    help="JSON config (configs/09_explore/<slug>.json)")
    ap.add_argument("--output-root", type=str, default=None,
                    help="default: outputs/09_explore/")
    ap.add_argument("--force", action="store_true",
                    help="既存出力 dir があっても上書き (default: 自動で suffix)")
    return ap.parse_args()


def build_run_dir(slug: str, output_root: Path, force: bool) -> Path:
    run_dir = output_root / slug
    if run_dir.exists() and not force:
        for suffix in range(1, 100):
            cand = run_dir.with_name(slug + f"_{suffix:02d}")
            if not cand.exists():
                run_dir = cand
                break
    ensure_dir(run_dir)
    return run_dir


def main() -> int:
    args = parse_args()
    cfg_path = Path(args.config).resolve()
    if not cfg_path.exists():
        log(f"[error] config not found: {cfg_path}")
        return 2
    cfg = ExploreConfig.from_file(cfg_path)

    diffusers_cfg = load_diffusers_probe_config()
    model_cfg = get_model_config(diffusers_cfg, MODEL_KEY)
    model_id = model_cfg["model_id"]
    device, dtype = pick_device_and_dtype(model_cfg)

    output_root = Path(args.output_root) if args.output_root \
                  else project_root() / "outputs" / "09_explore"
    run_dir = build_run_dir(cfg.slug, output_root, args.force)

    log("=== 09 prompt explore ===")
    log(f"  config  : {cfg_path}")
    log(f"  run_dir : {run_dir}")
    log(f"  model   : {model_id}")
    log(f"  device  : {device} / dtype: {dtype}")
    log(f"  prompt  : {cfg.prompt}")
    log(f"  neg     : {cfg.negative_prompt}")
    log(f"  seeds   : {cfg.seeds}")
    log(f"  size    : {cfg.size}")
    log(f"  steps   : {cfg.num_inference_steps}, guidance: {cfg.guidance_scale}")
    log(f"  tokens  : {cfg.target_tokens}")
    log(f"  blend?  : {bool(cfg.blend_with)}")
    log("")

    started_at = now_iso()
    t_overall = time.perf_counter()

    # pipeline load
    log("[load] StableDiffusionXLPipeline ...")
    try:
        from diffusers import StableDiffusionXLPipeline
        t0 = time.perf_counter()
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id, torch_dtype=dtype, use_safetensors=True,
            variant="fp16" if dtype in (torch.float16, torch.bfloat16) else None,
        )
    except Exception as e:
        log(f"[fatal] load failed: {type(e).__name__}: {e}")
        for line in hint_for_load_error(e, model_id):
            log(line)
        traceback.print_exc()
        return 1
    pipe = pipe.to(device)
    if model_cfg.get("vae_fp32_override", False) and dtype != torch.float32:
        apply_vae_fp32_override(pipe)
    if model_cfg.get("enable_attention_slicing_on_mps", True) and device == "mps":
        pipe.enable_attention_slicing()
    load_elapsed = time.perf_counter() - t0
    log(f"[load] done in {load_elapsed:.1f} s")

    selected = select_representatives(pipe.unet)
    log(f"[attn] selected {len(selected)} representative cross-attention modules")

    # config を出力 dir にコピー (再現性)
    write_json(run_dir / "config.json", cfg.as_dict())

    # 各 seed で実行
    seed_results: list[dict[str, Any]] = []
    for seed in cfg.seeds:
        seed_dir = run_dir / f"seed_{seed:04d}"
        try:
            r = run_one_seed(pipe, device, dtype, cfg, seed, selected, seed_dir)
            seed_results.append(r)
        except Exception as e:
            log(f"[error] seed {seed} failed: {type(e).__name__}: {e}")
            traceback.print_exc()
            seed_results.append({"seed": seed, "error": f"{type(e).__name__}: {e}"})

    # blend (optional)
    blend_result: dict[str, Any] = {"skipped": True}
    if cfg.blend_with:
        try:
            blend_result = run_blend(pipe, device, dtype, cfg, selected, run_dir)
        except Exception as e:
            log(f"[error] blend failed: {type(e).__name__}: {e}")
            traceback.print_exc()
            blend_result = {"skipped": False, "error": f"{type(e).__name__}: {e}"}

    total_elapsed = time.perf_counter() - t_overall

    # top-level summary
    write_json(run_dir / "summary.json", {
        "script": SCRIPT_NAME,
        "config_slug": cfg.slug,
        "config_path": str(cfg_path),
        "model_id": model_id,
        "device": device, "dtype": str(dtype),
        "started_at": started_at, "finished_at": now_iso(),
        "load_elapsed_sec": round(load_elapsed, 2),
        "total_elapsed_sec": round(total_elapsed, 2),
        "config": cfg.as_dict(),
        "seed_results": seed_results,
        "blend_result": blend_result,
        "selected_modules": [r["name"] for r in selected],
    })

    # summary.md (画像埋め込み)
    sm: list[str] = [f"# 09 prompt explore — {cfg.slug}", ""]
    sm.append(f"> SDXL Base 1.0 (MPS / fp32) で `{cfg_path.name}` を実行。")
    sm.append("")
    sm.append("## prompt")
    sm.append("")
    sm.append(f"- **positive**: `{cfg.prompt}`")
    sm.append(f"- **negative**: `{cfg.negative_prompt}`")
    sm.append(f"- size {cfg.size[0]}x{cfg.size[1]}, steps {cfg.num_inference_steps}, "
              f"guidance {cfg.guidance_scale}")
    sm.append(f"- target tokens: {cfg.target_tokens}")
    sm.append(f"- seeds: {cfg.seeds}")
    sm.append("")
    sm.append("## 最終画像 (seed ごと)")
    sm.append("")
    for r in seed_results:
        if "error" in r:
            sm.append(f"### seed {r['seed']} — FAILED: {r['error']}")
            sm.append("")
            continue
        sd = f"seed_{r['seed']:04d}"
        sm.append(f"### {sd} ({r['elapsed_sec']} s)")
        sm.append("")
        sm.append(md_img(f"{sd}/final.png", f"{sd} final"))
        sm.append("")
        sm.append(f"- index: [{sd}/index.md]({sd}/index.md)")
        sm.append(f"- trajectory: [{sd}/trajectory_grid.png]({sd}/trajectory_grid.png)")
        sm.append("")
    if cfg.blend_with and not blend_result.get("skipped"):
        sm.append("## blend")
        sm.append("")
        if "error" in blend_result:
            sm.append(f"- FAILED: {blend_result['error']}")
        else:
            sm.append(f"- ratio = {blend_result['ratio']}")
            sm.append(f"- other prompt = `{blend_result['other_prompt']}`")
            sm.append("")
            sm.append(md_img("blend/final.png", "blend final"))
            sm.append("")
            sm.append("- [blend/index.md](blend/index.md)")
        sm.append("")
    # 各 seed の主要 attention 図 (seed 0 のみ embed)
    if seed_results and "error" not in seed_results[0]:
        sd0 = f"seed_{seed_results[0]['seed']:04d}"
        sm.append(f"## 主要 attention grid (代表として {sd0} を埋め込み)")
        sm.append("")
        legacy_pngs = sorted((run_dir / sd0 / "attention" / "legacy").glob("*.png"))
        for lf in legacy_pngs:
            rel = lf.relative_to(run_dir)
            sm.append(f"### {lf.stem}")
            sm.append("")
            sm.append(md_img(str(rel), lf.name))
            sm.append("")
        per_token_pngs = sorted((run_dir / sd0 / "attention" / "per_token").glob("*.png"))
        if per_token_pngs:
            sm.append("### per-token grid (代表 3 つ)")
            sm.append("")
            for lf in per_token_pngs[:3]:
                rel = lf.relative_to(run_dir)
                sm.append(md_img(str(rel), lf.name))
                sm.append("")
            if len(per_token_pngs) > 3:
                sm.append(f"他 {len(per_token_pngs) - 3} 枚は `{sd0}/attention/per_token/` 参照")
                sm.append("")

    sm.append("## 実行情報")
    sm.append("")
    sm.append(f"- pipeline load: {load_elapsed:.1f} s")
    sm.append(f"- total elapsed: {total_elapsed:.1f} s")
    sm.append(f"- started_at: {started_at}")
    sm.append(f"- config copy: [config.json](config.json)")
    sm.append("")
    if any(r.get("tokens_missing") for r in seed_results if "error" not in r):
        sm.append("## 警告")
        sm.append("")
        for r in seed_results:
            if "error" not in r and r.get("tokens_missing"):
                sm.append(f"- seed {r['seed']}: prompt に含まれず heatmap が空: {r['tokens_missing']}")
        sm.append("")

    write_text(run_dir / "summary.md", "\n".join(sm) + "\n")

    log("")
    log("=== done ===")
    log(f"  total elapsed: {total_elapsed:.1f} s")
    log(f"  summary.md   : {run_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
