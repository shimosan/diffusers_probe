# 10_quickgen.py
#
# 汎用 quickgen script。--models と --prompt-sets を CLI で与えて
# (model x prompt_set) の組み合わせを一気に生成する。
#
# 設定:
#   - models       : scripts/model_sets.json (default_model, model_sets.{key: {base_model_id, ...,
#                    scheduler?, loras?}})。LoRA stacking と scheduler 差し替えに対応。
#   - prompt sets  : scripts/prompt_sets.json (default_prompt_set, prompt_sets.{key: {prompt, negative_prompt, seed}})
#
# 注: 01-07 (legacy single-shot) は引き続き diffusers_probe.json の common+models を読む。
#     10_quickgen は完全に model_sets.json + prompt_sets.json で完結する。
#
# 出力 (1 run = 1 subdir):
#   outputs/10_quickgen/<run_label>/
#     config.json                       使った model list + prompt_set list + 環境メタ情報
#     run.log                           main の標準出力ミラー
#     grid.png                          全作品を 1 枚に並べた一覧 (列=model, 行=prompt_set)
#     <model_key>__<prompt_set_key>.png 個別画像
#     <model_key>__<prompt_set_key>.json 個別 summary
#
# CLI 例:
#   python scripts/10_quickgen.py                                                   # default_model x default_prompt_set
#   python scripts/10_quickgen.py --models sd15,sdxl_base --prompt-sets witch       # 2x1
#   python scripts/10_quickgen.py --all-models --prompt-sets witch                  # 全 model x 1 prompt
#   python scripts/10_quickgen.py --models sdxl_turbo --all-prompt-sets             # 1 model x 全 prompt
#   python scripts/10_quickgen.py --models sd15,sdxl_base --prompt-sets witch,astronaut_horse --run-label demo01
#
# 注意:
#   - 同じ model に対して複数 prompt_set を回す場合、pipeline は 1 回だけ load する
#     (model 切替コストが大きいので model の外ループ、prompt_set の内ループ)。
#   - guidance_scale = 0 の model (turbo, flux_schnell) には negative_prompt を渡さない。
#   - 失敗した (model, prompt_set) はスキップして run.log に記録、grid からも除く。
#   - run_label を省略すると "YYYYMMDD_HHMMSS" を自動生成。

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw, ImageFont

from common import (
    apply_vae_fp32_override,
    get_model_set,
    get_prompt_set,
    hint_for_load_error,
    load_model_sets,
    load_prompt_sets,
    pick_device_and_dtype,
    project_root,
    resolve_quickgen_outputs_dir,
)

SCRIPT_NAME = "10_quickgen.py"


# ---------- CLI ----------

def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="10_quickgen.py",
        description="Generic quickgen: run (model x prompt_set) combinations from diffusers_probe.json + prompt_sets.json",
    )
    p.add_argument(
        "--models",
        type=str,
        default=None,
        help="comma-separated model keys (e.g. sd15,sdxl_base). default: cfg.default_model",
    )
    p.add_argument(
        "--prompt-sets",
        type=str,
        default=None,
        help="comma-separated prompt_set keys (e.g. witch,astronaut_horse). default: prompt_sets.default_prompt_set",
    )
    p.add_argument("--all-models", action="store_true", help="use all model keys in diffusers_probe.json")
    p.add_argument("--all-prompt-sets", action="store_true", help="use all prompt_set keys in prompt_sets.json")
    p.add_argument(
        "--run-label",
        type=str,
        default=None,
        help="subdir name under outputs/10_quickgen/. default: timestamp.",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="show available model keys / prompt_set keys and exit (no generation).",
    )
    return p.parse_args(argv)


def resolve_keys(args: argparse.Namespace, ms_cfg: dict[str, Any], ps_cfg: dict[str, Any]) -> tuple[list[str], list[str]]:
    all_model_keys = list(ms_cfg.get("model_sets", {}).keys())
    all_ps_keys = list(ps_cfg.get("prompt_sets", {}).keys())

    if args.all_models:
        model_keys = all_model_keys
    elif args.models:
        model_keys = [k.strip() for k in args.models.split(",") if k.strip()]
    else:
        default = ms_cfg.get("default_model")
        if not default:
            raise SystemExit("default_model not set in model_sets.json and --models not given")
        model_keys = [default]

    if args.all_prompt_sets:
        ps_keys = all_ps_keys
    elif args.prompt_sets:
        ps_keys = [k.strip() for k in args.prompt_sets.split(",") if k.strip()]
    else:
        default = ps_cfg.get("default_prompt_set")
        if not default:
            raise SystemExit("default_prompt_set not set in prompt_sets.json and --prompt-sets not given")
        ps_keys = [default]

    # validate
    for k in model_keys:
        if k not in all_model_keys:
            raise SystemExit(f"unknown model key: {k}. available: {', '.join(all_model_keys)}")
    for k in ps_keys:
        if k not in all_ps_keys:
            raise SystemExit(f"unknown prompt_set key: {k}. available: {', '.join(all_ps_keys)}")

    return model_keys, ps_keys


# ---------- logging (stdout mirror) ----------

class TeeLogger:
    """sys.stdout に書きつつ file にもミラーする最小 logger。flush もちゃんと両方に。"""

    def __init__(self, log_path: Path) -> None:
        self.terminal = sys.stdout
        self.log_path = log_path
        self.log = log_path.open("w", encoding="utf-8")

    def write(self, message: str) -> int:
        self.terminal.write(message)
        self.log.write(message)
        return len(message)

    def flush(self) -> None:
        self.terminal.flush()
        self.log.flush()

    def close(self) -> None:
        try:
            self.log.flush()
            self.log.close()
        except Exception:
            pass


# ---------- pipeline build / generate ----------

def apply_scheduler_override(pipe, scheduler_cfg: dict[str, Any]) -> None:
    """model_set.scheduler.{class, config_overrides} を pipe.scheduler に適用する。"""
    import diffusers

    cls_name = scheduler_cfg["class"]
    sched_cls = getattr(diffusers, cls_name, None)
    if sched_cls is None:
        raise AttributeError(f"scheduler class '{cls_name}' not found in diffusers namespace")
    overrides = scheduler_cfg.get("config_overrides", {}) or {}
    # from_config(config, **overrides) で diffusers の慣習に従い override する
    pipe.scheduler = sched_cls.from_config(pipe.scheduler.config, **overrides)
    extra = f" with overrides {overrides}" if overrides else ""
    print(f"[sched] scheduler -> {cls_name}{extra}")


def apply_loras(pipe, loras: list[dict[str, Any]]) -> None:
    """loras list を pipe に load + set_adapters + fuse する。複数 LoRA は同時 stack。"""
    if not loras:
        return
    adapter_names: list[str] = []
    weights: list[float] = []
    for i, lora in enumerate(loras):
        name = lora.get("name") or f"lora_{i}"
        repo = lora["repo"]
        weight_name = lora["weight_name"]
        scale = float(lora.get("scale", 1.0))
        print(f"[lora] load {name}: {repo} / {weight_name} (scale={scale})")
        pipe.load_lora_weights(repo, weight_name=weight_name, adapter_name=name)
        adapter_names.append(name)
        weights.append(scale)
    pipe.set_adapters(adapter_names, adapter_weights=weights)
    pipe.fuse_lora()
    print(f"[lora] fused {len(adapter_names)} adapter(s): {adapter_names}")


def build_pipeline(base_model_id: str, model_cfg: dict[str, Any], device: str, dtype: torch.dtype):
    """base_model_id から pipeline を組み立てて device に乗せ、optional な scheduler / LoRA / その他を適用する。"""
    # AutoPipeline でだいたいの model に対応。失敗時は DiffusionPipeline (model_index.json 経由) で再試行。
    from diffusers import AutoPipelineForText2Image, DiffusionPipeline

    load_kwargs: dict[str, Any] = {"torch_dtype": dtype, "use_safetensors": True}
    try:
        pipe = AutoPipelineForText2Image.from_pretrained(base_model_id, **load_kwargs)
    except Exception as e_auto:
        print(f"[load] AutoPipeline 失敗 ({type(e_auto).__name__}: {e_auto}). DiffusionPipeline で再試行")
        pipe = DiffusionPipeline.from_pretrained(base_model_id, **load_kwargs)

    pipe = pipe.to(device)

    # scheduler 差し替え (LoRA load の前に終わらせる: 一部 LoRA は scheduler config に依存)
    if "scheduler" in model_cfg:
        apply_scheduler_override(pipe, model_cfg["scheduler"])

    vae_fp32_override = bool(model_cfg.get("vae_fp32_override", False))
    if vae_fp32_override and dtype != torch.float32 and hasattr(pipe, "vae"):
        apply_vae_fp32_override(pipe)
        print("[vae ] vae_fp32_override 適用")

    if bool(model_cfg.get("enable_attention_slicing_on_mps", False)) and device == "mps":
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()
            print("[mps ] enable_attention_slicing() 有効化")

    if bool(model_cfg.get("enable_model_cpu_offload", False)):
        if hasattr(pipe, "enable_model_cpu_offload"):
            pipe.enable_model_cpu_offload()
            print("[mem ] enable_model_cpu_offload() 有効化")

    # LoRA は最後に: pipe を完全に組み終わってから fuse する
    apply_loras(pipe, list(model_cfg.get("loras", []) or []))

    return pipe


def generate(pipe, model_cfg: dict[str, Any], prompt: str, negative_prompt: str, seed: int, device: str) -> Image.Image:
    width = int(model_cfg.get("width", 1024))
    height = int(model_cfg.get("height", 1024))
    steps = int(model_cfg["num_inference_steps"])
    guidance = float(model_cfg["guidance_scale"])

    generator = torch.Generator(device="cpu" if device == "mps" else device).manual_seed(seed)

    call_kwargs: dict[str, Any] = dict(
        prompt=prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator,
    )
    if guidance > 0 and negative_prompt:
        call_kwargs["negative_prompt"] = negative_prompt
    if "max_sequence_length" in model_cfg:  # FLUX 系
        call_kwargs["max_sequence_length"] = int(model_cfg["max_sequence_length"])
    if "true_cfg_scale" in model_cfg:  # Qwen-Image
        call_kwargs["true_cfg_scale"] = float(model_cfg["true_cfg_scale"])

    result = pipe(**call_kwargs)
    return result.images[0]  # type: ignore[attr-defined]


# ---------- grid composition ----------

def compose_grid(
    cells: dict[tuple[str, str], Image.Image],
    model_keys: list[str],
    ps_keys: list[str],
    cell_size: int = 512,
    label_height: int = 28,
) -> Image.Image:
    """rows = prompt_sets, cols = models。各セルは同サイズに resize + 上部に "<model>__<prompt_set>" ラベル。
    欠損セル (生成失敗) は薄いグレー塗りでスキップ表示。
    """
    cols = len(model_keys)
    rows = len(ps_keys)
    canvas_w = cols * cell_size
    canvas_h = rows * (cell_size + label_height)
    canvas = Image.new("RGB", (canvas_w, canvas_h), (240, 240, 240))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size=14)
    except Exception:
        font = ImageFont.load_default()

    for r, ps_key in enumerate(ps_keys):
        for c, m_key in enumerate(model_keys):
            x0 = c * cell_size
            y0 = r * (cell_size + label_height)
            label = f"{m_key} | {ps_key}"
            draw.rectangle((x0, y0, x0 + cell_size, y0 + label_height), fill=(40, 40, 40))
            draw.text((x0 + 6, y0 + 6), label, fill=(255, 255, 255), font=font)
            img = cells.get((m_key, ps_key))
            if img is None:
                draw.text((x0 + 6, y0 + label_height + 6), "(skipped)", fill=(180, 60, 60), font=font)
                continue
            thumb = img.resize((cell_size, cell_size), Image.LANCZOS)
            canvas.paste(thumb, (x0, y0 + label_height))
    return canvas


# ---------- main ----------

def main(argv: list[str]) -> int:
    args = parse_args(argv)
    ms_cfg = load_model_sets()
    ps_cfg = load_prompt_sets()

    if args.list:
        print("model_sets:")
        for k, v in ms_cfg.get("model_sets", {}).items():
            base = v.get("base_model_id", "?")
            extras: list[str] = []
            if "loras" in v and v["loras"]:
                names = [str(l.get("name") or l.get("weight_name", "?")) for l in v["loras"]]
                extras.append(f"+lora:{','.join(names)}")
            if "scheduler" in v:
                extras.append(f"sched:{v['scheduler'].get('class', '?')}")
            tail = f"  [{' '.join(extras)}]" if extras else ""
            print(f"  {k:<26}  {base}{tail}")
        print("\nprompt_sets:")
        for k, v in ps_cfg.get("prompt_sets", {}).items():
            preview = v.get("prompt", "")[:60]
            print(f"  {k:<22}  {preview}{'...' if len(v.get('prompt', '')) > 60 else ''}")
        return 0

    model_keys, ps_keys = resolve_keys(args, ms_cfg, ps_cfg)

    run_label = args.run_label or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = resolve_quickgen_outputs_dir(run_label)

    # logger を立てて以降の print を run.log にもミラー
    logger = TeeLogger(out_dir / "run.log")
    sys.stdout = logger  # type: ignore[assignment]

    try:
        print("=== diffusers_probe / 10_quickgen ===")
        print(f"  run_label   : {run_label}")
        print(f"  out_dir     : {out_dir}")
        print(f"  models      : {model_keys}")
        print(f"  prompt_sets : {ps_keys}")
        print()

        cells: dict[tuple[str, str], Image.Image] = {}
        results: list[dict[str, Any]] = []
        wall_t0 = time.perf_counter()

        for m_key in model_keys:
            _, model_cfg = get_model_set(ms_cfg, m_key)
            base_model_id = model_cfg["base_model_id"]
            device, dtype = pick_device_and_dtype(model_cfg)

            print(f"---- model: {m_key} (base={base_model_id}) | device={device} dtype={dtype} ----")
            t_load0 = time.perf_counter()
            try:
                pipe = build_pipeline(base_model_id, model_cfg, device, dtype)
            except Exception as e:
                print(f"[error] {m_key} の load に失敗: {type(e).__name__}: {e}")
                for line in hint_for_load_error(e, base_model_id):
                    print(line)
                traceback.print_exc()
                continue
            load_elapsed = time.perf_counter() - t_load0
            print(f"[load] done in {load_elapsed:.2f} s")

            for ps_key in ps_keys:
                _, ps = get_prompt_set(ps_cfg, ps_key)
                prompt = ps["prompt"]
                negative_prompt = ps.get("negative_prompt", "")
                seed = int(ps.get("seed", 42))
                print(f"  [gen ] {m_key} x {ps_key} (seed={seed})")
                print(f"         prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
                t_gen0 = time.perf_counter()
                try:
                    image = generate(pipe, model_cfg, prompt, negative_prompt, seed, device)
                except Exception as e:
                    print(f"  [error] 生成失敗 {m_key} x {ps_key}: {type(e).__name__}: {e}")
                    traceback.print_exc()
                    continue
                gen_elapsed = time.perf_counter() - t_gen0
                print(f"  [gen ] done in {gen_elapsed:.2f} s")

                basename = f"{m_key}__{ps_key}"
                png_path = out_dir / f"{basename}.png"
                json_path = out_dir / f"{basename}.json"
                image.save(png_path)
                summary = {
                    "script": SCRIPT_NAME,
                    "run_label": run_label,
                    "model_key": m_key,
                    "base_model_id": base_model_id,
                    "prompt_set_key": ps_key,
                    "device": device,
                    "dtype": str(dtype),
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "width": int(model_cfg.get("width", 1024)),
                    "height": int(model_cfg.get("height", 1024)),
                    "num_inference_steps": int(model_cfg["num_inference_steps"]),
                    "guidance_scale": float(model_cfg["guidance_scale"]),
                    "seed": seed,
                    "load_time_sec": round(load_elapsed, 3),
                    "generation_time_sec": round(gen_elapsed, 3),
                    "loras": model_cfg.get("loras", []),
                    "scheduler": model_cfg.get("scheduler"),
                    "image_path": str(png_path.relative_to(project_root())),
                }
                with json_path.open("w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=2, ensure_ascii=False)
                print(f"  [save] {png_path.name} + {json_path.name}")

                cells[(m_key, ps_key)] = image
                results.append(summary)

            # 1 model 終わり: pipeline を解放してメモリを返す (MPS は cache を持つので明示 free)
            del pipe
            if device == "mps":
                try:
                    torch.mps.empty_cache()  # type: ignore[attr-defined]
                except Exception:
                    pass
            elif device == "cuda":
                torch.cuda.empty_cache()

        # grid
        if cells:
            print()
            print("[grid] composing grid.png ...")
            grid = compose_grid(cells, model_keys, ps_keys)
            grid.save(out_dir / "grid.png")
            print(f"[grid] saved -> {out_dir / 'grid.png'}")
        else:
            print("[grid] 全 cell が生成失敗。grid.png はスキップ。")

        # run config
        wall_elapsed = time.perf_counter() - wall_t0
        run_config = {
            "script": SCRIPT_NAME,
            "run_label": run_label,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "model_keys": model_keys,
            "prompt_set_keys": ps_keys,
            "wall_time_sec": round(wall_elapsed, 3),
            "successful": len(results),
            "expected": len(model_keys) * len(ps_keys),
            "results": results,
        }
        with (out_dir / "config.json").open("w", encoding="utf-8") as f:
            json.dump(run_config, f, indent=2, ensure_ascii=False)

        print()
        print(f"=== done: {len(results)}/{len(model_keys) * len(ps_keys)} 件成功, wall {wall_elapsed:.2f} s ===")
        return 0
    finally:
        sys.stdout = logger.terminal
        logger.close()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
