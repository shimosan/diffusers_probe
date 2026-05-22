# 05_flux1_schnell_generate.py
#
# FLUX.1-schnell の生成 script。few-step distilled FLUX (Apache 2.0 ライセンス、ただし HF gated)。
# 初回利用には https://huggingface.co/black-forest-labs/FLUX.1-schnell の規約承認と
# `hf auth login` が必要なことに注意。
#
# 設定: scripts/diffusers_probe.json
#   common.{prompt, negative_prompt, seed, width, height}
#   models.flux_schnell.{model_id, num_inference_steps=4, guidance_scale=0.0,
#                         max_sequence_length, bfloat16, enable_model_cpu_offload, fallback}
#
# schnell は CFG なし (guidance_scale=0.0)、negative_prompt も使用しない。
# bfloat16 が推奨 (FLUX のリリースノートに準拠)。
#
# 出力:
#   outputs/flux_schnell_generate.png
#   outputs/flux_schnell_generate_summary.json
#   outputs/flux_schnell_generate.txt

from __future__ import annotations

import time
import traceback

import torch

from common import (
    build_summary_base,
    get_common,
    get_model_config,
    hint_for_load_error,
    load_config,
    pick_device_and_dtype,
    write_outputs,
)

MODEL_KEY = "flux_schnell"
SCRIPT_NAME = "05_flux1_schnell_generate.py"
OUTPUT_BASENAME = "flux_schnell_generate"


def main() -> int:
    cfg = load_config()
    common = get_common(cfg)
    model_cfg = get_model_config(cfg, MODEL_KEY)

    model_id = model_cfg["model_id"]
    width = int(common.get("width", model_cfg.get("width", 1024)))
    height = int(common.get("height", model_cfg.get("height", 1024)))
    num_inference_steps = int(model_cfg["num_inference_steps"])
    guidance_scale = float(model_cfg["guidance_scale"])
    max_sequence_length = int(model_cfg.get("max_sequence_length", 256))

    prompt = common["prompt"]
    seed = int(common["seed"])

    attn_slicing_mps = bool(model_cfg.get("enable_attention_slicing_on_mps", True))
    cpu_offload = bool(model_cfg.get("enable_model_cpu_offload", False))

    device, dtype = pick_device_and_dtype(model_cfg)

    print("=== diffusers_probe / FLUX.1-schnell generate ===")
    print(f"  model_id        : {model_id}")
    print(f"  device          : {device}")
    print(f"  dtype           : {dtype}")
    print(f"  attn_slicing(MPS): {attn_slicing_mps}")
    print(f"  cpu_offload     : {cpu_offload}")
    print(f"  size            : {width} x {height}")
    print(f"  steps           : {num_inference_steps} (schnell は 1-4 steps が前提)")
    print(f"  guidance        : {guidance_scale} (schnell は CFG なし)")
    print(f"  max_seq_len     : {max_sequence_length}")
    print(f"  seed            : {seed}")
    print(f"  prompt          : {prompt}")
    print()
    print("[note] FLUX.1-schnell は HF gated repo です。事前に下記を完了してください:")
    print("       1) https://huggingface.co/black-forest-labs/FLUX.1-schnell で利用条件を承認")
    print("       2) ターミナルで `hf auth login` を実行")
    print("[note] 初回 download は数 GB ~ 十数 GB あり、download 時間が混ざります。")
    print()

    try:
        from diffusers import FluxPipeline  # pyright: ignore[reportPrivateImportUsage]
    except Exception as e:
        print(f"[error] diffusers の import に失敗: {type(e).__name__}: {e}")
        print("[hint ] FLUX には比較的新しい diffusers が必要です (>= 0.30)。")
        return 1

    print("[load] FluxPipeline.from_pretrained ...")
    t0 = time.perf_counter()
    try:
        pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=dtype)
    except Exception as e:
        print(f"[error] model load に失敗: {type(e).__name__}: {e}")
        for line in hint_for_load_error(e, model_id):
            print(line)
        traceback.print_exc()
        return 1

    if cpu_offload:
        try:
            pipe.enable_model_cpu_offload()
            print("[mem ] enable_model_cpu_offload() 有効化")
        except Exception as e:
            print(f"[warn] cpu_offload 失敗、to(device) にフォールバック: {e}")
            pipe = pipe.to(device)
    else:
        pipe = pipe.to(device)

    if attn_slicing_mps and device == "mps":
        try:
            pipe.enable_attention_slicing()
            print("[mps ] enable_attention_slicing() 有効化")
        except Exception as e:
            print(f"[warn] enable_attention_slicing() 失敗: {e}")

    load_elapsed = time.perf_counter() - t0
    print(f"[load] done in {load_elapsed:.2f} s (初回は download 時間を含む)")

    generator = torch.Generator(device="cpu" if device == "mps" else device).manual_seed(seed)

    def try_generate(_w: int, _h: int, _steps: int):
        print(f"[gen ] running pipeline ... ({_w}x{_h}, steps={_steps})")
        _t = time.perf_counter()
        _r = pipe(
            prompt=prompt,
            width=_w,
            height=_h,
            num_inference_steps=_steps,
            guidance_scale=guidance_scale,
            max_sequence_length=max_sequence_length,
            generator=generator,
        )
        return _r, time.perf_counter() - _t

    used_fallback = False
    try:
        result, gen_elapsed = try_generate(width, height, num_inference_steps)
    except Exception as e:
        print(f"[error] 生成に失敗: {type(e).__name__}: {e}")
        fb = model_cfg.get("fallback")
        if fb:
            print(f"[fallback] 低解像度/少ステップで再試行: {fb}")
            try:
                fb_w = int(fb["width"])
                fb_h = int(fb["height"])
                fb_steps = int(fb["num_inference_steps"])
                result, gen_elapsed = try_generate(fb_w, fb_h, fb_steps)
                width, height, num_inference_steps = fb_w, fb_h, fb_steps
                used_fallback = True
            except Exception as e2:
                print(f"[error] fallback も失敗: {type(e2).__name__}: {e2}")
                print("[gpu-server-candidate] FLUX.1-schnell は MPS では実用厳しい可能性 → GPU サーバー候補")
                traceback.print_exc()
                return 1
        else:
            traceback.print_exc()
            return 1

    image = result.images[0]  # type: ignore[attr-defined]
    print(f"[gen ] done in {gen_elapsed:.2f} s")

    total_elapsed = time.perf_counter() - t0

    extras = {
        "max_sequence_length": max_sequence_length,
        "enable_model_cpu_offload": cpu_offload,
        "attention_slicing_enabled": attn_slicing_mps and device == "mps",
        "used_fallback": used_fallback,
        "note": "FLUX.1-schnell: few-step distilled, no CFG, no negative_prompt",
    }
    summary = build_summary_base(
        script=SCRIPT_NAME,
        cfg=cfg,
        model_key=MODEL_KEY,
        model_id=model_id,
        device=device,
        dtype=dtype,
        prompt=prompt,
        negative_prompt="",
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        seed=seed,
        load_time_sec=load_elapsed,
        generation_time_sec=gen_elapsed,
        total_time_sec=total_elapsed,
        image_relpath=f"outputs/{OUTPUT_BASENAME}.png",
        extras=extras,
    )

    png_path, json_path, txt_path = write_outputs(
        image=image,
        summary=summary,
        output_basename=OUTPUT_BASENAME,
        prompt=prompt,
        negative_prompt="",
    )
    print(f"[save] image   -> {png_path}")
    print(f"[save] summary -> {json_path}")
    print(f"[save] prompt  -> {txt_path}")

    print()
    print("=== timing ===")
    print(f"  load time      : {load_elapsed:.2f} s (初回 download 含む)")
    print(f"  generation time: {gen_elapsed:.2f} s")
    print(f"  total time     : {total_elapsed:.2f} s")
    if used_fallback:
        print("  [!] fallback (低解像度/少 step) を使用しました")
    print()
    print("=== done ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
