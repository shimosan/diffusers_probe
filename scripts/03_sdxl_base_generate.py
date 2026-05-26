# 03_sdxl_base_generate.py
#
# Stable Diffusion XL Base 1.0 の生成 script。
#
# 設定: scripts/diffusers_probe.json
#   common.{prompt, negative_prompt, seed, width, height}
#   models.sdxl_base.{model_id, num_inference_steps, guidance_scale, dtype, ...}
#
# 注意:
#   SDXL の fp16 + MPS では VAE decode が黒画像になることがある。vae_fp32_override = true 既定。
#
# 出力:
#   outputs/00-07_legacy/sdxl_base_generate.png
#   outputs/00-07_legacy/sdxl_base_generate_summary.json
#   outputs/00-07_legacy/sdxl_base_generate.txt

from __future__ import annotations

import time
import traceback

import torch

from common import (
    apply_vae_fp32_override,
    build_summary_base,
    get_common,
    get_model_config,
    hint_for_load_error,
    load_config,
    pick_device_and_dtype,
    write_outputs,
)

MODEL_KEY = "sdxl_base"
SCRIPT_NAME = "03_sdxl_base_generate.py"
OUTPUT_BASENAME = "sdxl_base_generate"


def main() -> int:
    cfg = load_config()
    common = get_common(cfg)
    model_cfg = get_model_config(cfg, MODEL_KEY)

    model_id: str = model_cfg["model_id"]
    width = int(common.get("width", model_cfg.get("width", 1024)))
    height = int(common.get("height", model_cfg.get("height", 1024)))
    num_inference_steps = int(model_cfg["num_inference_steps"])
    guidance_scale = float(model_cfg["guidance_scale"])

    prompt = common["prompt"]
    negative_prompt = common.get("negative_prompt", "")
    seed = int(common["seed"])

    attn_slicing_mps = bool(model_cfg.get("enable_attention_slicing_on_mps", True))
    vae_fp32_override = bool(model_cfg.get("vae_fp32_override", True))

    device, dtype = pick_device_and_dtype(model_cfg)

    print("=== diffusers_probe / SDXL Base 1.0 generate ===")
    print(f"  model_id        : {model_id}")
    print(f"  device          : {device}")
    print(f"  dtype           : {dtype}")
    print(f"  vae_fp32_override: {vae_fp32_override}")
    print(f"  attn_slicing(MPS): {attn_slicing_mps}")
    print(f"  size            : {width} x {height}")
    print(f"  steps           : {num_inference_steps}")
    print(f"  guidance        : {guidance_scale}")
    print(f"  seed            : {seed}")
    print(f"  prompt          : {prompt}")
    if negative_prompt:
        print(f"  neg prompt      : {negative_prompt}")
    print()
    print("[note] 初回実行は Hugging Face cache への download 時間が混ざります。")
    print()

    try:
        from diffusers import StableDiffusionXLPipeline
    except Exception as e:
        print(f"[error] diffusers の import に失敗: {type(e).__name__}: {e}")
        return 1

    print("[load] StableDiffusionXLPipeline.from_pretrained ...")
    t0 = time.perf_counter()
    try:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            use_safetensors=True,
            variant="fp16" if dtype in (torch.float16, torch.bfloat16) else None,
        )
    except Exception as e:
        print(f"[error] model load に失敗: {type(e).__name__}: {e}")
        for line in hint_for_load_error(e, model_id):
            print(line)
        traceback.print_exc()
        return 1

    pipe = pipe.to(device)

    if vae_fp32_override and dtype != torch.float32:
        apply_vae_fp32_override(pipe)
        print("[vae ] vae_fp32_override 適用")

    if attn_slicing_mps and device == "mps":
        pipe.enable_attention_slicing()
        print("[mps ] enable_attention_slicing() 有効化")

    load_elapsed = time.perf_counter() - t0
    print(f"[load] done in {load_elapsed:.2f} s (初回は download 時間を含む)")

    generator = torch.Generator(device="cpu" if device == "mps" else device).manual_seed(seed)

    print("[gen ] running pipeline ...")
    t1 = time.perf_counter()
    try:
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
    except Exception as e:
        print(f"[error] 生成に失敗: {type(e).__name__}: {e}")
        traceback.print_exc()
        return 1
    gen_elapsed = time.perf_counter() - t1
    image = result.images[0]  # type: ignore[attr-defined]
    print(f"[gen ] done in {gen_elapsed:.2f} s")

    total_elapsed = time.perf_counter() - t0

    extras = {
        "vae_fp32_override": vae_fp32_override,
        "attention_slicing_enabled": attn_slicing_mps and device == "mps",
    }
    summary = build_summary_base(
        script=SCRIPT_NAME,
        cfg=cfg,
        model_key=MODEL_KEY,
        model_id=model_id,
        device=device,
        dtype=dtype,
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        seed=seed,
        load_time_sec=load_elapsed,
        generation_time_sec=gen_elapsed,
        total_time_sec=total_elapsed,
        image_relpath=f"outputs/00-07_legacy/{OUTPUT_BASENAME}.png",
        extras=extras,
    )

    png_path, json_path, txt_path = write_outputs(
        image=image,
        summary=summary,
        output_basename=OUTPUT_BASENAME,
        prompt=prompt,
        negative_prompt=negative_prompt,
    )
    print(f"[save] image   -> {png_path}")
    print(f"[save] summary -> {json_path}")
    print(f"[save] prompt  -> {txt_path}")

    print()
    print("=== timing ===")
    print(f"  load time      : {load_elapsed:.2f} s (初回 download 含む)")
    print(f"  generation time: {gen_elapsed:.2f} s")
    print(f"  total time     : {total_elapsed:.2f} s")
    print()
    print("=== done ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
