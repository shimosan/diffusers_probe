# 02_sd15_generate.py
#
# Stable Diffusion 1.5 を 2 パスで生成する script。
#
#   pass A : 1024x1024 / float32  (他モデルと同じ 1024 解像度で比較するため。fp16 だと UNet/VAE が NaN になる)
#   pass B : 512x512  / float16   (SD1.5 のネイティブ解像度・実用設定での参考画像)
#
# 同じ prompt / negative_prompt / seed を使うので、構図破綻と本来の品質の対比が見える。
#
# 出力 (2 セット):
#   outputs/sd15_generate_1024_fp32.png / _summary.json / .txt
#   outputs/sd15_generate_512_fp16.png  / _summary.json / .txt

from __future__ import annotations

import time
import traceback
from typing import cast

import torch

from common import (
    build_summary_base,
    get_common,
    get_model_config,
    hint_for_load_error,
    load_config,
    write_outputs,
)

MODEL_KEY = "sd15"
SCRIPT_NAME = "02_sd15_generate.py"


def load_pipeline(model_id: str, dtype: torch.dtype, device: str, *, attention_slicing: bool):
    from diffusers import StableDiffusionPipeline  # pyright: ignore[reportPrivateImportUsage]

    print(f"[load] StableDiffusionPipeline.from_pretrained (dtype={dtype}) ...")
    t0 = time.perf_counter()
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe = pipe.to(device)
    pipe.safety_checker = None
    if attention_slicing and device == "mps":
        pipe.enable_attention_slicing()
        print("[mps ] enable_attention_slicing() 有効化")
    elapsed = time.perf_counter() - t0
    print(f"[load] done in {elapsed:.2f} s")
    return pipe, elapsed


def run_pass(
    *,
    cfg: dict,
    pipe,
    pass_name: str,
    output_basename: str,
    model_id: str,
    device: str,
    dtype: torch.dtype,
    width: int,
    height: int,
    num_inference_steps: int,
    guidance_scale: float,
    prompt: str,
    negative_prompt: str,
    seed: int,
    load_time_sec: float,
    attention_slicing_on: bool,
) -> float:
    from diffusers.pipelines.stable_diffusion.pipeline_output import (
        StableDiffusionPipelineOutput,
    )

    print(f"=== pass {pass_name} : {width}x{height} / {dtype} ===")
    generator = torch.Generator(device="cpu" if device == "mps" else device).manual_seed(seed)

    t1 = time.perf_counter()
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt or None,
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )
    gen_elapsed = time.perf_counter() - t1
    result = cast(StableDiffusionPipelineOutput, result)
    image = result.images[0]
    nsfw_flags = result.nsfw_content_detected
    nsfw_detected = bool(nsfw_flags[0]) if (nsfw_flags is not None and len(nsfw_flags) > 0) else False
    print(f"[gen ] pass {pass_name} done in {gen_elapsed:.2f} s")
    if nsfw_detected:
        print("[warn] safety_checker NSFW 検出 → 黒塗り画像になります")

    extras = {
        "pass": pass_name,
        "safety_checker_enabled": False,
        "attention_slicing_enabled": attention_slicing_on,
        "nsfw_content_detected": nsfw_detected,
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
        load_time_sec=load_time_sec,
        generation_time_sec=gen_elapsed,
        total_time_sec=load_time_sec + gen_elapsed,
        image_relpath=f"outputs/{output_basename}.png",
        extras=extras,
    )

    png_path, json_path, txt_path = write_outputs(
        image=image,
        summary=summary,
        output_basename=output_basename,
        prompt=prompt,
        negative_prompt=negative_prompt,
    )
    print(f"[save] image   -> {png_path}")
    print(f"[save] summary -> {json_path}")
    print(f"[save] prompt  -> {txt_path}")
    return gen_elapsed


def main() -> int:
    cfg = load_config()
    common = get_common(cfg)
    model_cfg = get_model_config(cfg, MODEL_KEY)

    model_id: str = model_cfg["model_id"]
    num_inference_steps: int = int(model_cfg["num_inference_steps"])
    guidance_scale: float = float(model_cfg["guidance_scale"])

    prompt: str = common["prompt"]
    negative_prompt: str = common.get("negative_prompt", "")
    seed: int = int(common["seed"])

    if torch.cuda.is_available():
        device = "cuda"
    else:
        mps_backend = getattr(torch.backends, "mps", None)
        device = "mps" if (mps_backend is not None and mps_backend.is_available()) else "cpu"

    attention_slicing = bool(model_cfg.get("enable_attention_slicing_on_mps", True))

    print("=== diffusers_probe / SD1.5 generate (2-pass) ===")
    print(f"  model_id        : {model_id}")
    print(f"  device          : {device}")
    print(f"  steps           : {num_inference_steps}")
    print(f"  guidance        : {guidance_scale}")
    print(f"  seed            : {seed}")
    print(f"  prompt          : {prompt}")
    if negative_prompt:
        print(f"  neg prompt      : {negative_prompt}")
    print()
    print("  pass A : 1024x1024 / float32 (他モデルと同解像度比較用)")
    print("  pass B :  512x512 / float16 (SD1.5 ネイティブ解像度)")
    print()
    print("[note] 初回実行は Hugging Face cache への download 時間が混ざります。")
    print()

    # pass A: 1024 fp32
    try:
        pipe_a, load_a = load_pipeline(model_id, torch.float32, device, attention_slicing=attention_slicing)
    except Exception as e:
        print(f"[error] pass A load 失敗: {type(e).__name__}: {e}")
        for line in hint_for_load_error(e, model_id):
            print(line)
        traceback.print_exc()
        return 1
    try:
        gen_a = run_pass(
            cfg=cfg,
            pipe=pipe_a,
            pass_name="A",
            output_basename="sd15_generate_1024_fp32",
            model_id=model_id,
            device=device,
            dtype=torch.float32,
            width=1024,
            height=1024,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            load_time_sec=load_a,
            attention_slicing_on=(attention_slicing and device == "mps"),
        )
    except Exception as e:
        print(f"[error] pass A 生成失敗: {type(e).__name__}: {e}")
        traceback.print_exc()
        return 1
    del pipe_a

    # pass B: 512 fp16
    try:
        pipe_b, load_b = load_pipeline(model_id, torch.float16, device, attention_slicing=attention_slicing)
    except Exception as e:
        print(f"[error] pass B load 失敗: {type(e).__name__}: {e}")
        traceback.print_exc()
        return 1
    try:
        gen_b = run_pass(
            cfg=cfg,
            pipe=pipe_b,
            pass_name="B",
            output_basename="sd15_generate_512_fp16",
            model_id=model_id,
            device=device,
            dtype=torch.float16,
            width=512,
            height=512,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            load_time_sec=load_b,
            attention_slicing_on=(attention_slicing and device == "mps"),
        )
    except Exception as e:
        print(f"[error] pass B 生成失敗: {type(e).__name__}: {e}")
        traceback.print_exc()
        return 1

    print()
    print("=== timing summary ===")
    print(f"  pass A (1024 fp32) : load {load_a:.2f} s + gen {gen_a:.2f} s")
    print(f"  pass B ( 512 fp16) : load {load_b:.2f} s + gen {gen_b:.2f} s")
    print()
    print("=== done ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
