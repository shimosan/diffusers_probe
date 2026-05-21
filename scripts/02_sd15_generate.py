# 02_sd15_generate.py
#
# Stable Diffusion 1.5 の標準生成 script。普段使う default 動作はここ。
#
# 設定は scripts/diffusers_probe.json から:
#   common.{prompt, negative_prompt, seed}
#   models.sd15.{model_id, width, height, num_inference_steps, guidance_scale,
#                mps_dtype, cuda_dtype, cpu_dtype,
#                disable_safety_checker, enable_attention_slicing_on_mps, vae_fp32_override}
#
# 既定値は Apple Silicon / MPS のコミュニティ実用パターンに寄せてある:
#   - 基本 fp16 (memory 削減 & 速度)
#   - safety_checker を外す (CLIP-based の fp16 誤発火を避ける)
#   - attention slicing 有効
#   - vae_fp32_override は false (必要なら true で --no-half-vae 相当)
#
# 01_sd15_generate_smoke.py は超安全策 (全 fp32 + safety_checker 有り) で、
# 動作確認だけしたいときに使う。本 script はそれより速く・実用寄り。
#
# 出力:
#   outputs/sd15_generate.png
#   outputs/sd15_generate_summary.json
#   outputs/sd15_generate.txt

from __future__ import annotations

import json
import platform
import sys
import time
import traceback
from typing import cast

import torch

from common import (
    ensure_dir,
    get_common,
    get_model_config,
    load_config,
    project_root,
    resolve_outputs_dir,
)

MODEL_KEY = "sd15"
OUTPUT_BASENAME = "sd15_generate"

_DTYPE_MAP = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "float32": torch.float32,
    "fp32": torch.float32,
    "full": torch.float32,
}


def parse_dtype(name: str | None, default: torch.dtype) -> torch.dtype:
    if name is None:
        return default
    return _DTYPE_MAP.get(str(name).lower(), default)


def pick_device_and_dtype(model_cfg: dict) -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        return "cuda", parse_dtype(model_cfg.get("cuda_dtype"), torch.float16)
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps", parse_dtype(model_cfg.get("mps_dtype"), torch.float16)
    return "cpu", parse_dtype(model_cfg.get("cpu_dtype"), torch.float32)


def apply_vae_fp32_override(pipe) -> None:
    """VAE のみ fp32 化し、decode 直前で latent を fp32 にキャストする hook を入れる。
    AUTOMATIC1111 の --no-half-vae 相当。
    """
    pipe.vae = pipe.vae.to(dtype=torch.float32)
    original_decode = pipe.vae.decode

    def patched_decode(z, *args, **kwargs):
        z = z.to(dtype=torch.float32)
        return original_decode(z, *args, **kwargs)

    pipe.vae.decode = patched_decode  # type: ignore[method-assign]


def main() -> int:
    cfg = load_config()
    common = get_common(cfg)
    model_cfg = get_model_config(cfg, MODEL_KEY)

    model_id: str = model_cfg["model_id"]
    width: int = int(model_cfg["width"])
    height: int = int(model_cfg["height"])
    num_inference_steps: int = int(model_cfg["num_inference_steps"])
    guidance_scale: float = float(model_cfg["guidance_scale"])

    prompt: str = common["prompt"]
    negative_prompt: str = common.get("negative_prompt", "")
    seed: int = int(common["seed"])

    disable_safety = bool(model_cfg.get("disable_safety_checker", True))
    attn_slicing_mps = bool(model_cfg.get("enable_attention_slicing_on_mps", True))
    vae_fp32_override = bool(model_cfg.get("vae_fp32_override", False))

    outputs_dir = ensure_dir(resolve_outputs_dir())
    png_path = outputs_dir / f"{OUTPUT_BASENAME}.png"
    json_path = outputs_dir / f"{OUTPUT_BASENAME}_summary.json"
    txt_path = outputs_dir / f"{OUTPUT_BASENAME}.txt"

    device, dtype = pick_device_and_dtype(model_cfg)

    print("=== diffusers_probe / SD1.5 generate ===")
    print(f"  model_id        : {model_id}")
    print(f"  device          : {device}")
    print(f"  dtype           : {dtype}  (from models.{MODEL_KEY})")
    print(f"  safety_checker  : {'OFF (disabled)' if disable_safety else 'ON (default)'}")
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
    print("       生成時間の測定は 2 回目以降の値を参考にしてください。")
    print()

    try:
        # diffusers は _LazyModule 経由で公開しているので Pyright の
        # reportPrivateImportUsage が誤検知する。実行時の public API はこの形が正。
        from diffusers import StableDiffusionPipeline  # pyright: ignore[reportPrivateImportUsage]
        from diffusers.pipelines.stable_diffusion.pipeline_output import (
            StableDiffusionPipelineOutput,
        )
    except Exception as e:
        print(f"[error] diffusers の import に失敗しました: {type(e).__name__}: {e}")
        return 1

    print("[load] StableDiffusionPipeline.from_pretrained ...")
    t0 = time.perf_counter()
    try:
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
    except Exception as e:
        name = type(e).__name__
        msg = str(e)
        print(f"[error] model load に失敗しました: {name}: {msg}")
        if "401" in msg or "gated" in msg.lower() or "access" in msg.lower() or "Unauthorized" in msg:
            print("[hint] Hugging Face の access denied / gated repo の可能性があります。")
            print("       次を試してください:")
            print("         1) https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5 で利用条件を承認")
            print("         2) 端末で `hf auth login` を実行 (token はファイルに保存しないでください)")
        elif "ConnectionError" in name or "Timeout" in name or "Resolve" in msg:
            print("[hint] ネットワーク接続に問題がある可能性があります。")
        traceback.print_exc()
        return 1

    pipe = pipe.to(device)

    if disable_safety:
        pipe.safety_checker = None

    if vae_fp32_override and dtype != torch.float32:
        apply_vae_fp32_override(pipe)
        print("[vae ] vae_fp32_override 適用 (VAE のみ fp32、decode 前に latent を fp32 化)")

    if attn_slicing_mps and device == "mps":
        pipe.enable_attention_slicing()
        print("[mps ] enable_attention_slicing() 有効化")
    load_elapsed = time.perf_counter() - t0
    print(f"[load] done in {load_elapsed:.2f} s (初回は download 時間を含む)")

    generator = torch.Generator(device=device if device != "mps" else "cpu").manual_seed(seed)

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
        print(f"[error] 生成に失敗しました: {type(e).__name__}: {e}")
        traceback.print_exc()
        return 1
    gen_elapsed = time.perf_counter() - t1
    # pipe.__call__ は戻り値型注釈が無く、Pyright は tuple との union と推論する。
    # ここでは return_dict=True (既定) で呼び出しているので StableDiffusionPipelineOutput が返る。
    result = cast(StableDiffusionPipelineOutput, result)
    image = result.images[0]
    nsfw_flags = result.nsfw_content_detected
    nsfw_detected = bool(nsfw_flags[0]) if (nsfw_flags is not None and len(nsfw_flags) > 0) else False
    print(f"[gen ] done in {gen_elapsed:.2f} s")
    if nsfw_detected:
        print("[warn] safety_checker が NSFW を検出 → 画像は黒塗りになります。")
        print("       disable_safety_checker を true にするか、dtype を float32 に上げて再試行してください。")

    total_elapsed = time.perf_counter() - t0

    image.save(png_path)
    print(f"[save] image -> {png_path}")

    summary = {
        "script": "02_sd15_generate.py",
        "workspace_name": cfg.get("workspace_name", "diffusers_probe"),
        "model_key": MODEL_KEY,
        "model_id": model_id,
        "device": device,
        "dtype": str(dtype),
        "safety_checker_enabled": not disable_safety,
        "vae_fp32_override": vae_fp32_override,
        "attention_slicing_enabled": attn_slicing_mps and device == "mps",
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "seed": seed,
        "nsfw_content_detected": nsfw_detected,
        "load_time_sec": round(load_elapsed, 3),
        "generation_time_sec": round(gen_elapsed, 3),
        "total_time_sec": round(total_elapsed, 3),
        "image_path": str(png_path.relative_to(project_root())),
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[save] summary -> {json_path}")

    txt_lines = [
        f"script: 02_sd15_generate.py",
        f"model_id: {model_id}",
        f"prompt: {prompt}",
        f"negative_prompt: {negative_prompt}",
        f"image_path: {png_path}",
        f"summary_path: {json_path}",
    ]
    with txt_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(txt_lines) + "\n")
    print(f"[save] prompt+paths -> {txt_path}")

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
