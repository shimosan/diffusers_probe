# 01_sd15_generate_smoke.py
#
# Stable Diffusion 1.5 (Diffusers) で 1 枚画像を生成する smoke test (超安全策版)。
#
# このスクリプトは「とにかく素直に動かす」ことを優先するため、
# dtype / safety_checker / attention slicing / 生成パラメータすべてを
# コード中にハードコードしている。
# config (diffusers_probe.json) からは "common" セクション (prompt / negative_prompt / seed)
# だけを読み、models.sd15 の値も読まない。dtype 等を試したい人は 02_sd15_generate.py を使う。
#
# 方針 (hardcoded):
#   - device は cuda -> mps -> cpu の順で選択
#   - dtype: cuda=float16, mps=float32, cpu=float32
#       (MPS の fp32 は HF Diffusers 公式の基本例と同じ。重みの素も fp32)
#   - safety_checker: 触らない (デフォルト ON のまま)
#   - MPS のときだけ enable_attention_slicing()
#   - 512x512, 20 steps, guidance_scale=7.5
#
# 出力:
#   outputs/sd15_generate_smoke.png
#   outputs/sd15_generate_smoke_summary.json
#   outputs/sd15_generate_smoke.txt

from __future__ import annotations

import json
import platform
import sys
import time
import traceback
from typing import cast

import torch

from common import ensure_dir, get_common, load_config, project_root, resolve_outputs_dir

OUTPUT_BASENAME = "sd15_generate_smoke"

# ハードコードされた生成パラメータ (config からは読まない)
HARDCODED_MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
HARDCODED_WIDTH = 512
HARDCODED_HEIGHT = 512
HARDCODED_NUM_INFERENCE_STEPS = 20
HARDCODED_GUIDANCE_SCALE = 7.5


def pick_device_and_dtype() -> tuple[str, torch.dtype]:
    # 超安全策: MPS では fp32 を選ぶ。
    # 理由: SD1.5 を MPS + fp16 で動かすと safety_checker (CLIP) が誤発火し
    # 黒画像を返すケースが多いため。fp32 なら safety_checker をそのまま温存できる。
    if torch.cuda.is_available():
        return "cuda", torch.float16
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps", torch.float32
    return "cpu", torch.float32


def main() -> int:
    cfg = load_config()
    common = get_common(cfg)

    prompt: str = common["prompt"]
    negative_prompt: str = common.get("negative_prompt", "")
    seed: int = int(common["seed"])

    outputs_dir = ensure_dir(resolve_outputs_dir())
    png_path = outputs_dir / f"{OUTPUT_BASENAME}.png"
    json_path = outputs_dir / f"{OUTPUT_BASENAME}_summary.json"
    txt_path = outputs_dir / f"{OUTPUT_BASENAME}.txt"

    device, dtype = pick_device_and_dtype()

    print("=== diffusers_probe / SD1.5 generate smoke (safe) ===")
    print(f"  model_id  : {HARDCODED_MODEL_ID}")
    print(f"  device    : {device}")
    print(f"  dtype     : {dtype}  (hardcoded; MPS=fp32, CUDA=fp16, CPU=fp32)")
    print(f"  safety_checker: ON (デフォルトのまま)")
    print(f"  size      : {HARDCODED_WIDTH} x {HARDCODED_HEIGHT}")
    print(f"  steps     : {HARDCODED_NUM_INFERENCE_STEPS}")
    print(f"  guidance  : {HARDCODED_GUIDANCE_SCALE}")
    print(f"  seed      : {seed}")
    print(f"  prompt    : {prompt}")
    if negative_prompt:
        print(f"  neg prompt: {negative_prompt}")
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
        pipe = StableDiffusionPipeline.from_pretrained(HARDCODED_MODEL_ID, torch_dtype=dtype)
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
    if device == "mps":
        pipe.enable_attention_slicing()
        print("[mps] enable_attention_slicing() 有効化")
    load_elapsed = time.perf_counter() - t0
    print(f"[load] done in {load_elapsed:.2f} s (初回は download 時間を含む)")

    generator = torch.Generator(device=device if device != "mps" else "cpu").manual_seed(seed)

    print("[gen ] running pipeline ...")
    t1 = time.perf_counter()
    try:
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            width=HARDCODED_WIDTH,
            height=HARDCODED_HEIGHT,
            num_inference_steps=HARDCODED_NUM_INFERENCE_STEPS,
            guidance_scale=HARDCODED_GUIDANCE_SCALE,
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
        print("       fp32 では通常起きないはず。seed/prompt を変えて再試行するか")
        print("       02_sd15_generate.py で safety_checker を外して確認してください。")

    total_elapsed = time.perf_counter() - t0

    image.save(png_path)
    print(f"[save] image -> {png_path}")

    summary = {
        "script": "01_sd15_generate_smoke.py",
        "workspace_name": cfg.get("workspace_name", "diffusers_probe"),
        "model_id": HARDCODED_MODEL_ID,
        "device": device,
        "dtype": str(dtype),
        "safety_checker_enabled": True,
        "attention_slicing_enabled": device == "mps",
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": HARDCODED_WIDTH,
        "height": HARDCODED_HEIGHT,
        "num_inference_steps": HARDCODED_NUM_INFERENCE_STEPS,
        "guidance_scale": HARDCODED_GUIDANCE_SCALE,
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
        f"script: 01_sd15_generate_smoke.py (safe)",
        f"model_id: {HARDCODED_MODEL_ID}",
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
