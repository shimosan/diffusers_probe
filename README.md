# diffusers_probe

Diffusers による画像生成モデルの軽量 probe workspace。

## Purpose

2026 年度「情報AI基礎」講義デモ向けに、Hugging Face Diffusers を使って各種画像生成モデルの動作を観察・可視化する workspace。
最初は Stable Diffusion 1.5 から始め、段階的に対象モデルを増やしていく予定。

## Models

最初の対象:

- `stable-diffusion-v1-5/stable-diffusion-v1-5`

将来の追加候補（**今回は実装しない**）:

- `stabilityai/stable-diffusion-xl-base-1.0`（SDXL Base）
- `stabilityai/sdxl-turbo`（SDXL Turbo）
- `black-forest-labs/FLUX.1-schnell`（FLUX.1-schnell）
- `stabilityai/stable-diffusion-3.5-medium`（SD3.5 Medium）
- `Qwen/Qwen-Image`（Qwen-Image）

今回実装済みなのは **SD1.5 の smoke test のみ**。

## Repository structure

| Path | Git | Contents |
|---|---|---|
| `scripts/` | ✓ | 番号付き script 群（`common.py`, `00_env_check.py`, `01_sd15_generate_smoke.py` ...）|
| `docs/` | ✓ | 完成版の実験レポート md と参照画像 `docs/images/`（学生配布対象）|
| `outputs/` | ✗ | script の生成物（PNG / JSON / TXT 等）。再生成可能で永続性なし |
| `logs/` | ✗ | 実行ログと作業中の md ドラフト |
| `cache/`, `tmp/` | ✗ | 一時ファイル用 |

詳細な作業方針は [CLAUDE.md](CLAUDE.md) を参照。

## Python venvs

venv は用途別に分ける方針:

```text
~/.venvs/dfs2026-dev    scripts / exploration 用（今回はこちらを使う）
~/.venvs/dfs2026        将来の notebook / 学生用 slim 環境（今回は作成しない）
```

## Setup

dev venv を作成し、必要 package を入れる:

```bash
python3 -m venv ~/.venvs/dfs2026-dev
source ~/.venvs/dfs2026-dev/bin/activate
python -m pip install -U pip wheel setuptools
python -m pip install -U torch torchvision torchaudio
python -m pip install -U diffusers transformers accelerate safetensors pillow matplotlib pandas huggingface_hub ipykernel
python -m pip check
python -m pip freeze > requirements-dev.txt
```

Mac (Apple Silicon) では PyTorch wheel が MPS 対応で入る。CUDA 環境では公式の case に応じて index URL を切り替える。

## Run SD1.5

二系統の生成 script を用意してある。

### 01: 超安全策 (`01_sd15_generate_smoke.py`)

Diffusers / HF 公式の基本例に最も近い構成。dtype / safety_checker / attention slicing /
生成パラメータ (size, steps, guidance) は**コードにハードコード**してあり、config からは
`common.{prompt, negative_prompt, seed}` だけを読む (`models.sd15` は読まない)。

- MPS: `torch.float32` (公式 docs の基本例 + 重みの素 dtype と一致)
- CUDA: `torch.float16` (CUDA では fp16 が安定)
- CPU: `torch.float32`
- `safety_checker`: **ON のまま** (デフォルト挙動)
- MPS のときだけ `enable_attention_slicing()`

```bash
source ~/.venvs/dfs2026-dev/bin/activate
python scripts/00_env_check.py
python scripts/01_sd15_generate_smoke.py
```

出力:

```text
outputs/sd15_generate_smoke.png
outputs/sd15_generate_smoke_summary.json
outputs/sd15_generate_smoke.txt
```

### 02: 普段使う標準版 (`02_sd15_generate.py`)

Apple Silicon / MPS のコミュニティで広く使われる実用パターン。基本 fp16 + safety_checker 外し +
attention slicing + 任意で VAE のみ fp32 override (= `--no-half-vae` 相当)。
設定は `scripts/diffusers_probe.json` の `common` + `models.sd15` から読む。

```bash
source ~/.venvs/dfs2026-dev/bin/activate
python scripts/02_sd15_generate.py
```

`models.sd15` の中身:

```json
"models": {
  "sd15": {
    "model_id": "stable-diffusion-v1-5/stable-diffusion-v1-5",
    "width": 512,
    "height": 512,
    "num_inference_steps": 20,
    "guidance_scale": 7.5,

    "mps_dtype": "float16",
    "cuda_dtype": "float16",
    "cpu_dtype": "float32",
    "disable_safety_checker": true,
    "enable_attention_slicing_on_mps": true,
    "vae_fp32_override": false
  }
}
```

出力:

```text
outputs/sd15_generate.png
outputs/sd15_generate_summary.json
outputs/sd15_generate.txt
```

ターミナルには `device`, `dtype`, `safety_checker`, `model_id`, 各 elapsed time が表示される。

### 01 と 02 の使い分け

- **何かおかしい / 動作確認したい** → 01。何も悩まずに動く。
- **普段使う / 設定を変えて挙動を見たい** → 02。fp16/fp32 や safety_checker on/off を試せる。
- 講義デモで「公式デフォルト」と「速度最適化 + 罠の知識」を分けて見せるときも 01/02 を順に紹介する。

### 将来モデルを増やすとき

`scripts/diffusers_probe.json` の `models` 配下に `sdxl`, `flux_schnell` などのキーを足し、
対応する script (`03_sdxl_generate.py` 等) を書く。各 script は冒頭で `MODEL_KEY = "sdxl"` のように
対象キーを指定する。`common` (prompt / negative_prompt / seed) はモデル間で使い回せる。

## Hugging Face access

SD1.5 は通常 anonymous で download できるが、もし gated repo / access denied が起きた場合は:

```bash
hf auth login
```

でログインし、Hugging Face のモデルページで利用条件を承認しておく。
**token をファイルに保存したり、ターミナル出力に貼り付けたりしないこと。**

## 注意事項

- 初回実行はモデル download 時間（数百 MB）が混ざるため、**生成時間の測定は 2 回目以降の値を参考にする**こと。
- MPS で OOM や極端な遅さが出る場合は、一時的に `width=384, height=384, num_inference_steps=10` 等に下げて smoke 確認してよい（その場合は config と summary に実際の値を明記する）。
- 完成版レポートで使う画像は `outputs/` から `docs/images/` に **cp**（mv ではない）すること。`docs/` と `docs/images/` は Git 管理対象。

### 既知の挙動: SD1.5 + MPS + fp16 で safety_checker が誤発火する

Apple Silicon の MPS backend で SD1.5 を `torch.float16` で動かすと、画像 tensor 自体は
0–1 の健全な値（NaN なし）になるが、**safety_checker (CLIP-based) の内部計算が fp16 で不安定**になり、
NSFW を誤判定して**黒画像**を返すケースが多い（seed や prompt を変えても抜けにくい）。

この workspace ではこれを 2 つの script で住み分けている:

- **01 (smoke / safe)**: MPS で `float32` を選び、safety_checker は ON のまま保持する。黒画像にはならない。
- **02 (generate / 普段使い)**: MPS で `float16` を選び、`disable_safety_checker: true` で safety_checker を外す。
  これで誤発火の問題ごと回避する（コミュニティの実用標準パターン）。
  さらに本当に VAE が NaN を出す既知ケース対策に `vae_fp32_override: true` を用意してあり、
  これは AUTOMATIC1111 の `--no-half-vae` 相当（VAE のみ fp32 化 + decode 前に latent を fp32 化）。

参考（実測値、本機 M4 Max, 512×512, 20 step, seed=42）:

| script | dtype | safety_checker | generation time |
|---|---|---|---|
| 01 smoke | fp32 | ON | ~5.2 s |
| 02 generate (fp16, safety off) | fp16 | OFF | ~4.3 s |
| 02 generate (fp16 + vae_fp32) | fp16 + VAE fp32 | OFF | ~4.3 s |

CUDA 環境では `torch.float16` が安定して動作するため、上記の安全策は不要（02 をそのまま使えばよい）。

## License / Credits

`stable-diffusion-v1-5/stable-diffusion-v1-5` は CreativeML Open RAIL-M ライセンス。利用条件はモデルカードを参照。
