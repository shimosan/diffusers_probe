# 画像生成モデル比較 — scripts 00–07

Scripts:
- [`scripts/00_env_check.py`](../scripts/00_env_check.py)
- [`scripts/01_sd15_generate_smoke.py`](../scripts/01_sd15_generate_smoke.py)
- [`scripts/02_sd15_generate.py`](../scripts/02_sd15_generate.py)
- [`scripts/03_sdxl_base_generate.py`](../scripts/03_sdxl_base_generate.py)
- [`scripts/04_sdxl_turbo_generate.py`](../scripts/04_sdxl_turbo_generate.py)
- [`scripts/05_flux1_schnell_generate.py`](../scripts/05_flux1_schnell_generate.py)
- [`scripts/06_sd35_medium_generate.py`](../scripts/06_sd35_medium_generate.py)
- [`scripts/07_qwen_image_generate.py`](../scripts/07_qwen_image_generate.py)

最終更新: 2026-05-23
ステータス: 🟢 全 7 script 動作確認済み。AC 駆動・cache warmed の clean run で timing を確定。

---

## 1. このドキュメントの位置づけ

このリポジトリ ([diffusers_probe](../README.md)) は、Hugging Face Diffusers で **画像生成モデルの内部を観察する** ことが最終目的です。最初のステップとして、**主要 6 モデルを Mac (M4 Max, **MPS** = Metal Performance Shaders、Apple Silicon の GPU バックエンド) で実際に動かしてみて、どれが講義デモに使えるか**を見極めるのが本ドキュメントの内容です。

| script | モデル | 役割 |
|---|---|---|
| 00 | (なし) | 環境チェック |
| 01 | **SD1.5** = Stable Diffusion 1.5 | smoke test (超安全策、ハードコード) |
| 02 | SD1.5 | 標準生成 (1024 fp32 + 512 fp16 の 2 pass) |
| 03 | **SDXL** Base 1.0 = Stable Diffusion XL | 標準生成 |
| 04 | SDXL Turbo | 4-step 高速生成 |
| 05 | FLUX.1-schnell | 4-step 高速生成 (**HF** = Hugging Face gated) |
| 06 | **SD3.5** Medium = Stable Diffusion 3.5 | 標準生成 (HF gated) |
| 07 | Qwen-Image | 20B モデル、テキスト描画特化 |

略語の凡例 (本文中の初出時にも展開):
- **fp16 / fp32 / bf16**: それぞれ floating-point 16-bit / 32-bit / brain-float 16-bit (Google が深層学習用に提案した変種、fp16 と同じ 16 bit だが指数部を広げて表現範囲を確保)
- **VAE** = Variational Autoencoder (画素 ↔ latent 空間の符号化器)
- **UNet** = U-shaped neural network (encoder-decoder + skip connection、SD1.5/SDXL の denoising 中核)
- **CLIP** = Contrastive Language-Image Pretraining (OpenAI のテキスト-画像 alignment encoder)
- **CFG** = Classifier-Free Guidance (条件付け強度を強める手法)
- **MMDiT** = Multi-Modal Diffusion Transformer (FLUX/SD3.5/Qwen の新型 backbone)
- **ADD** = Adversarial Diffusion Distillation (SDXL Turbo の蒸留手法)
- **DDPM** = Denoising Diffusion Probabilistic Models (Ho+ 2020、SD1.5/SDXL が依拠する古典的**拡散モデル**)
- **Rectified Flow** = 直線補間経路の **flow matching** 手法 (Liu+ 2022、FLUX/SD3.5/Qwen が採用)。**DDPM とは数学的に別ファミリ** (Markov 拡散過程ではなく決定論的 flow)、ただし「ノイズから反復で生成」という運用面の見た目は似ているためコミュニティでは "diffusion-based" と総称されがち
- **VLM** = Vision Language Model (画像とテキストを同時に扱う LLM、例: Qwen2.5-VL)
- **HF gated**: モデルが Hugging Face 上で「利用条件 (license terms) への承認」を要求している状態。初回 download 前に該当 model の HF ページで Accept ボタンを押し、`hf auth login` でトークン認証を済ませる必要がある。本ドキュメント内では FLUX.1-schnell と SD3.5 Medium が該当。

実行順は 00 → 01 → 02 → 03 → 04 → 05 → 06 → 07。各 script は独立しています (script 間の出力連携はなし)。

> **note**: 本ドキュメントの数値・画像は 2026-05-23 の **clean run** スナップショットに基づきます (`runs/2026-05-23_clean/`)。条件:
> - **2 回目の実行**なので HF cache warmed (download 時間は load_time に含まれない)
> - AC 駆動・`caffeinate -i` 付きで idle sleep 抑制・他重 process なし
> - 全 7 script を順次実行、合計 24m15s

---

## 2. 全 scripts の共通設定

設定は [`scripts/diffusers_probe.json`](../scripts/diffusers_probe.json) に集約。

```json
{
  "common": {
    "prompt": "A small robot studying artificial intelligence in a university classroom, simple illustration",
    "negative_prompt": "blurry, low quality, distorted",
    "seed": 42,
    "width": 1024,
    "height": 1024
  },
  "models": {
    "sd15":         { "steps": 20, "guidance_scale": 7.5, "mps_dtype": "float32", ... },
    "sdxl_base":    { "steps": 30, "guidance_scale": 7.5, "mps_dtype": "float32", ... },
    "sdxl_turbo":   { "steps":  4, "guidance_scale": 0.0, "mps_dtype": "float32", ... },
    "flux_schnell": { "steps":  4, "guidance_scale": 0.0, "mps_dtype": "bfloat16", ... },
    "sd35_medium":  { "steps": 28, "guidance_scale": 4.5, "mps_dtype": "bfloat16", ... },
    "qwen_image":   { "steps": 30, "guidance_scale": 4.0, "mps_dtype": "bfloat16", ... }
  }
}
```

ポイント:

- **prompt / negative_prompt / seed / 解像度 (1024x1024) は全モデル共通**。比較条件を揃えるのが目的。
- **steps / guidance はモデル固有**: SDXL Turbo と FLUX.1-schnell は distilled モデルで 1-4 step が前提、**CFG (Classifier-Free Guidance、条件付け強度を強める手法)** も 0 が前提。SD3.5 は 4.5、Qwen は 4.0 など、モデルが想定する値を尊重。
- **dtype は MPS 上の実測で決定**: SDXL 系は MPS で fp16/bf16 が **NaN** (Not a Number、無効値) になるため fp32。FLUX / SD3.5 / Qwen は bf16 で安定動作。詳細は各 Chapter で説明。

共通ユーティリティは [`scripts/common.py`](../scripts/common.py) にあり、主に以下を提供:

- `load_config()` / `get_common()` / `get_model_config(key)` — config 読み込み
- `pick_device_and_dtype(model_cfg)` — **CUDA** (NVIDIA GPU) / **MPS** / cpu 自動選択 + dtype 解決
- `apply_vae_fp32_override(pipe)` — **VAE (Variational Autoencoder)** のみ fp32 化 (一部モデルで必要)
- `build_summary_base(...)` / `write_outputs(...)` — 02–07 で統一フォーマットの png + summary.json + .txt を保存
- `hint_for_load_error(exc, model_id)` — gated repo / OOM / 接続エラーのヒント抽出

---

## 3. Chapter 00 — 環境確認 ([`00_env_check.py`](../scripts/00_env_check.py))

### 目的

Python / 主要 package / device (MPS/CUDA) / venv / config を 1 画面で確認するだけ。**モデルのロードは行わない** (重い処理は起動チェックに混ぜない CLAUDE.md 方針)。

### 実装の要点

`importlib.import_module(name)` で torch / diffusers / transformers / accelerate / huggingface_hub / safetensors / PIL / matplotlib / pandas を列挙、未インストールでも止まらず "not installed" と表示。`torch.cuda.is_available()` / `torch.backends.mps.is_available()` で device 確認。

### 結果

初回計測時の環境:

```text
python  : 3.12.13
platform: macOS-26.3.1-arm64-arm-64bit
torch   : 2.12.0
diffusers   : 0.38.0
transformers: 5.9.0
mps available: True
cuda available: False
venv: ~/.venvs/dfs2026-dev
```

→ MacBook Pro 16 (M4 Max, 64GB) の MPS 環境。以降の script はすべてこの環境で実行。

---

## 4. Chapter 01 — SD1.5 smoke test ([`01_sd15_generate_smoke.py`](../scripts/01_sd15_generate_smoke.py))

### 目的

**「Diffusers が壊れずに動くこと」の最初の確認**。dtype / safety_checker / 解像度 / steps / guidance をすべて **コード中にハードコード** して、迷わず動かす。

### 実装の要点

```python
HARDCODED_MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
HARDCODED_WIDTH = HARDCODED_HEIGHT = 512
HARDCODED_NUM_INFERENCE_STEPS = 20
HARDCODED_GUIDANCE_SCALE = 7.5

def pick_device_and_dtype():
    if torch.cuda.is_available():
        return "cuda", torch.float16
    if torch.backends.mps.is_available():
        return "mps", torch.float32      # MPS は fp32
    return "cpu", torch.float32
```

- **MPS で fp32 を採用**: SD1.5 を MPS + fp16 で動かすと `safety_checker` (**CLIP** = Contrastive Language-Image Pretraining ベース) が誤発火して黒画像を返すことが多いため。fp32 なら safety_checker を温存できる。
- **`enable_attention_slicing()`** を MPS のみで有効化 (メモリ節約)。
- config からは `common.{prompt, negative_prompt, seed}` のみ読み、models セクションは読まない。

### 結果

| 項目 | 値 |
|---|---|
| device / dtype | mps / float32 |
| size | 512×512 |
| steps / guidance | 20 / 7.5 |
| load 時間 (cache hit) | **1.58 s** |
| generation 時間 | **5.19 s** |
| safety_checker | ON |

![sd15_generate_smoke](images/sd15_generate_smoke.png)

**Figure 1**: 01 smoke (SD1.5, 512×512, fp32, safety_checker ON) の出力。

→ 期待通り 1 体のロボットが教室で勉強する絵が出る。safety_checker が誤発火する場合はここで黒画像になる (今回は問題なし)。**「とにかく動くこと」がここで保証**できたので、以降の 02–07 では条件を変えて比較してよい状態に。

### 出力ファイル

- [runs/2026-05-23_clean/sd15_generate_smoke.png](../runs/2026-05-23_clean/sd15_generate_smoke.png) (退避保存)
- `outputs/sd15_generate_smoke.{png,_summary.json,.txt}`

---

## 5. Chapter 02 — SD1.5 標準生成 (2-pass) ([`02_sd15_generate.py`](../scripts/02_sd15_generate.py))

### このモデルが特殊な理由

**SD1.5 は 512×512 で訓練されたモデル**。他のモデル (SDXL 以降) は 1024×1024 ネイティブで、揃った解像度で比較できない問題があります。

このリポジトリでは「全モデル 1024×1024 で並べる」方針を採ったため、SD1.5 だけは **2 パス** で生成して両方残すことにしました:

- **Pass A**: 1024×1024 / fp32 (他モデルと並べる比較用)
- **Pass B**:  512×512 / fp16 (本来の SD1.5 の品質)

### 1024×1024 で fp32 が必須な理由

実は最初 1024 fp16 で動かしたところ、画像は**真っ黒**になりました。`vae_fp32_override` で VAE だけ fp32 にしても黒のまま。**UNet (U-shaped neural network、SD1.5/SDXL の denoising 中核ネットワーク)** 段階で latent が NaN になっており、SD1.5 を MPS fp16 で 1024 解像度に動かすこと自体が破綻します。fp32 全体化で回避できますが、SD1.5 の本来想定外サイズなので、構図そのものは破綻します (下記)。

### 結果 — Pass A: 1024×1024 fp32

| 項目 | 値 |
|---|---|
| device / dtype | mps / float32 |
| size | 1024×1024 |
| steps / guidance | 20 / 7.5 |
| load 時間 | 3.22 s |
| generation 時間 | **49.84 s** |

![sd15_1024_fp32](images/sd15_generate_1024_fp32.png)

**Figure 2**: SD1.5, 1024×1024, fp32。**複数のロボットが画面いっぱいに並ぶ構図破綻**。これは SD1.5 が 512 で訓練されているために起こる典型的な現象で、講義的には**「訓練解像度から外れるとモデルが破綻する」**ことを示す良い反例素材になります。

### 結果 — Pass B: 512×512 fp16

| 項目 | 値 |
|---|---|
| device / dtype | mps / float16 |
| size | 512×512 |
| steps / guidance | 20 / 7.5 |
| load 時間 | 3.24 s |
| generation 時間 | **4.36 s** |

![sd15_512_fp16](images/sd15_generate_512_fp16.png)

**Figure 3**: SD1.5, 512×512, fp16, safety_checker 無効。ネイティブ解像度。1 体のロボットが机に座る、SD1.5 本来の絵に。Pass A と全く同じ prompt / seed で出ているのに構図がまったく違うのが本質。

### 観察

1. **解像度がモデル品質を支配する場面がある**: 同じ seed・同じ prompt でも、SD1.5 が訓練されていない 1024 に拡張するだけで破綻する。これは scheduler や guidance 等のパラメータをいじっても解決しない。
2. **fp16 vs fp32 の選択**: SD1.5 は 512 なら fp16 で十分速い (4.36s vs 49.84s で **10 倍以上の差**)。fp32 は「本来の解像度から外れている = 構図がもう破綻している」ケースの救済にしか使えない。

### 講義での扱い

- **本命は Pass B (512×512)**: SD1.5 が想定する条件。
- **Pass A は教材として残す**: 「モデルを訓練解像度外で使うと壊れる」の例。

### 出力ファイル

- [runs/2026-05-23_clean/sd15_generate_1024_fp32.png](../runs/2026-05-23_clean/sd15_generate_1024_fp32.png) (Pass A)
- [runs/2026-05-23_clean/sd15_generate_512_fp16.png](../runs/2026-05-23_clean/sd15_generate_512_fp16.png) (Pass B)

---

## 6. Chapter 03 — SDXL Base 1.0 ([`03_sdxl_base_generate.py`](../scripts/03_sdxl_base_generate.py))

### 背景: SDXL とは

**SDXL = Stable Diffusion XL**。Stability AI が 2023 年に公開した、SD1.5 の正統な後継。UNet を大型化 (2.6B → 3.5B 程度の Latent Diffusion + 1.4B の VAE/text encoders) し、**ネイティブ 1024×1024** で訓練。プロンプト追随性と画質が大きく向上。

### MPS で fp16 / bf16 が動かない問題

最初の試行で次の道筋を辿りました:

1. **fp16 + `vae_fp32_override`** → `Input type (float) and bias type (c10::Half)` RuntimeError。SDXL pipeline 自身が `upcast_vae()` を呼ぶので、自前で VAE を fp32 化する hack と干渉。
2. **fp16 のみ、override 無し** → エラーは出ないが**画像が真っ黒** (UNet 段階で NaN)。
3. **bf16 + `variant="fp16"`** → やはり真っ黒。
4. **mps_dtype = `float32`** → ✅ 成功。

→ MPS の Metal kernel における SDXL UNet の fp16 / bf16 演算には数値精度問題があり、現状は fp32 強制が必要。`madebyollin/sdxl-vae-fp16-fix` の VAE を使えば fp16 で救える可能性がありますが今回は未検証。

### 結果

| 項目 | 値 |
|---|---|
| device / dtype | mps / float32 |
| size | 1024×1024 |
| steps / guidance | 30 / 7.5 |
| load 時間 (cache warmed) | 7.42 s |
| generation 時間 | **51.77 s** |

![sdxl_base](images/sdxl_base_generate.png)

**Figure 4**: SDXL Base 1.0, 1024×1024, fp32, 30 steps。1 体のロボットが教室の前で姿勢正しく立ち、手にはノートを持つ。構図、線、色、すべてが SD1.5 (Pass A) から劇的に改善。

### 観察

1. **品質が SD1.5 から大きく上がる**: ネイティブ 1024 で訓練されているので構図破綻なし。
2. **生成時間は SD1.5 (1024 fp32) と同等**: 51.77s vs 49.84s。SDXL は UNet 自体が大きいが、attention slicing と最適化で SD1.5 fp32 と同等の時間に収まる。
3. **fp32 強制のコスト**: fp16 が動けばさらに 1.5–2x 速くなる見込み。MPS の改善待ち。

### 出力ファイル

- [runs/2026-05-23_clean/sdxl_base_generate.png](../runs/2026-05-23_clean/sdxl_base_generate.png)

---

## 7. Chapter 04 — SDXL Turbo ([`04_sdxl_turbo_generate.py`](../scripts/04_sdxl_turbo_generate.py))

### 背景: distilled / few-step モデルとは

通常 Diffusion は 20–50 step の反復で画像を生成しますが、**ADD = Adversarial Diffusion Distillation** (敵対的拡散蒸留) などの手法で、SDXL の生成プロセスを 1–4 step に蒸留した版が SDXL Turbo。CFG (Classifier-Free Guidance) も使わない (`guidance_scale=0.0`) のが特徴。**講義デモで「待ち時間が短い」のは強い**ので採用候補として重要。

### 実装の要点

- `AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", ...)` で読み込み (内部的には SDXL pipeline と同じ class)。
- `num_inference_steps=4`, `guidance_scale=0.0`。
- `negative_prompt` も渡さない (Turbo 設計上使わない)。

### 結果

| 項目 | 値 |
|---|---|
| device / dtype | mps / float32 |
| size | 1024×1024 |
| steps / guidance | **4** / 0.0 |
| load 時間 (cache warmed) | 7.67 s |
| generation 時間 | **4.40 s** |

![sdxl_turbo](images/sdxl_turbo_generate.png)

**Figure 5**: SDXL Turbo, 1024×1024, fp32, 4 steps, no CFG。10 秒未満で 1024×1024 が出る。机に書類 + 椅子 + ロボット 1 体の教室の絵で、構図は問題なく、SDXL Base に比べると線がややくっきり (蒸留モデルの傾向)。

### 観察

1. **12 倍速**: SDXL Base 51.77s に対して Turbo は 4.40s。30 step → 4 step + CFG 無効 (1 step あたり 2 forward → 1 forward) で、ステップ数比 (7.5x) と CFG 効果 (≈2x) を合わせるとほぼ理論値 (実測はさらに速い)。
2. **品質は SDXL Base に肉薄**: 構図は同等、ディテールはやや単純化されている程度。**講義デモなら十分**。
3. **CFG 無しは prompt の細かい指示が効きにくい**: 「university classroom」「simple illustration」の指示反映度は SDXL Base のほうがやや上。

### 出力ファイル

- [runs/2026-05-23_clean/sdxl_turbo_generate.png](../runs/2026-05-23_clean/sdxl_turbo_generate.png)

---

## 8. Chapter 05 — FLUX.1-schnell ([`05_flux1_schnell_generate.py`](../scripts/05_flux1_schnell_generate.py))

### 背景: FLUX と Rectified Flow

Black Forest Labs (Stable Diffusion 原作者陣) が 2024 年に公開した、**12B パラメータの MMDiT = Multi-Modal Diffusion Transformer** (text token と image token を一つの sequence にして self-attention で処理する transformer、UNet の置き換え)。SDXL とは別系統で、**Rectified Flow** (Liu+ 2022、ノイズとデータを直線補間する flow matching 手法) を採用。`schnell` はその 4-step distilled 版 (Apache 2.0 ライセンスだが HF gated — 規約承認 + `hf auth login` が必要)。

### 実装の要点

- `FluxPipeline.from_pretrained(..., torch_dtype=torch.bfloat16)`。MPS で **bf16 が動く** (SDXL とは違って)。
- `guidance_scale=0.0`, `max_sequence_length=256`。
- `negative_prompt` は使わない。
- 初回 download は ~24 GB。`caffeinate -i` 付きで起動推奨 (sleep で TCP CLOSE_WAIT 化のリスク)。

### 結果

| 項目 | 値 |
|---|---|
| device / dtype | mps / **bfloat16** |
| size | 1024×1024 |
| steps / guidance | **4** / 0.0 |
| load 時間 (cache warmed) | 17.92 s |
| generation 時間 | **49.85 s** |

![flux_schnell](images/flux_schnell_generate.png)

**Figure 6**: FLUX.1-schnell, 1024×1024, bf16, 4 steps。可愛らしいロボットが机の前で本を読む、illustration スタイルの絵。SDXL 系と比べると線がより整理されており、色のフラットさで「simple illustration」のプロンプト指示によく追従。

### 観察

1. **MPS で bf16 が動く**: SDXL の fp32 強制とは対照的。MMDiT 系の方が MPS との相性が良いらしい。
2. **生成時間は SDXL Base と同程度 (49.85s vs 51.77s)**: 4 step だが、1 step あたりの計算量が SDXL の 30 step 分に近い (~13B パラメータ vs SDXL ~3B、step あたり ~4 倍重い)。SDXL Turbo (4.40s) よりは明らかに遅いが、品質は FLUX のほうが上。
3. **プロンプト追随性が高い**: 「simple illustration」がきっちり反映される。SDXL 系よりも text encoder (**T5-xxl** = Text-to-Text Transfer Transformer の最大版、~4.7B パラメータの Google 製 encoder-decoder LLM) が強力なため。
4. **download コストが大きい**: ~24 GB。再現実験のたびに `caffeinate -i` を忘れずに。

### 出力ファイル

- [runs/2026-05-23_clean/flux_schnell_generate.png](../runs/2026-05-23_clean/flux_schnell_generate.png)

---

## 9. Chapter 06 — SD3.5 Medium ([`06_sd35_medium_generate.py`](../scripts/06_sd35_medium_generate.py))

### 背景: SD3 ファミリ

Stability AI が FLUX と同じ MMDiT アーキテクチャを採用して 2024 年に出した次世代モデル。`Medium` は SD3.5 の中位版 (~2.5B 程度)。**HF gated** で `hf auth login` が必要。FLUX と SDXL のいいとこ取りを狙った設計。

### 実装の要点

- `StableDiffusion3Pipeline.from_pretrained(..., torch_dtype=torch.bfloat16)`。
- `num_inference_steps=28`, `guidance_scale=4.5` (Stability の推奨)。
- `negative_prompt` あり。

### 結果

| 項目 | 値 |
|---|---|
| device / dtype | mps / **bfloat16** |
| size | 1024×1024 |
| steps / guidance | 28 / 4.5 |
| load 時間 (cache warmed) | 8.27 s |
| generation 時間 | **191.04 s** |

![sd35_medium](images/sd35_medium_generate.png)

**Figure 7**: SD3.5 Medium, 1024×1024, bf16, 28 steps。背景の黒板には記号が並び、本棚も配置された情報量の多い教室。ロボットの顔の表情やキャラクター性は FLUX より素朴。

### 観察

1. **生成時間は中位 (191.04s ≒ 3 分)**: 4-step 蒸留モデル (Turbo / schnell) より遅く、SDXL Base よりも遅い。28 step を真面目に計算しているため。
2. **多人数構成や複雑なシーン記述に強い印象**: 黒板や本棚など、prompt にない要素まで自然に描かれる傾向。
3. **HF gated + 大きい download** (~17 GB): FLUX より小さいがそれでも事前 download 必須。

### 出力ファイル

- [runs/2026-05-23_clean/sd35_medium_generate.png](../runs/2026-05-23_clean/sd35_medium_generate.png)

---

## 10. Chapter 07 — Qwen-Image ([`07_qwen_image_generate.py`](../scripts/07_qwen_image_generate.py))

### 背景: Qwen-Image とは

Alibaba が 2024 年末に公開した **20B パラメータ** の画像生成モデル (Apache 2.0、非 gated)。**テキスト描画能力**を強く強化しているのが特徴 (中国語・英語の文字を絵の中にきれいに描ける)。MPS で動かす場合の挙動は今回が初めての検証。

### 実装の要点

- `QwenImagePipeline.from_pretrained(..., torch_dtype=torch.bfloat16)`。
- `num_inference_steps=30`, `true_cfg_scale=4.0` (Qwen-Image 独自の引数名)。
- 初回 download は **~58 GB**。当 workspace の最大モデル。

### 結果

| 項目 | 値 |
|---|---|
| device / dtype | mps / **bfloat16** |
| size | 1024×1024 |
| steps / guidance | 30 / 4.0 (true_cfg_scale) |
| load 時間 (cache warmed) | 53.16 s |
| generation 時間 | **980.07 s ≒ 16m20s** |

![qwen_image](images/qwen_image_generate.png)

**Figure 8**: Qwen-Image, 1024×1024, bf16, 30 steps。**黒板に「Artificial Intelligence」の文字が綺麗に描画**されている点に注目。FLUX も SDXL も英文をここまで明瞭には描けない。Qwen-Image の最大の強みであるテキスト描画能力が確認できる。

### 観察

1. **テキスト描画が突出**: 黒板の "Artificial Intelligence" がリーダブル。これは他のどのモデルにも無い特徴。
2. **生成時間は約 16 分**: **講義中のリアルタイム生成は無理**だが、デモ前に事前生成しておくなら現実的な範囲。
3. **モデルサイズ (~58 GB) が大きすぎる**: M4 Max 64GB でぎりぎり動く。CPU offload を使えば更に安定するかも (今回未検証)。
4. **`load_time` も 53s と長め**: 他モデルが 3-18s なのに対して。20B モデルの初期化と MPS 上の memory allocation に時間がかかる。

### 出力ファイル

- [runs/2026-05-23_clean/qwen_image_generate.png](../runs/2026-05-23_clean/qwen_image_generate.png)

---

## 11. モデル横断比較

### 11-1. 画像ギャラリー (同一 prompt / seed)

| | 出力 |
|---|---|
| **01 SD1.5 smoke** (512, fp32, safety_checker ON, 20 steps) | ![01](images/sd15_generate_smoke.png) |
| **02 SD1.5 Pass A** (1024, fp32, 20 steps) | ![02A](images/sd15_generate_1024_fp32.png) |
| **02 SD1.5 Pass B** (512, fp16, 20 steps) | ![02B](images/sd15_generate_512_fp16.png) |
| **03 SDXL Base** (1024, fp32, 30 steps) | ![03](images/sdxl_base_generate.png) |
| **04 SDXL Turbo** (1024, fp32, **4 steps**) | ![04](images/sdxl_turbo_generate.png) |
| **05 FLUX.1-schnell** (1024, bf16, **4 steps**) | ![05](images/flux_schnell_generate.png) |
| **06 SD3.5 Medium** (1024, bf16, 28 steps) | ![06](images/sd35_medium_generate.png) |
| **07 Qwen-Image** (1024, bf16, 30 steps) | ![07](images/qwen_image_generate.png) |

prompt: `"A small robot studying artificial intelligence in a university classroom, simple illustration"`
negative: `"blurry, low quality, distorted"`
seed: 42

### 11-2. 技術スペック・実行時間比較

(2026-05-23 clean run, MPS, M4 Max, AC 駆動, cache warmed, **2 回目の実行で download 時間含まず**)

| script | モデル | dtype | size | steps | guidance | load | **gen** | download |
|---|---|---|---|---:|---:|---:|---:|---:|
| 01 | SD1.5 smoke | fp32 | 512 | 20 | 7.5 | 1.58 s | 5.19 s | ~4 GB |
| 02-A | SD1.5 | fp32 | 1024 | 20 | 7.5 | 3.22 s | 49.84 s | (同上) |
| 02-B | SD1.5 | fp16 | 512 | 20 | 7.5 | 3.24 s | **4.36 s** ⚡ | (同上) |
| 03 | SDXL Base | fp32 | 1024 | 30 | 7.5 | 7.42 s | 51.77 s | ~14 GB |
| 04 | SDXL Turbo | fp32 | 1024 | 4 | 0.0 | 7.67 s | **4.40 s** ⚡ | ~14 GB |
| 05 | FLUX.1-schnell | bf16 | 1024 | 4 | 0.0 | 17.92 s | 49.85 s | ~24 GB |
| 06 | SD3.5 Medium | bf16 | 1024 | 28 | 4.5 | 8.27 s | 191.04 s | ~17 GB |
| 07 | Qwen-Image | bf16 | 1024 | 30 | 4.0 | 53.16 s | 980.07 s (16m20s) | **~58 GB** |

⚡ 講義デモで「数秒で 1 枚」のリアルタイム性が確保できるのは 02-B (SD1.5 512) と 04 (SDXL Turbo) の 2 つ。

### 11-3. アーキテクチャ・位置づけ比較

各モデルの内部構造の世代と、講義での扱い方を一覧。**SD1.5 系 → SDXL 系 → MMDiT 系** という年代順の進化と、**多 step + CFG vs 蒸留 (few-step + CFG なし)** の 2 軸で整理できる。

「生成手法」列について: **DDPM (拡散モデル) と Rectified Flow (flow matching) は数学的には別ファミリ**。前者はガウスノイズを段階的に加える Markov 過程の逆向き、後者はノイズとデータを直線補間して velocity を学ぶ手法。共通点は「ノイズから始めて反復的にデータへ近づける」サンプリング構造のみ。コミュニティではよく "diffusion-based" と総称されるが、本ドキュメントでは区別して書く。

| script | モデル | backbone | 生成手法 | text encoder | 講義での位置づけ |
|---|---|---|---|---|---|
| 01 | SD1.5 smoke | UNet (~860M) | DDPM | CLIP-L | 動作確認 (smoke test) |
| 02-A | SD1.5 (1024) | UNet (~860M) | DDPM | CLIP-L | **反例**: 訓練解像度外で破綻 |
| 02-B | SD1.5 (512) | UNet (~860M) | DDPM | CLIP-L | レガシー参照、高速 4 秒 |
| 03 | SDXL Base | UNet (~3.5B) | DDPM | CLIP-L + OpenCLIP-G | SDXL レシピ理解、Turbo の比較対象 |
| 04 | SDXL Turbo | UNet (SDXL と同じ) | **DDPM (蒸留)** | CLIP-L + OpenCLIP-G | **本命リアルタイムデモ** |
| 05 | FLUX.1-schnell | MMDiT (~12B) | **Rectified Flow (蒸留)** | T5-XXL + CLIP-L | 現代高品質枠 (要 HF gated) |
| 06 | SD3.5 Medium | MMDiT (~2.5B) | Rectified Flow | CLIP-L + CLIP-G + T5-XXL | もう一つの MMDiT 系列 (要 HF gated) |
| 07 | Qwen-Image | MMDiT (~20B) | Rectified Flow | **Qwen2.5-VL** (VLM) | テキスト描画特化、事前生成のみ |

蒸留 (distillation) 手法は表からは省略 (個別手法は Chapter 7 / Chapter 8 で言及)。代表的なものは:
- **SDXL Turbo**: ADD = Adversarial Diffusion Distillation ([Sauer+ 2023](https://arxiv.org/abs/2311.17042))。教師モデル (SDXL Base 30 step) の denoising 軌道を生徒 (Turbo) が 1-4 step で再現するよう学習させ、さらに discriminator を加えてシャープさを担保。
- **FLUX.1-schnell**: 手法詳細は非公開。step distillation 系と推定。

#### この表からの示唆 (講義での材料)

1. **アーキテクチャ + 生成手法の世代交代**: SD1.5 / SDXL 世代は **UNet + DDPM** (古典的な拡散モデル: ノイズを段階的に加える Markov 過程の逆向き)、FLUX / SD3.5 / Qwen 世代は **MMDiT + Rectified Flow** (拡散ではなく flow matching: ノイズとデータを直線補間して velocity を学ぶ別ファミリ)。**LLM と同じ transformer 系統に画像生成も収束**している。
2. **蒸留で 4 step 化**: SDXL Turbo と FLUX.1-schnell はいずれも蒸留版で 4 step + CFG なしを実現。**待ち時間を 1/10 にする現代の鍵**。
3. **text encoder の進化**: CLIP 1 本 (SD1.5) → CLIP 2 本 (SDXL) → CLIP + T5 (FLUX/SD3) → VLM (Qwen)。**プロンプト追随性とテキスト描画能力が比例的に向上**。
4. **モデル規模 ≠ Mac で使える**: SD3.5 Medium (2.5B) は MMDiT としては小さいが gen 3 分、Qwen-Image (20B) は 16 分。**規模はリアルタイム性のボトルネック**。

---

## 12. 講義での採用方針

### 12-1. 結論

**「待ち時間が短く、品質も高く、配布難度が低い」モデルを優先**します。優先順位:

1. **SDXL Turbo (04)** — メインのリアルタイムデモ用。1024×1024 を **4.4 秒**、講義中に何度でも prompt を変えて試せる。download も SDXL Base と共通の cache。
2. **SDXL Base (03)** — 「Turbo は 30 step → 4 step に蒸留した版」という関係を示すための比較対象。約 52 秒なので 1–2 回見せる程度。
3. **SD1.5 Pass B (02-B、512 fp16)** — レガシーモデル参照、4.4 秒と高速、講義の「最初の歴史紹介」スライド用。SDXL Turbo と並べて「512 と 1024 の解像度の差」を見せる対比にも使える。
4. **SD1.5 Pass A (02-A、1024 fp32)** — **教材としての反例**。「訓練解像度の外で動かすとモデルが破綻する」を視覚的に示す。
5. **FLUX.1-schnell (05)** — 「現代の高品質モデル」枠。MPS で 4-step **50 秒**。HF gated だが配布前提なら学生に `hf auth login` を踏ませてもよい。SDXL Base と同等時間で品質が一段上。
6. **SD3.5 Medium (06)** — 「もう一つの MMDiT 系列」枠。約 3 分。FLUX と比較する文脈で 1 回出す。デモのライブ再生成には遅め。
7. **Qwen-Image (07)** — 「テキスト描画特化のモデル」枠。AC 駆動で **16 分**。リアルタイム生成は無理だが、講義の休憩時間に 1 枚回す程度なら可能。事前生成した png を出すのが安全。

### 12-2. 学生に伝えるメッセージ

- 同じ prompt・同じ seed でも、**モデル選択次第で出力は劇的に変わる**: 構図、画風、テキスト描画能力すべて。Figure 1–8 がその証拠。
- **少ない step で同等品質を出す技術 (distillation)** がここ 2 年の進展の中心。SDXL Turbo / FLUX.1-schnell はその代表。
- **モデルの大きさ ≠ Mac で動かしやすさ**: Qwen-Image (20B) は Mac M4 Max でも 1 枚 16 分。普段使うモデルは「自分の machine で何分待てるか」で選ぶ。
- **同じ seed なら再現可能**: 本ドキュメントの画像は seed=42 で生成。再実行すれば (同じ環境なら) 同じ絵が出る。

### 12-3. 講義で実機を動かすときの推奨手順

```bash
# venv 起動
source ~/.venvs/dfs2026-dev/bin/activate

# 環境確認
python scripts/00_env_check.py

# まずスモークで動作確認 (5 秒)
python scripts/01_sd15_generate_smoke.py

# 本命のリアルタイムデモ (約 4 秒)
python scripts/04_sdxl_turbo_generate.py

# 学生に prompt を提案してもらって 04 を何度か実行
# (config の common.prompt を書き換えるか、コマンドライン引数化を後で追加)
```

事前準備 (講義前夜):

- HF cache を温める (`caffeinate -i` 付きで 01〜06 を 1 回ずつ動かす)。03-06 が cache hit 状態なら load 時間は 10-20 秒に抑えられる。
- AC 接続、他の重い process (Zoom 等) を閉じる。
- 07 Qwen-Image は事前生成した png を準備、講義中は実機実行しない (1 枚 16 分)。

---

## 13. 既知問題・注意事項

1. **MPS で SDXL は fp16 / bf16 共に動かない**: UNet または VAE が NaN を出す。fp32 強制で対処。`madebyollin/sdxl-vae-fp16-fix` で fp16 を救う案は未検証 (次の課題)。
2. **MPS の決定性**: `torch.use_deterministic_algorithms(True)` は MPS で完全サポート外だが、`seed=42` 固定 + `Generator(device="cpu")` の組み合わせなら、同 hardware / 同 OS / 同 venv で再現可能。dtype / OS / hardware / library version を変えると変わる可能性は残る。
3. **hf_xet の挙動**: 未完了 `.incomplete` ファイルは resume せず、再実行で 0 から取り直し。完了済ファイルは使い回される。
4. **大型 download (>10 GB) には `caffeinate -i` 必須**: Mac が sleep すると TCP CLOSE_WAIT 化、process が hang する。Qwen-Image (~58 GB) のような大型 download で注意。
5. **outputs/ は git 管理外**: 再生成可能 / 上書きされる。`runs/<日付>_*/` に `cp` して凍結保存している (今回は runner script `tmp/run_clean_2026-05-23.sh` で自動化)。

---

## 14. 次のステップ

### 短期

- [ ] SDXL を `sdxl-vae-fp16-fix` の VAE で fp16 試行 (MPS で SDXL を高速化できる可能性)
- [ ] 同 prompt・同 seed で multi-run して timing のばらつきを測る

### 中期 (探査拡張)

- [ ] **cross-attention の可視化** (text token → 画像 patch の対応)。Qwen3 の Chapter 06 と同じ要領で probe を追加。MMDiT 系は joint attention なので別実装が必要。
- [ ] **scheduler を変えた比較** (DDIM (Denoising Diffusion Implicit Models), Euler-a, DPM-Solver++ 等)。同じ seed・同じ step で見え方がどう変わるか。SD3 / FLUX 系は Rectified Flow なので scheduler 選択は別の意味を持つ。
- [ ] **同 prompt の seed ごとのばらつき**: seed 42 だけでなく seed 1–10 を並べてどれくらい変動するか。

### 長期 (notebook 化)

- [ ] [notebooks/](../notebooks/) に学生配布用 notebook を作る (slim venv `~/.venvs/dfs2026` で動作)。candidate: SDXL Turbo + prompt を変えて遊ぶ notebook。

---

## 15. 出力ファイル一覧

### scripts → 凍結保存先

凍結 run: `runs/2026-05-23_clean/` (AC + cache warmed + caffeinate)。`runs/` は git 管理外で Dropbox sync で個人保全。

| script | 主な出力 | runs/2026-05-23_clean |
|---|---|:---:|
| 00 | (標準出力のみ) | (なし) |
| 01 | sd15_generate_smoke.{png, _summary.json, .txt} | ✓ |
| 02-A | sd15_generate_1024_fp32.{png, _summary.json, .txt} | ✓ |
| 02-B | sd15_generate_512_fp16.{png, _summary.json, .txt} | ✓ |
| 03 | sdxl_base_generate.{png, _summary.json, .txt} | ✓ |
| 04 | sdxl_turbo_generate.{png, _summary.json, .txt} | ✓ |
| 05 | flux_schnell_generate.{png, _summary.json, .txt} | ✓ |
| 06 | sd35_medium_generate.{png, _summary.json, .txt} | ✓ |
| 07 | qwen_image_generate.{png, _summary.json, .txt} | ✓ |

### このドキュメントで使った画像

すべて `docs/images/` に `cp` 済 (md からの相対パス参照のため):

- [docs/images/sd15_generate_smoke.png](images/sd15_generate_smoke.png)
- [docs/images/sd15_generate_1024_fp32.png](images/sd15_generate_1024_fp32.png)
- [docs/images/sd15_generate_512_fp16.png](images/sd15_generate_512_fp16.png)
- [docs/images/sdxl_base_generate.png](images/sdxl_base_generate.png)
- [docs/images/sdxl_turbo_generate.png](images/sdxl_turbo_generate.png)
- [docs/images/flux_schnell_generate.png](images/flux_schnell_generate.png)
- [docs/images/sd35_medium_generate.png](images/sd35_medium_generate.png)
- [docs/images/qwen_image_generate.png](images/qwen_image_generate.png)
