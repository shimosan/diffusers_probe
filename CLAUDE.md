# CLAUDE.md

このファイルは、Claude Code がこのリポジトリで作業するときの作業方針をまとめたものです。

## プロジェクト概要

このリポジトリは、2026 年度「情報AI基礎」講義デモ用の **Diffusers probe workspace** です。
Hugging Face Diffusers の既存 API を使って、画像生成モデル（SD1.5 などの latent diffusion 系）の動作を観察・可視化することを目的とします。

最初の対象は以下です。

```text
stable-diffusion-v1-5/stable-diffusion-v1-5
```

この workspace は **probe 用**で、Diffusers のソースコードを改変して内部を追跡するための workspace ではありません。
今回は **pip install 版 Diffusers** だけを使い、source tracing や pipeline 改変はしません。

将来追加予定（今回は実装しない）:

```text
- SDXL Base / SDXL Turbo
- FLUX.1-schnell
- SD3.5 Medium
- Qwen-Image
- cross-attention 可視化
- notebook 化
```

---

## 役割

```text
diffusers_probe:
  pip install 版 Diffusers を使い、
  既存 pipeline (StableDiffusionPipeline 等)、
  必要に応じた軽い PyTorch hook により、
  生成プロセスや内部状態を観察する。
```

Diffusers の実装ファイル（`pipeline_stable_diffusion.py`, `unet_2d_condition.py`, attention processor 等）に
breakpoint を張る・改変する作業は別 workspace で行います（今回は未作成）。

---

## Python 環境

venv は用途別に分けます。

```text
~/.venvs/dfs2026-dev   scripts / exploration 用（**今回はこちらだけ作成・使用**）
~/.venvs/dfs2026       将来の notebook / 学生用 slim 環境（今回は作成しない）
```

管理ファイル:

```text
requirements-dev.txt   dfs2026-dev の pip freeze
```

activate:

```bash
source ~/.venvs/dfs2026-dev/bin/activate
```

Mac では MPS、CUDA 環境では CUDA を使います。

---

## モデルと cache の方針

モデル本体は workspace に置きません。Hugging Face cache に置きます。

```text
~/.cache/huggingface/hub/...
```

`safety_checker` の扱い:

- **01 (smoke)** ではデフォルトのまま (ON) 温存する。fp32 動作で誤発火を踏まない構成。
- **02 (generate)** では config の `disable_safety_checker` (既定 true) でオフにできる。
  fp16 で MPS の CLIP 誤発火を回避するために必要な、コミュニティの実用標準パターン。

今回は cross-attention の capture / 可視化は実装しません。

## config schema (将来モデル拡張を見据えた構造)

```text
diffusers_probe.json
  workspace_name
  default_model           # 現状のメインモデル key (例: "sd15")
  common                  # 全モデル共通 (prompt / negative_prompt / seed)
  models
    sd15                  # モデルキーごとに 1 dict
      model_id
      width / height / num_inference_steps / guidance_scale
      mps_dtype / cuda_dtype / cpu_dtype
      disable_safety_checker
      enable_attention_slicing_on_mps
      vae_fp32_override
    # 将来: sdxl, flux_schnell, sd35_medium, qwen_image を追加
```

各 script は冒頭で `MODEL_KEY = "sd15"` のように対象 key を指定し、`get_model_config(cfg, MODEL_KEY)`
で per-model dict を取得する。01 は per-model dict も読まずハードコードで動く。

---

## リポジトリ構成（想定）

```text
diffusers_probe/
  CLAUDE.md
  README.md
  requirements-dev.txt
  .gitignore
  .cursorignore
  .cursor/
    rules/
      project.mdc
      python.mdc
      diffusers_probe.mdc
  .vscode/
    settings.json
    launch.json
  scripts/
    common.py
    diffusers_probe.json
    00_env_check.py
    01_sd15_generate_smoke.py
  docs/
    images/
  outputs/
  logs/
```

`outputs/` / `logs/` / `cache/` / `tmp/` は Git 管理対象**外**。
`docs/` および `docs/images/` は Git 管理対象（学生配布用の完成品）。

script で `outputs/` に保存する場合は事前にディレクトリを作成すること:

```python
from pathlib import Path
Path("outputs").mkdir(parents=True, exist_ok=True)
```

---

## script の番号順を保つ

```text
00_env_check.py             環境チェック
01_sd15_generate_smoke.py   SD1.5 の超安全策・smoke版 (fp32 + safety_checker ON, ハードコード)
02_sd15_generate.py         SD1.5 の普段使う標準版 (fp16 + safety_checker OFF が既定、config で切替)
（03 番以降は将来用、今回は作らない）
```

01 と 02 の役割分担:

- **01 (smoke / safe)**: dtype / safety_checker / attention slicing / 生成パラメータはコード中にハードコード。
  config からは `common.{prompt, negative_prompt, seed}` のみ読む。Diffusers 公式の基本例に最も近い。
- **02 (generate / 普段使い)**: `common` に加えて `models.sd15` を読む。fp16 / fp32,
  safety_checker on/off, `vae_fp32_override` (--no-half-vae 相当) を切り替え可能。

番号付き script 群を、理由なく巨大な単一 script にまとめないこと。

---

## 出力ファイルの方針

軽量な出力は `outputs/` に保存して構いません。

SD1.5 smoke / generate の場合:

```text
outputs/sd15_generate_smoke.png             01 の生成画像
outputs/sd15_generate_smoke_summary.json    01 の実行条件・経過時間
outputs/sd15_generate_smoke.txt             01 の prompt と出力パス
outputs/sd15_generate.png                   02 の生成画像
outputs/sd15_generate_summary.json          02 の実行条件・経過時間
outputs/sd15_generate.txt                   02 の prompt と出力パス
```

完成品のレポート（docs/）は今回作成しません（今後 nb01 等を作る際に書く）。

---

## docs/ の方針

```text
outputs/      git 管理外。再生成可能、永続性なし。
logs/         git 管理外。実行ログ、作業中 md ドラフト。
docs/         git 管理。完成品の実験レポート md。
docs/images/  git 管理。docs/*.md が参照する figure を outputs/ から cp する。
```

- figure は `outputs/` から `docs/images/` へ **cp**（mv ではない）。outputs/ にも原本を残す。
- 1 script = 1 docs md。同じ script を複数回更新しても新ファイルを作らず、md を更新する。

今回は docs/ にレポートは作りません。

---

## notebook の方針

今回は notebook を作りません。

将来 notebook を作る場合:

- 各 notebook は **単体で完結する**設計にする（scripts/ への import / 参照は不可）。
- `~/.venvs/dfs2026`（学生用 slim venv）で動くことを目指す。
- モデル load / 生成 / 可視化を notebook 内に完結させる。

---

## コーディング方針

- pathlib.Path を使う。
- 調整可能なパラメータは `scripts/diffusers_probe.json` または script 冒頭にまとめる。
- 出力先ディレクトリは保存前に作成する。
- 進捗が分かる簡潔な print を入れる。
- device / dtype / model_id を明示する。
- マシン固有の絶対パスをハードコードしない。
- download / setup 用 script 以外に暗黙のネットワークアクセスを入れない。

講義デモ用なので、技巧的な実装よりも、読んで分かる実装を優先する。

---

## Git の注意

明示的な指示がない限り、広く stage しないこと。

避ける例:

```bash
git add .
```

推奨:

```bash
git add CLAUDE.md
git add scripts/01_sd15_generate_smoke.py
```

特に以下には注意:

```text
.vscode/settings.json   Cursor / Pyright 由来のローカル変更が入りやすい。エディタ設定変更タスク以外では commit しない。
outputs/                生成物。commit しない。
```

以下は commit しないこと:

```text
outputs/  runs/  logs/  cache/  tmp/
*.pt、大きな tensor、モデル重み、Hugging Face cache
.env / .env.* / token / secret / huggingface_token*
```

## 自律実行の禁止

明示的な指示があるまで以下を実行しない:

- `git add` / `git add .`
- `git commit` / `git commit --amend`
- `git push` / `git push --force`
- `git rebase` / `git reset --hard`
- `git restore` / `git checkout -- .`
- `git clean -f` / `git clean -fd`
- ブランチの作成・削除・リネーム

最後にユーザーへ報告するときは `git status -sb` の結果を提示するだけにする。

## commit の作業フロー（参考）

1. `git diff` / `git status` で変更内容を確認してユーザーに提示する
2. ユーザーの承認を得てから `git add`（対象ファイルを明示）
3. commit メッセージ案を提示する
4. ユーザーの承認を得てから `git commit`
5. `git push` はユーザーが明示的に要求した場合のみ実行する

commit メッセージは 1 行で、`add:` / `update:` / `fix:` / `remove:` などの prefix を使う。

---

## やってはいけないこと

- Diffusers のソースコードを改変する（今回は probe のみ）。
- モデル重みを repository に入れる。
- 大きな tensor を workspace 内に保存する。
- `/Users/<username>/...` のようなマシン固有の絶対パスを script に直書きする。
- 通常の probe script に暗黙の download 処理を追加する（pipeline 内の自動 download は除く）。
- 明示的な指示なしに `default_model_id` を変更する。
- 生成物（outputs/）を commit する（docs/ は例外で意図的に Git 管理対象）。
- token / secret をファイルやログに残す。
- 今回のスコープにない SDXL / FLUX / SD3.5 / Qwen-Image の実装を勝手に始める。

---

## 講義デモとしての優先事項

優先するもの:

```text
- 分かりやすさ
- 再現性
- 出力ファイルの意味の明確さ
- 短い prompt での安定動作
- 図や JSON による説明しやすさ
```

この workspace の目的は、生成品質を最大化することではなく、
画像生成モデルの内部計算を、実際の出力・図・JSON を通して見える形にすることです。
