# CLAUDE.md

このファイルは、Claude Code がこのリポジトリで作業するときの作業方針をまとめたものです。

## プロジェクト概要

このリポジトリは、2026 年度「情報AI基礎」講義デモ用の **Diffusers probe workspace** です。
Hugging Face Diffusers の既存 API を使って、複数の latent diffusion 系モデルの動作を観察・可視化します。

対象モデル (実装済):

```text
SD1.5            (01_smoke, 02_generate)
SDXL Base 1.0    (03)
SDXL Turbo       (04)
FLUX.1-schnell   (05)
SD3.5 Medium     (06)
Qwen-Image       (07)
```

特化機能 (実装済):

```text
08: SDXL Base の deep probe (legacy-style cross-attention grid for 3 年前 SD1.x 講義スライド対応)
09: SDXL Base の prompt 探索 (config 駆動、Phase C agent 自律探索可)
```

この workspace は **probe 用**で、Diffusers のソースコードを改変しません。
pip install 版 Diffusers + 必要に応じた軽い PyTorch hook で内部を観察します。

今後の予定:

```text
- 講義スライド用の curation (notes/<topic>-curated/)
- notebook 化 (学生向け slim venv 用、scripts/ への import 不可・単体完結設計)
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
breakpoint を張る・改変する作業は別 workspace で行う方針 (現時点で別 workspace 未作成、必要になったら作る)。

---

## Python 環境

venv は用途別に 2 つに分ける方針:

```text
~/.venvs/dfs2026-dev   scripts / exploration 用 (主要 dev venv、自由運用)
~/.venvs/dfs2026       notebook / 学生用 slim 環境 (厳密管理、原則変更不可)
```

管理ファイル:

```text
requirements.txt       dfs2026 の direct dependencies (slim、学生配布想定)
requirements-dev.txt   dfs2026-dev の pip freeze (lint / type-check 等の dev tool も含む)
```

activate:

```bash
source ~/.venvs/dfs2026-dev/bin/activate     # scripts 編集・実験
source ~/.venvs/dfs2026/bin/activate         # notebook 動作確認
```

Mac では MPS、CUDA 環境では CUDA を使います。

### venv 管理ポリシー (重要)

**dfs2026 と dfs2026-dev は扱いがまったく違う**ので注意:

#### `~/.venvs/dfs2026` (slim) — 厳密管理

学生に配布する想定の slim venv。再現性とパッケージ最小性のため **厳密管理**:

- **原則変更しない**。何かを足したい場面でも、まず「足さずに済む書き方はないか」を検討する。
- **変更が必要でも Claude / Cursor agent 側から提案しない**。「○○を install しますか?」のような **承認待ちの提案も原則禁止** (理由: 提案フローで無自覚に OK と言ってしまう事故を防ぐため)。
- ユーザーが明示的に「これを入れて」と言った場合のみ install。実施したら `requirements.txt` を必ず同期する。
- `~/.venvs/dfs2026/` 配下のファイル直接編集 (`bin/activate` 等) も明示指示時のみ。

#### `~/.venvs/dfs2026-dev` (dev) — 自由運用、報告必須

scripts / 実験 / lint 用の dev venv。**追加 install は自由に行ってよい** が:

- 追加 install したら **必ずユーザーに伝える** (何を入れたか・なぜ要るか・transitive deps の件数)。
- `requirements-dev.txt` を `pip freeze` で同期するかはユーザーと相談する。勝手に sync しない。
- `pip uninstall` / `pip install --upgrade` は autonomous には行わない (これは dfs2026 と共通)。

dev で追加するときの flow: (1) 現状確認 → (2) 何を入れるかと理由を提示 → (3) ユーザー承認 → (4) install → (5) 完了報告 (パッケージ名・transitive 件数) → (6) `requirements-dev.txt` の sync 要否を相談。

---

## モデルと cache の方針

モデル本体は workspace に置きません。Hugging Face cache に置きます。

```text
~/.cache/huggingface/hub/...
```

`safety_checker` の扱い (SD1.5 系のみ; SDXL / FLUX / SD3.5 / Qwen には safety_checker なし):

- **01 (smoke)** ではデフォルトのまま (ON) 温存する。fp32 動作で誤発火を踏まない構成。
- **02 (generate)** では config の `disable_safety_checker` (既定 true) でオフにできる。
  fp16 で MPS の CLIP 誤発火を回避するために必要な、コミュニティの実用標準パターン。

cross-attention の capture / 可視化は 08 (deep probe) と 09 (prompt explore) で実装済。

---

## フォルダ役割定義

git 管理対象と git 管理外 (gitignored) に分かれ、保存期間 (tier T0=永続 〜 T5=即時) で分類。

| dir | git | tier | 役割 |
|---|:-:|:-:|---|
| `scripts/` | ○ | — | Python worker (`*.py`) + shell runner (`run_*.sh`) + script-specific config template (`*_template.json`) |
| `lecture/` | ○ | — | 学生がまず読む講義本編 (フォーマット不問: ipynb / md / pdf / pptx 等) |
| `lecture/images/` | ○ | — | lecture/ で参照する figure (outputs/ から cp、必要になったら作る) |
| `docs/` | ○ | — | 補助 reference (フォーマット不問: ipynb / md / pdf / pptx 等) |
| `docs/images/` | ○ | — | docs/ で参照する figure (outputs/ から cp) |
| `inbox/` | × | T0 | 外部由来 永続資料 (3 年前 notebook、論文 PDF、共有された他人の素材など) |
| `notes/` | × | T1 | 内部生成 永続 (md 中心の知見記録: handoff、observation、curated 実験記録) |
| `runs/` | × | T2 | 凍結 archive (実験 1 セットの完全パッケージ: input config + 全 output) |
| `outputs/` | × | T3 | 現用生成物 (script の生成先、curation 中の md、`run.log` 同梱) |
| `tmp/` | × | T5 | 真の scratch (新 config draft、ephemeral log、実行中 work) |

### tier 別 削除タイミング

- **T0 (inbox/)**: 原則削除しない
- **T1 (notes/)**: 原則削除しない
- **T2 (runs/)**: 原則永続
- **T3 (outputs/)**: notes/ 昇格後 or disk pressure で判断
- **T5 (tmp/)**: 数日 〜 1 週間

### notes/ の中身ルール

- 単純 (1 md だけ) → `notes/YYYY-MM-DD_<slug>.md`
- 複雑 (md + 図 + 再現用 script など) → `notes/YYYY-MM-DD_<slug>/README.md` + 同 dir に flat 配置
- サブ dir は **1 階層厳守、困ったら 2 階層許容** (例外運用)
- 大量素材 (数十〜数百ファイル) は notes/ に持ち込まず、runs/ から参照

### graduation 流れ

```text
1. 試行錯誤    tmp/ で config 編集、debug log
2. 実行       runner → outputs/<topic>/<slug>/ (config.json + 全 output + run.log)
3. 凍結       cp -r outputs/<topic>/<slug>/ runs/<date>_<topic>/<slug>/
4. curation   notes/<date>_<topic>-curated/ で knowledge md (任意)
5. 公開       docs/<topic>.md に formal report (任意)
```

### 廃止された慣習

- `logs/` → 廃止 (実行ログは `outputs/<slug>/run.log` 同梱、永続 md は `notes/`、ephemeral log は `tmp/`)
- `configs/` → 廃止 (各 run dir に `config.json` として同梱、template は `scripts/<basename>_template.json` に)
- `runners/` → 結局作らず、shell wrapper は `scripts/run_*.sh` で同居

---

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
    sdxl_base / sdxl_turbo / flux_schnell / sd35_medium / qwen_image  # 同様の per-model dict (実装済)
```

各 script は冒頭で `MODEL_KEY = "sd15"` のように対象 key を指定し、`get_model_config(cfg, MODEL_KEY)`
で per-model dict を取得する。01 は per-model dict も読まずハードコードで動く。

---

## script で `outputs/` を作る注意

script で `outputs/` に保存する場合は事前にディレクトリを作成すること:

```python
from pathlib import Path
Path("outputs").mkdir(parents=True, exist_ok=True)
```

---

## script 一覧と命名規則

```text
00_env_check.py                 環境チェック
01_sd15_generate_smoke.py       SD1.5 smoke (fp32 + safety_checker ON、ハードコード)
02_sd15_generate.py             SD1.5 (fp16/fp32、config 駆動)
03_sdxl_base_generate.py        SDXL Base 1.0
04_sdxl_turbo_generate.py       SDXL Turbo
05_flux1_schnell_generate.py    FLUX.1-schnell
06_sd35_medium_generate.py      SD3.5 Medium
07_qwen_image_generate.py       Qwen-Image
08_sdxl_base_deep_probe.py      SDXL Base の deep probe (legacy-style attention grid)
09_prompt_explore.py            SDXL Base の prompt 探索 (config 駆動)
```

各 script に対応する shell runner: `scripts/run_<basename>.sh` (必要なものだけ用意)。
各 script の詳細な役割・パラメータは冒頭の docstring を参照。

番号付き script 群を、理由なく巨大な単一 script にまとめないこと。

---

## 出力ファイルの方針

軽量な出力は `outputs/` に保存して構いません。

**02-07 (single-shot 流、legacy)**: outputs/ ルートに `<basename>.png` + `_summary.json` + `.txt` の 3 set 直置き:

```text
outputs/sd15_generate_smoke.png
outputs/sd15_generate_smoke_summary.json
outputs/sd15_generate_smoke.txt
```

**08 以降 (subdir 流)**: `outputs/<basename>/<run_label>/` 配下に config + 全 output + `run.log` を同梱:

```text
outputs/09_explore/<config_slug>/
  config.json
  summary.md
  run.log
  seed_NNNN/
  ...
```

完成品レポート (docs/) は curation 完了時に作成する (未着手のものはまだ作らない)。

---

## lecture/ と docs/ の使い分け

両者ともフォーマット不問 (ipynb / md / pdf / pptx 等)。**内容で分ける**:

- `lecture/` — 学生がまず読む講義本編 (intro notebook、講義 slide deck 等)
- `docs/` — 補助 reference (script 単位の formal report、技術 probe、付録等)

どちらも公開対象なので個人情報・絶対パス・token を含めない。lecture/ から docs/ への figure 参照はせず、必要なら `lecture/images/` を作って完結させる。

---

## docs/ の方針

- figure は `outputs/` から `docs/images/` へ **cp** (mv ではない)。outputs/ にも原本を残す。
- 1 script = 1 docs md。同じ script を複数回更新しても新ファイルを作らず、md を更新する。

---

## notebook の方針

`lecture/` `docs/` どちらに置く ipynb (notebook) にも適用:

- 各 notebook は **単体で完結する**設計にする (scripts/ への import / 参照は不可)。
- `~/.venvs/dfs2026` (学生用 slim venv) で動くことを目指す。
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
outputs/  runs/  notes/  inbox/  cache/  tmp/  configs/
*.pt、大きな tensor、モデル重み、Hugging Face cache
.env / .env.* / token / secret / huggingface_token*
```

(`logs/` は廃止。`notes/` `inbox/` `configs/` を追加。`configs/` は廃止だが誤って復活しないよう gitignore。)

## 自律実行の禁止

明示的な指示があるまで以下を実行しない:

- `git add` / `git add .`
- `git commit` / `git commit --amend`
- `git push` / `git push --force`
- `git rebase` / `git reset --hard`
- `git restore` / `git checkout -- .`
- `git clean -f` / `git clean -fd`
- ブランチの作成・削除・リネーム

**venv 関連の autonomous 操作** (pip install / uninstall / freeze、venv 内のファイル直接編集など) は「Python 環境 → venv 管理ポリシー」節を参照。要点: **dfs2026 は実行禁止 + 提案も禁止**、**dfs2026-dev は実行禁止だが報告ありの追加は OK**、両方とも uninstall / upgrade は明示指示が必要。

最後にユーザーへ報告するときは `git status -sb` の結果を提示するだけにする。

## commit の作業フロー（参考）

1. `git diff` / `git status` で変更内容を確認してユーザーに提示する
2. ユーザーの承認を得てから `git add`（対象ファイルを明示）
3. commit メッセージ案を提示する
4. ユーザーの承認を得てから `git commit`
5. `git push` はユーザーが明示的に要求した場合のみ実行する

commit メッセージは `add:` / `update:` / `fix:` / `remove:` などの prefix を使う。

長さは **中庸** を目指す (簡潔すぎず、詳細を全部羅列もしない):

- subject (1 行目) は核となる変更だけ、50〜70 文字程度を目安。
- 「11 点 + 〜 + 〜 + 〜」のような長い列挙は subject に書かない。詳細が必要なら空行 + body に書く。
- 「全画像 outputs/... 保存 (... = N 枚、... スタイルに統一)」のような副次情報を subject に積まない。

---

## やってはいけないこと

- Diffusers のソースコードを改変する (この workspace は probe スコープ、改変は別 workspace で行う方針)。
- モデル重みを repository に入れる。
- 大きな tensor を workspace 内に保存する。
- `/Users/<username>/...` のようなマシン固有の絶対パスを script に直書きする。
- 通常の probe script に暗黙の download 処理を追加する（pipeline 内の自動 download は除く）。
- 明示的な指示なしに `scripts/diffusers_probe.json` の `default_model` を変更する。
- 生成物（outputs/）を commit する（docs/ は例外で意図的に Git 管理対象）。
- token / secret をファイルやログに残す。
- 廃止された慣習 dir (`logs/`, `configs/`, `runners/`) を新規に作らない (新 dir 名は フォルダ役割定義 参照)。

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
