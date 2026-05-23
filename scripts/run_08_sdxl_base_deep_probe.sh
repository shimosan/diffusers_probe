#!/bin/bash
# Runner: scripts/08_sdxl_base_deep_probe.py
#
# 使い方:
#   bash scripts/run_08_sdxl_base_deep_probe.sh             # full (default)
#   bash scripts/run_08_sdxl_base_deep_probe.sh --quick     # quick (動作確認)
#   bash scripts/run_08_sdxl_base_deep_probe.sh --skip-attn --skip-guidance
#   ARCHIVE=1 bash scripts/run_08_sdxl_base_deep_probe.sh   # 成功後に runs/ にコピー
#
# venv: ~/.venvs/dfs2026-dev
# stdout/stderr: 実行中 tmp/08_sdxl_base_deep_probe_<timestamp>.log、完了後 outputs/.../run.log に move

set -u

# repo root に移動 (このスクリプトは scripts/ にある)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

VENV="${HOME}/.venvs/dfs2026-dev"
if [ ! -f "${VENV}/bin/activate" ]; then
  echo "[error] venv not found: ${VENV}"
  echo "        scripts/00_env_check.py 相当の dev venv を作成してください"
  exit 1
fi

# shellcheck disable=SC1091
source "${VENV}/bin/activate"

TS="$(date +%Y-%m-%d_%H-%M-%S)"
LOG_DIR="tmp"
mkdir -p "$LOG_DIR"
LOG_PATH="${LOG_DIR}/08_sdxl_base_deep_probe_${TS}.log"

echo "=========================================="
echo " 08 SDXL Base deep probe"
echo " started : $(date '+%Y-%m-%d %H:%M:%S')"
echo " python  : $(which python)"
echo " repo    : ${REPO_ROOT}"
echo " log     : ${LOG_PATH}"
echo " args    : $*"
echo "=========================================="

CMD=(caffeinate -i python scripts/08_sdxl_base_deep_probe.py "$@")
echo "[cmd] ${CMD[*]}"
echo

START_EPOCH=$(date +%s)
# stdout/stderr を log にも残しつつ画面にも出す
"${CMD[@]}" 2>&1 | tee "${LOG_PATH}"
RC="${PIPESTATUS[0]}"
END_EPOCH=$(date +%s)
DUR=$((END_EPOCH - START_EPOCH))

echo
echo "=========================================="
echo " finished : $(date '+%Y-%m-%d %H:%M:%S')"
echo " elapsed  : ${DUR}s ($((DUR/60))m$((DUR%60))s)"
echo " exitcode : ${RC}"
echo "=========================================="

# 直近 run の summary.md を探して提示 + log を mv
LATEST_RUN_DIR="$(ls -dt outputs/08_sdxl_base_deep_probe/*/ 2>/dev/null | head -1)"
if [ -n "${LATEST_RUN_DIR}" ]; then
  # log を outputs/<run-dir>/run.log に move (失敗時は tmp/ に残す)
  if [ "${RC}" -eq 0 ] && [ -f "${LOG_PATH}" ]; then
    mv "${LOG_PATH}" "${LATEST_RUN_DIR%/}/run.log"
    echo
    echo "[log] moved to ${LATEST_RUN_DIR%/}/run.log"
  else
    echo
    echo "[log] kept in tmp/ for debug: ${LOG_PATH}"
  fi
  SUMMARY_MD="${LATEST_RUN_DIR%/}/summary.md"
  echo "[summary] ${SUMMARY_MD}"
  if [ -f "${SUMMARY_MD}" ]; then
    echo "[summary] head:"
    head -40 "${SUMMARY_MD}" | sed 's/^/  | /'
  else
    echo "[summary] (まだ生成されていません)"
  fi

  # ARCHIVE=1 かつ成功時のみ runs/ に cp
  if [ "${ARCHIVE:-0}" = "1" ] && [ "${RC}" -eq 0 ]; then
    DATE_LABEL="$(date +%Y-%m-%d)_sdxl_base_deep_probe"
    DEST="runs/${DATE_LABEL}"
    if [ -e "${DEST}" ]; then
      DEST="${DEST}_${TS}"
    fi
    echo
    echo "[archive] cp -r ${LATEST_RUN_DIR%/} ${DEST}"
    cp -r "${LATEST_RUN_DIR%/}" "${DEST}"
    echo "[archive] done -> ${DEST}"
  fi
fi

exit "${RC}"
