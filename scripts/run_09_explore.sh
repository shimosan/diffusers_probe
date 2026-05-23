#!/bin/bash
# Runner: scripts/09_prompt_explore.py
#
# 使い方:
#   bash scripts/run_09_explore.sh <path-to-config.json>
#   bash scripts/run_09_explore.sh <path-to-config.json> --force
#
# config はどこに置いてもよい (tmp/ で draft、runs/.../_configs/ から再利用、など)。
# 雛形は scripts/09_prompt_explore_template.json。
#
# venv: ~/.venvs/dfs2026-dev
# stdout/stderr: 実行中 tmp/09_explore_<slug>_<timestamp>.log、完了後 outputs/09_explore/<slug>/run.log に move
# 出力: outputs/09_explore/<slug>/

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

if [ $# -lt 1 ]; then
  echo "usage: bash scripts/run_09_explore.sh <config-path> [--force]" >&2
  echo "       e.g. bash scripts/run_09_explore.sh tmp/my_new.json" >&2
  echo "" >&2
  echo "tip: 新 config は scripts/09_prompt_explore_template.json を tmp/ に cp して編集" >&2
  exit 1
fi

CONFIG="$1"
shift || true
EXTRA_ARGS=("$@")

if [ ! -f "$CONFIG" ]; then
  echo "[error] config not found: $CONFIG" >&2
  exit 1
fi

VENV="${HOME}/.venvs/dfs2026-dev"
if [ ! -f "${VENV}/bin/activate" ]; then
  echo "[error] venv not found: ${VENV}" >&2
  exit 1
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

SLUG="$(basename "$CONFIG" .json)"
TS="$(date +%Y-%m-%d_%H-%M-%S)"
LOG_DIR="tmp"
mkdir -p "$LOG_DIR"
LOG_PATH="${LOG_DIR}/09_explore_${SLUG}_${TS}.log"

echo "=========================================="
echo " 09 prompt explore"
echo " started : $(date '+%Y-%m-%d %H:%M:%S')"
echo " python  : $(which python)"
echo " repo    : ${REPO_ROOT}"
echo " config  : ${CONFIG}"
echo " log     : ${LOG_PATH}"
echo " extra   : ${EXTRA_ARGS[*]+${EXTRA_ARGS[*]}}"
echo "=========================================="

CMD=(caffeinate -i python scripts/09_prompt_explore.py --config "$CONFIG" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"})
echo "[cmd] ${CMD[*]}"
echo

START_EPOCH=$(date +%s)
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

# 出力 dir を探して log を mv + summary.md を提示
LATEST_RUN_DIR="$(ls -dt outputs/09_explore/${SLUG}*/ 2>/dev/null | head -1)"
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
    head -30 "${SUMMARY_MD}" | sed 's/^/  | /'
  fi
else
  echo
  echo "[log] no matching output dir; kept in tmp/: ${LOG_PATH}"
fi

exit "${RC}"
