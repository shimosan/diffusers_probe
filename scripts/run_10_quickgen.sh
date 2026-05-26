#!/bin/bash
# Runner: scripts/10_quickgen.py
#
# 使い方:
#   bash scripts/run_10_quickgen.sh                                                    # default 1x1
#   bash scripts/run_10_quickgen.sh --models sd15,sdxl_base --prompt-sets witch
#   bash scripts/run_10_quickgen.sh --all-models --prompt-sets witch --run-label demo01
#   bash scripts/run_10_quickgen.sh --list                                             # 利用可能な key 一覧
#
# venv: ~/.venvs/dfs2026-dev
# 出力: outputs/10_quickgen/<run_label>/ (10_quickgen.py 内部で run.log を書く)
# caffeinate: 大 model x 多 prompt の組み合わせは長時間化するため有効化

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

VENV="${HOME}/.venvs/dfs2026-dev"
if [ ! -f "${VENV}/bin/activate" ]; then
  echo "[error] venv not found: ${VENV}" >&2
  exit 1
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

echo "=========================================="
echo " 10 quickgen"
echo " started : $(date '+%Y-%m-%d %H:%M:%S')"
echo " python  : $(which python)"
echo " repo    : ${REPO_ROOT}"
echo " args    : $*"
echo "=========================================="

START_EPOCH=$(date +%s)
caffeinate -i python scripts/10_quickgen.py "$@"
RC=$?
END_EPOCH=$(date +%s)
DUR=$((END_EPOCH - START_EPOCH))

echo
echo "=========================================="
echo " finished : $(date '+%Y-%m-%d %H:%M:%S')"
echo " elapsed  : ${DUR}s ($((DUR/60))m$((DUR%60))s)"
echo " exitcode : ${RC}"
echo "=========================================="
exit "${RC}"
