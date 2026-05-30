#!/usr/bin/env bash
# run_claude_code_s7.sh — Run S7 Tier-2 experiments via Claude Code CLI.
#
# Models: Sonnet 4.6, Opus 4.6, Opus 4.7, Opus 4.8
# Protocol aligned with claude_code_sonnet46_s7 / openhands_gpt4o_s7:
#   S7: 10 sessions, isolated workspace, max_turns=50
#
# No GPU required — runs against the Claude Code subscription/API only.
#
# Partial runs are resumed automatically: completed sessions are skipped and
# the workspace is restored from the last snapshot. Usage-limit / auth
# failures abort immediately (non-zero exit) instead of recording zeros.
#
# Prerequisites:
#   - Claude Code CLI on PATH (`curl -fsSL https://claude.ai/install.sh | bash`)
#   - Logged in (`claude /login`)
#   - AgingBench deps: `uv sync --project prototype --extra dev`
#
# Usage:
#   ./scripts/run_claude_code_s7.sh
#   SEEDS=3 ./scripts/run_claude_code_s7.sh                    # seeds 42,43,44 from yaml base
#   MODELS="opus-4.8" SEED_LIST="43,45" ./scripts/run_claude_code_s7.sh
#   MODELS="opus-4.6,opus-4.7" SEED_LIST="43,45" ./scripts/run_claude_code_s7.sh --detach
#   OUTPUT_ROOT=experiments/results/claude_code_s7 ./scripts/run_claude_code_s7.sh
#   SKIP_COMPLETED=0 ./scripts/run_claude_code_s7.sh
#   CLEAN=1 ./scripts/run_claude_code_s7.sh
#
# Multi-seed layout:
#   seed 42 (legacy, SEEDS=1):  {output}/{model}/{sut_id}/
#   explicit SEED_LIST:         {output}/{model}/seed_{N}/{sut_id}/

set -euo pipefail

# Avoid tee to /dev/stdout in non-TTY environments (CI, detached nohup).
export LOG_FILE="${LOG_FILE:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REGISTRY="${PROJECT_ROOT}/agingbench/registry/suts/claude_code"

# Defaults (override via env)
SEEDS="${SEEDS:-1}"
SEED_LIST="${SEED_LIST:-}"   # e.g. "43,45" — run each seed explicitly (for 3-seed avg with existing seed-42)
SESSIONS="${SESSIONS:-10}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/experiments/results/claude_code_s7}"
MODELS="${MODELS:-sonnet-4.6}"
CARD_FLAG="${CARD_FLAG:-"--card"}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
CLEAN="${CLEAN:-0}"
MODEL_DELAY_SEC="${MODEL_DELAY_SEC:-0}"   # pause between models (e.g. 60)

export PATH="${HOME}/.local/bin:${PATH}"

log() {
  local line
  line="$(printf '[%s] %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*")"
  if [[ -n "${LOG_FILE}" ]]; then
    printf '%s\n' "${line}" >> "${LOG_FILE}"
  else
    printf '%s\n' "${line}"
  fi
}
die() { log "ERROR: $*" >&2; exit 1; }

# ---- preflight -------------------------------------------------------------

command -v uv >/dev/null 2>&1 || die "uv not found. Install: https://docs.astral.sh/uv/"

log "Syncing dependencies (prototype + dev/pytest)..."
uv sync --project "${PROJECT_ROOT}" --extra dev --quiet

command -v claude >/dev/null 2>&1 || die "Claude Code CLI not found. Install: curl -fsSL https://claude.ai/install.sh | bash"

if ! claude auth status 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); sys.exit(0 if d.get('loggedIn') else 1)"; then
  die "Claude Code not authenticated. Run: claude /login"
fi

run_agingbench() {
  uv run --project "${PROJECT_ROOT}" agingbench "$@"
}

sut_for_model() {
  local model="$1"
  case "${model}" in
    sonnet-4.6) echo "${REGISTRY}/claude_code_sonnet46_s7.yaml" ;;
    opus-4.6)   echo "${REGISTRY}/claude_code_opus46_s7_seed45.yaml" ;;
    opus-4.7)   echo "${REGISTRY}/claude_code_opus47_s7_seed45.yaml" ;;
    opus-4.8)   echo "${REGISTRY}/claude_code_opus48_s7_seed45.yaml" ;;
    *) die "Unknown model shorthand: ${model} (use sonnet-4.6, opus-4.6, opus-4.7, or opus-4.8)" ;;
  esac
}

sut_id_from_yaml() {
  python3 -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['sut_id'])" "$1"
}

metrics_file_for_run() {
  local out_dir="$1" sut_id="$2"
  echo "${out_dir}/${sut_id}/metrics.json"
}

is_run_complete() {
  local metrics_file="$1" expected_sessions="$2"
  [[ -f "${metrics_file}" ]] || return 1
  python3 - "${metrics_file}" "${expected_sessions}" <<'PY'
import json, sys
path, expected = sys.argv[1], int(sys.argv[2])
with open(path) as f:
    d = json.load(f)
if d.get("phase") == "phase_0_stub":
    sys.exit(1)
if d.get("m0") is None:
    sys.exit(1)
if d.get("run_status") != "complete":
    sys.exit(1)
if len(d.get("per_session") or []) < expected:
    sys.exit(1)
sys.exit(0)
PY
}

partial_sessions_done() {
  local metrics_file="$1"
  [[ -f "${metrics_file}" ]] || { echo 0; return; }
  python3 - "${metrics_file}" <<'PY'
import json, sys
path = sys.argv[1]
try:
    with open(path) as f:
        d = json.load(f)
except (OSError, json.JSONDecodeError):
    print(0)
else:
    print(len(d.get("per_session") or []))
PY
}

output_dir_for() {
  local model="$1" seed="${2:-}"
  if [[ -n "${seed}" ]]; then
    echo "${OUTPUT_ROOT}/${model}/seed_${seed}"
  else
    echo "${OUTPUT_ROOT}/${model}"
  fi
}

run_one_model() {
  local model="$1" seed="${2:-}" sut_yaml="$3"
  local out_dir sut_id metrics_file tmp_sut="" effective_yaml="${sut_yaml}"

  out_dir="$(output_dir_for "${model}" "${seed}")"
  sut_id="$(sut_id_from_yaml "${sut_yaml}")"
  metrics_file="$(metrics_file_for_run "${out_dir}" "${sut_id}")"

  if [[ -n "${seed}" ]]; then
    tmp_sut="$(mktemp --suffix=.yaml)"
    sed "s/^seed:.*/seed: ${seed}/" "${sut_yaml}" > "${tmp_sut}"
    effective_yaml="${tmp_sut}"
  fi

  if [[ "${SKIP_COMPLETED}" == "1" ]] && is_run_complete "${metrics_file}" "${SESSIONS}"; then
    log "SKIP (complete): ${model} seed=${seed:-yaml} -> ${metrics_file}"
    SKIP=$((SKIP + 1))
    [[ -n "${tmp_sut}" ]] && rm -f "${tmp_sut}"
    return 0
  fi

  local done
  done="$(partial_sessions_done "${metrics_file}")"
  if [[ "${done}" -gt 0 ]]; then
    log "RESUME  model=${model}  seed=${seed:-yaml}  sessions_done=${done}/${SESSIONS}"
  else
    log "RUN  model=${model}  seed=${seed:-yaml}  sessions=${SESSIONS}  sut=${effective_yaml}"
  fi

  set +e
  # Paper-release CLI: curated S7 uses scenario.yaml default_sessions (10).
  # Do not pass --sessions here (requires --generated in this tree).
  run_agingbench run \
    --scenario s7_research_notes \
    --sut "${effective_yaml}" \
    --seeds 1 \
    --output "${out_dir}" \
    ${CARD_FLAG:+"${CARD_FLAG}"}
  local rc=$?
  set -e
  [[ -n "${tmp_sut}" ]] && rm -f "${tmp_sut}"

  TOTAL=$((TOTAL + 1))
  if [[ ${rc} -ne 0 ]]; then
    FAIL=$((FAIL + 1))
    log "FAILED (exit ${rc}): ${model} seed=${seed:-yaml} — partial progress saved; re-run to resume"
    return 1
  fi
  log "OK: ${model} seed=${seed:-yaml} -> ${out_dir}"
  return 0
}

run_batch() {
  LOG_FILE="${LOG_FILE:-${OUTPUT_ROOT}/run.log}"
  mkdir -p "${OUTPUT_ROOT}"

  log "Claude Code S7 batch starting"
  log "  models:    ${MODELS}"
  log "  sessions:  ${SESSIONS}"
  if [[ -n "${SEED_LIST}" ]]; then
    log "  seed_list: ${SEED_LIST}"
  else
    log "  seeds:     ${SEEDS} (sequential from yaml base seed)"
  fi
  log "  output:    ${OUTPUT_ROOT}"
  log "  skip done: ${SKIP_COMPLETED}"
  log "  clean:     ${CLEAN}"
  log "  model_delay_sec: ${MODEL_DELAY_SEC}"
  CLAUDE_VERSION="$(claude --version 2>/dev/null | head -1 || true)"
  log "  claude:    ${CLAUDE_VERSION:-unknown}"

  if [[ "${CLEAN}" == "1" ]]; then
    IFS=',' read -r -a CLEAN_MODEL_ARR <<< "${MODELS}"
    for model in "${CLEAN_MODEL_ARR[@]}"; do
      model="$(echo "${model}" | xargs)"
      target="${OUTPUT_ROOT}/${model}"
      if [[ -d "${target}" ]]; then
        log "CLEAN  removing ${target}"
        rm -rf "${target}"
      fi
    done
  fi

  TOTAL=0
  FAIL=0
  SKIP=0

  IFS=',' read -r -a MODEL_ARR <<< "${MODELS}"

  model_idx=0
  for model in "${MODEL_ARR[@]}"; do
    model="$(echo "${model}" | xargs)"
    if [[ "${model_idx}" -gt 0 && "${MODEL_DELAY_SEC}" -gt 0 ]]; then
      log "WAIT  ${MODEL_DELAY_SEC}s before next model (${model})"
      sleep "${MODEL_DELAY_SEC}"
    fi
    model_idx=$((model_idx + 1))
    sut_yaml="$(sut_for_model "${model}")"
    [[ -f "${sut_yaml}" ]] || die "Missing SUT config: ${sut_yaml}"

    if [[ -n "${SEED_LIST}" ]]; then
      IFS=',' read -r -a SEED_ARR <<< "${SEED_LIST}"
      for seed in "${SEED_ARR[@]}"; do
        seed="$(echo "${seed}" | xargs)"
        run_one_model "${model}" "${seed}" "${sut_yaml}" || break
      done
    elif [[ "${SEEDS}" -gt 1 ]]; then
      out_dir="${OUTPUT_ROOT}/${model}"
      sut_id="$(sut_id_from_yaml "${sut_yaml}")"
      if [[ "${SKIP_COMPLETED}" == "1" ]]; then
        all_done=1
        for ((i=0; i<SEEDS; i++)); do
          mf="${out_dir}/s7_research_notes/${sut_id}/seed_${i}/metrics.json"
          if [[ ! -f "${mf}" ]] || ! is_run_complete "${mf}" "${SESSIONS}"; then
            all_done=0
            break
          fi
        done
        if [[ "${all_done}" == "1" ]]; then
          log "SKIP (complete): ${model} / ${SEEDS} seeds"
          SKIP=$((SKIP + 1))
          continue
        fi
      fi
      log "RUN  model=${model}  seeds=${SEEDS}  sessions=${SESSIONS}"
      set +e
      run_agingbench run \
        --scenario s7_research_notes \
        --sut "${sut_yaml}" \
        --seeds "${SEEDS}" \
        --output "${out_dir}" \
        ${CARD_FLAG:+"${CARD_FLAG}"}
      rc=$?
      set -e
      TOTAL=$((TOTAL + 1))
      if [[ ${rc} -ne 0 ]]; then
        FAIL=$((FAIL + 1))
        log "FAILED (exit ${rc}): ${model}"
        break
      fi
      log "OK: ${model} -> ${out_dir}"
    else
      run_one_model "${model}" "" "${sut_yaml}" || break
    fi
  done

  log "Done: $((TOTAL - FAIL))/${TOTAL} runs succeeded, ${SKIP} skipped"
  [[ ${FAIL} -eq 0 ]] || exit 1
}

if [[ "${1:-}" == "--detach" ]]; then
  mkdir -p "${OUTPUT_ROOT}"
  export LOG_FILE="${OUTPUT_ROOT}/run.log"
  nohup env MODELS="${MODELS}" SEED_LIST="${SEED_LIST}" SEEDS="${SEEDS}" SESSIONS="${SESSIONS}" \
    OUTPUT_ROOT="${OUTPUT_ROOT}" SKIP_COMPLETED="${SKIP_COMPLETED}" CARD_FLAG="${CARD_FLAG}" \
    MODEL_DELAY_SEC="${MODEL_DELAY_SEC}" \
    bash "${BASH_SOURCE[0]}" >> "${OUTPUT_ROOT}/nohup.out" 2>&1 &
  echo $! > "${OUTPUT_ROOT}/run.pid"
  echo "Detached Claude Code S7 batch (PID $(cat "${OUTPUT_ROOT}/run.pid"))"
  echo "  log: ${OUTPUT_ROOT}/run.log"
  echo "  tail: tail -f ${OUTPUT_ROOT}/run.log"
  exit 0
fi

run_batch
