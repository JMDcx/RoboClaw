#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=./common.sh
source "${SCRIPT_DIR}/common.sh"

usage() {
  cat <<'USAGE'
Usage:
  scripts/docker/matrix.sh list-profiles
  scripts/docker/matrix.sh build <instance> [--profiles profile1,profile2]
  scripts/docker/matrix.sh start-dev <instance> [--profiles profile1,profile2]
  scripts/docker/matrix.sh run-task <instance> [--profiles profile1,profile2] -- <roboclaw args...>

Default matrix profiles: ubuntu2204-ros2,ubuntu2404-ros2
USAGE
}

run_for_profiles() {
  local action="${1}"
  local instance="${2}"
  shift 2

  local profile
  for profile in "$@"; do
    printf '\n=== %s :: %s ===\n' "${profile}" "${action}"
    case "${action}" in
      build)
        "${SCRIPT_DIR}/build-image.sh" --profile "${profile}" "${instance}"
        ;;
      start-dev)
        "${SCRIPT_DIR}/start-dev.sh" --profile "${profile}" "${instance}"
        ;;
      *)
        die "unsupported action: ${action}"
        ;;
    esac
  done
}

run_task_matrix() {
  local instance="${1}"
  local profiles_csv="${2}"
  shift 2

  [ "$#" -gt 0 ] || die "run-task requires a command after --"

  local profiles=()
  while IFS= read -r profile; do
    [ -n "${profile}" ] && profiles+=("${profile}")
  done < <(split_profiles_csv "${profiles_csv}")

  local profile
  for profile in "${profiles[@]}"; do
    printf '\n=== %s :: build ===\n' "${profile}"
    "${SCRIPT_DIR}/build-image.sh" --profile "${profile}" "${instance}"
  done

  local logs_dir="${ROBOCLAW_DOCKER_HOME}/matrix-logs"
  mkdir -p "${logs_dir}"

  local pids=()
  local log_files=()
  for profile in "${profiles[@]}"; do
    local log_file="${logs_dir}/${instance}-${profile}-$(date +%Y%m%d-%H%M%S).log"
    log_files+=("${profile}:${log_file}")
    (
      printf 'PROFILE=%s\n' "${profile}"
      printf 'IMAGE=%s\n' "$(image_ref "${instance}" "${profile}")"
      printf 'INSTANCE_DIR=%s\n\n' "$(instance_dir "${instance}" "${profile}")"
      "${SCRIPT_DIR}/run-task.sh" --profile "${profile}" "${instance}" "$@"
    ) >"${log_file}" 2>&1 &
    pids+=("$!")
  done

  local overall_status=0
  local idx=0
  for pid in "${pids[@]}"; do
    local profile="${profiles[${idx}]}"
    local log_file="${log_files[${idx}]#*:}"
    if wait "${pid}"; then
      printf '\n=== %s :: success ===\n' "${profile}"
    else
      printf '\n=== %s :: failed ===\n' "${profile}"
      overall_status=1
    fi
    cat "${log_file}"
    idx=$((idx + 1))
  done

  return "${overall_status}"
}

COMMAND="${1:-}"
[ -n "${COMMAND}" ] || {
  usage
  exit 1
}
shift

case "${COMMAND}" in
  list-profiles)
    list_docker_profiles
    exit 0
    ;;
  build|start-dev|run-task)
    ;;
  *)
    usage
    exit 1
    ;;
esac

INSTANCE="${1:-}"
require_instance "${INSTANCE}"
shift || true

PROFILES_CSV="${DEFAULT_MATRIX_PROFILES}"
if [ "${1:-}" = "--profiles" ]; then
  [ -n "${2:-}" ] || die "missing value for --profiles"
  split_profiles_csv "${2}" >/dev/null
  PROFILES_CSV="${2}"
  shift 2
fi

profiles=()
while IFS= read -r profile; do
  [ -n "${profile}" ] && profiles+=("${profile}")
done < <(split_profiles_csv "${PROFILES_CSV}")
[ "${#profiles[@]}" -gt 0 ] || die "no docker profiles resolved"

case "${COMMAND}" in
  build|start-dev)
    run_for_profiles "${COMMAND}" "${INSTANCE}" "${profiles[@]}"
    ;;
  run-task)
    [ "${1:-}" = "--" ] || die "run-task expects -- before the roboclaw command"
    shift
    run_task_matrix "${INSTANCE}" "$(join_profiles_csv "${profiles[@]}")" "$@"
    ;;
esac
