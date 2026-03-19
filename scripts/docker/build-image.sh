#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=./common.sh
source "${SCRIPT_DIR}/common.sh"

PROFILE="${DEFAULT_DOCKER_PROFILE}"
if [ "${1:-}" = "--profile" ]; then
  [ -n "${2:-}" ] || die "missing value for --profile"
  PROFILE="$(docker_profile "${2}")"
  shift 2
fi

INSTANCE="${1:-}"
require_instance "${INSTANCE}"
require_clean_git
configure_proxy_env

build_args=()
for key in HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy; do
  value="${!key:-}"
  if [ -n "${value}" ]; then
    build_args+=(--build-arg "${key}=${value}")
  fi
done

export DOCKER_BUILDKIT=1
docker build \
  --network=host \
  --build-arg "BASE_IMAGE=$(docker_profile_base_image "${PROFILE}")" \
  --build-arg "ROBOCLAW_DOCKER_PROFILE=${PROFILE}" \
  --build-arg "ROBOCLAW_INSTALL_ROS2=$(docker_profile_installs_ros2 "${PROFILE}")" \
  --build-arg "ROBOCLAW_ROS2_DISTRO=$(docker_profile_ros_distro "${PROFILE}")" \
  --build-arg "ROBOCLAW_ROS2_CONTROL_PYTHON=$(docker_profile_control_bridge_python "${PROFILE}")" \
  "${build_args[@]}" \
  -t "$(image_ref "${INSTANCE}" "${PROFILE}")" \
  -t "$(dev_image_ref "${INSTANCE}" "${PROFILE}")" \
  "${REPO_ROOT}"
