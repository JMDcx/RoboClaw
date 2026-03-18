#!/usr/bin/env bash
set -euo pipefail

if [ "${ROBOCLAW_ROS2_DISTRO:-none}" != "none" ] && [ -f "/opt/ros/${ROBOCLAW_ROS2_DISTRO}/setup.sh" ]; then
  # shellcheck disable=SC1090
  source "/opt/ros/${ROBOCLAW_ROS2_DISTRO}/setup.sh"
fi

exec /usr/local/bin/roboclaw-real "$@"
