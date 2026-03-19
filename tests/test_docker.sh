#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1

INSTANCE_NAME="${INSTANCE_NAME:-docker-test}"
SCRIPT_DIR="$(pwd)/scripts/docker"
# shellcheck source=../scripts/docker/common.sh
source "${SCRIPT_DIR}/common.sh"

echo "=== Running Docker matrix validation smoke ==="
STATUS_OUTPUT=$("${SCRIPT_DIR}/matrix.sh" run-task "${INSTANCE_NAME}" -- status 2>&1) || true

echo "$STATUS_OUTPUT"

echo ""
echo "=== Validating output ==="
PASS=true

check() {
    if echo "$STATUS_OUTPUT" | grep -q "$1"; then
        echo "  PASS: found '$1'"
    else
        echo "  FAIL: missing '$1'"
        PASS=false
    fi
}

check "ubuntu2204-ros2 :: success"
check "ubuntu2404-ros2 :: success"
check "RoboClaw Status"
check "Config:"
check "Workspace:"

echo ""
if $PASS; then
    echo "=== All checks passed ==="
else
    echo "=== Some checks FAILED ==="
    exit 1
fi

echo ""
echo "=== Cleanup ==="
while IFS= read -r profile; do
    [ -n "$profile" ] || continue
    rm -rf "$(instance_dir "$INSTANCE_NAME" "$profile")"
done < <(split_profiles_csv "${DEFAULT_MATRIX_PROFILES}")
rm -rf "${ROBOCLAW_DOCKER_HOME}/matrix-logs/${INSTANCE_NAME}-"*
echo "Done."
