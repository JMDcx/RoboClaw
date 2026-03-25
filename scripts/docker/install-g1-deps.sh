#!/usr/bin/env bash
set -euo pipefail

IMAGE="roboclaw"
ROBOCLAW_HOME="${HOME}/.roboclaw"
BUILD_JOBS="2"
CONTAINER_ROBOCLAW_HOME="/root/.roboclaw"
G1_ROOT="${CONTAINER_ROBOCLAW_HOME}/g1"
PYTHON_DIR="${G1_ROOT}/python"

usage() {
  cat <<USAGE
Usage: $0 [--image IMAGE] [--roboclaw-home PATH] [--jobs N]

Install Unitree G1 runtime dependencies for the current Docker workflow.
This script keeps the main RoboClaw image unchanged. It launches a helper
container and installs CycloneDDS plus unitree_sdk2py into the mounted
~/.roboclaw volume so the result survives container recreation.

Options:
  --image IMAGE           Docker image to use (default: roboclaw)
  --roboclaw-home PATH    Host path mounted as /root/.roboclaw (default: ~/.roboclaw)
  --jobs N                Parallel build jobs for CycloneDDS (default: 2)
  -h, --help              Show this help message
USAGE
}

while [ $# -gt 0 ]; do
  case "$1" in
    --image)
      IMAGE="${2:-}"
      shift 2
      ;;
    --roboclaw-home)
      ROBOCLAW_HOME="${2:-}"
      shift 2
      ;;
    --jobs)
      BUILD_JOBS="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required" >&2
  exit 1
fi

mkdir -p "${ROBOCLAW_HOME}"

if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  echo "Docker image '${IMAGE}' not found. Build it first with: docker build -t ${IMAGE} ." >&2
  exit 1
fi

echo "Installing G1 dependencies into ${ROBOCLAW_HOME} using image ${IMAGE}"

docker run --rm \
  -v "${ROBOCLAW_HOME}:${CONTAINER_ROBOCLAW_HOME}" \
  --entrypoint /bin/bash \
  "${IMAGE}" -lc "
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends build-essential cmake git ca-certificates
rm -rf /var/lib/apt/lists/*

G1_ROOT='${G1_ROOT}'
PYTHON_DIR='${PYTHON_DIR}'
BUILD_JOBS='${BUILD_JOBS}'
mkdir -p \"\${G1_ROOT}/src\" \"\${PYTHON_DIR}\"

if [ ! -d \"\${G1_ROOT}/src/cyclonedds/.git\" ]; then
  git clone --depth 1 --branch releases/0.10.x https://github.com/eclipse-cyclonedds/cyclonedds \"\${G1_ROOT}/src/cyclonedds\"
fi

mkdir -p \"\${G1_ROOT}/src/cyclonedds/build\" \"\${G1_ROOT}/src/cyclonedds/install\"
cd \"\${G1_ROOT}/src/cyclonedds/build\"
cmake .. -DCMAKE_INSTALL_PREFIX=../install
cmake --build . --target install -j\"\${BUILD_JOBS}\"

export CYCLONEDDS_HOME=\"\${G1_ROOT}/src/cyclonedds/install\"
export CMAKE_PREFIX_PATH=\"\${CYCLONEDDS_HOME}\"
export LD_LIBRARY_PATH=\"\${CYCLONEDDS_HOME}/lib:\${LD_LIBRARY_PATH:-}\"
export PATH=\"\${CYCLONEDDS_HOME}/bin:\${PATH}\"
export PYTHONPATH=\"\${PYTHON_DIR}:\${PYTHONPATH:-}\"

if [ ! -d \"\${G1_ROOT}/src/unitree_sdk2_python/.git\" ]; then
  git clone --depth 1 https://github.com/unitreerobotics/unitree_sdk2_python.git \"\${G1_ROOT}/src/unitree_sdk2_python\"
fi

python3 -m pip install --no-cache-dir --default-timeout 300 --retries 10 --target \"\${PYTHON_DIR}\" --ignore-installed \"\${G1_ROOT}/src/unitree_sdk2_python\"

mkdir -p "\${PYTHON_DIR}/unitree_sdk2py/utils/lib"
cp -f "\${G1_ROOT}/src/unitree_sdk2_python/unitree_sdk2py/utils/lib/"* "\${PYTHON_DIR}/unitree_sdk2py/utils/lib/"

python3 - <<'PY'
from pathlib import Path
path = Path('/root/.roboclaw/g1/python/unitree_sdk2py/__init__.py')
text = path.read_text(encoding='utf-8')
if 'from . import idl, utils, core, rpc, go2, b2' in text:
    path.write_text(
        'from . import idl, utils, core, rpc, go2\n\n'
        '__all__ = [\n'
        '    \"idl\",\n'
        '    \"utils\",\n'
        '    \"core\",\n'
        '    \"rpc\",\n'
        '    \"go2\",\n'
        ']\n',
        encoding='utf-8',
    )

lib_dir = Path('/root/.roboclaw/g1/python/unitree_sdk2py/utils/lib')
assert lib_dir.exists() and any(lib_dir.iterdir()), 'unitree_sdk2py CRC native libs were not copied'
PY

cat > \"\${G1_ROOT}/env.sh\" <<'ENVEOF'
export CYCLONEDDS_HOME=/root/.roboclaw/g1/src/cyclonedds/install
export CMAKE_PREFIX_PATH=/root/.roboclaw/g1/src/cyclonedds/install
export LD_LIBRARY_PATH=/root/.roboclaw/g1/src/cyclonedds/install/lib:\${LD_LIBRARY_PATH:-}
export PATH=/root/.roboclaw/g1/src/cyclonedds/install/bin:\${PATH}
export PYTHONPATH=/root/.roboclaw/g1/python:\${PYTHONPATH:-}
ENVEOF

. \"\${G1_ROOT}/env.sh\"
python3 -c 'import cyclonedds, unitree_sdk2py; print("g1 deps ok")'
"

cat <<EOF

G1 dependencies installed successfully.

Before running RoboClaw in Docker with G1, source the env file inside the container:
  . /root/.roboclaw/g1/env.sh

Example one-off status check:
  docker run --rm -it \
    -v ~/.roboclaw:/root/.roboclaw \
    --entrypoint /bin/bash \
    roboclaw -lc '. /root/.roboclaw/g1/env.sh && roboclaw status'

Example agent session with host DDS networking:
  docker run --rm -it \
    --network host \
    -v ~/.roboclaw:/root/.roboclaw \
    --entrypoint /bin/bash \
    roboclaw -lc '. /root/.roboclaw/g1/env.sh && roboclaw agent'
EOF
