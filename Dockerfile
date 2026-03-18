ARG BASE_IMAGE=python:3.11-noble
FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG ROBOCLAW_DOCKER_PROFILE=ubuntu2404
ARG ROBOCLAW_INSTALL_ROS2=0
ARG ROBOCLAW_ROS2_DISTRO=none
ENV ROBOCLAW_ROS2_DISTRO=${ROBOCLAW_ROS2_DISTRO}

# Install Node.js 20 for the WhatsApp bridge
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates gnupg git locales lsb-release && \
    mkdir -p /etc/apt/keyrings && \
    curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg && \
    echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_20.x nodistro main" > /etc/apt/sources.list.d/nodesource.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends nodejs && \
    rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir --upgrade pip uv

RUN if [ "${ROBOCLAW_INSTALL_ROS2}" = "1" ]; then \
      locale-gen en_US en_US.UTF-8 && \
      update-locale LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 && \
      curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | gpg --dearmor -o /etc/apt/keyrings/ros-archive-keyring.gpg && \
      . /etc/os-release && \
      echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu ${VERSION_CODENAME} main" > /etc/apt/sources.list.d/ros2.list && \
      apt-get update && \
      apt-get install -y --no-install-recommends \
        "ros-${ROBOCLAW_ROS2_DISTRO}-ros-base" \
        python3-argcomplete \
        python3-colcon-common-extensions && \
      rm -rf /var/lib/apt/lists/*; \
    fi

LABEL roboclaw.docker_profile="${ROBOCLAW_DOCKER_PROFILE}"
LABEL roboclaw.ros2_distro="${ROBOCLAW_ROS2_DISTRO}"

WORKDIR /app

# Install Python dependencies first (cached layer)
COPY pyproject.toml README.md LICENSE ./
RUN mkdir -p roboclaw bridge && touch roboclaw/__init__.py && \
    uv pip install --system --no-cache . && \
    rm -rf roboclaw bridge

# Copy the full source and install
COPY roboclaw/ roboclaw/
COPY bridge/ bridge/
RUN uv pip install --system --no-cache .

RUN mv /usr/local/bin/roboclaw /usr/local/bin/roboclaw-real
COPY scripts/docker/roboclaw-wrapper.sh /usr/local/bin/roboclaw
RUN chmod +x /usr/local/bin/roboclaw

# Build the WhatsApp bridge
WORKDIR /app/bridge
RUN npm install && npm run build
WORKDIR /app

# Create config directory
RUN mkdir -p /root/.roboclaw

# Gateway default port
EXPOSE 18790

ENTRYPOINT ["roboclaw"]
CMD ["status"]
