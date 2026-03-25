# Docker Installation

This guide is the Docker installation path for RoboClaw.

If you do not want Docker, use [INSTALLATION.md](./INSTALLATION.md).

## 1. Prerequisites

Start from a clean clone:

```bash
git clone https://github.com/MINT-SJTU/RoboClaw.git
cd RoboClaw
```

## 2. Build the Docker Image

```bash
docker build -t roboclaw .
```

## 3. Initialize RoboClaw

```bash
docker run -v ~/.roboclaw:/root/.roboclaw --rm roboclaw onboard
```

## 4. Configure the Model Provider

Edit `~/.roboclaw/config.json` on your host to add API keys or provider settings. See [INSTALLATION.md](./INSTALLATION.md#5-configure-the-model-provider) for provider details.

## 5. Verify Inside Docker

```bash
docker run -v ~/.roboclaw:/root/.roboclaw --rm roboclaw status
```

Check that:

- `Config` is shown as `✓`
- `Workspace` is shown as `✓`
- the current `Model` is correct

## 6. Run the Agent

```bash
docker run -v ~/.roboclaw:/root/.roboclaw --rm roboclaw agent -m "hello"
```

## 7. Run the Gateway

```bash
docker run -v ~/.roboclaw:/root/.roboclaw -p 18790:18790 roboclaw gateway
```

## 8. Docker Compose

You can also use Docker Compose:

```bash
docker compose run --rm roboclaw-cli onboard     # first-time setup
docker compose up -d roboclaw-gateway             # start gateway
docker compose run --rm roboclaw-cli agent -m "Hello!"
docker compose logs -f roboclaw-gateway           # view logs
```

Current compose roles:

- `roboclaw-cli`: normal interactive CLI container
- `roboclaw-gateway`: normal gateway container
- `roboclaw-g1-cli`: G1-ready CLI with DDS env sourced and `network_mode: host`
- `roboclaw-g1-agent`: G1-ready interactive agent container
- `roboclaw-g1-gateway`: G1-ready gateway container


## 9. Optional: Unitree G1 Simulation Dependencies

If you want to control a Unitree G1 Isaac Lab simulation from Docker, keep the main image unchanged and install the extra G1 runtime dependencies into your mounted `~/.roboclaw` volume:

```bash
bash scripts/docker/install-g1-deps.sh
```

This installs:

- `CycloneDDS` under `~/.roboclaw/g1/src/cyclonedds/install`
- `unitree_sdk2py` under `~/.roboclaw/g1/python`
- an env file at `~/.roboclaw/g1/env.sh`

Then run RoboClaw by sourcing that env file inside the container.

Status check:

```bash
docker run --rm -it   -v ~/.roboclaw:/root/.roboclaw   --entrypoint /bin/bash   roboclaw -lc '. /root/.roboclaw/g1/env.sh && roboclaw status'
```

Agent with host DDS networking:

```bash
docker run --rm -it   --network host   -v ~/.roboclaw:/root/.roboclaw   --entrypoint /bin/bash   roboclaw -lc '. /root/.roboclaw/g1/env.sh && roboclaw agent'
```

Compose equivalents:

```bash
docker compose run --rm roboclaw-g1-cli
docker compose run --rm roboclaw-g1-cli '. /root/.roboclaw/g1/env.sh && roboclaw status'
docker compose run --rm roboclaw-g1-agent
docker compose up -d roboclaw-g1-gateway
```

For G1 simulation, start `unitree_sim_isaaclab` on the host first, then use the `g1_*` embodied actions inside RoboClaw.
