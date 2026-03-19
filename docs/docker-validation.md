# Docker Validation Workflow

Use this workflow for reproducible acceptance runs. It builds immutable images and runs the same
task across the matrix.

## Profiles

Default validation profiles:

- `ubuntu2204-ros2`
- `ubuntu2404-ros2`

Each profile is built from a clean Git worktree and tagged with the instance name, profile, and
current short commit hash.

## Build

Build the matrix for one instance:

```bash
./scripts/docker/matrix.sh build devbox
```

The build entrypoint requires a clean Git worktree. If the worktree is dirty, the build stops
before any image is produced.

## Run a Validation Task

Run the same RoboClaw command across the matrix:

```bash
./scripts/docker/matrix.sh run-task devbox -- status
./scripts/docker/matrix.sh run-task devbox -- agent -m "hello" --no-markdown
```

For embodied validation, use a bounded command sequence such as:

```bash
./scripts/docker/matrix.sh run-task devbox -- agent -m "I want to connect a real robot. Please guide me step by step."
```

Then continue the conversation with the device and setup facts that RoboClaw requests.

## Acceptance Notes

- Build uses host networking.
- Runtime uses host networking.
- Proxy values are discovered on the remote host and propagated into the build and runtime.
- The validation containers use the immutable image tags; they do not bind-mount the host repo.
- Instance-local calibration is prepared under `~/.roboclaw-docker/instances/<instance>--<profile>/calibration/`.
- Session metadata and generated deployment facts persist `/dev/serial/by-id/...` identifiers only.

## Cleanup

The matrix workflow is designed to keep build cache available. Avoid removing image layers unless
you intentionally want a cold rebuild.
