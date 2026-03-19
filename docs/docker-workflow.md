# Docker Dev Workflow

Use this workflow when you want fast iteration on the latest host source without rebuilding on every Python edit.

The dev workflow is separate from the immutable validation workflow:

- Dev containers are bind-mounted to the host RoboClaw repo.
- Dev containers run the latest host source through `PYTHONPATH` precedence.
- Validation images stay clean, commit-tagged, and rebuild only when dependencies change.

## Dev Containers

Start a long-lived dev container for one instance and profile:

```bash
./scripts/docker/start-dev.sh devbox --profile ubuntu2404-ros2
./scripts/docker/exec-dev.sh devbox --profile ubuntu2404-ros2
```

The dev container mounts the host repo at `/roboclaw-source` and uses it first on `PYTHONPATH`,
so ordinary source edits are visible immediately without rebuilding the image.

The dev container still uses the same isolated instance state under:

```text
~/.roboclaw-docker/instances/<instance>--<profile>/
```

That instance directory stores:

- `config.json`
- `workspace/`
- `calibration/`
- runtime user state under `home/`

The bootstrap step seeds instance-local calibration from the host's canonical
`~/.roboclaw/calibration/` tree when it exists. If that tree is empty but a
legacy lerobot calibration cache exists, the bootstrap step imports it once into
the instance-local canonical layout.

## When To Rebuild

Use `build-image.sh` only when the runtime environment changes:

- Dockerfile changes
- ROS/system dependency changes
- explicit Python dependency changes
- release or acceptance validation

For normal Python source edits, keep the dev container running and rerun the command you are
testing.

## Dependencies

`scservo_sdk` ships as part of the RoboClaw source tree, so the Docker image no
longer depends on a host-side site-packages drop.

## Related Validation Workflow

The immutable matrix acceptance workflow lives in:

- [Docker Validation Workflow](./docker-validation.md)
