# Embodied Workspace Policy

`roboclaw/embodied/` is framework code. It holds generic protocols, reusable robot manifests, shared sensor types, runtime abstractions, transport contracts, and common procedure definitions.

Concrete user setups must live in this workspace under `embodied/`, not in the package source tree.

## Rule

- Edit `roboclaw/embodied/` only when changing generic framework behavior.
- Put user-, lab-, or demo-specific embodied assets under `embodied/` in this workspace.
- Reuse built-in robot and sensor ids when available instead of copying framework definitions.

## Recommended Workspace Layout

```text
embodied/
  README.md
  intake/
  robots/
  sensors/
  assemblies/
  deployments/
  adapters/
    ros2/
  simulators/
    worlds/
    scenarios/
  notes/
  _templates/
```

## Generation Flow

1. Discover the robot, sensors, ROS2 interfaces, and target backends.
2. Write an intake note under `embodied/intake/`.
3. Reuse built-in component manifests where possible.
4. Generate setup-specific files under `embodied/assemblies`, `embodied/deployments`, `embodied/adapters`, and `embodied/simulators`.
5. If a robot or sensor is local-only and not reusable enough for framework, define it under `embodied/robots` or `embodied/sensors`.
6. Keep ids stable across later iterations so the same setup can be refined incrementally.

## Ownership Boundary

- Framework examples should not hardcode one user's SO101, Piper, xArm, or humanoid setup.
- A workspace assembly may reference a built-in robot id such as `so101`, but the specific ROS2 namespace, topics, camera mounts, and deployment connection values belong in workspace files.
- Workspace loader convention: generated Python files should export one of `ROBOT`, `SENSOR`, `ASSEMBLY`, `DEPLOYMENT`, `ADAPTER`, `WORLD`, `SCENARIO`, or their plural forms.
- Workspace contract metadata: generated Python files should also define `WORKSPACE_ASSET = WorkspaceAssetContract(...)` with `kind`, `schema_version`, `export_convention`, and `migration_policy`.
- Files under `embodied/_templates/` are intentionally minimal scaffolds, not ready-made arm demos. Replace the placeholders from intake facts instead of copying them verbatim.
