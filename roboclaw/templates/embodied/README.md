# Workspace Embodied Assets

This directory is where RoboClaw should generate or refine setup-specific embodied files for the active user.

## Purpose

- `robots/`: local-only robot manifests that should not live in framework yet
- `sensors/`: local-only sensor manifests that should not live in framework yet
- `assemblies/`: robot plus sensor attachments and available execution targets
- `deployments/`: site-specific connection values, device paths, namespaces, and safety overrides
- `adapters/`: transport-specific adapter bindings and entrypoints for this setup
- `simulators/`: world and scenario files for local simulation targets
- `intake/`: discovery notes captured from user conversation or environment inspection
- `_templates/`: intentionally minimal scaffolds for generated Python files

## Policy

- Prefer importing reusable framework definitions from `roboclaw.embodied.*`.
- Do not copy framework manifests into workspace unless the robot or sensor is truly local-only.
- Update these files instead of touching `roboclaw/embodied/` when the change is specific to one user's equipment.
- Workspace Python files are discovered by export name. Use `ROBOT`, `SENSOR`, `ASSEMBLY`, `DEPLOYMENT`, `ADAPTER`, `WORLD`, `SCENARIO`, or the plural form of each.
- Each generated file should include `WORKSPACE_ASSET = WorkspaceAssetContract(...)` with schema version and migration policy so the loader can validate and migrate safely.
- Treat `_templates/` as fragments to adapt after intake. They should not hardcode one robot class, one camera mount, or one simulator backend.
