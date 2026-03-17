---
name: embodiment-authoring
description: Workspace-first rules for generating embodied assemblies, deployments, adapters, and simulator assets.
always: true
---

# Embodiment Authoring

## Policy

- Treat `roboclaw/embodied/` as framework code.
- Only edit framework code when changing generic contracts, reusable component manifests, shared runtime logic, or transport/procedure abstractions.
- Do not put user-specific assemblies, deployments, adapters, scenarios, or lab configs under `roboclaw/embodied/`.
- Generate concrete embodiment assets under `~/.roboclaw/workspace/embodied/` for real user setups.

## Workspace-First Flow

1. Read `EMBODIED.md` in the active workspace before creating embodied files.
2. Capture discovered facts in `embodied/intake/<slug>.md`.
3. Reuse built-in robot and sensor ids when they already exist in framework code.
4. Create or update only the workspace files needed for this setup:
   - `embodied/robots/`
   - `embodied/sensors/`
   - `embodied/assemblies/`
   - `embodied/deployments/`
   - `embodied/adapters/`
   - `embodied/simulators/`
5. Keep ids stable so later chat turns can refine the same setup instead of generating a new one.

## Boundaries

- `robots/` in framework may contain reusable robot manifests such as supported open-source bodies.
- Attachment placement, ROS2 namespaces, deployment connection params, lab safety limits, and simulator worlds are setup-specific and belong in workspace assets.
- If a setup needs a new robot manifest that is not reusable enough for framework, create it in workspace first.

## Scaffolding

- Prefer reading and adapting files under `embodied/_templates/` in the workspace instead of inventing structure from scratch.
- The templates are skeletal on purpose. Do not preserve placeholder capability families, mounts, attachment ids, simulator backends, or safety defaults unless intake facts confirm them.
- Use export names that the workspace loader can discover: `ROBOT`, `SENSOR`, `ASSEMBLY`, `DEPLOYMENT`, `ADAPTER`, `WORLD`, `SCENARIO`, or the plural form of each.
- The intake note should record:
  - robot model and embodiment type
  - sensors and mounting points
  - ROS2 packages, nodes, namespaces, topics, actions, services
  - real vs sim targets
  - deployment-specific connection facts
  - safety or calibration constraints
