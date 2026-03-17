"""Default embodied catalog and registries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from roboclaw.embodied.definition.components.robots import RobotRegistry, SO101_ROBOT
from roboclaw.embodied.definition.components.sensors import RGB_CAMERA, SensorRegistry
from roboclaw.embodied.definition.systems.assemblies import AssemblyRegistry
from roboclaw.embodied.definition.systems.deployments import DeploymentRegistry
from roboclaw.embodied.definition.systems.simulators import SimulatorRegistry
from roboclaw.embodied.execution.integration.adapters import AdapterRegistry
from roboclaw.embodied.execution.integration.bridges import (
    BridgeRegistry,
    DEFAULT_DOMAIN_BRIDGES,
)
from roboclaw.embodied.execution.orchestration.procedures import (
    DEFAULT_PROCEDURES,
    ProcedureRegistry,
)

if TYPE_CHECKING:
    from roboclaw.embodied.workspace import WorkspaceInspectOptions, WorkspaceLoadReport


@dataclass(frozen=True)
class EmbodiedCatalog:
    """One place to access the built-in embodied registries."""

    robots: RobotRegistry
    sensors: SensorRegistry
    assemblies: AssemblyRegistry
    bridges: BridgeRegistry
    adapters: AdapterRegistry
    procedures: ProcedureRegistry
    deployments: DeploymentRegistry
    simulators: SimulatorRegistry


def build_default_catalog() -> EmbodiedCatalog:
    """Build the default embodied catalog.

    This catalog contains reusable framework definitions only. Concrete
    assemblies, deployments, adapters, and simulator scenarios should be
    generated under the active workspace.
    """

    robots = RobotRegistry()
    robots.register(SO101_ROBOT)

    sensors = SensorRegistry()
    sensors.register(RGB_CAMERA)

    assemblies = AssemblyRegistry()

    bridges = BridgeRegistry()
    for bridge in DEFAULT_DOMAIN_BRIDGES:
        bridges.register(bridge)

    adapters = AdapterRegistry()

    procedures = ProcedureRegistry()
    for procedure in DEFAULT_PROCEDURES:
        procedures.register(procedure)

    deployments = DeploymentRegistry()

    simulators = SimulatorRegistry()

    return EmbodiedCatalog(
        robots=robots,
        sensors=sensors,
        assemblies=assemblies,
        bridges=bridges,
        adapters=adapters,
        procedures=procedures,
        deployments=deployments,
        simulators=simulators,
    )


def build_catalog(
    workspace: Path | None = None,
    workspace_options: WorkspaceInspectOptions | None = None,
) -> EmbodiedCatalog:
    """Build the framework catalog and optionally merge workspace assets."""

    catalog = build_default_catalog()
    if workspace is None:
        return catalog

    from roboclaw.embodied.workspace import load_workspace_assets

    return load_workspace_assets(catalog, workspace, options=workspace_options)


def inspect_workspace(
    workspace: Path,
    options: WorkspaceInspectOptions | None = None,
) -> WorkspaceLoadReport:
    """Inspect workspace assets using loader static contract checks."""

    from roboclaw.embodied.workspace import inspect_workspace_assets

    return inspect_workspace_assets(workspace, options=options)
