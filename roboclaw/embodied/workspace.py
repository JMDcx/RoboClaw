"""Workspace asset loading for embodied catalogs."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from types import ModuleType

from roboclaw.embodied.catalog import EmbodiedCatalog

try:
    from enum import StrEnum
except ImportError:  # pragma: no cover - Python < 3.11 fallback for local tooling.
    class StrEnum(str, Enum):
        """Fallback for Python versions without enum.StrEnum."""


WORKSPACE_SCHEMA_VERSION = "1.0"
SUPPORTED_WORKSPACE_SCHEMA_VERSIONS = (WORKSPACE_SCHEMA_VERSION,)


class WorkspaceAssetKind(StrEnum):
    """Supported workspace asset kinds."""

    ROBOT = "robot"
    SENSOR = "sensor"
    ASSEMBLY = "assembly"
    ADAPTER = "adapter"
    DEPLOYMENT = "deployment"
    WORLD = "world"
    SCENARIO = "scenario"


class WorkspaceExportConvention(StrEnum):
    """Export variable convention for one workspace asset file."""

    AUTO = "AUTO"
    ROBOT = "ROBOT"
    ROBOTS = "ROBOTS"
    SENSOR = "SENSOR"
    SENSORS = "SENSORS"
    ASSEMBLY = "ASSEMBLY"
    ASSEMBLIES = "ASSEMBLIES"
    ADAPTER = "ADAPTER"
    ADAPTERS = "ADAPTERS"
    DEPLOYMENT = "DEPLOYMENT"
    DEPLOYMENTS = "DEPLOYMENTS"
    WORLD = "WORLD"
    WORLDS = "WORLDS"
    SCENARIO = "SCENARIO"
    SCENARIOS = "SCENARIOS"


class WorkspaceMigrationPolicy(StrEnum):
    """How loader should handle unsupported schema versions."""

    STRICT = "strict"
    ACCEPT_UNSUPPORTED = "accept_unsupported"


class WorkspaceIssueLevel(StrEnum):
    """Validation issue severity."""

    WARNING = "warning"
    ERROR = "error"


class WorkspaceValidationStage(StrEnum):
    """Validation stage where one issue is produced."""

    IMPORT = "import"
    SCHEMA = "schema"
    CATALOG_CONFLICT = "catalog_conflict"
    LINT = "lint"


class WorkspaceLintProfile(StrEnum):
    """Lint profile controlling strictness of workspace checks."""

    BASIC = "basic"
    STRICT = "strict"


@dataclass(frozen=True)
class WorkspaceProvenance:
    """Source metadata for generated workspace assets."""

    source: str = "workspace"
    generator: str | None = None
    generated_by: str | None = None
    generated_at: str | None = None
    source_session: str | None = None
    source_path: str | None = None

    def __post_init__(self) -> None:
        if not self.source.strip():
            raise ValueError("Workspace provenance source cannot be empty.")
        for field_name in ("generator", "generated_by", "generated_at", "source_session", "source_path"):
            value = getattr(self, field_name)
            if value is not None and not value.strip():
                raise ValueError(f"Workspace provenance {field_name} cannot be empty when specified.")


@dataclass(frozen=True)
class WorkspaceAssetContract:
    """Metadata contract for one workspace asset module."""

    kind: WorkspaceAssetKind
    schema_version: str = WORKSPACE_SCHEMA_VERSION
    export_convention: WorkspaceExportConvention = WorkspaceExportConvention.AUTO
    migration_policy: WorkspaceMigrationPolicy = WorkspaceMigrationPolicy.STRICT
    provenance: WorkspaceProvenance = field(default_factory=WorkspaceProvenance)
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.schema_version.strip():
            raise ValueError("Workspace asset schema_version cannot be empty.")


@dataclass(frozen=True)
class WorkspaceValidationIssue:
    """One static validation issue for workspace loading."""

    level: WorkspaceIssueLevel
    stage: WorkspaceValidationStage
    code: str
    path: str
    message: str


@dataclass(frozen=True)
class WorkspaceStagedAsset:
    """Staged workspace asset metadata for dry-run reporting."""

    kind: WorkspaceAssetKind
    asset_id: str
    path: str
    module_name: str
    schema_version: str
    provenance: WorkspaceProvenance


@dataclass(frozen=True)
class WorkspaceInspectOptions:
    """Options for workspace inspection and dry-run behavior."""

    lint_profile: WorkspaceLintProfile = WorkspaceLintProfile.BASIC
    require_contract_metadata: bool = False
    require_provenance_metadata: bool = False
    fail_on_warnings: bool = False
    include_staged_assets: bool = True


@dataclass(frozen=True)
class WorkspaceLoadReport:
    """Validation and staging report for workspace assets and dry-run."""

    root: str
    schema_version: str = WORKSPACE_SCHEMA_VERSION
    lint_profile: WorkspaceLintProfile = WorkspaceLintProfile.BASIC
    loaded_counts: dict[WorkspaceAssetKind, int] = field(default_factory=dict)
    staged_assets: tuple[WorkspaceStagedAsset, ...] = field(default_factory=tuple)
    issues: tuple[WorkspaceValidationIssue, ...] = field(default_factory=tuple)

    @property
    def has_errors(self) -> bool:
        return any(issue.level == WorkspaceIssueLevel.ERROR for issue in self.issues)

    @property
    def has_warnings(self) -> bool:
        return any(issue.level == WorkspaceIssueLevel.WARNING for issue in self.issues)

    @property
    def stage_counts(self) -> dict[WorkspaceValidationStage, int]:
        counts = {stage: 0 for stage in WorkspaceValidationStage}
        for issue in self.issues:
            counts[issue.stage] += 1
        return counts


@dataclass(frozen=True)
class _WorkspaceGroupSpec:
    kind: WorkspaceAssetKind
    relative_dir: tuple[str, ...]
    singular_export: WorkspaceExportConvention
    plural_export: WorkspaceExportConvention

    @property
    def exports(self) -> tuple[str, str]:
        return (self.singular_export.value, self.plural_export.value)

    @property
    def allowed_conventions(self) -> tuple[WorkspaceExportConvention, ...]:
        return (WorkspaceExportConvention.AUTO, self.singular_export, self.plural_export)


@dataclass(frozen=True)
class _StagedAsset:
    kind: WorkspaceAssetKind
    asset_id: str
    path: Path
    module_name: str
    contract: WorkspaceAssetContract
    value: object


_GROUP_SPECS = (
    _WorkspaceGroupSpec(
        kind=WorkspaceAssetKind.ROBOT,
        relative_dir=("robots",),
        singular_export=WorkspaceExportConvention.ROBOT,
        plural_export=WorkspaceExportConvention.ROBOTS,
    ),
    _WorkspaceGroupSpec(
        kind=WorkspaceAssetKind.SENSOR,
        relative_dir=("sensors",),
        singular_export=WorkspaceExportConvention.SENSOR,
        plural_export=WorkspaceExportConvention.SENSORS,
    ),
    _WorkspaceGroupSpec(
        kind=WorkspaceAssetKind.ASSEMBLY,
        relative_dir=("assemblies",),
        singular_export=WorkspaceExportConvention.ASSEMBLY,
        plural_export=WorkspaceExportConvention.ASSEMBLIES,
    ),
    _WorkspaceGroupSpec(
        kind=WorkspaceAssetKind.ADAPTER,
        relative_dir=("adapters",),
        singular_export=WorkspaceExportConvention.ADAPTER,
        plural_export=WorkspaceExportConvention.ADAPTERS,
    ),
    _WorkspaceGroupSpec(
        kind=WorkspaceAssetKind.DEPLOYMENT,
        relative_dir=("deployments",),
        singular_export=WorkspaceExportConvention.DEPLOYMENT,
        plural_export=WorkspaceExportConvention.DEPLOYMENTS,
    ),
    _WorkspaceGroupSpec(
        kind=WorkspaceAssetKind.WORLD,
        relative_dir=("simulators", "worlds"),
        singular_export=WorkspaceExportConvention.WORLD,
        plural_export=WorkspaceExportConvention.WORLDS,
    ),
    _WorkspaceGroupSpec(
        kind=WorkspaceAssetKind.SCENARIO,
        relative_dir=("simulators", "scenarios"),
        singular_export=WorkspaceExportConvention.SCENARIO,
        plural_export=WorkspaceExportConvention.SCENARIOS,
    ),
)

_REQUIRED_FIELDS_BY_KIND: dict[WorkspaceAssetKind, tuple[str, ...]] = {
    WorkspaceAssetKind.ROBOT: ("id", "robot_type", "primitives"),
    WorkspaceAssetKind.SENSOR: ("id", "kind", "description"),
    WorkspaceAssetKind.ASSEMBLY: ("id", "robots", "execution_targets"),
    WorkspaceAssetKind.ADAPTER: ("id", "assembly_id", "transport", "supported_targets"),
    WorkspaceAssetKind.DEPLOYMENT: ("id", "assembly_id", "target_id"),
    WorkspaceAssetKind.WORLD: ("id", "simulator", "description"),
    WorkspaceAssetKind.SCENARIO: ("id", "assembly_id", "target_id", "world_id"),
}


def inspect_workspace_assets(
    workspace: Path,
    options: WorkspaceInspectOptions | None = None,
) -> WorkspaceLoadReport:
    """Inspect workspace assets with static checks and lint metadata."""

    normalized_options = _normalize_inspect_options(options)
    root = workspace.expanduser().resolve() / "embodied"
    staged, report = _collect_workspace_assets(root, normalized_options)
    if not normalized_options.include_staged_assets:
        return replace(report, staged_assets=tuple())
    del staged
    return report


def load_workspace_assets(
    catalog: EmbodiedCatalog,
    workspace: Path,
    options: WorkspaceInspectOptions | None = None,
) -> EmbodiedCatalog:
    """Load workspace-generated embodied assets into an existing catalog."""

    normalized_options = _normalize_inspect_options(options)
    root = workspace.expanduser().resolve() / "embodied"
    staged_assets, report = _collect_workspace_assets(root, normalized_options)
    report = _check_catalog_conflicts(catalog, staged_assets, report)

    if report.has_errors or (normalized_options.fail_on_warnings and report.has_warnings):
        raise ValueError(
            _format_workspace_errors(
                report,
                fail_on_warnings=normalized_options.fail_on_warnings,
            )
        )

    for asset in staged_assets:
        _register_staged_asset(catalog, asset)
    return catalog


def _collect_workspace_assets(
    root: Path,
    options: WorkspaceInspectOptions,
) -> tuple[list[_StagedAsset], WorkspaceLoadReport]:
    counts = {spec.kind: 0 for spec in _GROUP_SPECS}
    if not root.exists():
        return [], WorkspaceLoadReport(
            root=str(root),
            lint_profile=options.lint_profile,
            loaded_counts=counts,
        )

    issues: list[WorkspaceValidationIssue] = []
    staged_assets: list[_StagedAsset] = []

    for spec in _GROUP_SPECS:
        group_assets = _load_group(root.joinpath(*spec.relative_dir), spec, issues, options)
        staged_assets.extend(group_assets)
        counts[spec.kind] = len(group_assets)

    staged_report_assets = tuple(
        WorkspaceStagedAsset(
            kind=item.kind,
            asset_id=item.asset_id,
            path=str(item.path),
            module_name=item.module_name,
            schema_version=item.contract.schema_version,
            provenance=item.contract.provenance,
        )
        for item in staged_assets
    )
    report = WorkspaceLoadReport(
        root=str(root),
        lint_profile=options.lint_profile,
        loaded_counts=counts,
        staged_assets=staged_report_assets if options.include_staged_assets else tuple(),
        issues=tuple(issues),
    )
    return staged_assets, report


def _load_group(
    root: Path,
    spec: _WorkspaceGroupSpec,
    issues: list[WorkspaceValidationIssue],
    options: WorkspaceInspectOptions,
) -> list[_StagedAsset]:
    if not root.exists():
        return []

    staged: list[_StagedAsset] = []
    seen_ids: dict[str, Path] = {}
    for path in sorted(root.rglob("*.py")):
        if path.name == "__init__.py":
            continue

        module = _try_load_module(path, issues)
        if module is None:
            continue

        contract = _read_asset_contract(module, spec, path, issues, options)
        if contract is None:
            continue

        exports = _read_exports(module, spec, contract, path, issues)
        if not exports:
            continue

        for item in exports:
            if not _validate_asset_shape(item, spec, path, issues):
                continue
            asset_id = _read_asset_id(item)
            if asset_id is None:
                issues.append(
                    WorkspaceValidationIssue(
                        level=WorkspaceIssueLevel.ERROR,
                        stage=WorkspaceValidationStage.SCHEMA,
                        code="ASSET_ID_MISSING",
                        path=str(path),
                        message=(
                            f"{spec.kind.value} export must define a non-empty string 'id' attribute."
                        ),
                    )
                )
                continue

            if asset_id in seen_ids:
                issues.append(
                    WorkspaceValidationIssue(
                        level=WorkspaceIssueLevel.ERROR,
                        stage=WorkspaceValidationStage.SCHEMA,
                        code="DUPLICATE_ASSET_ID",
                        path=str(path),
                        message=(
                            f"Duplicate {spec.kind.value} id '{asset_id}' also defined in "
                            f"'{seen_ids[asset_id]}'."
                        ),
                    )
                )
                continue

            seen_ids[asset_id] = path
            staged.append(
                _StagedAsset(
                    kind=spec.kind,
                    asset_id=asset_id,
                    path=path,
                    module_name=module.__name__,
                    contract=contract,
                    value=item,
                )
            )

    return staged


def _read_asset_contract(
    module: ModuleType,
    spec: _WorkspaceGroupSpec,
    path: Path,
    issues: list[WorkspaceValidationIssue],
    options: WorkspaceInspectOptions,
) -> WorkspaceAssetContract | None:
    if hasattr(module, "WORKSPACE_ASSET"):
        value = getattr(module, "WORKSPACE_ASSET")
        if not isinstance(value, WorkspaceAssetContract):
            issues.append(
                WorkspaceValidationIssue(
                    level=WorkspaceIssueLevel.ERROR,
                    stage=WorkspaceValidationStage.SCHEMA,
                    code="INVALID_WORKSPACE_ASSET_TYPE",
                    path=str(path),
                    message="WORKSPACE_ASSET must be an instance of WorkspaceAssetContract.",
                )
            )
            return None
        contract = value
    else:
        contract = WorkspaceAssetContract(
            kind=spec.kind,
            provenance=WorkspaceProvenance(
                source="implicit_contract",
                source_path=str(path),
            ),
        )
        _append_contract_metadata_issue(
            issues=issues,
            path=path,
            options=options,
            message=(
                "WORKSPACE_ASSET metadata is missing; use explicit metadata to stabilize dry-run and migration."
            ),
        )

    if contract.kind != spec.kind:
        issues.append(
            WorkspaceValidationIssue(
                level=WorkspaceIssueLevel.ERROR,
                stage=WorkspaceValidationStage.SCHEMA,
                code="ASSET_KIND_MISMATCH",
                path=str(path),
                message=(
                    f"WORKSPACE_ASSET.kind '{contract.kind.value}' does not match folder kind "
                    f"'{spec.kind.value}'."
                ),
            )
        )
        return None

    if contract.export_convention not in spec.allowed_conventions:
        issues.append(
            WorkspaceValidationIssue(
                level=WorkspaceIssueLevel.ERROR,
                stage=WorkspaceValidationStage.SCHEMA,
                code="INVALID_EXPORT_CONVENTION",
                path=str(path),
                message=(
                    f"Export convention '{contract.export_convention.value}' is invalid for "
                    f"{spec.kind.value}; use one of {tuple(item.value for item in spec.allowed_conventions)}."
                ),
            )
        )
        return None

    if contract.schema_version not in SUPPORTED_WORKSPACE_SCHEMA_VERSIONS:
        level = (
            WorkspaceIssueLevel.WARNING
            if contract.migration_policy == WorkspaceMigrationPolicy.ACCEPT_UNSUPPORTED
            else WorkspaceIssueLevel.ERROR
        )
        issues.append(
            WorkspaceValidationIssue(
                level=level,
                stage=WorkspaceValidationStage.SCHEMA,
                code="UNSUPPORTED_SCHEMA_VERSION",
                path=str(path),
                message=(
                    f"Schema version '{contract.schema_version}' is not supported by loader "
                    f"{SUPPORTED_WORKSPACE_SCHEMA_VERSIONS}; migration policy is "
                    f"'{contract.migration_policy.value}'."
                ),
            )
        )
        if level == WorkspaceIssueLevel.ERROR:
            return None

    _lint_provenance(contract, path, issues, options)
    return contract


def _read_exports(
    module: ModuleType,
    spec: _WorkspaceGroupSpec,
    contract: WorkspaceAssetContract,
    path: Path,
    issues: list[WorkspaceValidationIssue],
) -> tuple[object, ...]:
    singular, plural = spec.exports
    if contract.export_convention == spec.singular_export:
        convention = singular
    elif contract.export_convention == spec.plural_export:
        convention = plural
    else:
        convention = _resolve_auto_export_convention(module, singular, plural, path, issues)
        if convention is None:
            return ()

    if not hasattr(module, convention):
        issues.append(
            WorkspaceValidationIssue(
                level=WorkspaceIssueLevel.ERROR,
                stage=WorkspaceValidationStage.SCHEMA,
                code="MISSING_EXPORT",
                path=str(path),
                message=f"Expected export '{convention}' is missing.",
            )
        )
        return ()

    value = getattr(module, convention)
    if convention == plural:
        if not isinstance(value, (tuple, list)):
            issues.append(
                WorkspaceValidationIssue(
                    level=WorkspaceIssueLevel.ERROR,
                    stage=WorkspaceValidationStage.SCHEMA,
                    code="INVALID_PLURAL_EXPORT_TYPE",
                    path=str(path),
                    message=f"Export '{plural}' must be a tuple or list.",
                )
            )
            return ()
        return tuple(value)
    return (value,)


def _resolve_auto_export_convention(
    module: ModuleType,
    singular: str,
    plural: str,
    path: Path,
    issues: list[WorkspaceValidationIssue],
) -> str | None:
    has_singular = hasattr(module, singular)
    has_plural = hasattr(module, plural)
    if has_singular and has_plural:
        issues.append(
            WorkspaceValidationIssue(
                level=WorkspaceIssueLevel.ERROR,
                stage=WorkspaceValidationStage.SCHEMA,
                code="AMBIGUOUS_EXPORT",
                path=str(path),
                message=f"Use either '{singular}' or '{plural}', not both.",
            )
        )
        return None
    if has_plural:
        return plural
    if has_singular:
        return singular
    issues.append(
        WorkspaceValidationIssue(
            level=WorkspaceIssueLevel.ERROR,
            stage=WorkspaceValidationStage.SCHEMA,
            code="MISSING_EXPORT",
            path=str(path),
            message=f"Expected export '{singular}' or '{plural}'.",
        )
    )
    return None


def _check_catalog_conflicts(
    catalog: EmbodiedCatalog,
    staged_assets: list[_StagedAsset],
    report: WorkspaceLoadReport,
) -> WorkspaceLoadReport:
    issues = list(report.issues)
    for asset in staged_assets:
        if _is_id_present_in_catalog(catalog, asset):
            issues.append(
                WorkspaceValidationIssue(
                    level=WorkspaceIssueLevel.ERROR,
                    stage=WorkspaceValidationStage.CATALOG_CONFLICT,
                    code="CATALOG_ID_CONFLICT",
                    path=str(asset.path),
                    message=(
                        f"{asset.kind.value} id '{asset.asset_id}' already exists in the active catalog."
                    ),
                )
            )
    return WorkspaceLoadReport(
        root=report.root,
        schema_version=report.schema_version,
        lint_profile=report.lint_profile,
        loaded_counts=report.loaded_counts,
        staged_assets=report.staged_assets,
        issues=tuple(issues),
    )


def _is_id_present_in_catalog(catalog: EmbodiedCatalog, asset: _StagedAsset) -> bool:
    try:
        if asset.kind == WorkspaceAssetKind.ROBOT:
            catalog.robots.get(asset.asset_id)
            return True
        if asset.kind == WorkspaceAssetKind.SENSOR:
            catalog.sensors.get(asset.asset_id)
            return True
        if asset.kind == WorkspaceAssetKind.ASSEMBLY:
            catalog.assemblies.get(asset.asset_id)
            return True
        if asset.kind == WorkspaceAssetKind.ADAPTER:
            catalog.adapters.get(asset.asset_id)
            return True
        if asset.kind == WorkspaceAssetKind.DEPLOYMENT:
            catalog.deployments.get(asset.asset_id)
            return True
        if asset.kind == WorkspaceAssetKind.WORLD:
            catalog.simulators.get_world(asset.asset_id)
            return True
        if asset.kind == WorkspaceAssetKind.SCENARIO:
            catalog.simulators.get_scenario(asset.asset_id)
            return True
    except KeyError:
        return False
    return False


def _register_staged_asset(catalog: EmbodiedCatalog, asset: _StagedAsset) -> None:
    if asset.kind == WorkspaceAssetKind.ROBOT:
        catalog.robots.register(asset.value)
        return
    if asset.kind == WorkspaceAssetKind.SENSOR:
        catalog.sensors.register(asset.value)
        return
    if asset.kind == WorkspaceAssetKind.ASSEMBLY:
        catalog.assemblies.register(asset.value)
        return
    if asset.kind == WorkspaceAssetKind.ADAPTER:
        catalog.adapters.register(asset.value)
        return
    if asset.kind == WorkspaceAssetKind.DEPLOYMENT:
        catalog.deployments.register(asset.value)
        return
    if asset.kind == WorkspaceAssetKind.WORLD:
        catalog.simulators.register_world(asset.value)
        return
    if asset.kind == WorkspaceAssetKind.SCENARIO:
        catalog.simulators.register_scenario(asset.value)
        return
    raise ValueError(f"Unsupported workspace asset kind '{asset.kind.value}'.")


def _read_asset_id(item: object) -> str | None:
    value = getattr(item, "id", None)
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _validate_asset_shape(
    item: object,
    spec: _WorkspaceGroupSpec,
    path: Path,
    issues: list[WorkspaceValidationIssue],
) -> bool:
    required_fields = _REQUIRED_FIELDS_BY_KIND[spec.kind]
    missing = [field_name for field_name in required_fields if not hasattr(item, field_name)]
    if not missing:
        return True

    issues.append(
        WorkspaceValidationIssue(
            level=WorkspaceIssueLevel.ERROR,
            stage=WorkspaceValidationStage.SCHEMA,
            code="ASSET_SCHEMA_MISMATCH",
            path=str(path),
            message=(
                f"{spec.kind.value} export is missing required fields {tuple(missing)} "
                f"for expected schema {tuple(required_fields)}."
            ),
        )
    )
    return False


def _append_contract_metadata_issue(
    issues: list[WorkspaceValidationIssue],
    path: Path,
    options: WorkspaceInspectOptions,
    message: str,
) -> None:
    level = WorkspaceIssueLevel.ERROR if options.require_contract_metadata else WorkspaceIssueLevel.WARNING
    issues.append(
        WorkspaceValidationIssue(
            level=level,
            stage=WorkspaceValidationStage.LINT,
            code="CONTRACT_METADATA_MISSING",
            path=str(path),
            message=message,
        )
    )


def _lint_provenance(
    contract: WorkspaceAssetContract,
    path: Path,
    issues: list[WorkspaceValidationIssue],
    options: WorkspaceInspectOptions,
) -> None:
    provenance = contract.provenance
    missing_fields: list[str] = []
    if provenance.generator is None:
        missing_fields.append("generator")
    if provenance.generated_by is None:
        missing_fields.append("generated_by")
    if provenance.generated_at is None:
        missing_fields.append("generated_at")

    if not missing_fields:
        return

    level = WorkspaceIssueLevel.ERROR if options.require_provenance_metadata else WorkspaceIssueLevel.WARNING
    issues.append(
        WorkspaceValidationIssue(
            level=level,
            stage=WorkspaceValidationStage.LINT,
            code="PROVENANCE_METADATA_MISSING",
            path=str(path),
            message=(
                f"WORKSPACE_ASSET.provenance is missing fields {tuple(missing_fields)}. "
                "Populate provenance to improve dry-run traceability."
            ),
        )
    )


def _normalize_inspect_options(options: WorkspaceInspectOptions | None) -> WorkspaceInspectOptions:
    if options is None:
        options = WorkspaceInspectOptions()
    if options.lint_profile == WorkspaceLintProfile.STRICT:
        return WorkspaceInspectOptions(
            lint_profile=WorkspaceLintProfile.STRICT,
            require_contract_metadata=True,
            require_provenance_metadata=True,
            fail_on_warnings=True,
            include_staged_assets=options.include_staged_assets,
        )
    return options


def _try_load_module(path: Path, issues: list[WorkspaceValidationIssue]) -> ModuleType | None:
    try:
        return _load_module(path)
    except Exception as exc:  # pragma: no cover - defensive loader guard.
        issues.append(
            WorkspaceValidationIssue(
                level=WorkspaceIssueLevel.ERROR,
                stage=WorkspaceValidationStage.IMPORT,
                code="IMPORT_ERROR",
                path=str(path),
                message=f"Could not import module: {exc}",
            )
        )
        return None


def _format_workspace_errors(report: WorkspaceLoadReport, fail_on_warnings: bool = False) -> str:
    errors = [issue for issue in report.issues if issue.level == WorkspaceIssueLevel.ERROR]
    warnings = [issue for issue in report.issues if issue.level == WorkspaceIssueLevel.WARNING]
    lines = [f"Workspace asset validation failed for '{report.root}':"]
    for issue in errors:
        lines.append(f"- [error/{issue.stage.value}/{issue.code}] {issue.path}: {issue.message}")
    if fail_on_warnings:
        for issue in warnings:
            lines.append(f"- [warning/{issue.stage.value}/{issue.code}] {issue.path}: {issue.message}")
    return "\n".join(lines)


def _load_module(path: Path) -> ModuleType:
    module_name = "roboclaw_workspace_" + "_".join(path.with_suffix("").parts[-6:])
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load workspace module from '{path}'.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
