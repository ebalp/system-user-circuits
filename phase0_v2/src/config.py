"""Configuration loading and validation for Phase 0 v2."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SystemTemplate:
    """A system prompt wrapping template with a rhetorical system style."""
    name: str       # "bare", "compliance", "authority", "persona", "safety"
    template: str   # contains {system_instruction}


@dataclass
class UserTemplate:
    """A user prompt wrapping template with style."""
    name: str       # "with_instruction", "authority", etc.
    template: str   # contains {user_instruction} and {task}


@dataclass
class Task:
    """A synthetic task to evaluate."""
    id: str
    prompt: str


@dataclass
class ApiConfig:
    """API configuration."""
    timeout: int
    max_retries: int
    concurrent_per_model: int


@dataclass
class GenerationConfig:
    """Generation parameters for API calls."""
    temperature: float
    max_tokens: int
    instances_per_cell: int


@dataclass
class CounterbalancingConfig:
    """Counterbalancing settings."""
    enabled: bool
    require_invertible: bool = False


@dataclass
class ThresholdsConfig:
    """Thresholds for go/no-go decision."""
    hierarchy_index: float
    conflict_resolution: float
    asymmetry_warning: float


@dataclass
class TaskSourceConfig:
    """Configuration for a task source (synthetic or wildchat)."""
    file: str
    k_tasks_per_conflict: int | None  # None = use all


@dataclass
class ExperimentConfig:
    """Complete experiment configuration for Phase 0 v2."""
    api: ApiConfig
    models: list[str]
    system_templates: dict[str, SystemTemplate]  # keyed by name
    user_templates: dict[str, UserTemplate]       # keyed by name
    tasks: list[Task]
    conditions: list[str]
    counterbalancing: CounterbalancingConfig
    generation: GenerationConfig
    condition_c_system_styles: list[str]
    default_system_style: str
    default_user_style: str
    user_styles_to_test: list[str]
    thresholds: ThresholdsConfig
    task_sources: dict[str, TaskSourceConfig]
    seed: int


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate Phase 0 v2 configuration from YAML file.

    Args:
        path: Path to the experiment.yaml configuration file.

    Returns:
        ExperimentConfig with all settings.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If config is invalid or validation fails.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    config_dir = path.parent

    # Parse API config
    api_data = data["api"]
    api = ApiConfig(
        timeout=api_data["timeout"],
        max_retries=api_data["max_retries"],
        concurrent_per_model=api_data.get("concurrent_per_model", 3),
    )

    # Parse system templates
    system_templates: dict[str, SystemTemplate] = {}
    for name, template_str in data["system_templates"].items():
        system_templates[name] = SystemTemplate(name=name, template=template_str)

    # Parse user templates
    user_templates: dict[str, UserTemplate] = {}
    for name, template_str in data["user_templates"].items():
        user_templates[name] = UserTemplate(name=name, template=template_str)

    # Parse task sources
    task_sources: dict[str, TaskSourceConfig] = {}
    for source_name, source_data in data.get("task_sources", {}).items():
        task_sources[source_name] = TaskSourceConfig(
            file=source_data["file"],
            k_tasks_per_conflict=source_data.get("k_tasks_per_conflict"),
        )

    # Load synthetic tasks from the referenced YAML file
    tasks: list[Task] = []
    if "synthetic" in task_sources:
        tasks_file = config_dir / task_sources["synthetic"].file
        if tasks_file.exists():
            with open(tasks_file, "r") as f:
                tasks_data = yaml.safe_load(f)
            for t in tasks_data.get("tasks", []):
                tasks.append(Task(id=t["id"], prompt=t["prompt"]))

    # Parse generation config
    gen_data = data["generation"]
    generation = GenerationConfig(
        temperature=gen_data["temperature"],
        max_tokens=gen_data["max_tokens"],
        instances_per_cell=gen_data["instances_per_cell"],
    )

    # Parse counterbalancing
    cb_data = data["counterbalancing"]
    counterbalancing = CounterbalancingConfig(
        enabled=cb_data["enabled"],
        require_invertible=cb_data.get("require_invertible", False),
    )

    # Parse thresholds
    th_data = data["thresholds"]
    thresholds = ThresholdsConfig(
        hierarchy_index=th_data["hierarchy_index"],
        conflict_resolution=th_data["conflict_resolution"],
        asymmetry_warning=th_data["asymmetry_warning"],
    )

    config = ExperimentConfig(
        api=api,
        models=data["models"],
        system_templates=system_templates,
        user_templates=user_templates,
        tasks=tasks,
        conditions=data["conditions"],
        counterbalancing=counterbalancing,
        generation=generation,
        condition_c_system_styles=data["condition_c_system_styles"],
        default_system_style=data["default_system_style"],
        default_user_style=data["default_user_style"],
        user_styles_to_test=data["user_styles_to_test"],
        thresholds=thresholds,
        task_sources=task_sources,
        seed=data.get("seed", 42),
    )

    # Validate
    errors = validate_config(config)
    if errors:
        error_msg = "Configuration validation failed:\n" + "\n".join(
            f"  - {e}" for e in errors
        )
        raise ValueError(error_msg)

    return config


def validate_config(config: ExperimentConfig) -> list[str]:
    """Validate configuration and return list of errors (empty if valid)."""
    errors: list[str] = []

    # Validate system templates have {system_instruction} placeholder
    for name, tmpl in config.system_templates.items():
        if "{system_instruction}" not in tmpl.template:
            errors.append(
                f"System template '{name}' missing {{system_instruction}} placeholder"
            )

    # Validate user templates have {user_instruction} and {task} placeholders
    for name, tmpl in config.user_templates.items():
        if "{user_instruction}" not in tmpl.template:
            errors.append(
                f"User template '{name}' missing {{user_instruction}} placeholder"
            )
        if "{task}" not in tmpl.template:
            errors.append(
                f"User template '{name}' missing {{task}} placeholder"
            )

    # Validate condition_c_system_styles exist in system_templates
    for style in config.condition_c_system_styles:
        if style not in config.system_templates:
            errors.append(
                f"System style '{style}' in condition_c_system_styles not found "
                f"in system_templates. Available: {sorted(config.system_templates.keys())}"
            )

    # Validate user_styles_to_test exist in user_templates
    for style in config.user_styles_to_test:
        if style not in config.user_templates:
            errors.append(
                f"Style '{style}' in user_styles_to_test not found "
                f"in user_templates. Available: {sorted(config.user_templates.keys())}"
            )

    # Validate default_system_style exists in system_templates
    if config.default_system_style not in config.system_templates:
        errors.append(
            f"Default system style '{config.default_system_style}' not found in system_templates. "
            f"Available: {sorted(config.system_templates.keys())}"
        )

    # Validate default_user_style exists in user_templates
    if config.default_user_style not in config.user_templates:
        errors.append(
            f"Default user style '{config.default_user_style}' not found in user_templates. "
            f"Available: {sorted(config.user_templates.keys())}"
        )

    return errors
