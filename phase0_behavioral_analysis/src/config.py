"""
Configuration loading and validation for Phase 0 Behavioral Analysis.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ConstraintOption:
    """A single option within a constraint type."""
    name: str           # e.g., "english", "spanish"
    value: str          # e.g., "English", "Spanish" (for template substitution)
    expected_value: str  # e.g., "en", "es" (for classifier matching)


@dataclass
class ConstraintType:
    """A constraint type with templates and options pool.
    
    This is the new structure that separates template definitions from
    individual options, allowing easy addition of new options without
    duplicating template text.
    
    Attributes:
        name: Identifier for the constraint type (e.g., "language", "format")
        instruction_template: Template with {value} placeholder for instructions
        negative_template: Template for negative instruction text
        classifier: Name of the classifier to use (e.g., "language", "format", "yaml")
        options: List of available options for this constraint type
    """
    name: str                           # e.g., "language", "format"
    instruction_template: str           # e.g., "respond in {value}"
    negative_template: str              # e.g., "Do not use any other language"
    classifier: str                     # e.g., "language", "format", "yaml"
    options: list[ConstraintOption] = field(default_factory=list)
    
    # Internal cache for O(1) option lookup
    _options_map: dict[str, ConstraintOption] = field(default_factory=dict, repr=False, compare=False)
    
    def __post_init__(self):
        """Build the options lookup map after initialization."""
        self._options_map = {opt.name: opt for opt in self.options}
    
    def get_option(self, name: str) -> ConstraintOption:
        """Get option by name with O(1) lookup.
        
        Args:
            name: The name of the option to retrieve.
            
        Returns:
            The ConstraintOption with the given name.
            
        Raises:
            ValueError: If the option name is not found in this constraint type.
        """
        if name in self._options_map:
            return self._options_map[name]
        available = list(self._options_map.keys())
        raise ValueError(
            f"Option '{name}' not found in constraint type '{self.name}'. "
            f"Available options: {available}"
        )
    
    def render_instruction(self, option: ConstraintOption) -> str:
        """Render instruction text by substituting option value into template.
        
        Args:
            option: The option whose value should be substituted.
            
        Returns:
            The instruction template with {value} replaced by the option's value.
        """
        return self.instruction_template.format(value=option.value)


@dataclass
class ExperimentPair:
    """A pair of options to test within a constraint type.
    
    This dataclass represents a specific pair of constraint options that
    should be tested against each other in the experiment. It references
    a constraint type by name and specifies which two options from that
    type's options pool should be compared.
    
    Attributes:
        constraint_type: Reference to the ConstraintType.name that contains the options
        option_a: Name of the first option to test (must exist in constraint type's options)
        option_b: Name of the second option to test (must exist in constraint type's options)
    
    Example:
        ExperimentPair(
            constraint_type="language",
            option_a="english",
            option_b="spanish"
        )
    """
    constraint_type: str  # Reference to ConstraintType.name
    option_a: str         # First option name
    option_b: str         # Second option name


@dataclass
class SystemTemplate:
    """A system prompt template with strength level.
    
    System templates define how system prompts are structured at different
    strength levels (e.g., weak, medium, strong). They contain placeholders
    for instruction and negative text that get substituted during prompt
    generation.
    
    Attributes:
        name: Identifier for the strength level (e.g., "medium", "strong")
        template: Template string with {instruction} and {negative} placeholders
    
    Example:
        SystemTemplate(
            name="medium",
            template="You must {instruction}. {negative} under any circumstances."
        )
    """
    name: str      # e.g., "medium", "strong"
    template: str  # e.g., "You must {instruction}. {negative} under any circumstances."


@dataclass
class UserTemplate:
    """A user prompt template with style.
    
    User templates define how user prompts are structured with different
    styles (e.g., standard, jailbreak, polite). They contain placeholders
    for instruction and task that get substituted during prompt generation.
    
    Attributes:
        name: Identifier for the style (e.g., "standard", "jailbreak", "polite")
        template: Template string with {instruction} and/or {task} placeholders
    
    Example:
        UserTemplate(
            name="standard",
            template="{instruction}. {task}"
        )
    """
    name: str      # e.g., "standard", "jailbreak", "polite"
    template: str  # e.g., "{instruction}. {task}"


@dataclass
class Task:
    """A task to evaluate."""
    id: str
    prompt: str


@dataclass
class ApiConfig:
    """API configuration."""
    timeout: int
    max_retries: int


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


@dataclass
class ThresholdsConfig:
    """Thresholds for go/no-go decision."""
    hierarchy_index: float
    conflict_resolution: float
    asymmetry_warning: float


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    api: ApiConfig
    models: list[str]
    constraint_types: list[ConstraintType] = field(default_factory=list)
    experiment_pairs: list[ExperimentPair] = field(default_factory=list)
    system_templates: dict[str, str] = field(default_factory=dict)
    user_templates: dict[str, str] = field(default_factory=dict)
    tasks: list[Task] = field(default_factory=list)
    conditions: list[str] = field(default_factory=list)
    counterbalancing: CounterbalancingConfig | None = None
    generation: GenerationConfig | None = None
    condition_c_strengths: list[str] = field(default_factory=list)
    default_strength: str = "medium"
    default_user_style: str = "standard"
    user_styles_to_test: list[str] = field(default_factory=list)
    thresholds: ThresholdsConfig | None = None


def load_config(path: str | Path) -> ExperimentConfig:
    """
    Load and validate configuration from YAML file.
    
    Supports both the legacy configuration structure (using `constraints` and
    `strength_templates`) and the new refactored structure (using `constraint_types`,
    `experiment_pairs`, `system_templates`, and `user_templates`).
    
    Args:
        path: Path to the YAML configuration file.
        
    Returns:
        ExperimentConfig with all settings.
        
    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If config is invalid or validation fails.
        
    **Validates: Requirements 7.1, 7.2, 7.3, 7.4**
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    
    # Parse API config
    api_data = data['api']
    api = ApiConfig(
        timeout=api_data['timeout'],
        max_retries=api_data['max_retries']
    )
    
    # Parse constraint types
    constraint_types = [_parse_constraint_type(ct) for ct in data.get('constraint_types', [])]
    
    # Parse experiment pairs
    experiment_pairs = [_parse_experiment_pair(ep) for ep in data.get('experiment_pairs', [])]
    
    # Parse new system templates (if present)
    system_templates = data.get('system_templates', {})
    
    # Parse new user templates (if present)
    user_templates = data.get('user_templates', {})
    
    # Parse tasks
    tasks = [Task(id=t['id'], prompt=t['prompt']) for t in data['tasks']]
    
    # Parse generation config
    gen_data = data['generation']
    generation = GenerationConfig(
        temperature=gen_data['temperature'],
        max_tokens=gen_data['max_tokens'],
        instances_per_cell=gen_data['instances_per_cell']
    )
    
    # Parse counterbalancing
    cb_data = data['counterbalancing']
    counterbalancing = CounterbalancingConfig(enabled=cb_data['enabled'])
    
    # Parse thresholds
    th_data = data['thresholds']
    thresholds = ThresholdsConfig(
        hierarchy_index=th_data['hierarchy_index'],
        conflict_resolution=th_data['conflict_resolution'],
        asymmetry_warning=th_data['asymmetry_warning']
    )
    
    # Parse user style settings
    default_user_style = data.get('default_user_style', 'standard')
    user_styles_to_test = data.get('user_styles_to_test', [])
    
    config = ExperimentConfig(
        api=api,
        models=data['models'],
        constraint_types=constraint_types,
        experiment_pairs=experiment_pairs,
        system_templates=system_templates,
        user_templates=user_templates,
        tasks=tasks,
        conditions=data['conditions'],
        counterbalancing=counterbalancing,
        generation=generation,
        condition_c_strengths=data['condition_c_strengths'],
        default_strength=data['default_strength'],
        default_user_style=default_user_style,
        user_styles_to_test=user_styles_to_test,
        thresholds=thresholds
    )
    
    # Validate configuration
    errors = validate_config(config)
    if errors:
        error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        raise ValueError(error_msg)
    
    return config


def _parse_constraint_type(data: dict[str, Any]) -> ConstraintType:
    """
    Parse a constraint type from config data.
    
    Args:
        data: Dictionary containing constraint type configuration.
        
    Returns:
        ConstraintType object with parsed options.
        
    **Validates: Requirements 1.1, 1.2, 1.3, 1.4**
    """
    options = [
        ConstraintOption(
            name=opt['name'],
            value=opt['value'],
            expected_value=opt['expected_value']
        )
        for opt in data.get('options', [])
    ]
    
    return ConstraintType(
        name=data['name'],
        instruction_template=data['instruction_template'],
        negative_template=data['negative_template'],
        classifier=data['classifier'],
        options=options
    )


def _parse_experiment_pair(data: dict[str, Any]) -> ExperimentPair:
    """
    Parse an experiment pair from config data.
    
    Args:
        data: Dictionary containing experiment pair configuration.
        
    Returns:
        ExperimentPair object.
        
    **Validates: Requirements 2.1**
    """
    return ExperimentPair(
        constraint_type=data['constraint_type'],
        option_a=data['option_a'],
        option_b=data['option_b']
    )


def get_constraint_type_by_name(config: ExperimentConfig, name: str) -> ConstraintType:
    """Get a constraint type by its name."""
    for ct in config.constraint_types:
        if ct.name == name:
            return ct
    raise ValueError(f"Unknown constraint type: {name}")


# =============================================================================
# Configuration Validation
# =============================================================================

# Set of valid classifier names that can be referenced in constraint types
VALID_CLASSIFIERS = {'language', 'format', 'yaml', 'starting_word'}


def validate_config(config: ExperimentConfig) -> list[str]:
    """
    Validate configuration and return list of errors (empty if valid).
    
    This function performs comprehensive validation of the experiment
    configuration, checking for:
    - Valid classifier references in constraint types
    - Valid option references in experiment pairs
    - Required placeholders in templates
    - Valid strength level and user style references
    
    Args:
        config: The ExperimentConfig to validate.
        
    Returns:
        A list of error messages. Empty list indicates valid configuration.
        
    Example:
        >>> errors = validate_config(config)
        >>> if errors:
        ...     for error in errors:
        ...         print(f"Error: {error}")
        ... else:
        ...     print("Configuration is valid")
    
    **Validates: Requirements 1.5, 2.2, 2.4, 7.1, 7.2, 7.3, 7.4**
    """
    errors = []
    
    # Build lookup map for constraint types
    constraint_map = {ct.name: ct for ct in config.constraint_types}
    
    # Validate constraint types
    errors.extend(_validate_constraint_types(config.constraint_types))
    
    # Validate experiment pairs
    errors.extend(_validate_experiment_pairs(config.experiment_pairs, constraint_map))
    
    # Validate system templates
    errors.extend(_validate_system_templates(config.system_templates))
    
    # Validate user templates
    errors.extend(_validate_user_templates(config.user_templates))
    
    # Validate strength levels
    errors.extend(_validate_strength_levels(config))
    
    # Validate user styles
    errors.extend(_validate_user_styles(config))
    
    return errors


def _validate_constraint_types(constraint_types: list[ConstraintType]) -> list[str]:
    """
    Validate constraint types have valid classifiers and templates.
    
    Checks:
    - Classifier references a valid classifier name
    - instruction_template contains {value} placeholder
    
    Args:
        constraint_types: List of ConstraintType objects to validate.
        
    Returns:
        List of error messages for invalid constraint types.
        
    **Validates: Requirements 1.5, 7.2**
    """
    errors = []
    
    for ct in constraint_types:
        # Check classifier is valid
        if ct.classifier not in VALID_CLASSIFIERS:
            errors.append(
                f"Unknown classifier '{ct.classifier}' in constraint type '{ct.name}'. "
                f"Valid classifiers: {sorted(VALID_CLASSIFIERS)}"
            )
        
        # Check instruction_template has {value} placeholder
        if '{value}' not in ct.instruction_template:
            errors.append(
                f"Missing {{value}} placeholder in instruction_template for "
                f"constraint type '{ct.name}'"
            )
    
    return errors


def _validate_experiment_pairs(
    experiment_pairs: list[ExperimentPair],
    constraint_map: dict[str, ConstraintType]
) -> list[str]:
    """
    Validate experiment pairs reference valid constraint types and options.
    
    Checks:
    - constraint_type references an existing ConstraintType
    - option_a exists in the constraint type's options pool
    - option_b exists in the constraint type's options pool
    
    Args:
        experiment_pairs: List of ExperimentPair objects to validate.
        constraint_map: Dict mapping constraint type names to ConstraintType objects.
        
    Returns:
        List of error messages for invalid experiment pairs.
        
    **Validates: Requirements 2.2, 2.4, 7.3**
    """
    errors = []
    
    for pair in experiment_pairs:
        # Check constraint type exists
        if pair.constraint_type not in constraint_map:
            errors.append(
                f"Unknown constraint type '{pair.constraint_type}' in experiment pair. "
                f"Available constraint types: {sorted(constraint_map.keys())}"
            )
            continue
        
        ct = constraint_map[pair.constraint_type]
        option_names = {opt.name for opt in ct.options}
        
        # Check option_a exists
        if pair.option_a not in option_names:
            errors.append(
                f"Unknown option '{pair.option_a}' in experiment pair for "
                f"constraint type '{pair.constraint_type}'. "
                f"Available options: {sorted(option_names)}"
            )
        
        # Check option_b exists
        if pair.option_b not in option_names:
            errors.append(
                f"Unknown option '{pair.option_b}' in experiment pair for "
                f"constraint type '{pair.constraint_type}'. "
                f"Available options: {sorted(option_names)}"
            )
    
    return errors


def _validate_system_templates(system_templates: dict[str, str]) -> list[str]:
    """
    Validate system templates have required placeholders.
    
    System templates must contain both {instruction} and {negative} placeholders.
    
    Args:
        system_templates: Dict mapping strength names to template strings.
        
    Returns:
        List of error messages for invalid system templates.
        
    **Validates: Requirements 7.2**
    """
    errors = []
    
    for name, template in system_templates.items():
        if '{instruction}' not in template:
            errors.append(
                f"Missing {{instruction}} placeholder in system template '{name}'"
            )
        if '{negative}' not in template:
            errors.append(
                f"Missing {{negative}} placeholder in system template '{name}'"
            )
    
    return errors


def _validate_user_templates(user_templates: dict[str, str]) -> list[str]:
    """
    Validate user templates have at least one required placeholder.
    
    User templates must contain at least {instruction} or {task} placeholder.
    
    Args:
        user_templates: Dict mapping style names to template strings.
        
    Returns:
        List of error messages for invalid user templates.
        
    **Validates: Requirements 7.2**
    """
    errors = []
    
    for name, template in user_templates.items():
        has_instruction = '{instruction}' in template
        has_task = '{task}' in template
        
        if not has_instruction and not has_task:
            errors.append(
                f"User template '{name}' must have {{instruction}} or {{task}} placeholder"
            )
    
    return errors


def _validate_strength_levels(config: ExperimentConfig) -> list[str]:
    """
    Validate strength levels reference existing system templates.
    
    Checks that all strength levels in condition_c_strengths exist in
    system_templates, and that default_strength exists in system_templates.
    
    Args:
        config: The ExperimentConfig to validate.
        
    Returns:
        List of error messages for invalid strength level references.
        
    **Validates: Requirements 7.1**
    """
    errors = []
    
    # Skip validation if no system_templates defined (using legacy config)
    if not config.system_templates:
        return errors
    
    # Check condition_c_strengths
    for strength in config.condition_c_strengths:
        if strength not in config.system_templates:
            errors.append(
                f"Strength level '{strength}' in condition_c_strengths not found "
                f"in system_templates. Available: {sorted(config.system_templates.keys())}"
            )
    
    # Check default_strength
    if config.default_strength and config.default_strength not in config.system_templates:
        errors.append(
            f"Default strength '{config.default_strength}' not found in system_templates. "
            f"Available: {sorted(config.system_templates.keys())}"
        )
    
    return errors


def _validate_user_styles(config: ExperimentConfig) -> list[str]:
    """
    Validate user styles reference existing user templates.
    
    Checks that all user styles in user_styles_to_test exist in user_templates,
    and that default_user_style exists in user_templates.
    
    Args:
        config: The ExperimentConfig to validate.
        
    Returns:
        List of error messages for invalid user style references.
        
    **Validates: Requirements 7.1**
    """
    errors = []
    
    # Skip validation if no user_templates defined (using legacy config)
    if not config.user_templates:
        return errors
    
    # Check user_styles_to_test
    for style in config.user_styles_to_test:
        if style not in config.user_templates:
            errors.append(
                f"User style '{style}' in user_styles_to_test not found "
                f"in user_templates. Available: {sorted(config.user_templates.keys())}"
            )
    
    # Check default_user_style
    if config.default_user_style and config.default_user_style not in config.user_templates:
        errors.append(
            f"Default user style '{config.default_user_style}' not found in user_templates. "
            f"Available: {sorted(config.user_templates.keys())}"
        )
    
    return errors
