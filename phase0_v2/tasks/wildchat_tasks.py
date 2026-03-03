"""WildChat task loader with category tagging."""

import json
from dataclasses import dataclass
from pathlib import Path

from phase0_v2.conflicts.compatibility import TASK_CATEGORIES, is_compatible


@dataclass
class WildChatTask:
    wildchat_id: str
    prompt: str
    category: str  # one of TASK_CATEGORIES

    @property
    def task_id(self) -> str:
        return f"wc_{self.wildchat_id[:12]}"


# Keyword heuristics for category tagging
_CODING_KEYWORDS = {
    "code", "function", "python", "javascript", "java", "html", "css",
    "program", "debug", "api", "sql", "database", "algorithm", "git",
    "compile", "syntax", "variable", "class ", "import ", "def ",
    "script", "terminal", "bash", "linux", "docker", "react", "vue",
    "node", "typescript", "rust", "golang", "c++", "swift",
}

_MATH_KEYWORDS = {
    "calculate", "equation", "formula", "integral", "derivative",
    "algebra", "geometry", "trigonometry", "probability", "statistics",
    "matrix", "vector", "theorem", "proof", "solve for",
    "math", "arithmetic", "calculus",
}


def classify_task(prompt: str) -> str:
    """Classify a WildChat prompt into a task category using keyword heuristics."""
    lower = prompt.lower()

    # Check coding first (highest priority for incompatibility)
    if any(kw in lower for kw in _CODING_KEYWORDS):
        return "coding"

    if any(kw in lower for kw in _MATH_KEYWORDS):
        return "math"

    # Check for list/bullet requests
    if any(phrase in lower for phrase in ["list of", "give me a list", "name some", "top 10", "top 5"]):
        return "list"

    # Check for creative writing
    if any(phrase in lower for phrase in ["write a story", "write a poem", "creative writing", "fiction", "novel", "screenplay"]):
        return "creative"

    # Check for game/fiction
    if any(phrase in lower for phrase in ["game", "rpg", "character", "quest", "dungeon", "fantasy world"]):
        return "game"

    # Check for essay/analysis
    if any(phrase in lower for phrase in ["essay", "analyze", "thesis", "argument", "discuss the"]):
        return "essay"

    # Check for QA
    if lower.startswith(("what ", "how ", "why ", "when ", "where ", "who ", "is ", "can ", "does ")):
        return "qa"

    return "general"


def load_wildchat_tasks(jsonl_path: str | Path) -> list[WildChatTask]:
    """Load and tag WildChat prompts from JSONL file."""
    path = Path(jsonl_path)
    tasks = []
    with open(path) as f:
        for line in f:
            record = json.loads(line)
            prompt = record["clean_prompt"]
            wildchat_id = record["wildchat_id"]
            category = classify_task(prompt)
            tasks.append(WildChatTask(
                wildchat_id=wildchat_id,
                prompt=prompt,
                category=category,
            ))
    return tasks


def filter_compatible_tasks(
    tasks: list[WildChatTask], conflict_id: str
) -> list[WildChatTask]:
    """Return only tasks compatible with the given conflict."""
    return [t for t in tasks if is_compatible(conflict_id, t.category)]
