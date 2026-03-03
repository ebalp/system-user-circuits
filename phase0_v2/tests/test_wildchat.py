"""Test WildChat task loader and category classification."""

import pytest
from phase0_v2.tasks.wildchat_tasks import classify_task, WildChatTask
from phase0_v2.conflicts.compatibility import TASK_CATEGORIES


class TestClassifyTask:
    def test_coding_detection(self):
        assert classify_task("Write a Python function to sort a list") == "coding"
        assert classify_task("Debug this JavaScript code") == "coding"

    def test_math_detection(self):
        assert classify_task("Solve this equation: 2x + 3 = 7") == "math"
        assert classify_task("Calculate the integral of x^2") == "math"

    def test_creative_detection(self):
        assert classify_task("Write a story about a dragon") == "creative"

    def test_general_fallback(self):
        """Non-matching prompts should fall back to 'general'."""
        result = classify_task("Tell me about the Roman Empire's influence on architecture")
        assert result in TASK_CATEGORIES

    def test_all_categories_valid(self):
        """classify_task must always return a valid TASK_CATEGORY."""
        test_prompts = [
            "Write code", "Solve math", "Write a poem",
            "List 10 animals", "What is gravity?",
            "Play a game", "Write an essay", "Hello world"
        ]
        for prompt in test_prompts:
            cat = classify_task(prompt)
            assert cat in TASK_CATEGORIES, f"'{prompt}' -> '{cat}' not in TASK_CATEGORIES"


class TestWildChatTask:
    def test_task_id_format(self):
        t = WildChatTask(wildchat_id="abc123def456xyz", prompt="test", category="general")
        assert t.task_id.startswith("wc_")
        assert len(t.task_id) == 15  # "wc_" + 12 chars
