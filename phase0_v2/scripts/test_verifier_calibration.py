"""Calibrate verifier thresholds by comparing constrained vs unconstrained model responses.

For each gradable constraint, queries the model WITH and WITHOUT the constraint
across multiple tasks. This produces:
  - "trying" scores: model was told to follow the constraint
  - "ignoring" scores: model was NOT told about the constraint (baseline)

The gap between trying and ignoring determines a good threshold.

Run: HF_TOKEN=hf_... uv run python phase0_v2/scripts/test_verifier_calibration.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from phase0_v2.src.api_client import HFClient
from phase0_v2.conflicts.verify_utils import (
    score_alphabetical_word_start,
    score_all_alliteration,
    score_no_consecutive_first_letter,
    score_sentence_chaining,
    score_no_sentence_chaining,
    score_paragraph_bookend,
    score_no_paragraph_bookend,
    score_title_case,
    score_sentence_case,
    score_indent_stairs,
    score_each_word_on_new_line,
    score_alternating_odd_even_syllables,
    score_max_word_repeat,
)

MODEL = "meta-llama/Llama-3.1-8B-Instruct"

TASKS = [
    "Explain how photosynthesis works.",
    "Describe the causes and effects of the French Revolution.",
    "Write a short guide on how to make sourdough bread.",
    "Explain the difference between TCP and UDP protocols.",
]

# Each test: constraint name, system prompt to apply, score function, extra args
TESTS = [
    {
        "name": "alphabetical_word_start",
        "system": "Each word in your response must start with the next letter of the alphabet, looping back to 'A' after 'Z'.",
        "score_fn": score_alphabetical_word_start,
    },
    {
        "name": "all_alliteration",
        "system": "Start every pair of consecutive words with the same first letter (alliteration throughout).",
        "score_fn": score_all_alliteration,
    },
    {
        "name": "no_consecutive_first_letter",
        "system": "No two consecutive words should start with the same letter.",
        "score_fn": score_no_consecutive_first_letter,
    },
    {
        "name": "sentence_chaining",
        "system": "The last word of each sentence must be the first word of the next sentence.",
        "score_fn": score_sentence_chaining,
    },
    {
        "name": "no_sentence_chaining",
        "system": "Each sentence must begin with a completely different word than where the previous sentence ended.",
        "score_fn": score_no_sentence_chaining,
    },
    {
        "name": "paragraph_bookend",
        "system": "Each paragraph must end with the same word it started with.",
        "score_fn": score_paragraph_bookend,
    },
    {
        "name": "no_paragraph_bookend",
        "system": "Each paragraph must end with a completely different word than it started with.",
        "score_fn": score_no_paragraph_bookend,
    },
    {
        "name": "title_case",
        "system": "Write your entire response in Title Case.",
        "score_fn": score_title_case,
    },
    {
        "name": "sentence_case",
        "system": "Write in standard sentence case. Only capitalize the first word of each sentence.",
        "score_fn": score_sentence_case,
    },
    {
        "name": "indent_stairs",
        "system": "Each line must be indented more than the previous one, creating a staircase pattern.",
        "score_fn": score_indent_stairs,
    },
    {
        "name": "each_word_new_line",
        "system": "Write each word on a new line.",
        "score_fn": score_each_word_on_new_line,
    },
    {
        "name": "alternating_syllables",
        "system": "Words must alternate between odd and even syllable counts throughout your response.",
        "score_fn": score_alternating_odd_even_syllables,
    },
    {
        "name": "max_word_repeat (N=2)",
        "system": "Do not repeat any word more than 2 times in your response.",
        "score_fn": lambda text: score_max_word_repeat(text, 2),
    },
]


def query(client, system_prompt, user_prompt):
    """Query the model and return response text."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    resp = client.chat_completion(MODEL, messages, temperature=0.0, max_tokens=512)
    if resp.error:
        print(f"  ERROR: {resp.error}")
        return ""
    return resp.content


def main():
    client = HFClient()
    n_tasks = len(TASKS)
    n_tests = len(TESTS)
    total_queries = n_tests * n_tasks * 2  # constrained + unconstrained

    print(f"Model: {MODEL}")
    print(f"Tasks: {n_tasks}")
    print(f"Constraints: {n_tests}")
    print(f"Total queries: {total_queries}")
    print("=" * 90)

    results = []

    for ti, test in enumerate(TESTS):
        name = test["name"]
        score_fn = test["score_fn"]
        system = test["system"]

        trying_scores = []
        ignoring_scores = []

        for i, task in enumerate(TASKS):
            resp_constrained = query(client, system, task)
            s_trying = score_fn(resp_constrained) if resp_constrained else None

            resp_unconstrained = query(client, None, task)
            s_ignoring = score_fn(resp_unconstrained) if resp_unconstrained else None

            if s_trying is not None:
                trying_scores.append(s_trying)
            if s_ignoring is not None:
                ignoring_scores.append(s_ignoring)

        avg_t = sum(trying_scores) / len(trying_scores) if trying_scores else 0
        avg_i = sum(ignoring_scores) / len(ignoring_scores) if ignoring_scores else 0
        print(f"[{ti+1}/{n_tests}] {name:<30}  trying={avg_t:.3f}  ignoring={avg_i:.3f}  gap={avg_t - avg_i:+.3f}")

        if trying_scores and ignoring_scores:
            avg_trying = sum(trying_scores) / len(trying_scores)
            avg_ignoring = sum(ignoring_scores) / len(ignoring_scores)
            min_trying = min(trying_scores)
            max_ignoring = max(ignoring_scores)
            gap = avg_trying - avg_ignoring

            results.append({
                "name": name,
                "avg_trying": avg_trying,
                "avg_ignoring": avg_ignoring,
                "min_trying": min_trying,
                "max_ignoring": max_ignoring,
                "gap": gap,
                "trying_scores": trying_scores,
                "ignoring_scores": ignoring_scores,
            })

    # Summary table
    print(f"\n\n{'=' * 90}")
    print("SUMMARY TABLE")
    print(f"{'=' * 90}")
    print(f"{'Constraint':<30} {'Trying':>8} {'Ignoring':>9} {'Gap':>7} {'MinTry':>8} {'MaxIgn':>8} {'Suggested':>10}")
    print(f"{'─' * 30} {'─' * 8} {'─' * 9} {'─' * 7} {'─' * 8} {'─' * 8} {'─' * 10}")

    for r in results:
        # Suggest threshold: midpoint between min_trying and max_ignoring
        # If gap is negative or tiny, mark as "discard?"
        if r["min_trying"] <= r["max_ignoring"]:
            # Distributions overlap — threshold can't cleanly separate
            suggested = f"  OVERLAP"
        else:
            midpoint = (r["min_trying"] + r["max_ignoring"]) / 2
            suggested = f"  {midpoint:.2f}"

        print(f"{r['name']:<30} {r['avg_trying']:>8.3f} {r['avg_ignoring']:>9.3f} "
              f"{r['gap']:>+7.3f} {r['min_trying']:>8.3f} {r['max_ignoring']:>8.3f} {suggested:>10}")

    print(f"\n{'=' * 90}")
    print("Done.")


if __name__ == "__main__":
    main()
