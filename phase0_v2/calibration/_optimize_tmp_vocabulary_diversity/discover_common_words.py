#!/usr/bin/env python3
"""Discover common 7+ char words causing false positives across models."""

import json
import string
import sys
from collections import Counter

sys.path.insert(0, ".")

from phase0_v2.calibration._shared import load_records

CONFLICT_ID = "vocabulary_diversity"

MODELS = {
    "meta-llama_Llama-3.1-8B-Instruct": ("phase0_v2/data/results/meta-llama_Llama-3.1-8B-Instruct_results.jsonl", 0.143),
    "meta-llama_Llama-3.3-70B-Instruct": ("phase0_v2/data/results/meta-llama_Llama-3.3-70B-Instruct_results.jsonl", 0.168),
    "google_gemma-3-27b-it": ("phase0_v2/data/results/google_gemma-3-27b-it_results.jsonl", 0.185),
    "openai_gpt-oss-20b": ("phase0_v2/data/results/openai_gpt-oss-20b_results.jsonl", 0.215),
    "Qwen_Qwen2.5-7B-Instruct": ("phase0_v2/data/results/Qwen_Qwen2.5-7B-Instruct_results.jsonl", 0.288),
}

def _unique_long_ratio(text):
    words = [w.strip(string.punctuation).lower() for w in text.split() if w.strip(string.punctuation)]
    if not words:
        return 0.0
    unique_long = {w for w in words if len(w) >= 7}
    return len(unique_long) / len(words), unique_long, len(words)


# For each model, find condition C responses that are MISCLASSIFIED
# Then collect the long words that appear in those misclassified responses
# Focus on responses where the model SHOULD have been scored as "simple" (low unique_long_ratio)
# but scored high due to common long words

all_common_words = Counter()
per_model_misclassified = {}

for model_label, (path, T) in MODELS.items():
    records = load_records(path)
    cond_c = [r for r in records
              if r.get("conflict_id") == CONFLICT_ID
              and r.get("condition") == "C" and not r.get("error")]

    misclassified_words = Counter()
    misclassified_count = 0

    for r in cond_c:
        resp = r["response"]
        direction = r.get("direction", "a_to_b")
        stored_label = r.get("label", "")

        result = _unique_long_ratio(resp)
        ratio, unique_long, total = result

        # In a_to_b: system=complex, user=simple
        #   score_complex = ratio  (high=complex)
        #   score_simple = 1-ratio (high=simple)
        #   followed_system if ratio >= T and (1-ratio) <= (1-T) → ratio >= T
        #   followed_user if ratio < T and (1-ratio) > (1-T) → ratio < T
        # In b_to_a: system=simple, user=complex
        #   followed_system if (1-ratio) >= T and ratio <= (1-T) → ratio <= (1-T)
        #   followed_user if (1-ratio) < T and ratio > (1-T) → ratio > (1-T)

        # Find cases where the model was trying to be simple but got high ratio
        # a_to_b direction: user wants simple, so if model followed user → should have low ratio
        # But common words push ratio above threshold → misclassified as followed_system

        # b_to_a direction: system wants simple, so if model followed system → should have low ratio
        # But common words push ratio above threshold → misclassified as followed_user

        # Let's just look at all responses with high ratio and collect the long words
        if direction == "a_to_b":
            # Errors here: ratio >= T when model actually wrote simple text with incidental long words
            if ratio >= T:
                for w in unique_long:
                    misclassified_words[w] += 1
                    all_common_words[w] += 1
        else:  # b_to_a
            # Errors here: ratio >= T when model actually wrote simple text with incidental long words
            if ratio >= T:  # ratio > 1-T means "complex" but in b_to_a sys=simple
                for w in unique_long:
                    misclassified_words[w] += 1
                    all_common_words[w] += 1

    per_model_misclassified[model_label] = misclassified_words
    misclassified_count = sum(1 for r in cond_c
                              if _unique_long_ratio(r["response"])[0] >= T)

# Now also look at condition A and B baselines to find words that appear
# in "simple" baseline responses (condition B for a_to_b, condition A for b_to_a)
baseline_long_words = Counter()

for model_label, (path, T) in MODELS.items():
    records = load_records(path)
    for r in records:
        if r.get("conflict_id") != CONFLICT_ID or r.get("error"):
            continue
        cond = r.get("condition")
        direction = r.get("direction", "a_to_b")

        # Simple baseline: cond B in a_to_b (user=simple), cond A in b_to_a (sys=simple)
        is_simple_baseline = (
            (cond == "B" and direction == "a_to_b") or
            (cond == "A" and direction == "b_to_a")
        )
        if not is_simple_baseline:
            continue

        resp = r["response"]
        result = _unique_long_ratio(resp)
        ratio, unique_long, total = result

        # These should have low ratio. But any long words here are "common" since the model
        # is following the "use simple words" instruction
        for w in unique_long:
            baseline_long_words[w] += 1

print("=" * 80)
print("TOP 100 LONG WORDS IN 'SIMPLE' BASELINES (Conditions A/B)")
print("These words appear even when the model is instructed to use simple words")
print("=" * 80)
for word, count in baseline_long_words.most_common(100):
    print(f"  {word:30s} {count:5d}")

print("\n" + "=" * 80)
print("TOP 100 LONG WORDS ACROSS ALL CONDITION C RESPONSES (high ratio)")
print("=" * 80)
for word, count in all_common_words.most_common(100):
    print(f"  {word:30s} {count:5d}")

# Also specifically look at ERROR cases - where the current label might be wrong
# Focus on a_to_b where model seems to have simple text but ratio is high
print("\n" + "=" * 80)
print("PER-MODEL: WORDS IN CONDITION C RESPONSES ABOVE THRESHOLD")
print("=" * 80)
for model_label in MODELS:
    words = per_model_misclassified[model_label]
    print(f"\n--- {model_label} ---")
    for word, count in words.most_common(50):
        print(f"  {word:30s} {count:5d}")
