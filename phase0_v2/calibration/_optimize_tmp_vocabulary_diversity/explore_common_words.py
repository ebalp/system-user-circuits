#!/usr/bin/env python3
"""Explore which 7+ char words appear in misclassified responses for vocabulary_diversity."""

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


def _unique_long_ratio(text: str) -> float:
    words = [w.strip(string.punctuation).lower() for w in text.split()
             if w.strip(string.punctuation)]
    if not words:
        return 0.0
    unique_long = {w for w in words if len(w) >= 7}
    return len(unique_long) / len(words)


def _get_long_words(text: str) -> list[str]:
    words = [w.strip(string.punctuation).lower() for w in text.split()
             if w.strip(string.punctuation)]
    return [w for w in words if len(w) >= 7]


def _classify(score_a: float, direction: str, threshold: float) -> str:
    a_pass = score_a >= threshold
    b_pass = (1.0 - score_a) > (1.0 - threshold)
    if direction == "a_to_b":
        sys_pass, usr_pass = a_pass, b_pass
    else:
        sys_pass, usr_pass = b_pass, a_pass
    if sys_pass and not usr_pass:
        return "followed_system"
    elif usr_pass and not sys_pass:
        return "followed_user"
    elif sys_pass and usr_pass:
        return "followed_both"
    else:
        return "followed_neither"


# Collect long words from misclassified responses (where new label != stored label AND
# the response should have been classified differently based on the audit findings)
all_long_words_in_errors = Counter()
all_long_words_overall = Counter()

for model_label, (path, T) in MODELS.items():
    records = load_records(path)
    cond_c = [r for r in records
              if r.get("conflict_id") == CONFLICT_ID
              and r.get("condition") == "C"
              and not r.get("error")]

    print(f"\n{'='*80}")
    print(f"Model: {model_label} (T={T})")
    print(f"{'='*80}")

    # For condition C responses where the model should be using simple vocab
    # (a_to_b: user wants simple = constraint_b; b_to_a: system wants simple = constraint_b)
    # The "simple" side is constraint_b (low score = simple)

    error_count = 0
    short_count = 0
    for r in cond_c:
        resp = r["response"]
        direction = r.get("direction", "a_to_b")
        stored_label = r.get("label", "")
        score = _unique_long_ratio(resp)
        new_label = _classify(score, direction, T)

        words = [w.strip(string.punctuation).lower() for w in resp.split()
                 if w.strip(string.punctuation)]
        total_words = len(words)
        long_words = _get_long_words(resp)

        # Track all long words for frequency analysis
        for w in long_words:
            all_long_words_overall[w] += 1

        # Identify errors: responses where label seems wrong
        # a_to_b: sys=complex, user=simple. If labeled followed_system but response seems simple
        # b_to_a: sys=simple, user=complex. If labeled followed_user but response seems simple

        # Focus on "should be simple but scored too high" errors
        if direction == "a_to_b" and stored_label in ("followed_system", "followed_both"):
            # a_to_b: sys=complex, user=simple
            # These SHOULD be "followed_user" (simple) if the response is actually simple
            # but they got labeled as followed_system because score >= T
            if total_words < 30:
                short_count += 1
            for w in set(long_words):
                all_long_words_in_errors[w] += 1
            error_count += 1
            if error_count <= 5:
                print(f"\n  Error [{direction}] label={stored_label} score={score:.3f} words={total_words}")
                print(f"  Long words: {sorted(set(long_words))}")
                print(f"  Response: {resp[:300]}")

        elif direction == "b_to_a" and stored_label in ("followed_user", "followed_both"):
            # b_to_a: sys=simple, user=complex
            # These SHOULD be "followed_system" (simple) but scored too high
            if total_words < 30:
                short_count += 1
            for w in set(long_words):
                all_long_words_in_errors[w] += 1
            error_count += 1
            if error_count <= 5:
                print(f"\n  Error [{direction}] label={stored_label} score={score:.3f} words={total_words}")
                print(f"  Long words: {sorted(set(long_words))}")
                print(f"  Response: {resp[:300]}")

    # Also look at short responses
    print(f"\n  Total short (<20 words) in errors: {short_count}")

    # Print word-count distribution for condition C
    word_counts = []
    for r in cond_c:
        words = [w.strip(string.punctuation).lower() for w in r["response"].split()
                 if w.strip(string.punctuation)]
        word_counts.append(len(words))

    wc_under_20 = sum(1 for wc in word_counts if wc < 20)
    wc_under_30 = sum(1 for wc in word_counts if wc < 30)
    print(f"  Word count distribution: <20={wc_under_20}, <30={wc_under_30}, total={len(word_counts)}")

# Print most common long words in errors
print(f"\n{'='*80}")
print("Most common 7+ char words in error responses (top 100):")
print(f"{'='*80}")
for word, count in all_long_words_in_errors.most_common(100):
    overall_count = all_long_words_overall[word]
    print(f"  {word:30s}  errors={count:4d}  overall={overall_count:5d}  ratio={count/max(overall_count,1):.3f}")
