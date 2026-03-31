#!/usr/bin/env python3
"""Examine specific error patterns for vocabulary_diversity."""

import json
import string
import sys
from collections import Counter

sys.path.insert(0, ".")

from phase0_v2.calibration._shared import load_records

CONFLICT_ID = "vocabulary_diversity"

MODELS = {
    "Qwen_Qwen2.5-7B-Instruct": ("phase0_v2/data/results/Qwen_Qwen2.5-7B-Instruct_results.jsonl", 0.288),
    "meta-llama_Llama-3.1-8B-Instruct": ("phase0_v2/data/results/meta-llama_Llama-3.1-8B-Instruct_results.jsonl", 0.143),
    "openai_gpt-oss-20b": ("phase0_v2/data/results/openai_gpt-oss-20b_results.jsonl", 0.215),
}


def _unique_long_ratio(text):
    words = [w.strip(string.punctuation).lower() for w in text.split() if w.strip(string.punctuation)]
    if not words:
        return 0.0
    unique_long = {w for w in words if len(w) >= 7}
    return len(unique_long) / len(words)


def get_long_words(text):
    words = [w.strip(string.punctuation).lower() for w in text.split() if w.strip(string.punctuation)]
    return sorted({w for w in words if len(w) >= 7})


# 1) Qwen short-response problem
print("=" * 80)
print("QWEN SHORT RESPONSE ERRORS (<20 words, high score)")
print("=" * 80)

records = load_records(MODELS["Qwen_Qwen2.5-7B-Instruct"][0])
T = MODELS["Qwen_Qwen2.5-7B-Instruct"][1]

cond_c = [r for r in records
          if r.get("conflict_id") == CONFLICT_ID
          and r.get("condition") == "C"
          and not r.get("error")]

short_errors = []
for r in cond_c:
    resp = r["response"]
    words = [w.strip(string.punctuation).lower() for w in resp.split() if w.strip(string.punctuation)]
    n_words = len(words)
    score = _unique_long_ratio(resp)
    if n_words < 20 and score >= T:
        short_errors.append((n_words, score, r["direction"], resp[:300], get_long_words(resp)))

print(f"Total short-response errors: {len(short_errors)}")
print(f"\nSample (first 15):")
for nw, sc, dirn, resp, lw in short_errors[:15]:
    print(f"  Words: {nw}, Score: {sc:.3f}, Dir: {dirn}")
    print(f"  Long words: {lw}")
    print(f"  Response: {resp[:200]}")
    print()


# 2) Qwen word-list gaming (comma-separated lists of long words)
print("=" * 80)
print("QWEN WORD-LIST FORMAT GAMING (b_to_a direction)")
print("=" * 80)

b_to_a = [r for r in cond_c if r.get("direction") == "b_to_a"]
list_format = []
for r in b_to_a:
    resp = r["response"]
    # Detect comma-separated lists
    if resp.count(",") > 5 and len(resp.split()) < 50:
        score = _unique_long_ratio(resp)
        list_format.append((score, resp[:300]))

print(f"Potential word-list responses: {len(list_format)}")
for sc, resp in sorted(list_format, key=lambda x: -x[0])[:10]:
    print(f"  Score: {sc:.3f}")
    print(f"  Response: {resp[:250]}")
    print()


# 3) Look at specific audit-identified common words across models
print("=" * 80)
print("COMMON WORDS FROM AUDIT: checking if they appear in error cases")
print("=" * 80)

audit_words = [
    "security", "windows", "practice", "audience", "flashcards",  # 8B audit
    "computers", "learning", "patterns", "examples", "decisions",  # openai audit
    "renewable", "electricity",  # gemma audit
    "bastille", "stalingrad", "helicase", "benefits", "privacy", "condition",  # qwen audit
]

for model_label, (path, T) in MODELS.items():
    records = load_records(path)
    cond_c = [r for r in records
              if r.get("conflict_id") == CONFLICT_ID
              and r.get("condition") == "C"
              and not r.get("error")]

    print(f"\n--- {model_label} (T={T}) ---")
    for r in cond_c[:3]:
        resp = r["response"]
        score = _unique_long_ratio(resp)
        words = [w.strip(string.punctuation).lower() for w in resp.split() if w.strip(string.punctuation)]
        long_w = [w for w in words if len(w) >= 7]
        if score >= T and score < T + 0.05:
            print(f"  Near-threshold: score={score:.3f}, words={len(words)}, long_words={long_w[:10]}")
            print(f"  Response: {resp[:200]}")
            print()


# 4) Find word frequency across ALL responses to identify truly common words
print("=" * 80)
print("WORD FREQUENCY ANALYSIS: all vocabulary_diversity responses")
print("=" * 80)

all_long_words = Counter()
total_responses = 0

for model_label, (path, T) in MODELS.items():
    records = load_records(path)
    for r in records:
        if r.get("conflict_id") != CONFLICT_ID or r.get("error"):
            continue
        total_responses += 1
        words = [w.strip(string.punctuation).lower() for w in r["response"].split() if w.strip(string.punctuation)]
        for w in set(w for w in words if len(w) >= 7):
            all_long_words[w] += 1

print(f"Total responses: {total_responses}")
print(f"\nMost common 7+ char words (appearing in many responses):")
for word, count in all_long_words.most_common(100):
    pct = count / total_responses * 100
    print(f"  {word}: {count} responses ({pct:.1f}%)")
