#!/usr/bin/env python3
"""Find specific misclassified responses and their common long words."""

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

def _tokenize(text):
    words = [w.strip(string.punctuation).lower() for w in text.split() if w.strip(string.punctuation)]
    return words

def _unique_long_ratio(text):
    words = _tokenize(text)
    if not words:
        return 0.0, set(), 0
    unique_long = {w for w in words if len(w) >= 7}
    return len(unique_long) / len(words), unique_long, len(words)


# For each model, look at condition C error cases — where label looks wrong
# Focus on a_to_b direction where audit says incidental long words cause errors
# In a_to_b: system=complex, user=simple
# An error is: model wrote simple text but scored high because of common words

for model_label, (path, T) in MODELS.items():
    records = load_records(path)
    cond_c = [r for r in records
              if r.get("conflict_id") == CONFLICT_ID
              and r.get("condition") == "C" and not r.get("error")]

    print(f"\n{'='*80}")
    print(f"MODEL: {model_label} (T={T})")
    print(f"{'='*80}")

    # Look at responses near the threshold boundary
    near_threshold = []
    for r in cond_c:
        resp = r["response"]
        ratio, unique_long, total = _unique_long_ratio(resp)
        distance = abs(ratio - T)
        if distance < 0.05:  # Near threshold
            near_threshold.append({
                "direction": r.get("direction"),
                "ratio": ratio,
                "total_words": total,
                "unique_long": sorted(unique_long),
                "label": r.get("label"),
                "response": resp[:300],
                "distance": distance,
            })

    near_threshold.sort(key=lambda x: x["distance"])

    print(f"\n  Near-threshold responses ({len(near_threshold)} within T±0.05):")
    for i, item in enumerate(near_threshold[:15]):
        print(f"\n  [{i+1}] dir={item['direction']}, ratio={item['ratio']:.4f}, "
              f"words={item['total_words']}, label={item['label']}")
        print(f"      Long words: {item['unique_long'][:20]}")
        print(f"      Response: {item['response'][:200]}...")

    # Focus on specific error patterns from audit
    # For 8B: a_to_b errors from incidental words like security, windows, practice, audience, flashcards
    # For Qwen: short responses where topic words inflate ratio
    # For gemma: incidental words like renewable, security, electricity
    # For openai: common words computers, learning, patterns, examples, decisions

    # Count long words in misclassified baseline responses
    # (condition B a_to_b where model tried simple but has some long words)
    simple_baseline_words = Counter()
    simple_baseline_count = 0
    for r in records:
        if r.get("conflict_id") != CONFLICT_ID or r.get("error"):
            continue
        cond = r.get("condition")
        direction = r.get("direction", "a_to_b")
        # Simple baselines
        if (cond == "B" and direction == "a_to_b") or (cond == "A" and direction == "b_to_a"):
            words = _tokenize(r["response"])
            long_words = {w for w in words if len(w) >= 7}
            for w in long_words:
                simple_baseline_words[w] += 1
            simple_baseline_count += 1

    print(f"\n  Simple baseline responses: {simple_baseline_count}")
    print(f"  Top 30 long words in simple baselines:")
    for w, c in simple_baseline_words.most_common(30):
        print(f"    {w:25s} {c:4d}")

    # Now look specifically at condition C a_to_b responses that are near threshold
    # These are the ones where "simple" text gets pushed above T by common words
    atob_above_T = []
    btoa_above_T = []
    for r in cond_c:
        resp = r["response"]
        ratio, unique_long, total = _unique_long_ratio(resp)
        direction = r.get("direction")

        if direction == "a_to_b" and T - 0.03 < ratio < T + 0.03:
            atob_above_T.append((r, ratio, unique_long, total))
        elif direction == "b_to_a" and T - 0.03 < ratio < T + 0.03:
            btoa_above_T.append((r, ratio, unique_long, total))

    print(f"\n  a_to_b near threshold (T±0.03): {len(atob_above_T)}")
    for r, ratio, ulw, total in atob_above_T[:5]:
        print(f"    ratio={ratio:.4f}, words={total}, long={sorted(ulw)[:15]}")
        print(f"    label={r.get('label')}")
        print(f"    response: {r['response'][:200]}")

    print(f"\n  b_to_a near threshold (T±0.03): {len(btoa_above_T)}")
    for r, ratio, ulw, total in btoa_above_T[:5]:
        print(f"    ratio={ratio:.4f}, words={total}, long={sorted(ulw)[:15]}")
        print(f"    label={r.get('label')}")
        print(f"    response: {r['response'][:200]}")
