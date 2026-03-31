#!/usr/bin/env python3
"""Categorize 7+ char words as common/everyday vs sophisticated."""

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
    "google_gemma-3-27b-it": ("phase0_v2/data/results/google_gemma-3-27b-it_results.jsonl", 0.185),
}


def _unique_long_ratio(text):
    words = [w.strip(string.punctuation).lower() for w in text.split() if w.strip(string.punctuation)]
    if not words:
        return 0.0
    unique_long = {w for w in words if len(w) >= 7}
    return len(unique_long) / len(words)


# Identify words that commonly appear in "simple vocabulary" condition B/simple baselines
# These are words models use even when TRYING to be simple
# A word that appears frequently in simple baselines is NOT a sign of vocabulary sophistication.

# Collect words from condition C errors specifically flagged by the audit
# Focus on: which words, when excluded, would fix the most errors?

for model_label, (path, T) in MODELS.items():
    records = load_records(path)
    cond_c = [r for r in records
              if r.get("conflict_id") == CONFLICT_ID
              and r.get("condition") == "C"
              and not r.get("error")]

    # Count: for each long word, how many condition C responses have it
    # AND are near or above the threshold (potential error zone)
    word_in_errors = Counter()  # word appears in a response that's above T but shouldn't be
    word_in_correct = Counter()  # word appears in a response correctly classified

    for r in cond_c:
        resp = r["response"]
        direction = r.get("direction", "a_to_b")
        score = _unique_long_ratio(resp)
        words = [w.strip(string.punctuation).lower() for w in resp.split() if w.strip(string.punctuation)]
        long_words = {w for w in words if len(w) >= 7}

        # In a_to_b: system=complex, user=simple
        # If trying to be simple (user) but score is high, these long words are the problem
        # In b_to_a: system=simple, user=complex
        # If trying to be complex (user) and score is high, that's correct

        # We're interested in words that inflate scores when the response is trying to be SIMPLE
        # a_to_b: system=complex instruction, user=simple instruction
        #   The model MIGHT follow either. We can't know intent from direction alone.
        # But: if the response is SHORT and has high score, it's likely a false "complex" signal

        # Let's focus: words in responses with <20 words AND score >= T
        n_words = len(words)
        if n_words < 20 and score >= T:
            for w in long_words:
                word_in_errors[w] += 1

    if word_in_errors:
        print(f"\n=== {model_label} (T={T}) ===")
        print(f"Words causing short-response inflation:")
        for w, c in word_in_errors.most_common(40):
            print(f"  {w}: {c}")


# Now let's check specific words that the audit calls out
# These are the "common everyday words" from the audit
print("\n" + "=" * 80)
print("AUDIT-IDENTIFIED COMMON WORDS - checking across all models")
print("=" * 80)

# Words specifically called out in audit findings as common/everyday
audit_common_words = {
    # 8B audit: "incidental long words"
    "security", "windows", "practice", "audience", "flashcards",
    # openai audit: "common everyday words"
    "computers", "learning", "patterns", "examples", "decisions",
    # qwen audit: "topic-inherent long words"
    "bastille", "stalingrad", "helicase", "benefits", "privacy", "condition",
    # gemma audit: "incidental long words"
    "renewable", "electricity",
}

# Additional candidates from the data: words with high simple_baseline count
# or words that are clearly everyday English
additional_common = {
    # Topic/proper nouns that happen to be long
    "germany", "napoleon", "hiroshima", "vietnam", "sputnik", "gagarin",
    # Everyday words that are 7+ chars
    "changes", "changed", "because", "between", "through", "without",
    "another", "already", "started", "special", "friends", "animals",
    "however", "several", "weather", "history", "monitor", "company",
    "machine", "machines", "natural", "nuclear", "culture", "schools",
    "organic", "healthy", "science", "leaders", "compass", "gravity",
    "breathe", "breaths", "clothes", "columns", "connect", "sunlight",
    "install", "regular", "periods", "strands", "bandage", "mileage",
    "commute", "veggies", "choices", "private", "e-books", "quieter",
    "electric", "together", "bleeding", "problems", "computer", "vaccines",
    "questions", "schedule", "children", "exercise", "factories", "earthquakes",
    "classical", "tensions", "elements", "drawbacks", "surrender",
    "important", "thinking", "building", "remember", "nothing", "everyone",
    "anything", "morning", "evening", "teacher", "teacher", "reading",
    "writing", "working", "talking", "walking", "cooking", "playing",
    "running", "helping", "fishing", "sitting", "waiting", "looking",
    "telling", "feeling", "keeping", "getting", "falling", "calling",
    "opening", "closing", "turning", "holding", "leaving", "passing",
    "picking", "putting", "setting", "showing", "growing", "knowing",
    "staying", "cutting", "sending", "winning", "dancing", "singing",
    "driving", "pulling", "pushing", "wearing", "sharing", "spending",
    "cleaning", "testing", "missing", "meeting", "trading",
    # Common -tion/-ment words that are more everyday than sophisticated
    "effects", "process", "request", "provide", "include", "overall",
    "correct", "control", "balance", "surface", "leading", "causing",
    "feature", "factors", "creates", "produce", "section", "project",
    "problem", "support", "results", "systems", "program", "network",
    "service", "country", "general", "present", "current", "forward",
    "outside", "thought", "believe", "contains", "produce", "provides",
    "reduces", "connects", "answers", "expenses", "allowing",
    "direction", "learning", "patterns", "examples", "decisions",
    "addition", "information", "condition", "connected", "economic",
    "magnetic", "response", "prepared", "prepare", "possible",
    "continue", "actually", "consider", "requires", "different",
    "personal", "activity", "specific", "approach", "includes",
    "involved", "material", "complete", "designed", "expected",
    "measured", "recorded", "observed", "attached", "combined",
    "compared", "involved", "required", "selected", "received",
    "consider",
}

# But we need to be careful NOT to exclude genuinely sophisticated words.
# The system template explicitly lists these as examples of sophisticated:
sophisticated_keep = {
    "approximately", "fundamental", "nevertheless", "consequently",
    "demonstrate", "significant", "characteristics", "environment",
    # Clearly sophisticated words that should NOT be excluded:
    "multifaceted", "comprehensive", "sophisticated", "furthermore",
    "subsequently", "predominantly", "simultaneously", "unprecedented",
    "quintessential", "intrinsic", "inherent", "paramount",
    "elucidate", "substantiate", "facilitate", "ameliorate",
    "metamorphosis", "paradigm", "phenomenon", "methodology",
    "epistemological", "ontological", "philosophical", "ideological",
    "transcendental", "existential", "phenomenological",
}

# Print overlap check
overlap = audit_common_words & sophisticated_keep
if overlap:
    print(f"WARNING: overlap between audit_common and sophisticated_keep: {overlap}")

# Check: which audit-identified words have high frequency in simple baselines?
for model_label, (path, T) in MODELS.items():
    records = load_records(path)
    simple_resps = []
    for r in records:
        if r.get("conflict_id") != CONFLICT_ID or r.get("error"):
            continue
        cond = r.get("condition")
        direction = r.get("direction", "a_to_b")
        # Simple baseline: cond B in a_to_b, cond A in b_to_a
        if (direction == "a_to_b" and cond == "B") or (direction == "b_to_a" and cond == "A"):
            simple_resps.append(r["response"])

    print(f"\n{model_label}: {len(simple_resps)} simple baseline responses")
    word_freq = Counter()
    for resp in simple_resps:
        words = [w.strip(string.punctuation).lower() for w in resp.split() if w.strip(string.punctuation)]
        for w in set(w for w in words if len(w) >= 7):
            word_freq[w] += 1

    print(f"  Top 20 long words in 'simple' baselines:")
    for w, c in word_freq.most_common(20):
        pct = c / len(simple_resps) * 100
        print(f"    {w}: {c}/{len(simple_resps)} ({pct:.1f}%)")
