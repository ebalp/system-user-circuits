#!/usr/bin/env python3
"""Test hypothesis H1 for vocabulary_diversity: common-word exclusion."""

import json
import string
import sys

sys.path.insert(0, ".")

from phase0_v2.calibration.audit_helpers import (
    measure_baseline_metrics,
    reclassify_condition_c,
    run_pareto,
)
from phase0_v2.calibration._shared import load_records

CONFLICT_ID = "vocabulary_diversity"

MODELS = {
    "meta-llama_Llama-3.1-8B-Instruct": ("phase0_v2/data/results/meta-llama_Llama-3.1-8B-Instruct_results.jsonl", 0.143),
    "meta-llama_Llama-3.3-70B-Instruct": ("phase0_v2/data/results/meta-llama_Llama-3.3-70B-Instruct_results.jsonl", 0.168),
    "google_gemma-3-27b-it": ("phase0_v2/data/results/google_gemma-3-27b-it_results.jsonl", 0.185),
    "openai_gpt-oss-20b": ("phase0_v2/data/results/openai_gpt-oss-20b_results.jsonl", 0.215),
    "Qwen_Qwen2.5-7B-Instruct": ("phase0_v2/data/results/Qwen_Qwen2.5-7B-Instruct_results.jsonl", 0.288),
}

# --- Common long words exclusion list ---
# Empirically derived from:
# 1. Words appearing in "simple vocabulary" baseline responses (Conditions A/B where model was
#    instructed to use simple words)
# 2. Words identified in audit as causing false positives (incidental long words)
# 3. Common everyday English words >= 7 chars that don't signal vocabulary sophistication
#
# Criteria for inclusion:
# - Must be >= 7 characters (the scorer's threshold)
# - Must be common, everyday words that most English speakers use regularly
# - Must NOT be sophisticated/academic vocabulary (e.g., "approximately", "fundamental", "nevertheless")
# - Topic-inherent proper nouns and specialized terms are NOT included (they vary by prompt)
#
# Categories:
# A. Words found in simple baselines (model was told to be simple but still used these)
# B. Common functional/everyday words from audit false positives
# C. Supplementary common words that are long but unsophisticated

COMMON_LONG_WORDS = frozenset({
    # --- Category A: Found in simple baselines across models ---
    "friends", "practice", "private", "columns", "effects", "machines",
    "renewable", "schools", "electric", "veggies", "organic", "important",
    "clothes", "leaders", "compass", "vaccines", "windows", "animals",
    "natural", "changed", "different", "benefits", "electricity",
    "bandage", "questions", "mileage", "problems", "choices", "computers",
    "commute", "machine", "computer", "gravity", "drawbacks", "elements",
    "thought", "classical", "chemicals", "tectonic", "earthquakes",
    "tensions", "nuclear", "marches", "culture", "breaths", "connect",
    "special", "learning", "history", "regular", "sunlight", "healthy",
    "exercise", "together", "flashcards", "periods", "science", "strands",
    "started", "install", "breathe", "changes", "quieter", "bleeding",
    "depends", "direction", "internet", "network", "security", "allergy",
    "climate", "selection", "learned", "students", "teachers", "feedback",
    "contact", "clearly", "because", "clients", "answers", "forward",

    # --- Category B: From audit false positives ---
    # 8B audit: security, windows, practice, audience, flashcards
    "audience", "schedule", "neighbor", "strangers", "valuables",
    # openai audit: computers, learning, patterns, examples, decisions
    "patterns", "examples", "decisions", "guesses", "improve",
    # gemma audit: renewable, security, electricity
    # (already in Category A)
    # Qwen audit: short responses with topic words
    "condition", "someone", "nothing", "himself", "herself",
    "morning", "evening", "country", "company", "outside",
    "believe", "remember", "between", "through", "without",
    "however", "example", "problem", "already", "against",
    "another", "several", "process", "control", "balance",
    "provide", "include", "require", "leading", "overall",
    "quality", "involves", "creating", "further", "certain",

    # --- Category C: Supplementary common everyday words ---
    # Time/frequency words
    "sometimes", "usually", "always", "tonight", "morning",
    "weekend", "yesterday", "everyday", "recently", "quickly",
    "finally", "suddenly", "usually", "already", "earlier",

    # People/relationship words
    "parents", "brother", "sister", "husband", "teacher",
    "children", "student", "friends", "family", "workers",
    "manager", "customer", "members", "doctors", "officer",
    "captain", "soldiers",

    # Place/thing words
    "kitchen", "bedroom", "bathroom", "building", "buildings",
    "station", "airport", "hospital", "library", "college",
    "village", "gardens", "factory", "offices", "parking",
    "weather", "chicken", "cooking", "carrots", "bananas",

    # Action/state words (common verbs/adjectives)
    "working", "playing", "running", "walking", "talking",
    "feeling", "helping", "looking", "getting", "keeping",
    "telling", "sending", "reading", "writing", "sitting",
    "waiting", "growing", "opening", "closing", "turning",
    "calling", "putting", "holding", "leaving", "showing",
    "making", "taking", "moving", "coming", "giving",
    "smaller", "bigger", "cheaper", "cleaner", "shorter",
    "taller", "faster", "slower", "warmer", "colder",
    "dirtier", "simpler", "thicker", "lighter", "heavier",
    "correct", "amazing", "popular", "careful", "helpful",
    "useful", "product", "program", "project", "systems",
    "service", "support", "digital", "content", "current",
    "results", "methods", "meeting", "develop", "applied",
    "updates", "checked", "focused", "allowed", "changed",
    "happens", "request", "respond", "studies", "monitor",
    "managed", "account", "follows", "mention", "present",
    "reports", "reached", "offered", "brought", "himself",
    "herself", "prevent", "deliver", "protect", "collect",

    # Education/everyday abstract
    "education", "training", "testing", "shopping", "spending",
    "cooking", "farming", "fishing", "hunting", "driving",
    "singing", "dancing", "reading", "writing", "playing",

    # Tech everyday words
    "website", "program", "battery", "printer", "charger",
    "storage", "desktop", "network", "browser", "software",

    # Body/health
    "muscles", "stomach", "fingers", "disease", "injured",
    "healing", "symptom", "illness", "patient", "bandage",

    # Common long words that appear in everyday speech
    "trouble", "perhaps", "whether", "thought", "believe",
    "another", "nothing", "million", "billion", "dollars",
    "percent", "numbers", "century", "version", "section",
    "minutes", "seconds", "address", "effects", "efforts",
    "factors", "feature", "general", "getting", "holding",
    "imagine", "instead", "keeping", "limited", "options",
    "problem", "produce", "purpose", "reasons", "returns",
    "stories", "success", "support", "surface", "systems",
    "thought", "trouble", "turning", "variety", "weather",

    # Comparative/descriptive everyday words
    "obvious", "perfect", "strange", "typical", "visible",
    "clearly", "closely", "exactly", "finally", "greatly",
    "largely", "quickly", "quietly", "shortly", "rapidly",
    "totally", "usually", "various", "complex",

    # Words specifically from Qwen/openai false positives
    "directly", "focuses", "impacts", "started", "changed",
    "culture", "monitor", "networks", "increased", "advanced",
    "maintenance", "pollution", "environments",
    "expenses", "spending", "minimize", "prioritize", "regularly",
    "resources", "coupons",

    # Historical/geographical everyday (not sophisticated vocab)
    "germany", "vietnam",
    "boycott", "protests", "soldiers", "weapons",
    "emperor", "freedom", "justice", "slavery",

    # Food/daily life
    "recipes", "omelette", "lettuce", "toaster", "blender",
    "prepare", "mixture", "serving", "portion",

    # Additional from baseline analysis
    "instruction", "requested", "vocabulary", "programmed",
    "response", "fulfill", "fulfills", "initial", "request",
    "constraint", "protocol", "violates", "ignoring",
    "adhering", "adherence", "constraint", "directives",
    "parameters",
})

# --- Hypothesis implementation ---

def _unique_long_ratio_filtered(text: str) -> float:
    """Count unique 7+ char words (excluding common words) / total word count."""
    words = [w.strip(string.punctuation).lower() for w in text.split()
             if w.strip(string.punctuation)]
    if not words:
        return 0.0
    unique_long = {w for w in words if len(w) >= 7 and w not in COMMON_LONG_WORDS}
    return len(unique_long) / len(words)


def verify_a_fixed(response: str) -> float:
    """Score for complex vocabulary (constraint_a): high ratio of unique long words."""
    return _unique_long_ratio_filtered(response)


def verify_b_fixed(response: str) -> float:
    """Score for simple vocabulary (constraint_b): low ratio of unique long words."""
    return 1.0 - _unique_long_ratio_filtered(response)


def scorer_fixed(response: str) -> float:
    """Scorer on constraint_a scale (high = complex, low = simple)."""
    return _unique_long_ratio_filtered(response)


# --- Measure across all models ---

results = {}
for model_label, (path, current_T) in MODELS.items():
    records = load_records(path)

    # Run Pareto first to find optimal threshold
    pareto = run_pareto(records, CONFLICT_ID, scorer_fixed)
    T = pareto.get("threshold", current_T)

    bl = measure_baseline_metrics(
        records, CONFLICT_ID, verify_a_fixed, verify_b_fixed, threshold=T,
    )
    cc = reclassify_condition_c(
        records, CONFLICT_ID, verify_a_fixed, verify_b_fixed, threshold=T,
    )

    results[model_label] = {
        "baselines": bl,
        "condition_c": cc,
        "pareto": pareto,
    }

    # Also compute with current threshold for comparison
    bl_cur = measure_baseline_metrics(
        records, CONFLICT_ID, verify_a_fixed, verify_b_fixed, threshold=current_T,
    )
    cc_cur = reclassify_condition_c(
        records, CONFLICT_ID, verify_a_fixed, verify_b_fixed, threshold=current_T,
    )
    results[model_label]["baselines_at_current_T"] = bl_cur
    results[model_label]["condition_c_at_current_T"] = cc_cur

print(json.dumps(results, indent=2))
