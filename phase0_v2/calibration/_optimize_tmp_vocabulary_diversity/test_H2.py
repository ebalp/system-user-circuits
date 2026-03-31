#!/usr/bin/env python3
"""Test hypothesis H2 for vocabulary_diversity.

H2: Common-word exclusion + min word count floor.
(a) If total words < 20, return 0.0 (favoring simple).
(b) After collecting words, filter out COMMON_LONG_WORDS before counting unique long words.
"""

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
# These are 7+ character words that are common, everyday English — NOT signs
# of vocabulary sophistication. Empirically derived from:
# 1. Words appearing in "simple vocabulary" baseline responses
# 2. Words identified in audit error cases as "incidental" or "topic-inherent"
# 3. Common English words (pronouns, conjunctions, prepositions, common verbs/nouns)
#
# EXCLUDED from this list (genuinely sophisticated):
# approximately, fundamental, nevertheless, consequently, demonstrate,
# significant, characteristics, environment, multifaceted, comprehensive,
# sophisticated, furthermore, subsequently, predominantly, simultaneously,
# unprecedented, quintessential, intrinsic, inherent, paramount, etc.

COMMON_LONG_WORDS = frozenset({
    # --- Common function words / connectors (7+ chars) ---
    "because", "between", "through", "without", "another", "already",
    "nothing", "everyone", "anything", "someone", "something", "however",
    "several", "whether", "although", "against", "besides", "perhaps",
    "outside", "forward", "further", "together", "whatever", "wherever",
    "whenever", "neither", "herself", "himself", "itself", "ourselves",
    "themselves", "yourself",

    # --- Common everyday nouns ---
    "history", "machine", "machines", "company", "computer", "computers",
    "animals", "friends", "schools", "clothes", "weather", "morning",
    "evening", "teacher", "children", "country", "windows", "problem",
    "problems", "leaders", "culture", "nuclear", "science", "privacy",
    "security", "monitor", "compass", "gravity", "columns", "bandage",
    "mileage", "organic", "healthy", "veggies", "breaths", "sunlight",
    "strands", "surface", "project", "network", "service", "program",
    "general", "present", "current", "thought", "section", "support",
    "results", "balance", "feature", "factors", "control", "systems",
    "shelter", "climate", "vaccine", "vaccines", "pattern", "patterns",
    "example", "examples", "website", "picture", "kitchen", "bedroom",
    "library", "college", "meeting", "company", "account", "address",
    "message", "parking", "package", "printer", "battery", "blanket",
    "cabinet", "clothes", "costume", "cushion", "curtain", "deposit",
    "diamond", "display", "earring", "factory", "fitness", "flashlight",
    "garbage", "glasses", "grandma", "grandpa", "harvest", "highway",
    "holiday", "husband", "invoice", "jewelry", "journal", "kingdom",
    "laundry", "leather", "luggage", "mansion", "memoirs", "million",
    "mineral", "monitor", "morning", "nearest", "neutral", "package",
    "passage", "pilgrim", "pioneer", "plastic", "playful", "printer",
    "program", "publish", "quarter", "routine", "scholar", "subject",
    "stomach", "storage", "teacher", "therapy", "traffic", "trouble",
    "uniform", "vehicle", "version", "village", "vitamin", "volcano",
    "weather",

    # --- Common adjectives ---
    "special", "natural", "regular", "private", "correct", "overall",
    "different", "personal", "specific", "possible", "complete", "electric",
    "electric", "physical", "positive", "negative", "material", "national",
    "original", "standard", "internal", "external", "critical", "chemical",
    "familiar", "peaceful", "powerful", "grateful", "creative", "flexible",
    "moderate", "portable", "pleasant", "relevant", "valuable", "reliable",

    # --- Common verbs and verb forms ---
    "provide", "include", "include", "produce", "prepare", "believe",
    "contain", "created", "creates", "involve", "require", "support",
    "suppose", "suggest", "develop", "explain", "express", "imagine",
    "improve", "managed", "mention", "noticed", "offered", "perform",
    "promise", "protect", "receive", "related", "release", "remains",
    "removed", "replace", "request", "satisfy", "started", "thought",
    "allowed", "applied", "attempt", "becomes", "belongs", "changed",
    "claimed", "collect", "combine", "compare", "connect", "consist",
    "contain", "deliver", "depends", "despite", "divided", "entered",
    "examine", "existed", "follows", "granted", "handles", "happens",
    "helpful", "ignored", "imposed", "install", "invited", "measure",
    "observe", "offered", "operate", "pointed", "reached", "recover",
    "reduces", "related", "remains", "reminds", "reports", "resolve",
    "respond", "returns", "reveals", "rewards", "scatter", "selling",
    "serving", "settled", "shifted", "similar", "spreads", "squeeze",
    "stopped", "succeed", "suffers", "suggest", "survive", "touches",
    "towards", "trigger",

    # --- Common -ing forms of simple verbs ---
    "looking", "working", "getting", "running", "talking", "walking",
    "cooking", "playing", "helping", "fishing", "sitting", "waiting",
    "telling", "feeling", "keeping", "falling", "calling", "opening",
    "closing", "turning", "holding", "leaving", "passing", "picking",
    "putting", "setting", "showing", "growing", "knowing", "staying",
    "cutting", "sending", "winning", "dancing", "singing", "driving",
    "pulling", "pushing", "wearing", "sharing", "spending", "cleaning",
    "testing", "missing", "trading", "reading", "writing", "causing",
    "leading", "causing", "reading", "writing", "burning", "drawing",
    "finding", "joining", "kissing", "lasting", "lending", "marking",
    "packing", "raising", "resting", "rolling", "saving", "shaking",
    "smiling", "smoking", "solving", "tasting", "warming", "washing",
    "growing", "talking", "looking", "walking",

    # --- Common -ed forms ---
    "changed", "started", "created", "allowed", "applied", "claimed",
    "entered", "existed", "granted", "ignored", "imposed", "invited",
    "managed", "noticed", "offered", "pointed", "reached", "related",
    "removed", "reports", "settled", "shifted", "stopped", "succeed",
    "touched", "arrived", "assumed", "blocked", "brought", "checked",
    "damaged", "decided", "defined", "enjoyed", "escaped", "figured",
    "focused", "founded", "gathered", "guarded", "handled", "happens",
    "injured", "learned", "limited", "located", "married", "ordered",
    "painted", "pleased", "powered", "pressed", "printed", "rescued",
    "retired", "rotated", "sampled", "secured", "shipped", "skipped",
    "slipped", "smashed", "spotted", "studied", "trusted", "updated",
    "visited", "watched", "wrapped", "blocked", "carried", "covered",
    "crossed", "dropped", "enabled", "engaged", "expired", "exposed",
    "guessed", "hunted", "knocked", "laughed", "matched", "mounted",
    "plugged", "praised", "pressed", "printed", "puzzled", "refused",
    "scanned", "shocked", "sparked", "squared", "starred", "stepped",
    "stuffed", "swapped", "tracked", "treated", "trimmed", "twisted",
    "wrapped",

    # --- Common topic-specific words that appear in simple responses ---
    "effects", "process", "changes", "results", "answers", "choices",
    "periods", "details", "reasons", "numbers", "records", "species",
    "methods", "sources", "options", "figures", "efforts", "designs",
    "demands", "targets", "markets", "devices", "signals", "aspects",
    "borders", "channel", "degrees", "charges", "battles", "beliefs",
    "flights", "formats", "muscles", "profits", "threats", "volumes",
    "weights",

    # --- Common proper nouns / place names ---
    "germany", "america", "england", "britain", "vietnam", "napoleon",
    "bastille", "hiroshima", "nagasaki", "stalingrad", "sputnik",
    "gagarin", "soviets", "america", "russian", "chinese", "english",
    "spanish", "italian", "african", "european", "american",

    # --- Common domain words that are NOT sophisticated ---
    "benefit", "benefits", "practice", "learning", "schedule", "computer",
    "exercise", "question", "questions", "decision", "decisions",
    "building", "remember", "response", "addition", "condition",
    "economic", "magnetic", "electric", "internet", "function",
    "industry", "audience", "strategy", "evidence", "material",
    "feedback", "document", "calendar", "category", "currency",
    "customer", "database", "deadline", "delivery", "election",
    "employee", "engineer", "entrance", "envelope", "estimate",
    "exchange", "exercise", "frontier", "handbook", "hardware",
    "identity", "imagined", "invested", "lifetime", "language",
    "listened", "location", "magazine", "majority", "measured",
    "medicine", "movement", "neighbor", "notebook", "observed",
    "opponent", "organize", "overcome", "passport", "password",
    "patience", "planning", "platform", "politics", "populace",
    "position", "practice", "pressure", "property", "purchase",
    "reaction", "register", "religion", "resource", "sandwich",
    "scenario", "software", "solution", "surprise", "training",
    "treasure", "universe", "warranty",

    # --- Common compound/derived forms ---
    "renewable", "non-renewable", "electricity", "earthquake",
    "earthquakes", "classical", "tensions", "elements", "drawbacks",
    "surrender", "important", "factories", "connected", "connected",
    "direction", "machines", "provides", "contains", "connects",
    "expenses", "allowing", "prepared", "continue", "actually",
    "consider", "requires", "includes", "involved", "designed",
    "expected", "measured", "recorded", "observed", "attached",
    "combined", "compared", "required", "selected", "received",
    "features", "traditionally", "typically", "basically", "hopefully",
    "directly", "recently", "commonly", "properly", "actively",
    "actually", "entirely", "honestly", "possibly", "probably",
    "recently", "usually", "readily", "largely", "closely",
    "finally", "clearly", "briefly", "exactly", "quickly",
    "quietly", "rapidly", "quietly", "shortly", "shortly",
    "roughly", "loosely", "tightly",

    # --- Common -tion/-sion words that are everyday ---
    "information", "connection", "education", "attention", "location",
    "direction", "election", "creation", "relation", "solution",
    "reaction", "addition", "position", "tradition", "situation",
    "condition", "question", "function", "decision", "division",
    "occasion", "rotation", "vacation", "variation", "inflation",
    "fraction", "adoption", "donation", "emission", "invasion",
    "mountain", "fountain", "curtains", "portions", "sessions",
    "versions", "sections", "passions", "missions", "tensions",
    "lessons",

    # --- Additional from error analysis ---
    "flashcards", "commute", "quieter", "e-books",
    "blitzkrieg", "urbanization", "isolation", "flexibility",
    "distractions", "reliability", "performance", "maintenance",
    "prioritize", "allocate", "resource",

    # --- Words that appear in meta-commentary ---
    "response", "request", "vocabulary", "programmed", "designed",
    "requested",
})


MIN_WORD_COUNT = 20


def _unique_long_ratio_h2(text: str) -> float:
    """H2: unique long word ratio with common-word exclusion + min word floor."""
    words = [w.strip(string.punctuation).lower() for w in text.split()
             if w.strip(string.punctuation)]
    if not words:
        return 0.0
    if len(words) < MIN_WORD_COUNT:
        return 0.0
    unique_long = {w for w in words if len(w) >= 7 and w not in COMMON_LONG_WORDS}
    return len(unique_long) / len(words)


def verify_a_fixed(response: str) -> float:
    """Score for complex vocabulary (constraint_a scale)."""
    return _unique_long_ratio_h2(response)


def verify_b_fixed(response: str) -> float:
    """Score for simple vocabulary (inverted)."""
    return 1.0 - _unique_long_ratio_h2(response)


def scorer_fixed(response: str) -> float:
    """Scorer on constraint_a scale for Pareto analysis."""
    return _unique_long_ratio_h2(response)


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

    print(f"\n{'='*60}", file=sys.stderr)
    print(f"{model_label}", file=sys.stderr)
    print(f"  Pareto T: {T:.3f} (current: {current_T})", file=sys.stderr)
    print(f"  BA: {bl['ba']:.4f}" if bl['ba'] else "  BA: N/A", file=sys.stderr)
    print(f"  SBR_a: {bl['sbr_a']:.4f}, UCR_a: {bl['ucr_a']:.4f}" if bl['sbr_a'] is not None else "  SBR_a: N/A", file=sys.stderr)
    print(f"  SBR_b: {bl['sbr_b']:.4f}, UCR_b: {bl['ucr_b']:.4f}" if bl['sbr_b'] is not None else "  SBR_b: N/A", file=sys.stderr)
    print(f"  Condition C: {cc['new_labels']}", file=sys.stderr)
    print(f"  Changed: {cc['changed']}, Contradiction: {cc['contradiction_pct']:.2f}%", file=sys.stderr)
    print(f"  Pareto: BA={pareto.get('ba', 'N/A')}, d_norm={pareto.get('d_norm', 'N/A')}", file=sys.stderr)

print(json.dumps(results, indent=2))
