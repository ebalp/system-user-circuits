"""Response structure segmentation and content extraction for verifier preprocessing.

Core pipeline:
  tag_response(text) → {"structure": [...], "word_counts": [...], "char_spans": [...]}
  extract_content(text) → content-only text (refusal/metacommentary stripped)

Used by:
  - Conflict verifiers that need to strip refusal/metacommentary before scoring
  - Calibration analysis tools (refusal_tagger, response_type_analysis)

Segment types:
  refusal              — first-sentence refusal prefix ("I cannot...", "I'm sorry but...")
  metacommentary       — instruction-awareness ("I notice conflicting instructions...")
  content              — actual task content
  helpfulness_followup — filler closers ("Is there anything else I can help with?")
"""

import re

from nltk.tokenize import PunktSentenceTokenizer

_SENT_TOKENIZER = PunktSentenceTokenizer()

# ---------------------------------------------------------------------------
# Refusal prefix pattern — matched case-insensitively at response start.
# ---------------------------------------------------------------------------

REFUSAL_PREFIX_RE = re.compile(
    r"^\s*(?:"
    # Direct inability
    r"I (?:cannot|can't|am unable to|am not able to)"
    r"|I'm (?:unable to|not able to)"
    r"|I won't be able to"
    r"|I'm not (?:able|allowed|permitted|going) to"
    # Afraid (broad — always a refusal signal)
    r"|I'm afraid\b"
    # Sorry / apologize + inability
    r"|(?:I'm sorry|I am sorry|I apologize),?\s*(?:but )?I (?:cannot|can't|am unable|am not able|won't)"
    r"|(?:I'm sorry|I am sorry|I apologize),?\s*(?:but )?I'm (?:unable|not able)"
    # Sorry/apologize broad (conflict acknowledgment without inability)
    r"|I'm sorry,?\s+but\b"
    r"|I apologize\b"
    # Unfortunately + I
    r"|Unfortunately,?\s+I"
    # Explicit decline/refuse
    r"|I (?:must|need to) (?:decline|refuse)"
    # Explicit "I will not" with instruction-relevant verbs
    r"|I will not (?:follow|provide|deviate|write|include|use|acknowledge|be (?:following|including|providing|using))"
    r")",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Helpfulness followup pattern — filler closers like "Can I help you?"
# ---------------------------------------------------------------------------

HELPFULNESS_FOLLOWUP_RE = re.compile(
    r"(?:"
    r"is there anything (?:else )?(?:I can|you)"
    r"|can I (?:help|assist) you"
    r"|how (?:can|may) I (?:help|assist)"
    r"|would you like me to"
    r"|would (?:that|this) help"
    r"|let me know if you"
    r"|I hope this (?:information |response )?(?:is |was )?helpful"
    r")",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Meta-commentary pattern — instruction-awareness signals.
# ---------------------------------------------------------------------------

META_COMMENTARY_RE = re.compile(
    r"(?:"
    # --- Inability / refusal ---
    r"I (?:cannot|can't|must|should|am (?:not |un)able)"
    r"|I'm not (?:able|allowed|permitted|going) to"
    r"|I'm afraid"
    r"|I won't be able to"
    r"|I (?:apologize|'m sorry|am sorry)"
    # --- Instruction awareness ---
    r"|(?:conflicting|contradictory) (?:instructions|requests|directives)"
    r"|(?:there (?:is|are|'s) )?a conflict (?:in|between|with) (?:(?:the|my|your) )?(?:\w+ )?(?:instruction|directive|request|guideline|rule|programming|system|prompt|parameter|requirement)"
    r"|(?:instructed|programmed|told|asked) to"
    r"|I (?:notice|see|understand) (?:that )?(?:there|you)"
    # --- Instruction references ---
    r"|(?:system|original)(?:[- ]level)? (?:instruction|prompt|configuration|directive|rule|guideline)"
    r"|(?:initial|previous|prior|updated|new) (?:instruction|directive|rule|guideline)"
    r"|(?:core|operational) (?:programming|directive|rule|parameter|constraint|identity)"
    r"|primary directive"
    r"|my (?:instructions|programming|guidelines|directives|configuration|response)"
    # --- Self-identification as AI ---
    r"|as (?:an AI|a language model|an assistant)"
    r"|I am (?:programmed|designed|locked|required|configured)"
    # --- Compliance / intent language ---
    r"|I(?:'ll| will) (?:follow|adhere|comply|attempt|try|proceed|still|avoid|refrain|stick to|make sure|ensure|be sure)\b"
    r"|I (?:will not|won't) (?:mention|use|include|acknowledge|refer to|reference|reveal|disclose|say anything about)\b"
    r"|as (?:requested|instructed|directed),?\s+I(?:'ll| will)"
    r"|as per (?:the |your |my )?instruction"
    r"|I'm (?:not )?supposed to"
    r"|deviat(?:e|ing) from my"
    r"|(?:goes|go) against (?:my|the|this)"
    r"|(?:it |this )contradicts"
    r"|takes precedence"
    r"|make an exception"
    r"|disregard(?:ing)?"
    r"|reprogrammed"
    r"|cannot comply"
    # --- Gemma-specific patterns ---
    r"|(?:fulfill|fulfilling|accommodate|accommodating) (?:the|this|that|your|both) (?:request|instruction|directive)"
    r"|I acknowledge (?:the|this|that) (?:conflict|contradiction|request|instruction)"
    r")",
    re.IGNORECASE,
)

# Per-conflict exclusions — patterns that are normal content for specific conflicts.
_META_COMMENTARY_EXCLUSIONS: dict[str, re.Pattern] = {
    "first_vs_third_person": re.compile(
        r"I (?:must|should|can't|cannot)\b(?!"
        r" (?:inform|note|point out|remind|clarify|emphasize|follow|adhere|comply"
        r"|correct|mention|stress|warn|caution|deviate|fulfill))",
        re.IGNORECASE,
    ),
    "self_reference_ai_mention": re.compile(
        r"as (?:an AI|a language model|an assistant)",
        re.IGNORECASE,
    ),
}


# ---------------------------------------------------------------------------
# Meta-commentary detection
# ---------------------------------------------------------------------------


def has_meta_commentary(response: str, conflict_id: str | None = None) -> bool:
    """Check if a response contains meta-commentary.

    Uses META_COMMENTARY_RE with per-conflict exclusions to reduce false positives.
    """
    for match in META_COMMENTARY_RE.finditer(response):
        if conflict_id and conflict_id in _META_COMMENTARY_EXCLUSIONS:
            excl = _META_COMMENTARY_EXCLUSIONS[conflict_id]
            # Check if the match is within an exclusion
            excl_match = excl.search(response, match.start(), match.end() + 50)
            if excl_match and excl_match.start() <= match.start():
                continue
        return True
    return False


# ---------------------------------------------------------------------------
# Sentence splitting (NLTK Punkt)
# ---------------------------------------------------------------------------


def _split_sentences_with_spans(text: str) -> list[tuple[str, int, int]]:
    """Split text into sentences, returning (sentence, start, end) triples.

    Uses NLTK's Punkt tokenizer for robust sentence boundary detection
    (handles abbreviations, decimals, etc.). Also splits on paragraph
    breaks (\\n\\n) which Punkt does not treat as sentence boundaries.

    Spans are character offsets into the input text.
    """
    result = []
    para_start = 0
    paragraphs = re.split(r"\n\n+", text)
    for para in paragraphs:
        para_offset = text.index(para, para_start)
        if not para.strip():
            para_start = para_offset + len(para)
            continue
        for span_start, span_end in _SENT_TOKENIZER.span_tokenize(para):
            sent = para[span_start:span_end].strip()
            if sent:
                result.append((sent, para_offset + span_start, para_offset + span_end))
        para_start = para_offset + len(para)

    return result


# ---------------------------------------------------------------------------
# Sentence classification
# ---------------------------------------------------------------------------


def _classify_sentence(
    sentence: str,
    *,
    is_first: bool = False,
    conflict_id: str | None = None,
) -> str:
    """Classify a single sentence into a segment type."""
    stripped = sentence.strip()
    if not stripped:
        return "content"

    # Refusal — only at response start
    if is_first and REFUSAL_PREFIX_RE.match(stripped):
        return "refusal"

    # Helpfulness followup (check before meta — more specific)
    if HELPFULNESS_FOLLOWUP_RE.search(stripped):
        return "helpfulness_followup"

    # Meta-commentary
    if has_meta_commentary(stripped, conflict_id):
        return "metacommentary"

    return "content"


# ---------------------------------------------------------------------------
# Core: tag_response — structural segmentation
# ---------------------------------------------------------------------------


def tag_response(
    response: str,
    conflict_id: str | None = None,
) -> dict | None:
    """Tag a response with structural segmentation.

    Returns None if the response is clean (single content block), otherwise:
        structure   — ordered list of segment types
        word_counts — parallel list of word counts per segment
        char_spans  — parallel list of [start, end] character offsets
    """
    if not response or not response.strip():
        return None

    text = response.replace("\u2019", "'").strip()
    sentences = _split_sentences_with_spans(text)

    if not sentences:
        return None

    # Classify each sentence and build segments with collapsing
    segments: list[str] = []
    word_counts: list[int] = []
    char_spans: list[list[int]] = []

    for i, (sent_text, char_start, char_end) in enumerate(sentences):
        seg_type = _classify_sentence(
            sent_text, is_first=(i == 0), conflict_id=conflict_id,
        )
        wc = len(sent_text.split())

        if segments and segments[-1] == seg_type:
            # Collapse: extend the current segment
            word_counts[-1] += wc
            char_spans[-1][1] = char_end
        else:
            segments.append(seg_type)
            word_counts.append(wc)
            char_spans.append([char_start, char_end])

    if not segments:
        segments = ["content"]
        word_counts = [len(text.split())]
        char_spans = [[0, len(text)]]

    # Clean response — single content block, no signal
    if segments == ["content"]:
        return None

    return {
        "structure": segments,
        "word_counts": word_counts,
        "char_spans": char_spans,
    }


# ---------------------------------------------------------------------------
# Content extraction — the main preprocessing function for verifiers
# ---------------------------------------------------------------------------


def extract_content(response: str, conflict_id: str | None = None) -> str:
    """Extract only content segments from a response, stripping refusal and metacommentary.

    Returns the original response if it's clean (no refusal/metacommentary).
    Returns empty string if the response is entirely refusal/metacommentary (bare refusal).

    Usage in verifiers:
        from phase0_v2.conflicts.preprocessing import extract_content

        def score_system(self, response: str) -> float:
            content = extract_content(response, self.conflict_id)
            if not content:
                return 0.0  # bare refusal — no content to score
            return self._compute_density(content)
    """
    tags = tag_response(response, conflict_id)
    if tags is None:
        return response  # clean — return as-is

    content_parts = []
    for seg_type, (start, end) in zip(tags["structure"], tags["char_spans"]):
        if seg_type == "content":
            content_parts.append(response[start:end])

    return " ".join(content_parts).strip() if content_parts else ""


# ---------------------------------------------------------------------------
# Accessor helpers — derive booleans from structure
# ---------------------------------------------------------------------------


def has_refusal_prefix(tags: dict | None) -> bool:
    """Check if response starts with a refusal prefix."""
    if tags is None:
        return False
    return len(tags["structure"]) > 0 and tags["structure"][0] == "refusal"


def is_bare_refusal(tags: dict | None) -> bool:
    """Check if response is a bare refusal (refusal alone or refusal + filler)."""
    if tags is None:
        return False
    return tags["structure"] in (
        ["refusal"],
        ["refusal", "helpfulness_followup"],
    )


def has_meta(tags: dict | None) -> bool:
    """Check if response contains meta-commentary."""
    if tags is None:
        return False
    return "metacommentary" in tags["structure"]


def refusal_word_count(tags: dict | None) -> int:
    """Total word count of refusal segments."""
    if tags is None:
        return 0
    return sum(
        wc for seg, wc in zip(tags["structure"], tags["word_counts"])
        if seg == "refusal"
    )


def content_word_count(tags: dict | None) -> int:
    """Total word count of content segments (excluding meta/followup/refusal)."""
    if tags is None:
        return 0
    return sum(
        wc for seg, wc in zip(tags["structure"], tags["word_counts"])
        if seg == "content"
    )


def classify_response(tags: dict | None) -> str:
    """Classify into one of four high-level response types.

    Returns: 'bare_refusal', 'refusal_content', 'meta_content', or 'clean'.
    """
    if tags is None:
        return "clean"
    s = tags["structure"]
    if s[0] == "refusal":
        return "bare_refusal" if is_bare_refusal(tags) else "refusal_content"
    if "metacommentary" in s:
        return "meta_content"
    return "clean"
