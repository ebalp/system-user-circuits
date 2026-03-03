"""Verification helpers for system-user conflict checks."""

import json
import re
import string
from collections import Counter

import emoji
import nltk
import syllapy

# Ensure NLTK data is available
for _res, _pkg in [("tokenizers/punkt_tab", "punkt_tab")]:
    try:
        nltk.data.find(_res)
    except LookupError:
        nltk.download(_pkg, quiet=True)


def split_sentences(text: str) -> list[str]:
    """Split text into sentences using NLTK sent_tokenize."""
    return [s.strip() for s in nltk.sent_tokenize(text) if s.strip()]


def count_words(text: str) -> int:
    """Return number of words using NLTK RegexpTokenizer."""
    tokenizer = nltk.tokenize.RegexpTokenizer(r"\w+")
    return len(tokenizer.tokenize(text))


def count_unique_words(text: str) -> int:
    """Return number of unique words (lowercased, punctuation stripped)."""
    return len(set(w.strip(string.punctuation).lower() for w in text.split() if w.strip()))


WORD_POOL = [
    "western", "signal", "spot", "bottom", "administration", "welcome",
    "agency", "wish", "press", "president", "brush", "beat", "growth",
    "bone", "equal", "region", "performance", "walk", "film", "rock",
    "total", "ease", "establishment", "parking", "plenty", "claim", "trade",
    "street", "decision", "agreement", "coach", "brain", "style", "brown",
    "procedure", "speed", "valuable", "session", "district", "dinner",
    "joke", "plate", "motor", "spend", "difference", "examination", "horse",
    "curve", "bother", "possibility", "activity", "hello", "background",
    "author", "actor", "bicycle", "throat", "character", "increase", "file",
    "inspector", "potential", "building", "shoe", "garden", "interview",
    "recognition", "spiritual", "sandwich", "passenger", "response",
    "variation", "candy", "guest", "price", "convert", "mouth", "song",
    "suspect", "roof", "refrigerator", "jury", "engineering", "crew",
    "description", "score", "letter", "suggestion", "national", "hall",
    "theory", "story", "history", "medium", "glass", "stomach", "ability",
    "village", "city", "confidence", "priest", "point", "body", "secret",
    "noise", "warning", "round", "flower", "permission", "prompt", "abuse",
    "save", "border", "drive", "meal", "confusion", "living", "significance",
    "creative", "blame", "housing", "drink", "silver", "damage",
    "environment", "savings", "tourist", "post", "grandmother", "push",
    "final", "swim", "stuff", "funeral", "source", "tradition", "snow",
    "distance", "sensitive", "major", "click", "period", "expression",
    "repeat", "closet", "sail", "clothes", "duty", "step", "jump",
    "professional", "front", "inside", "subject", "balance", "adult",
    "sample", "wedding", "king", "wife", "camp", "safe", "fault", "shame",
    "capital", "record", "swing", "minimum", "machine", "lead", "salary",
    "affair", "stage", "access", "chain", "kick", "airport", "philosophy",
    "chest", "place", "advertising", "rent", "tour", "construction", "war",
    "spray", "task", "friend", "promotion", "surround", "purpose",
    "conflict", "requirement", "hole", "junior", "catch", "wall", "position",
    "respect", "coat", "teach", "resolve", "employee", "market", "serve",
    "tone", "union", "river", "concept", "recipe", "reserve", "proof",
    "independent", "assignment", "amount", "edge", "check", "estimate",
    "stable", "delivery", "mirror", "representative", "nature", "fruit",
    "town", "upper", "stay", "neck", "network", "league", "signature",
    "importance", "engineer", "external", "simple", "student", "shift",
    "lady", "community", "youth", "skirt", "blind", "disease", "positive",
    "calm", "tune", "preference", "presentation", "thought", "effort",
    "implement", "floor", "stranger", "grade", "tennis", "collection",
    "register", "divide", "chair", "combine", "extension", "frame", "wave",
    "mouse", "counter", "resolution", "discussion", "accident", "dress",
    "hearing", "layer", "profile", "answer", "teacher", "belt", "equivalent",
    "image", "risk", "remote", "produce", "sand", "punch", "title",
    "mortgage", "number", "extent", "opinion", "dance", "material", "leader",
    "muscle", "variety", "director", "calendar", "pace", "consequence",
    "doctor", "share", "career", "force", "aspect", "respond", "reality",
    "impact", "news", "series", "mother", "strike", "month", "entertainment",
    "clue", "natural", "conversation", "earth", "percentage", "budget",
    "beginning", "young", "store", "value", "nurse", "tower", "camera",
    "panic", "basket", "chart", "feedback", "reputation", "exercise", "yard",
    "collar", "plant", "passion", "spread", "ticket", "island", "object",
    "proposal", "heat", "resident", "politics", "expert", "salt",
    "inspection", "couple", "dependent", "chicken", "currency", "scheme",
    "employment", "manager", "cover", "relative", "rate", "program",
    "bridge", "talk", "vehicle", "substance", "advantage", "death",
    "tomorrow", "request", "church", "forever", "debt", "following",
    "sector", "economics", "bench", "solid", "income", "honey", "grocery",
    "form", "model", "farm", "skill", "policy", "husband", "sink", "driver",
    "leather", "boat", "brick", "rush", "location", "manufacturer",
    "occasion", "introduction", "category", "office", "pride", "client",
    "anybody", "individual", "interest", "profession", "resource",
    "chocolate", "formal", "abroad", "associate", "surgery", "team", "path",
    "initial", "demand", "contest", "contribution", "channel", "discipline",
    "concert", "effective", "industry", "metal", "minute", "rest",
    "argument", "health", "investment", "lesson", "marriage", "evidence",
    "benefit", "affect", "special", "payment", "obligation", "smile",
    "addition", "towel", "soil", "internet", "entry", "family",
    "grandfather", "tank", "climate", "volume", "poet", "screen", "charity",
    "tooth", "mention", "reveal", "court", "freedom", "sport", "classroom",
    "carry", "distribution", "country", "stretch", "delay", "plastic",
    "worry", "goal", "election", "midnight", "inflation", "challenge",
    "coast", "campaign", "jacket", "visual", "weather", "cable", "buddy",
    "historian", "sympathy", "tension", "person", "usual", "worth",
    "physical", "raise", "writing", "party", "spring", "physics", "concern",
    "change", "target", "room", "bird", "normal", "meaning", "leadership",
    "ambition", "essay", "repair", "night", "drawing", "phase", "anger",
    "personality", "storage", "selection", "contract", "station", "tongue",
    "truth", "group", "move", "light", "mission", "shop", "alternative",
    "agent", "airline", "craft", "fuel", "partner", "entrance", "article",
    "summer", "extreme", "hospital", "fall", "piano", "gap", "report",
    "wind", "shine", "perception", "reference", "treat", "term", "status",
    "strategy", "enthusiasm", "concentrate", "travel", "business", "end",
    "employ", "brave", "process", "general", "highway", "psychology",
    "conference", "show", "weight", "club", "zone", "tonight", "excuse",
    "landscape", "satisfaction", "disaster", "prior", "visit", "idea",
    "comparison", "winner", "lake", "prize", "struggle", "safety",
    "conclusion", "strain", "measurement", "train", "insurance", "tree",
    "course", "slice", "patience", "escape", "royal", "childhood",
    "picture", "improvement", "pitch", "transition", "committee", "teaching",
    "complex", "people", "original", "data", "reading", "bunch", "judgment",
    "painting", "player", "north", "carpet", "officer", "clock", "baby",
    "assumption", "bill", "double", "finish", "brilliant", "math",
    "restaurant", "virus", "event", "failure", "intention", "pressure",
    "explanation", "angle", "efficiency", "habit", "chance", "transportation",
    "flow", "injury", "surprise",
]


def count_alliterative_words(text: str) -> int:
    """Count words participating in alliterative runs (consecutive words sharing first letter)."""
    words = [w.strip(string.punctuation).lower() for w in text.split()
             if w.strip(string.punctuation) and w.strip(string.punctuation)[0].isalpha()]
    if len(words) < 2:
        return 0
    count = 0
    in_run = False
    for i in range(len(words) - 1):
        if words[i][0] == words[i + 1][0]:
            if in_run:
                count += 1
            else:
                count += 2
            in_run = True
        else:
            in_run = False
    return count


def count_numbers(text: str) -> int:
    """Count digit runs after stripping punctuation (e.g. 3.14 counts as one)."""
    text = text.translate(str.maketrans("", "", string.punctuation))
    return len(re.findall(r"\d+", text))


def has_emoji(text: str) -> bool:
    """True if text contains emoji."""
    return emoji.emoji_count(text) > 0


def _is_emoji_char(c: str) -> bool:
    """True if c is an emoji (single character)."""
    return emoji.is_emoji(c)


def sentence_ends_with_emoji(text: str) -> bool:
    """True if every sentence ends with an emoji."""
    sentences = split_sentences(text)
    for i, sentence in enumerate(sentences):
        stripped = sentence.translate(str.maketrans("", "", string.punctuation)).strip()
        if not stripped:
            return False
        last_char = stripped[-1]
        second_last = stripped[-2] if len(stripped) > 1 else stripped[-1]
        if _is_emoji_char(last_char) or _is_emoji_char(second_last):
            continue
        if i < len(sentences) - 1:
            next_stripped = sentences[i + 1].translate(
                str.maketrans("", "", string.punctuation)
            ).strip()
            if not next_stripped:
                return False
            if _is_emoji_char(next_stripped[0]):
                continue
        return False
    return True


def has_bullet_points(text: str) -> bool:
    """True if text has line-starting * or - bullets."""
    return bool(re.search(r"^\s*[-*]\s", text, re.MULTILINE))


def has_sub_bullets(text: str) -> bool:
    """True if text has * bullet points with - sub-bullets under each.

    Detects lines starting with '* ' followed by '- ' sub-bullet lines
    (indented or not). Handles markdown **bold** headers without false splits.
    """
    lines = text.split("\n")
    bullet_count = 0
    current_has_sub = False
    all_have_subs = True

    for line in lines:
        stripped = line.strip()
        if re.match(r"^\*\s", stripped):
            # New bullet point — check if previous bullet had a sub-bullet
            if bullet_count > 0 and not current_has_sub:
                all_have_subs = False
            bullet_count += 1
            current_has_sub = False
        elif re.match(r"^\s*-\s", stripped) and bullet_count > 0:
            current_has_sub = True

    # Check last bullet
    if bullet_count > 0 and not current_has_sub:
        all_have_subs = False

    return bullet_count >= 2 and all_have_subs


def no_bullets(text: str) -> bool:
    """True if no bullet-point formatting (* or - list items at line start)."""
    # Check for * bullet at start of line (not **bold**)
    if re.search(r"^\s*\*\s", text, re.MULTILINE):
        return False
    # Check for - list items at start of line
    if re.search(r"^\s*-\s+\w", text, re.MULTILINE):
        return False
    return True


def has_html_tags(text: str) -> bool:
    """True if text contains an HTML tag."""
    return bool(re.search(r"<[a-zA-Z][^>]*>", text))


def has_italics_html(text: str) -> bool:
    """True if text has <i> or <em> (open or close)."""
    return "<i>" in text or "<em>" in text or "</i>" in text or "</em>" in text


def has_italics_thesis_followed_by_text(text: str) -> bool:
    """True if there is at least one italic block (HTML) with non-empty content followed by non-empty text."""
    idx = text.find("<i>")
    if idx == -1:
        idx = text.find("<em>")
        if idx == -1:
            return False
    value = text[idx:]
    end_thesis = value.find("</i>")
    if end_thesis == -1:
        end_thesis = value.find("</em>")
        if end_thesis == -1:
            return False
    thesis = value[3:end_thesis] if value.startswith("<i>") else value[4:end_thesis]
    if not thesis.strip():
        return False
    close_len = 4 if value[end_thesis:].startswith("</i>") else 5
    after_block = value[end_thesis + close_len :]
    return after_block.strip() != ""


def score_each_word_on_new_line(text: str) -> float:
    """Ratio of lines to words (1.0 when each word is on its own line)."""
    value = text.translate(str.maketrans("", "", string.punctuation))
    lines = [ln for ln in value.strip().split("\n") if ln.strip()]
    words = value.strip().split()
    if not words:
        return 0.0
    return min(len(lines) / len(words), 1.0)


def each_word_on_new_line(text: str) -> float:
    """Score: ratio of lines to words."""
    return score_each_word_on_new_line(text)


def not_each_word_on_new_line(text: str) -> float:
    """Score: 1.0 - each_word_on_new_line score. Anti-correlated."""
    return 1.0 - score_each_word_on_new_line(text)

not_each_word_on_new_line.is_inverted = True  # type: ignore[attr-defined]


def normal_paragraphs(text: str) -> bool:
    """True if multiple paragraphs or single block with spaces."""
    return "\n\n" in text or (text.count("\n") < 2 and " " in text)


def no_quotes(text: str) -> bool:
    """True if text has no double quotes and no standalone single quotes (contractions allowed)."""
    if '"' in text:
        return False
    return not bool(re.search(r"(?<!\w)'(?!\w)|^'|'$", text))


def nested_quotes_3_levels(text: str) -> bool:
    """True if quotes nest to at least 3 levels, alternating \" and '."""
    levels = []
    min_levels = 3
    reached_depth = 0
    current_depth = 0
    for char in text:
        if len(levels) != 0 and char == levels[-1]:
            levels.pop()
            current_depth -= 1
            if reached_depth - current_depth >= min_levels:
                return True
        elif char == '"' or char == "'":
            levels.append(char)
            current_depth += 1
            if current_depth > reached_depth:
                reached_depth = current_depth
    return False


def nested_parentheses_5_levels(text: str) -> bool:
    """True if ()[]{} nest at least 5 levels deep with correct matching."""
    levels = []
    min_levels = 5
    max_depth = 0
    for char in text:
        if char in "([{":
            levels.append(char)
            if len(levels) > max_depth:
                max_depth = len(levels)
        elif char in ")]}":
            if levels and (
                (levels[-1] == "(" and char == ")")
                or (levels[-1] == "[" and char == "]")
                or (levels[-1] == "{" and char == "}")
            ):
                levels.pop()
                if max_depth >= min_levels and len(levels) < max_depth:
                    return True
            else:
                levels = []
                max_depth = 0
    return False


def no_nesting(text: str) -> bool:
    """True if text has no (, [, or {."""
    return "(" not in text and "[" not in text and "{" not in text


def score_indent_stairs(text: str) -> float:
    """Fraction of line transitions with strictly increasing indentation."""
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if len(lines) < 2:
        return 0.0
    transitions = len(lines) - 1
    good = 0
    for i in range(transitions):
        curr_indent = len(lines[i]) - len(lines[i].lstrip(" "))
        next_indent = len(lines[i + 1]) - len(lines[i + 1].lstrip(" "))
        if next_indent > curr_indent:
            good += 1
    return good / transitions


def indent_stairs(text: str) -> float:
    """Score: fraction of line transitions with increasing indent."""
    return score_indent_stairs(text)


def not_indent_stairs(text: str) -> float:
    """Score: 1.0 - indent_stairs score. Anti-correlated."""
    return 1.0 - score_indent_stairs(text)

not_indent_stairs.is_inverted = True  # type: ignore[attr-defined]


def no_whitespace(text: str) -> bool:
    """True if text has no whitespace."""
    return not any(c.isspace() for c in text)


def has_whitespace(text: str) -> bool:
    """True if text has space or newline."""
    return " " in text or "\n" in text


_TITLE_CASE_EXCEPTIONS = {
    "a", "an", "the", "and", "but", "or", "nor", "for", "so", "yet",
    "at", "by", "in", "of", "on", "to", "up", "as",
}


def score_title_case(text: str) -> float:
    """Fraction of major words correctly title-cased."""
    words = text.split()
    if not words:
        return 1.0
    checked = 0
    good = 0
    for i, w in enumerate(words):
        clean = w.strip(string.punctuation)
        if not clean or not clean[0].isalpha():
            continue
        if i > 0 and clean.lower() in _TITLE_CASE_EXCEPTIONS:
            continue
        checked += 1
        if len(clean) == 1:
            if clean.isupper():
                good += 1
        else:
            if clean[0].isupper() and clean[1:].islower():
                good += 1
    return good / checked if checked else 1.0


def is_title_case(text: str) -> float:
    """Score: fraction of major words correctly title-cased."""
    return score_title_case(text)


def score_sentence_case(text: str) -> float:
    """Fraction of sentences in correct sentence case (first word caps, rest lowercase)."""
    sentences = split_sentences(text)
    checked = 0
    good = 0
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        words = sent.split()
        if not words:
            continue
        checked += 1
        ok = True
        first_clean = words[0].strip(string.punctuation)
        if first_clean and first_clean[0].isalpha() and not first_clean[0].isupper():
            ok = False
        if ok:
            for w in words[1:]:
                clean = w.strip(string.punctuation)
                if not clean or not clean[0].isalpha():
                    continue
                if len(clean) > 1 and clean[0].isupper() and clean[1:].islower():
                    ok = False
                    break
        if ok:
            good += 1
    return good / checked if checked else 1.0


def is_sentence_case(text: str) -> float:
    """Score: fraction of sentences in correct sentence case."""
    return score_sentence_case(text)


def has_template_markers(text: str) -> bool:
    """True if text contains My Answer:, My Conclusion:, Future Outlook:."""
    return "My Answer:" in text and "My Conclusion:" in text and "Future Outlook:" in text


def no_template(text: str) -> bool:
    """True if text does not contain any of the template markers."""
    return not has_template_markers(text)


_FIRST_PERSON_RE = re.compile(r"\b(I|me|my|mine|myself)\b", re.IGNORECASE)
_THIRD_PERSON_RE = re.compile(
    r"\b(he|she|they|him|her|them|his|hers|their|theirs|"
    r"himself|herself|themselves|one's|oneself)\b",
    re.IGNORECASE,
)


def score_first_person(text: str) -> float:
    """Ratio of first-person to (first + third) person pronouns.

    Returns 0.0 when no pronouns of either type are found.
    """
    first = len(_FIRST_PERSON_RE.findall(text))
    third = len(_THIRD_PERSON_RE.findall(text))
    total = first + third
    if total == 0:
        return 0.0
    return first / total


def score_third_person(text: str) -> float:
    """Ratio of third-person to (first + third) person pronouns.

    Returns 0.0 when no pronouns of either type are found.
    """
    first = len(_FIRST_PERSON_RE.findall(text))
    third = len(_THIRD_PERSON_RE.findall(text))
    total = first + third
    if total == 0:
        return 0.0
    return third / total


def pronoun_count(text: str) -> int:
    """Count total pronoun occurrences (case-insensitive, word-boundary matched)."""
    pronouns = [
        "i", "me", "my", "mine", "myself",
        "we", "us", "our", "ours", "ourselves",
        "he", "him", "his", "himself",
        "she", "her", "hers", "herself",
        "they", "them", "their", "theirs", "themselves",
        "it", "its", "itself",
        "you", "your", "yours", "yourself", "yourselves",
    ]
    lower = text.lower()
    count = 0
    for p in pronouns:
        count += len(re.findall(r"\b" + re.escape(p) + r"\b", lower))
    return count


def word_in_text(word: str, text: str) -> bool:
    """True if word appears in text as a whole word (case-insensitive)."""
    return bool(re.search(r"\b" + re.escape(word.lower()) + r"\b", text.lower()))


def no_word_in_text(word: str, text: str) -> bool:
    """True if word does not appear in text as a whole word (case-insensitive)."""
    return not word_in_text(word, text)


def count_word_occurrences(word: str, text: str) -> int:
    """Count whole-word matches for word in text (case-insensitive)."""
    if not word.strip():
        return 0
    pattern = re.compile(r"\b" + re.escape(word.strip()) + r"\b", re.IGNORECASE)
    return len(pattern.findall(text))


AI_DISCLAIMER_ENDING = "Note: This is AI-generated content and should not be considered professional advice."


def ends_with_ai_disclaimer(text: str) -> bool:
    """True if text ends with the standard AI disclaimer."""
    return text.strip().endswith(AI_DISCLAIMER_ENDING)


def no_disclaimer_caveat(text: str) -> bool:
    """True if text has no disclaimer/caveat phrases."""
    lower = text.lower()
    return (
        "ai-generated" not in lower
        and "professional advice" not in lower
        and "disclaimer" not in lower
        and "caveat" not in lower
        and "should not be considered" not in lower
    )


def all_sentences_max_n_words(text: str, n: int) -> bool:
    """True if every sentence has at most n words."""
    for s in split_sentences(text):
        if count_words(s) > n:
            return False
    return True


def any_sentence_longer_than_n_words(text: str, n: int) -> bool:
    """True if some sentence has more than n words."""
    for s in split_sentences(text):
        if count_words(s) > n:
            return True
    return False


def all_sentences_min_n_words(text: str, n: int) -> bool:
    """True if there is at least one sentence and every sentence has strictly more than n words."""
    sents = split_sentences(text)
    if not sents:
        return False
    return all(count_words(s) > n for s in sents)


def response_repeated_twice(text: str) -> bool:
    """True if text looks like the same block repeated twice."""
    t = text.strip()
    if len(t) < 20:
        return False
    halves = re.split(r"\n\n+", t, maxsplit=1)
    if len(halves) == 2:
        first, second = halves[0].strip(), halves[1].strip()
    else:
        mid = len(t) // 2
        first, second = t[:mid].strip(), t[mid:].strip()

    def _norm(s: str) -> str:
        return " ".join(s.split())

    return _norm(first) == _norm(second)


def is_valid_json_object(text: str) -> bool:
    """True if text parses as a single JSON object."""
    t = text.strip()
    if not t or not t.startswith("{"):
        return False
    try:
        parsed = json.loads(t)
        return isinstance(parsed, dict)
    except (json.JSONDecodeError, ValueError):
        return False


def _detect_lang(segment: str) -> str | None:
    """Return ISO 639-1 code for segment or None if detection fails. Requires langdetect."""
    try:
        from langdetect import detect
        return detect(segment)
    except Exception:
        return None


# langdetect confuses closely related Romance languages on short texts
_SPANISH_FAMILY = {"es", "ca", "pt"}


def response_has_english_and_language(text: str, second_lang_code: str) -> bool:
    """True if text has at least one segment in English and one in second_lang_code (by paragraph)."""
    segments = [s.strip() for s in text.split("\n\n") if s.strip() and len(s.strip()) >= 20]
    if len(segments) < 2:
        return False
    # Build the set of acceptable codes for the target language
    if second_lang_code == "es":
        accept = _SPANISH_FAMILY
    else:
        accept = {second_lang_code}
    has_en = False
    has_target = False
    for seg in segments:
        code = _detect_lang(seg)
        if code == "en":
            has_en = True
        if code in accept:
            has_target = True
    return has_en and has_target


def response_english_only(text: str) -> bool:
    """True if all non-tiny segments detect as English. Requires langdetect."""
    segments = [s.strip() for s in text.split("\n\n") if s.strip() and len(s.strip()) >= 15]
    if not segments:
        return True
    for seg in segments:
        code = _detect_lang(seg)
        if code is not None and code != "en":
            return False
    return True


def make_keyword_in_nth_sentence_verifier(keyword: str, n: int):
    """Return a verifier that checks keyword appears as a whole word in the n-th sentence."""
    pattern = re.compile(r"\b" + re.escape(keyword) + r"\b", re.IGNORECASE)

    def _v(response: str) -> bool:
        sents = split_sentences(response)
        if len(sents) < n:
            return False
        return bool(pattern.search(sents[n - 1]))

    return _v


def _split_into_sentences(line: str) -> list[str]:
    """Split line into sentences using NLTK sent_tokenize."""
    return split_sentences(line)


def has_sentences_and_bullets(text: str) -> bool:
    """True if response includes at least two sentences followed by at least two lines that start with *."""
    lines = text.split("\n")
    in_sentences = True
    count_sentences = 0
    count_bullets = 0
    for line in lines:
        if line.strip().startswith("*"):
            in_sentences = False
            if count_sentences < 2:
                return False
            count_bullets += 1
        elif in_sentences:
            sents = _split_into_sentences(line.strip())
            count_sentences += len(sents)
        else:
            return False
    return count_bullets >= 2


def three_sentences_same_char_count(text: str) -> bool:
    """True if text has exactly 3 sentences with identical character counts."""
    sents = split_sentences(text)
    if len(sents) != 3:
        return False
    return len(sents[0]) == len(sents[1]) == len(sents[2])


def check_incrementing_word_count(text: str, increment: int) -> bool:
    """True if each sentence has exactly `increment` more words than the previous one."""
    sents = split_sentences(text)
    if len(sents) < 2:
        return False
    for i in range(len(sents) - 1):
        if count_words(sents[i + 1]) - count_words(sents[i]) != increment:
            return False
    return True


def score_alphabetical_word_start(text: str) -> float:
    """Fraction of consecutive word pairs where the second word advances to the next alphabet letter.

    Measures sequential progression (each word → next letter) rather than absolute
    positional alignment, so a single extra/missing word doesn't ruin the entire score.
    """
    words = [w.strip(string.punctuation) for w in text.split() if w.strip(string.punctuation).isalpha()]
    if len(words) < 2:
        return 0.0
    alphabet = string.ascii_lowercase
    pairs = len(words) - 1
    good = 0
    for i in range(pairs):
        curr_letter = alphabet.index(words[i][0].lower())
        next_letter = alphabet.index(words[i + 1][0].lower())
        if next_letter == (curr_letter + 1) % 26:
            good += 1
    return good / pairs


def check_alphabetical_word_start(text: str) -> float:
    """Score: fraction of consecutive pairs advancing to the next alphabet letter."""
    return score_alphabetical_word_start(text)


_VOWELS = set("aeiou")
_CONSONANTS = set("bcdfghjklmnpqrstvwxyz")


def check_consonant_clusters(text: str) -> bool:
    """True if every word contains at least one consonant cluster."""
    words = text.lower().strip().split()
    consonants = set("bcdfghjklmnpqrstvwxyz")
    for word in words:
        cluster = False
        for i in range(len(word) - 1):
            if word[i] in consonants and word[i + 1] in consonants:
                cluster = True
                break
        if not cluster:
            return False
    return True


def score_sentence_chaining(text: str) -> float:
    """Fraction of sentence transitions where last word equals first word of next."""
    sents = split_sentences(text)
    if len(sents) < 2:
        return 0.0
    transitions = len(sents) - 1
    punct_space = "".join(string.punctuation) + " "
    good = 0
    for i in range(transitions):
        last_words = sents[i].rstrip(punct_space).split()
        first_words = sents[i + 1].lstrip(punct_space).split()
        if last_words and first_words and last_words[-1].lower() == first_words[0].lower():
            good += 1
    return good / transitions


def check_sentence_chaining(text: str) -> float:
    """Score: fraction of sentence transitions that chain."""
    return score_sentence_chaining(text)


def check_no_consecutive_first_letter(text: str) -> float:
    """Score: 1.0 - alliteration score. Anti-correlated with check_all_alliteration."""
    return 1.0 - score_all_alliteration(text)

check_no_consecutive_first_letter.is_inverted = True  # type: ignore[attr-defined]


def _is_palindrome(word: str) -> bool:
    return word == word[::-1]


def check_palindromes(text: str, min_count: int = 10, min_length: int = 5) -> bool:
    """True if text contains at least min_count palindromic words (occurrences) of length ≥ min_length."""
    value = text.translate(str.maketrans("", "", string.punctuation))
    words = value.lower().split()
    palindromes = [w for w in words if len(w) >= min_length and _is_palindrome(w)]
    return len(palindromes) >= min_count


def score_paragraph_bookend(text: str) -> float:
    """Fraction of non-empty paragraphs that bookend (first word == last word)."""
    paragraphs = [p.strip().lower() for p in re.split(r"\n\n+", text) if p.strip()]
    if not paragraphs:
        return 0.0
    punct_space = "".join(string.punctuation) + " "
    good = 0
    for paragraph in paragraphs:
        words = paragraph.strip(punct_space).split()
        if not words:
            continue
        if words[0] == words[-1]:
            good += 1
    return good / len(paragraphs)


def check_paragraph_bookend(text: str) -> float:
    """Score: fraction of paragraphs that bookend."""
    return score_paragraph_bookend(text)


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n ** 0.5) + 1, 2):
        if n % i == 0:
            return False
    return True


def check_prime_length_words(text: str) -> bool:
    """True if every word has prime-number character length."""
    value = text.translate(str.maketrans("", "", string.punctuation))
    words = value.split()
    return all(_is_prime(len(w)) for w in words)


def score_max_word_repeat(text: str, max_repeats: int) -> float:
    """Fraction of unique words within the repeat limit."""
    words = [w.strip(string.punctuation).lower() for w in text.split() if w.strip(string.punctuation)]
    if not words:
        return 1.0
    counts = Counter(words)
    total = len(counts)
    good = sum(1 for c in counts.values() if c <= max_repeats)
    return good / total


def check_max_word_repeat(text: str, max_repeats: int) -> float:
    """Score: fraction of unique words within the repeat limit."""
    return score_max_word_repeat(text, max_repeats)


def check_one_vowel_type(text: str) -> bool:
    """True if every alphabetic word contains at most one distinct vowel letter."""
    words = [w.strip(string.punctuation).lower() for w in text.split() if w.strip(string.punctuation).isalpha()]
    if not words:
        return False
    for word in words:
        vowels_in_word = {ch for ch in word if ch in _VOWELS}
        if len(vowels_in_word) > 1:
            return False
    return True

def check_equal_sentence_word_count(text: str) -> bool:
    """True if text has ≥2 sentences and all sentences have the same word count."""
    sents = split_sentences(text)
    if len(sents) < 2:
        return False
    lengths = [count_words(s) for s in sents]
    return len(set(lengths)) == 1


def check_strictly_increasing_sentence_lengths(text: str) -> bool:
    """True if text has ≥3 sentences with strictly increasing character counts."""
    sents = split_sentences(text)
    if len(sents) < 3:
        return False
    for i in range(len(sents) - 1):
        if len(sents[i + 1]) <= len(sents[i]):
            return False
    return True


def score_all_alliteration(text: str) -> float:
    """Fraction of consecutive word pairs sharing the same first letter."""
    words = [w.strip(string.punctuation) for w in text.split() if w.strip(string.punctuation) and w.strip(string.punctuation)[0].isalpha()]
    if len(words) < 2:
        return 0.0
    pairs = len(words) - 1
    matches = sum(1 for i in range(pairs) if words[i][0].lower() == words[i + 1][0].lower())
    return matches / pairs


def check_all_alliteration(text: str) -> float:
    """Score: fraction of consecutive word pairs with alliteration."""
    return score_all_alliteration(text)


def check_no_consonant_clusters(text: str) -> bool:
    """True if no alphabetic word contains two consecutive consonants."""
    words = [w.strip(string.punctuation).lower() for w in text.split() if w.strip(string.punctuation).isalpha()]
    if not words:
        return False
    for word in words:
        prev_consonant = False
        for ch in word:
            if ch in _CONSONANTS:
                if prev_consonant:
                    return False
                prev_consonant = True
            else:
                prev_consonant = False
    return True


def check_no_sentence_chaining(text: str) -> float:
    """Score: 1.0 - chaining score. Anti-correlated with check_sentence_chaining."""
    return 1.0 - score_sentence_chaining(text)

check_no_sentence_chaining.is_inverted = True  # type: ignore[attr-defined]


def check_no_palindromes(text: str, min_length: int = 3) -> bool:
    """True if no alphabetic word of length ≥ min_length is a palindrome."""
    words = {w.strip(string.punctuation).lower() for w in text.split() if w.strip(string.punctuation).isalpha()}
    return not any(len(w) >= min_length and _is_palindrome(w) for w in words)


def check_no_paragraph_bookend(text: str) -> float:
    """Score: 1.0 - bookend score. Anti-correlated with check_paragraph_bookend."""
    return 1.0 - score_paragraph_bookend(text)

check_no_paragraph_bookend.is_inverted = True  # type: ignore[attr-defined]


def check_even_length_words(text: str) -> bool:
    """True if every alphabetic word has an even character length."""
    words = [w.strip(string.punctuation).lower() for w in text.split() if w.strip(string.punctuation).isalpha()]
    if not words:
        return False
    return all(len(w) % 2 == 0 for w in words)


def check_min_word_repeat(text: str, min_count: int) -> bool:
    """True if at least one word appears ≥ min_count times (case-insensitive)."""
    words = [w.strip(string.punctuation).lower() for w in text.split() if w.strip(string.punctuation)]
    if not words:
        return False
    counts = Counter(words)
    return any(c >= min_count for c in counts.values())


def check_multi_vowel_words(text: str) -> bool:
    """True if every alphabetic word contains at least two distinct vowel letters."""
    words = [w.strip(string.punctuation).lower() for w in text.split() if w.strip(string.punctuation).isalpha()]
    if not words:
        return False
    for word in words:
        if len({ch for ch in word if ch in _VOWELS}) < 2:
            return False
    return True


def check_under_word_count(text: str, under_n: int) -> bool:
    """True if text has fewer than under_n words."""
    return count_words(text) < under_n


def score_alternating_odd_even_syllables(text: str) -> float:
    """Fraction of consecutive word pairs with alternating odd/even syllable parity."""
    words = text.translate(str.maketrans("", "", string.punctuation)).lower().split()
    syllables = [syllapy.count(word) % 2 for word in words if word.strip()]
    if len(syllables) < 2:
        return 1.0  # vacuously true (no pairs to check)
    pairs = len(syllables) - 1
    good = sum(1 for i in range(pairs) if syllables[i] != syllables[i + 1])
    return good / pairs


def check_alternating_odd_even_syllables(text: str) -> float:
    """Score: fraction of consecutive pairs with alternating syllable parity."""
    return score_alternating_odd_even_syllables(text)


def check_not_alternating_odd_even_syllables(text: str) -> float:
    """Score: 1.0 - alternating score. Anti-correlated with check_alternating_odd_even_syllables."""
    return 1.0 - score_alternating_odd_even_syllables(text)

check_not_alternating_odd_even_syllables.is_inverted = True  # type: ignore[attr-defined]
