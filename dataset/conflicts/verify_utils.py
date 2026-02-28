"""Verification helpers for system-user conflict checks."""

import json
import re
import string
from collections import Counter

import emoji
import syllapy


def count_words(text: str) -> int:
    """Return number of space-separated words."""
    return len(text.split())


def count_unique_words(text: str) -> int:
    """Return number of unique words (lowercased, punctuation stripped)."""
    return len(set(w.strip(string.punctuation).lower() for w in text.split() if w.strip()))


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
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
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
    """True if split on * has at least two parts and each part after the first contains -."""
    parts = text.split("*")
    if len(parts) < 2:
        return False
    for segment in parts[1:]:
        if "-" not in segment:
            return False
    return True


def no_bullets(text: str) -> bool:
    """True if no * bullets and no - list items."""
    return "*" not in text and not re.search(r"^\s*-\s+\w", text, re.MULTILINE)


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


def each_word_on_new_line(text: str) -> bool:
    """True if after stripping punctuation, number of non-empty lines equals number of words."""
    value = text.translate(str.maketrans("", "", string.punctuation))
    lines = value.strip().split("\n")
    while "" in lines:
        lines.remove("")
    return len(lines) == len(value.strip().split())


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


def indent_stairs(text: str) -> bool:
    """True if each non-empty line has strictly more leading spaces than the previous."""
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if len(lines) < 2:
        return False
    for i in range(len(lines) - 1):
        curr_indent = len(lines[i]) - len(lines[i].lstrip(" "))
        next_indent = len(lines[i + 1]) - len(lines[i + 1].lstrip(" "))
        if next_indent <= curr_indent:
            return False
    return True


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


def is_title_case(text: str) -> bool:
    """True if major words are capitalized (first letter upper, rest lower) per title case rules."""
    words = text.split()
    if not words:
        return True
    for i, w in enumerate(words):
        clean = w.strip(string.punctuation)
        if not clean or not clean[0].isalpha():
            continue
        if i > 0 and clean.lower() in _TITLE_CASE_EXCEPTIONS:
            continue
        if len(clean) == 1:
            if not clean.isupper():
                return False
        else:
            if not (clean[0].isupper() and clean[1:].islower()):
                return False
    return True


def is_sentence_case(text: str) -> bool:
    """True if each sentence starts with uppercase and mid-sentence words are not title-cased."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        words = sent.split()
        if not words:
            continue
        first_clean = words[0].strip(string.punctuation)
        if first_clean and first_clean[0].isalpha() and not first_clean[0].isupper():
            return False
        for w in words[1:]:
            clean = w.strip(string.punctuation)
            if not clean or not clean[0].isalpha():
                continue
            if len(clean) > 1 and clean[0].isupper() and clean[1:].islower():
                return False
    return True


def has_template_markers(text: str) -> bool:
    """True if text contains My Answer:, My Conclusion:, Future Outlook:."""
    return "My Answer:" in text and "My Conclusion:" in text and "Future Outlook:" in text


def no_template(text: str) -> bool:
    """True if text does not contain My Answer:."""
    return "My Answer:" not in text


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
    sents = re.split(r"[.!?]+", text)
    for s in sents:
        s = s.strip()
        if not s:
            continue
        if count_words(s) > n:
            return False
    return True


def any_sentence_longer_than_n_words(text: str, n: int) -> bool:
    """True if some sentence has more than n words."""
    sents = re.split(r"[.!?]+", text)
    for s in sents:
        s = s.strip()
        if not s:
            continue
        if count_words(s) > n:
            return True
    return False


def all_sentences_min_n_words(text: str, n: int) -> bool:
    """True if there is at least one sentence and every sentence has strictly more than n words."""
    sents = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
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


def response_has_english_and_language(text: str, second_lang_code: str) -> bool:
    """True if text has at least one segment in English and one in second_lang_code (by paragraph)."""
    segments = [s.strip() for s in text.split("\n\n") if s.strip() and len(s.strip()) >= 20]
    if len(segments) < 2:
        return False
    codes = set()
    for seg in segments:
        code = _detect_lang(seg)
        if code:
            codes.add(code)
    return "en" in codes and second_lang_code in codes


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
        sents = re.split(r"[.!?]+", response)
        sents = [s.strip() for s in sents if s.strip()]
        if len(sents) < n:
            return False
        return bool(pattern.search(sents[n - 1]))

    return _v


def _split_into_sentences(line: str) -> list[str]:
    """Split line into sentences by terminal punctuation (. ! ?), not just period."""
    return [s.strip() for s in re.split(r"[.!?]+", line) if s.strip()]


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
    sents = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    if len(sents) != 3:
        return False
    return len(sents[0]) == len(sents[1]) == len(sents[2])


def check_incrementing_word_count(text: str, increment: int) -> bool:
    """True if each sentence has exactly `increment` more words than the previous one."""
    sents = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    if len(sents) < 2:
        return False
    for i in range(len(sents) - 1):
        if count_words(sents[i + 1]) - count_words(sents[i]) != increment:
            return False
    return True


def check_alphabetical_word_start(text: str) -> bool:
    """True if each word starts with the next letter of the alphabet (A→Z cycling, case-insensitive)."""
    words = [w.strip(string.punctuation) for w in text.split() if w.strip(string.punctuation).isalpha()]
    if not words:
        return False
    alphabet = string.ascii_lowercase
    start_idx = alphabet.index(words[0][0].lower())
    for i, word in enumerate(words):
        expected = alphabet[(start_idx + i) % 26]
        if word[0].lower() != expected:
            return False
    return True


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


def check_sentence_chaining(text: str) -> bool:
    """True if the last word of each sentence equals the first word of the next."""
    sents = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    if len(sents) < 2:
        return False
    punct_space = "".join(string.punctuation) + " "
    for i in range(len(sents) - 1):
        last_words = sents[i].rstrip(punct_space).split()
        first_words = sents[i + 1].lstrip(punct_space).split()
        if not last_words or not first_words:
            return False
        if last_words[-1].lower() != first_words[0].lower():
            return False
    return True


def check_no_consecutive_first_letter(text: str) -> bool:
    """True if no two consecutive words share the same first letter."""
    words = text.lower().translate(str.maketrans("", "", string.punctuation)).split()
    while "" in words:
        words.remove("")
    if len(words) < 2:
        return True
    for i in range(len(words) - 1):
        if words[i] and words[i + 1] and words[i][0] == words[i + 1][0]:
            return False
    return True


def _is_palindrome(word: str) -> bool:
    return word == word[::-1]


def check_palindromes(text: str, min_count: int = 10, min_length: int = 5) -> bool:
    """True if text contains at least min_count palindromic words (occurrences) of length ≥ min_length."""
    value = text.translate(str.maketrans("", "", string.punctuation))
    words = value.lower().split()
    palindromes = [w for w in words if len(w) >= min_length and _is_palindrome(w)]
    return len(palindromes) >= min_count


def check_paragraph_bookend(text: str) -> bool:
    """True if each paragraph ends with the same word it started with."""
    paragraphs = text.split("\n")
    punct_space = "".join(string.punctuation) + " "
    for paragraph in paragraphs:
        paragraph = paragraph.strip().lower()
        if not paragraph:
            continue
        words = paragraph.strip(punct_space).split()
        if not words:
            continue
        if words[0] != words[-1]:
            return False
    return True


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


def check_max_word_repeat(text: str, max_repeats: int) -> bool:
    """True if no word appears more than max_repeats times (case-insensitive)."""
    words = [w.strip(string.punctuation).lower() for w in text.split() if w.strip(string.punctuation)]
    if not words:
        return True
    counts = Counter(words)
    return all(c <= max_repeats for c in counts.values())


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
    sents = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    if len(sents) < 2:
        return False
    lengths = [count_words(s) for s in sents]
    return len(set(lengths)) == 1


def check_strictly_increasing_sentence_lengths(text: str) -> bool:
    """True if text has ≥3 sentences with strictly increasing character counts."""
    sents = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    if len(sents) < 3:
        return False
    for i in range(len(sents) - 1):
        if len(sents[i + 1]) <= len(sents[i]):
            return False
    return True


def check_all_alliteration(text: str) -> bool:
    """True if every pair of consecutive words shares the same first letter (case-insensitive)."""
    words = [w.strip(string.punctuation) for w in text.split() if w.strip(string.punctuation) and w.strip(string.punctuation)[0].isalpha()]
    if len(words) < 2:
        return False
    for i in range(len(words) - 1):
        if words[i][0].lower() != words[i + 1][0].lower():
            return False
    return True


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


def check_no_sentence_chaining(text: str) -> bool:
    """True if text has ≥2 sentences and no consecutive sentences share last/first word."""
    sents = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    if len(sents) < 2:
        return False
    for i in range(len(sents) - 1):
        words_curr = [w.strip(string.punctuation) for w in sents[i].split() if w.strip(string.punctuation)]
        words_next = [w.strip(string.punctuation) for w in sents[i + 1].split() if w.strip(string.punctuation)]
        if not words_curr or not words_next:
            continue
        if words_curr[-1].lower() == words_next[0].lower():
            return False
    return True


def check_no_palindromes(text: str, min_length: int = 3) -> bool:
    """True if no alphabetic word of length ≥ min_length is a palindrome."""
    words = {w.strip(string.punctuation).lower() for w in text.split() if w.strip(string.punctuation).isalpha()}
    return not any(len(w) >= min_length and _is_palindrome(w) for w in words)


def check_no_paragraph_bookend(text: str) -> bool:
    """True if no paragraph ends with the same word it started with."""
    punct_space = "".join(string.punctuation) + " "
    for paragraph in text.split("\n"):
        paragraph = paragraph.strip().lower()
        if not paragraph:
            continue
        words = paragraph.strip(punct_space).split()
        if not words or len(words) < 2:
            continue
        if words[0] == words[-1]:
            return False
    return True


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


def check_alternating_odd_even_syllables(text: str) -> bool:
    """True if words alternate odd/even syllable parity."""
    words = text.translate(str.maketrans("", "", string.punctuation)).lower().split()
    syllables = [syllapy.count(word) % 2 for word in words if word.strip()]
    return all(syllables[i] != syllables[i + 1] for i in range(len(syllables) - 1))


def check_not_alternating_odd_even_syllables(text: str) -> bool:
    """True if words do NOT strictly alternate odd/even syllables (natural English; user constraint)."""
    return not check_alternating_odd_even_syllables(text)
