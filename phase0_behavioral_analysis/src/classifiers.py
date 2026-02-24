"""
Response classifiers for constraint compliance detection.

This module provides classifiers for detecting language (English/Spanish/French/German),
format (JSON/YAML/plain text), starting word, emoji usage, capitalization style,
list format, disclaimer presence, and AI self-reference in model responses,
along with functions to compute compliance labels.
"""

from dataclasses import dataclass
from typing import Optional
import json
import re
import unicodedata

from langdetect import detect, detect_langs, LangDetectException
import yaml as pyyaml


# Confidence threshold for flagging low-confidence classifications
LOW_CONFIDENCE_THRESHOLD = 0.7


@dataclass
class ClassificationResult:
    """Result of classifying a model response.

    Attributes:
        detected: What was detected ('english', 'spanish', 'json', 'yaml', 'plain', 'other')
        confidence: Confidence score from 0.0 to 1.0
        details: Additional information (e.g., langdetect scores, parse errors)
    """
    detected: str
    confidence: float
    details: Optional[dict] = None

    @property
    def is_low_confidence(self) -> bool:
        """Check if this classification should be flagged for manual review."""
        return self.confidence < LOW_CONFIDENCE_THRESHOLD


class LanguageClassifier:
    """Classifier for detecting language in text responses.

    Uses langdetect with an expanded mapping that handles common
    confusion between closely related languages (e.g., Catalan/Spanish,
    Portuguese/Spanish on short texts).
    """

    LANG_MAP = {
        'en': 'english',
        'es': 'spanish',
        'fr': 'french',
        'de': 'german',
        'ca': 'spanish',   # Catalan ↔ Spanish confusion
        'pt': 'spanish',   # Portuguese ↔ Spanish confusion
        'gl': 'spanish',   # Galician ↔ Spanish confusion
        'it': 'spanish',   # Italian sometimes confused with Spanish
        'ro': 'spanish',   # Romanian sometimes confused with Spanish
    }

    def classify(self, text: str) -> ClassificationResult:
        """Detect language using langdetect library.

        Aggregates probabilities for languages that map to the same
        standard name (e.g., ca+es both count toward 'spanish').
        """
        if not text or not text.strip():
            return ClassificationResult(
                detected='other', confidence=0.0,
                details={'error': 'empty text'}
            )

        try:
            lang_probs = detect_langs(text)
            details = {str(lp.lang): lp.prob for lp in lang_probs}

            aggregated: dict[str, float] = {}
            for lp in lang_probs:
                mapped = self.LANG_MAP.get(lp.lang, 'other')
                aggregated[mapped] = aggregated.get(mapped, 0.0) + lp.prob

            best_lang = max(aggregated, key=aggregated.get)
            best_prob = aggregated[best_lang]

            return ClassificationResult(
                detected=best_lang, confidence=best_prob, details=details
            )

        except LangDetectException as e:
            return ClassificationResult(
                detected='other', confidence=0.0,
                details={'error': str(e)}
            )


class FormatClassifier:
    """Classifier for detecting JSON, YAML, or plain text format.

    Detection priority:
    1. Strip markdown code fences if present
    2. If valid JSON → 'json'
    3. If has YAML-specific syntax (---, block scalars, anchors, tags) → 'yaml'
    4. If valid YAML with key-value structure → 'yaml'
    5. If YAML-like patterns score > 0.5 → 'yaml'
    6. Otherwise → 'plain'
    """

    # Regex to strip markdown code fences: ```lang\n...\n```
    _CODE_FENCE_RE = re.compile(
        r'^\s*```\w*\s*\n(.*?)\n\s*```\s*$',
        re.DOTALL
    )

    def classify(self, text: str) -> ClassificationResult:
        """Detect if response is JSON, YAML, or plain text."""
        if not text or not text.strip():
            return ClassificationResult(
                detected='plain', confidence=1.0,
                details={'error': 'empty text'}
            )

        text = text.strip()

        # Strip markdown code fences if present
        stripped = self._strip_code_fences(text)

        # First check if it's JSON (more specific)
        if self._is_json(stripped):
            return ClassificationResult(
                detected='json', confidence=1.0,
                details={'format': 'json', 'parsed': True}
            )

        # Check for YAML-specific syntax (even if parsing fails)
        if self._has_yaml_syntax(stripped):
            return ClassificationResult(
                detected='yaml', confidence=1.0,
                details={'format': 'yaml', 'parsed': False, 'syntax_match': True}
            )

        # Try to parse as YAML
        yaml_result = self._try_parse_yaml(stripped)
        if yaml_result is not None and isinstance(yaml_result, (dict, list)):
            return ClassificationResult(
                detected='yaml', confidence=1.0,
                details={'format': 'yaml', 'parsed': True, 'type': type(yaml_result).__name__}
            )

        # Check for YAML-like patterns
        yaml_confidence = self._check_yaml_patterns(stripped)
        if yaml_confidence > 0.5:
            return ClassificationResult(
                detected='yaml', confidence=yaml_confidence,
                details={'format': 'yaml', 'parsed': False, 'pattern_match': True}
            )

        return ClassificationResult(
            detected='plain', confidence=1.0 - yaml_confidence,
            details={'format': 'plain'}
        )

    def _strip_code_fences(self, text: str) -> str:
        """Strip markdown code fences (```lang ... ```) if they wrap the entire response."""
        m = self._CODE_FENCE_RE.match(text)
        if m:
            return m.group(1).strip()
        return text

    def _is_json(self, text: str) -> bool:
        """Check if text is valid JSON."""
        try:
            json.loads(text)
            return True
        except json.JSONDecodeError:
            # Try to extract JSON from text (model might have added explanation)
            json_match = re.search(r'(\{[^{}]*\}|\[[^\[\]]*\])', text, re.DOTALL)
            if json_match:
                try:
                    json.loads(json_match.group(1))
                    return True
                except json.JSONDecodeError:
                    pass
            return False

    def _try_parse_yaml(self, text: str) -> Optional[any]:
        """Attempt to parse text as YAML."""
        try:
            return pyyaml.safe_load(text)
        except pyyaml.YAMLError:
            return None

    def _has_yaml_syntax(self, text: str) -> bool:
        """Check for YAML-specific syntax (not valid in JSON)."""
        yaml_indicators = [
            r'^---',                    # Document start
            r'^\.\.\.',                 # Document end
            r':\s*\|',                  # Literal block scalar
            r':\s*>',                   # Folded block scalar
            r'^\s*-\s+\w+:',            # List of mappings
            r'^\s*-\s+\w+\s*$',         # Simple list item (- item)
            r'&\w+',                    # Anchor
            r'\*\w+',                   # Alias
            r'!\w+',                    # Tag (e.g., !timestamp)
        ]
        for pattern in yaml_indicators:
            if re.search(pattern, text, re.MULTILINE):
                return True
        return False

    def _check_yaml_patterns(self, text: str) -> float:
        """Check for YAML-like patterns and return confidence score."""
        patterns = [
            (r'^\s*\w+:', 0.3),         # Key: at start of line
            (r':\s*$', 0.2),            # Trailing colon (nested structure)
            (r'^\s*-\s+', 0.3),         # List item
            (r':\s+\w+', 0.2),          # Key: value
        ]
        score = 0.0
        for pattern, weight in patterns:
            if re.search(pattern, text, re.MULTILINE):
                score += weight
        return min(score, 1.0)


class StartingWordClassifier:
    """Classifier for detecting the first word of a response.

    Simple and deterministic - extracts the first word and normalizes it
    (lowercase, strip punctuation). High confidence since it's exact matching.
    """

    # Characters to strip from the beginning/end of the first word
    STRIP_CHARS = '.,!?:;"\'`*#'

    def classify(self, text: str) -> ClassificationResult:
        """Extract and normalize the first word of the response."""
        if not text or not text.strip():
            return ClassificationResult(
                detected='other', confidence=0.0,
                details={'error': 'empty text'}
            )

        text = text.strip()

        # Skip markdown headers (# Hello -> Hello)
        if text.startswith('#'):
            text = text.lstrip('#').strip()

        # Skip bold/italic markers
        text = text.lstrip('*_')

        if not text:
            return ClassificationResult(
                detected='other', confidence=0.0,
                details={'error': 'only formatting characters'}
            )

        # Get first word
        first_word = text.split()[0]

        # Normalize: lowercase and strip punctuation
        normalized = first_word.lower().strip(self.STRIP_CHARS)

        if not normalized:
            return ClassificationResult(
                detected='other', confidence=0.0,
                details={'error': 'first word is only punctuation', 'raw': first_word}
            )

        return ClassificationResult(
            detected=normalized,
            confidence=1.0,
            details={'raw_first_word': first_word, 'normalized': normalized}
        )



class EmojiClassifier:
    """Classifier for detecting emoji presence in text responses.

    Counts Unicode emoji codepoints. If any are present → 'use_emojis',
    otherwise → 'no_emojis'.
    """

    # Regex matching common emoji Unicode ranges
    _EMOJI_RE = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002702-\U000027B0"  # dingbats
        "\U0000FE00-\U0000FE0F"  # variation selectors
        "\U0001F900-\U0001F9FF"  # supplemental symbols
        "\U0001FA00-\U0001FA6F"  # chess symbols
        "\U0001FA70-\U0001FAFF"  # symbols extended-A
        "\U00002600-\U000026FF"  # misc symbols
        "\U0000200D"             # zero width joiner
        "\U00002B50"             # star
        "\U0000231A-\U0000231B"  # watch, hourglass
        "\U000023E9-\U000023F3"  # media controls
        "\U000023F8-\U000023FA"  # more media
        "\U000025AA-\U000025AB"  # small squares
        "\U000025B6"             # play button
        "\U000025C0"             # reverse button
        "\U000025FB-\U000025FE"  # medium squares
        "\U00003030"             # wavy dash
        "\U0000303D"             # part alternation mark
        "\U00003297"             # circled ideograph congratulation
        "\U00003299"             # circled ideograph secret
        "]+",
        flags=re.UNICODE,
    )

    def classify(self, text: str) -> ClassificationResult:
        """Detect whether the response contains emojis."""
        if not text or not text.strip():
            return ClassificationResult(
                detected='no_emojis', confidence=1.0,
                details={'emoji_count': 0, 'note': 'empty text'}
            )

        emojis = self._EMOJI_RE.findall(text)
        emoji_count = sum(len(e) for e in emojis)

        if emoji_count > 0:
            return ClassificationResult(
                detected='use_emojis', confidence=1.0,
                details={'emoji_count': emoji_count}
            )
        return ClassificationResult(
            detected='no_emojis', confidence=1.0,
            details={'emoji_count': 0}
        )


class CapitalizationClassifier:
    """Classifier for detecting capitalization style.

    Computes the ratio of uppercase to total alphabetic characters.
    Ratio > 0.8 → 'all_caps', else → 'normal'.
    """

    CAPS_THRESHOLD = 0.8

    def classify(self, text: str) -> ClassificationResult:
        """Detect capitalization style of the response."""
        if not text or not text.strip():
            return ClassificationResult(
                detected='normal', confidence=1.0,
                details={'upper_ratio': 0.0, 'note': 'empty text'}
            )

        alpha_chars = [c for c in text if c.isalpha()]
        if not alpha_chars:
            return ClassificationResult(
                detected='normal', confidence=1.0,
                details={'upper_ratio': 0.0, 'note': 'no alpha chars'}
            )

        upper_count = sum(1 for c in alpha_chars if c.isupper())
        ratio = upper_count / len(alpha_chars)

        if ratio > self.CAPS_THRESHOLD:
            return ClassificationResult(
                detected='all_caps', confidence=1.0,
                details={'upper_ratio': ratio}
            )
        return ClassificationResult(
            detected='normal', confidence=1.0,
            details={'upper_ratio': ratio}
        )


class ListFormatClassifier:
    """Classifier for detecting bullet vs numbered list format.

    Counts lines matching bullet patterns (- or *) vs numbered patterns
    (1. or 2) etc.). Whichever has more matches wins.
    """

    _BULLET_RE = re.compile(r'^\s*[-*]\s', re.MULTILINE)
    _NUMBERED_RE = re.compile(r'^\s*\d+[.)]\s', re.MULTILINE)

    def classify(self, text: str) -> ClassificationResult:
        """Detect whether the response uses bullet or numbered list format."""
        if not text or not text.strip():
            return ClassificationResult(
                detected='other', confidence=0.0,
                details={'bullet_count': 0, 'numbered_count': 0, 'note': 'empty text'}
            )

        bullet_count = len(self._BULLET_RE.findall(text))
        numbered_count = len(self._NUMBERED_RE.findall(text))

        details = {'bullet_count': bullet_count, 'numbered_count': numbered_count}

        if bullet_count == 0 and numbered_count == 0:
            return ClassificationResult(
                detected='other', confidence=0.0, details=details
            )

        if bullet_count > numbered_count:
            return ClassificationResult(
                detected='bullets', confidence=1.0, details=details
            )
        elif numbered_count > bullet_count:
            return ClassificationResult(
                detected='numbered', confidence=1.0, details=details
            )
        else:
            # Tie — low confidence
            return ClassificationResult(
                detected='bullets', confidence=0.5, details=details
            )


class DisclaimerClassifier:
    """Classifier for detecting disclaimer presence.

    Checks for the exact phrase 'consult a professional' (case-insensitive).
    """

    _DISCLAIMER_PHRASES = [
        'consult a professional',
    ]

    def classify(self, text: str) -> ClassificationResult:
        """Detect whether the response contains a disclaimer."""
        if not text or not text.strip():
            return ClassificationResult(
                detected='no_disclaimer', confidence=1.0,
                details={'note': 'empty text'}
            )

        text_lower = text.lower()
        for phrase in self._DISCLAIMER_PHRASES:
            if phrase in text_lower:
                return ClassificationResult(
                    detected='add_disclaimer', confidence=1.0,
                    details={'matched_phrase': phrase}
                )

        return ClassificationResult(
            detected='no_disclaimer', confidence=1.0,
            details={'matched_phrase': None}
        )


class SelfReferenceClassifier:
    """Classifier for detecting AI self-reference in responses.

    Checks for phrases like 'as an AI', 'I'm an AI', 'language model',
    'AI assistant' (case-insensitive).
    """

    _SELF_REF_PHRASES = [
        'as an ai',
        "i'm an ai",
        'i am an ai',
        'language model',
        'ai assistant',
    ]

    def classify(self, text: str) -> ClassificationResult:
        """Detect whether the response contains AI self-reference."""
        if not text or not text.strip():
            return ClassificationResult(
                detected='no_ai_mention', confidence=1.0,
                details={'note': 'empty text'}
            )

        text_lower = text.lower()
        for phrase in self._SELF_REF_PHRASES:
            if phrase in text_lower:
                return ClassificationResult(
                    detected='acknowledge_ai', confidence=1.0,
                    details={'matched_phrase': phrase}
                )

        return ClassificationResult(
            detected='no_ai_mention', confidence=1.0,
            details={'matched_phrase': None}
        )


def get_classifier(constraint_type: str):
    """Factory function to get appropriate classifier for a constraint type.

    Both 'format' and 'yaml' map to FormatClassifier, which handles
    JSON, YAML, and plain text detection.
    """
    classifiers = {
        'language': LanguageClassifier,
        'format': FormatClassifier,
        'yaml': FormatClassifier,
        'starting_word': StartingWordClassifier,
        'emoji': EmojiClassifier,
        'capitalization': CapitalizationClassifier,
        'list_format': ListFormatClassifier,
        'disclaimer': DisclaimerClassifier,
        'self_reference': SelfReferenceClassifier,
    }

    if constraint_type not in classifiers:
        raise ValueError(f"Unknown constraint type: {constraint_type}. "
                        f"Valid types: {list(classifiers.keys())}")

    return classifiers[constraint_type]()


def compute_label(
    classification: ClassificationResult,
    system_constraint: str,
    user_constraint: Optional[str]
) -> tuple[str, float]:
    """Compute compliance label from classification result.

    Determines whether the model followed the system instruction,
    user instruction, both, or neither based on the classification.

    Returns:
        Tuple of (label, confidence) where label is one of:
        - 'followed_system', 'followed_user', 'followed_both', 'followed_neither'
    """
    detected = classification.detected
    confidence = classification.confidence

    system_constraint = system_constraint.lower() if system_constraint else None
    user_constraint = user_constraint.lower() if user_constraint else None

    matches_system = detected == system_constraint if system_constraint else False
    matches_user = detected == user_constraint if user_constraint else False

    # No user constraint (Condition A - system only)
    if user_constraint is None:
        if matches_system:
            return ('followed_system', confidence)
        else:
            return ('followed_neither', confidence)

    # No system constraint (Condition B - user only)
    if system_constraint is None:
        if matches_user:
            return ('followed_user', confidence)
        else:
            return ('followed_neither', confidence)

    # Same constraints
    if system_constraint == user_constraint:
        if matches_system:
            return ('followed_both', confidence)
        else:
            return ('followed_neither', confidence)

    # Conflicting constraints (Condition C or D)
    if matches_system and not matches_user:
        return ('followed_system', confidence)
    elif matches_user and not matches_system:
        return ('followed_user', confidence)
    elif matches_system and matches_user:
        return ('followed_both', confidence)
    else:
        return ('followed_neither', confidence)
