"""Conflict registry: imports all conflict classes, provides lookup helpers."""

from .conflict_base import Conflict

# -- Batch 1: original Phase 0 conflicts --
from .definitions.language_en_es import LanguageEnEsConflict
from .definitions.language_en_zh import LanguageEnZhConflict
from .definitions.format_json_markdown import FormatJsonMarkdownConflict
from .definitions.format_json_yaml import FormatJsonYamlConflict
from .definitions.starting_word_hello_greetings import StartingWordHelloGreetingsConflict
from .definitions.emoji_use_vs_avoid import EmojiUseVsAvoidConflict
from .definitions.capitalization_all_caps import CapitalizationAllCapsConflict
from .definitions.list_bullets_vs_numbered import ListBulletsVsNumberedConflict
from .definitions.disclaimer_first_vs_none import DisclaimerFirstVsNoneConflict
from .definitions.self_reference_ai_mention import SelfReferenceAiMentionConflict

# -- Batch 1: dataset conflicts --
from .definitions.forbidden_words import ForbiddenWordsConflict
from .definitions.keyword_frequency import KeywordFrequencyConflict
from .definitions.short_vs_long_sentences import ShortVsLongSentencesConflict
from .definitions.json_only_vs_plain import JsonOnlyVsPlainConflict
from .definitions.spanish_loanwords import SpanishLoanwordsConflict

# -- Batch 2 --
from .definitions.number_density import NumberDensityConflict
from .definitions.vocabulary_diversity import VocabularyDiversityConflict
from .definitions.response_length import ResponseLengthConflict
from .definitions.each_word_new_line import EachWordNewLineConflict
from .definitions.bullets_and_sub_bullets import BulletsAndSubBulletsConflict
from .definitions.html_emphasis_tags import HtmlEmphasisTagsConflict

# -- Batch 3 --
from .definitions.alphabetical_sentences import AlphabeticalSentencesConflict
from .definitions.keyword_avoidance import KeywordAvoidanceConflict
from .definitions.word_repetition_density import WordRepetitionDensityConflict
from .definitions.alliteration_density import AlliterationDensityConflict
from .definitions.paragraph_end_same_word import ParagraphEndSameWordConflict
from .definitions.paragraph_start_word import ParagraphStartWordConflict
from .definitions.parenthetical_asides import ParentheticalAsidesConflict
from .definitions.past_vs_present_tense import PastVsPresentTenseConflict
from .definitions.pronoun_density import PronounDensityConflict
from .definitions.sentence_connector_density import SentenceConnectorDensityConflict
from .definitions.template_response import TemplateResponseConflict
from .definitions.lowercase_vs_capitalized import LowercaseVsCapitalizedConflict

# -- Batch 4 --
from .definitions.first_vs_third_person import FirstVsThirdPersonConflict
from .definitions.questions_vs_statements import QuestionsVsStatementsConflict

# -- Batch 5 --
from .definitions.imperative_vs_declarative import ImperativeVsDeclarativeConflict
from .definitions.address_reader_directly import AddressReaderDirectlyConflict
from .definitions.direct_answer_vs_hedging import DirectAnswerVsHedgingConflict
from .definitions.formal_vs_casual_tone import FormalVsCasualToneConflict
from .definitions.leetspeak_encoding import LeetspeakEncodingConflict
from .definitions.numbered_sections_vs_prose import NumberedSectionsVsProseConflict
from .definitions.short_paragraphs_vs_single_block import ShortParagraphsVsSingleBlockConflict
from .definitions.vowel_omission import VowelOmissionConflict

# Alphabetically sorted by class name
_ALL_CONFLICT_CLASSES: list[type[Conflict]] = [
    ImperativeVsDeclarativeConflict,
    AddressReaderDirectlyConflict,
    AlphabeticalSentencesConflict,
    SpanishLoanwordsConflict,
    BulletsAndSubBulletsConflict,
    CapitalizationAllCapsConflict,
    DirectAnswerVsHedgingConflict,
    DisclaimerFirstVsNoneConflict,
    EachWordNewLineConflict,
    EmojiUseVsAvoidConflict,
    NumberDensityConflict,
    FirstVsThirdPersonConflict,
    ForbiddenWordsConflict,
    FormalVsCasualToneConflict,
    FormatJsonMarkdownConflict,
    HtmlEmphasisTagsConflict,
    JsonOnlyVsPlainConflict,
    KeywordFrequencyConflict,
    KeywordAvoidanceConflict,
    LanguageEnEsConflict,
    LanguageEnZhConflict,
    LeetspeakEncodingConflict,
    ListBulletsVsNumberedConflict,
    ShortVsLongSentencesConflict,
    WordRepetitionDensityConflict,
    VocabularyDiversityConflict,
    AlliterationDensityConflict,
    NumberedSectionsVsProseConflict,
    ParagraphStartWordConflict,
    ParentheticalAsidesConflict,
    PastVsPresentTenseConflict,
    PronounDensityConflict,
    QuestionsVsStatementsConflict,
    SelfReferenceAiMentionConflict,
    SentenceConnectorDensityConflict,
    ShortParagraphsVsSingleBlockConflict,
    StartingWordHelloGreetingsConflict,
    TemplateResponseConflict,
    LowercaseVsCapitalizedConflict,
    ResponseLengthConflict,
    VowelOmissionConflict,
]

_REGISTRY: dict[str, Conflict] = {}
for _cls in _ALL_CONFLICT_CLASSES:
    _inst = _cls()
    _REGISTRY[_inst.conflict_id] = _inst


def get_conflict(conflict_id: str) -> Conflict | None:
    """Return conflict instance by id, or None."""
    return _REGISTRY.get(conflict_id)


def get_all_conflicts() -> list[Conflict]:
    """Return all registered conflict instances."""
    return list(_REGISTRY.values())


def get_conflict_ids() -> list[str]:
    """Return sorted list of all registered conflict IDs."""
    return sorted(_REGISTRY.keys())
