"""Conflict registry: imports all conflict classes, provides lookup helpers."""

from .conflict_base import Conflict

# -- Batch 1: original Phase 0 conflicts --
from .definitions.language_en_es import LanguageEnEsConflict
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
from .definitions.keyword_exact_count import KeywordExactCountConflict
from .definitions.keyword_frequency import KeywordFrequencyConflict
from .definitions.max_sentence_length import MaxSentenceLengthConflict
from .definitions.json_only_vs_plain import JsonOnlyVsPlainConflict
from .definitions.repeat_answer_twice import RepeatAnswerTwiceConflict
from .definitions.bilingual_english_plus import BilingualEnglishPlusConflict

# -- Batch 2 --
from .definitions.exact_number_count import ExactNumberCountConflict
from .definitions.min_unique_words import MinUniqueWordsConflict
from .definitions.word_count_range import WordCountRangeConflict
from .definitions.stairs_indent import StairsIndentConflict
from .definitions.each_word_new_line import EachWordNewLineConflict
from .definitions.bullets_and_sub_bullets import BulletsAndSubBulletsConflict
from .definitions.italics_thesis import ItalicsThesisConflict

# -- Batch 3 --
from .definitions.alphabetical_sentences import AlphabeticalSentencesConflict
from .definitions.keyword_in_early_sentence import KeywordInEarlySentenceConflict
from .definitions.max_word_repeat import MaxWordRepeatConflict
from .definitions.no_consecutive_first_letter import NoConsecutiveFirstLetterConflict
from .definitions.odd_even_syllables import OddEvenSyllablesConflict
from .definitions.paragraph_end_same_word import ParagraphEndSameWordConflict
from .definitions.paragraph_start_same_word import ParagraphStartSameWordConflict
from .definitions.parenthetical_asides import ParentheticalAsidesConflict
from .definitions.past_vs_present_tense import PastVsPresentTenseConflict
from .definitions.pronoun_density import PronounDensityConflict
from .definitions.sentence_chaining import SentenceChainingConflict
from .definitions.template_response import TemplateResponseConflict
from .definitions.title_case_vs_sentence_case import TitleCaseVsSentenceCaseConflict

# -- Batch 4 --
from .definitions.first_vs_third_person import FirstVsThirdPersonConflict
from .definitions.questions_vs_statements import QuestionsVsStatementsConflict

# -- Batch 5 --
from .definitions.active_vs_passive_voice import ActiveVsPassiveVoiceConflict
from .definitions.address_reader_directly import AddressReaderDirectlyConflict
from .definitions.direct_answer_vs_hedging import DirectAnswerVsHedgingConflict
from .definitions.formal_vs_casual_tone import FormalVsCasualToneConflict
from .definitions.numbered_sections_vs_prose import NumberedSectionsVsProseConflict
from .definitions.short_paragraphs_vs_single_block import ShortParagraphsVsSingleBlockConflict

# Alphabetically sorted by class name
_ALL_CONFLICT_CLASSES: list[type[Conflict]] = [
    ActiveVsPassiveVoiceConflict,
    AddressReaderDirectlyConflict,
    AlphabeticalSentencesConflict,
    BilingualEnglishPlusConflict,
    BulletsAndSubBulletsConflict,
    CapitalizationAllCapsConflict,
    DirectAnswerVsHedgingConflict,
    DisclaimerFirstVsNoneConflict,
    EachWordNewLineConflict,
    EmojiUseVsAvoidConflict,
    ExactNumberCountConflict,
    FirstVsThirdPersonConflict,
    ForbiddenWordsConflict,
    FormalVsCasualToneConflict,
    FormatJsonMarkdownConflict,
    ItalicsThesisConflict,
    JsonOnlyVsPlainConflict,
    KeywordExactCountConflict,
    KeywordFrequencyConflict,
    KeywordInEarlySentenceConflict,
    LanguageEnEsConflict,
    ListBulletsVsNumberedConflict,
    MaxSentenceLengthConflict,
    MaxWordRepeatConflict,
    MinUniqueWordsConflict,
    NoConsecutiveFirstLetterConflict,
    NumberedSectionsVsProseConflict,
    OddEvenSyllablesConflict,
    ParagraphStartSameWordConflict,
    ParentheticalAsidesConflict,
    PastVsPresentTenseConflict,
    PronounDensityConflict,
    QuestionsVsStatementsConflict,
    RepeatAnswerTwiceConflict,
    SelfReferenceAiMentionConflict,
    SentenceChainingConflict,
    ShortParagraphsVsSingleBlockConflict,
    StairsIndentConflict,
    StartingWordHelloGreetingsConflict,
    TemplateResponseConflict,
    TitleCaseVsSentenceCaseConflict,
    WordCountRangeConflict,
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
