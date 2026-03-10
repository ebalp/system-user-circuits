"""Tests for language_en_zh conflict: contract, scoring, edge cases."""

import pytest
from phase0_v2.conflicts.registry import get_conflict
from phase0_v2.conflicts.definitions.language_en_zh import (
    _is_english,
    _is_chinese,
    LanguageEnZhConflict,
)


@pytest.fixture
def conflict():
    c = get_conflict("language_en_zh")
    assert c is not None
    return c


# -- Contract tests --


class TestContract:
    def test_registered(self, conflict):
        assert conflict.conflict_id == "language_en_zh"

    def test_counterbalance_quality(self, conflict):
        assert conflict.counterbalance_quality == "full"

    def test_arg_keys_empty(self, conflict):
        assert conflict.arg_keys == [] or conflict.get_instruction_args_keys() == []

    def test_sample_args_empty(self, conflict):
        assert conflict.sample_args() == {}

    def test_build_direction_a(self, conflict):
        sys_a = conflict.build_system_prompt(direction="a")
        usr_a = conflict.build_user_conflict_prompt(direction="a")
        assert sys_a and usr_a

    def test_build_direction_b(self, conflict):
        sys_b = conflict.build_system_prompt(direction="b")
        usr_b = conflict.build_user_conflict_prompt(direction="b")
        assert sys_b and usr_b

    def test_templates_no_placeholders(self, conflict):
        sys_a = conflict.build_system_prompt(direction="a")
        usr_a = conflict.build_user_conflict_prompt(direction="a")
        assert "{" not in sys_a
        assert "{" not in usr_a

    def test_verify_fns_are_bool(self, conflict):
        result = conflict.score_system("Hello world", direction="a")
        assert isinstance(result, float)
        assert result in (0.0, 1.0)

    def test_counterbalance_symmetry(self, conflict):
        """Direction a system = direction b user (and vice versa)."""
        assert conflict._verify_system_fn is conflict._verify_inverse_user_fn
        assert conflict._verify_user_fn is conflict._verify_inverse_system_fn


# -- Scorer tests --


class TestIsEnglish:
    def test_pure_english(self):
        assert _is_english("This is a simple English sentence about science.")

    def test_long_english(self):
        text = (
            "Machine learning is a subset of artificial intelligence that enables "
            "systems to learn and improve from experience without being explicitly "
            "programmed. It focuses on developing algorithms that can access data "
            "and use it to learn for themselves."
        )
        assert _is_english(text)

    def test_chinese_text_not_english(self):
        assert not _is_english("这是一个简单的中文句子。机器学习是人工智能的一个分支。")

    def test_traditional_chinese_not_english(self):
        assert not _is_english("這是一個簡單的中文句子。機器學習是人工智能的一個分支。")

    def test_empty_string(self):
        assert not _is_english("")

    def test_whitespace_only(self):
        assert not _is_english("   ")


class TestIsChinese:
    def test_simplified_chinese(self):
        assert _is_chinese("机器学习是人工智能的一个重要分支，它使计算机系统能够从数据中学习和改进。")

    def test_traditional_chinese(self):
        assert _is_chinese("機器學習是人工智能的一個重要分支，它使計算機系統能夠從數據中學習和改進。")

    def test_chinese_with_numbers_and_punctuation(self):
        text = "1. 光合作用是植物利用太阳能将二氧化碳和水转化为葡萄糖和氧气的过程。"
        assert _is_chinese(text)

    def test_pure_english_not_chinese(self):
        assert not _is_chinese("This is a simple English sentence about science.")

    def test_empty_string(self):
        assert not _is_chinese("")

    def test_whitespace_only(self):
        assert not _is_chinese("   ")

    def test_short_chinese(self):
        assert _is_chinese("你好世界")


# -- Edge cases --


class TestEdgeCases:
    def test_mutual_exclusion(self):
        """English and Chinese detectors should not both return True."""
        english_text = "Hello, this is a test sentence in English."
        chinese_text = "你好，这是一个中文测试句子。"

        assert _is_english(english_text) and not _is_chinese(english_text)
        assert _is_chinese(chinese_text) and not _is_english(chinese_text)

    def test_mixed_text_mostly_chinese(self):
        """Text with some English words embedded in Chinese should be detected as Chinese."""
        text = "机器学习（Machine Learning）是人工智能的一个分支，它使计算机能够从数据中学习。"
        assert _is_chinese(text)

    def test_verify_system_direction_a(self, ):
        """Direction a: system = English."""
        c = LanguageEnZhConflict()
        assert c.verify_followed_system("This is English text.", direction="a")
        assert not c.verify_followed_system("这是中文文本。", direction="a")

    def test_verify_user_direction_a(self):
        """Direction a: user = Chinese."""
        c = LanguageEnZhConflict()
        assert c.verify_followed_user("这是中文文本，关于科学的内容。", direction="a")
        assert not c.verify_followed_user("This is English text.", direction="a")

    def test_verify_system_direction_b(self):
        """Direction b: system = Chinese."""
        c = LanguageEnZhConflict()
        assert c.verify_followed_system("这是中文文本，关于科学的内容。", direction="b")
        assert not c.verify_followed_system("This is English text.", direction="b")

    def test_verify_user_direction_b(self):
        """Direction b: user = English."""
        c = LanguageEnZhConflict()
        assert c.verify_followed_user("This is English text.", direction="b")
        assert not c.verify_followed_user("这是中文文本。", direction="b")
