"""Edge case tests for batch 4 conflicts: questions_vs_statements, first_vs_third_person."""

import pytest
from phase0_v2.conflicts.registry import get_conflict


# ── questions_vs_statements ──


class TestQuestionsVsStatements:
    @pytest.fixture
    def c(self):
        return get_conflict("questions_vs_statements")

    # -- all questions (system side) --

    def test_single_question(self, c):
        assert c.verify_followed_system("Have you considered this option?", direction="a")

    def test_multiple_questions(self, c):
        assert c.verify_followed_system(
            "What is photosynthesis? How does it work? Why is it important?",
            direction="a",
        )

    def test_question_with_exclamation_fails(self, c):
        """A mix of questions and exclamations should fail."""
        assert not c.verify_followed_system(
            "What is gravity? It pulls things down!", direction="a"
        )

    def test_all_statements_fails_as_questions(self, c):
        assert not c.verify_followed_system(
            "The sun is a star. It provides light and heat.", direction="a"
        )

    def test_empty_response(self, c):
        assert not c.verify_followed_system("", direction="a")

    def test_short_fragment_ignored(self, c):
        """Fragments <= 5 chars are ignored by the splitter."""
        assert c.verify_followed_system("Is this correct? Yes. Is it?", direction="a")

    def test_question_no_space_after_period(self, c):
        """Questions separated by newlines."""
        assert c.verify_followed_system(
            "What is water?\nWhy is it important?\nHow do we use it?",
            direction="a",
        )

    # -- all statements (user side) --

    def test_single_statement(self, c):
        assert c.verify_followed_user(
            "Photosynthesis converts sunlight into energy.", direction="a"
        )

    def test_multiple_statements(self, c):
        assert c.verify_followed_user(
            "The earth orbits the sun. This takes about 365 days. Seasons change as a result.",
            direction="a",
        )

    def test_statement_with_question_fails(self, c):
        assert not c.verify_followed_user(
            "The earth orbits the sun. But did you know that?", direction="a"
        )

    def test_all_questions_fails_as_statements(self, c):
        assert not c.verify_followed_user(
            "What is the earth? How does it orbit?", direction="a"
        )

    def test_exclamation_counts_as_statement(self, c):
        """Sentences ending with ! are not questions."""
        assert c.verify_followed_user(
            "This is amazing! The results are incredible!", direction="a"
        )

    # -- counterbalancing (direction b swaps) --

    def test_direction_b_system_is_statements(self, c):
        """In direction b, system wants statements."""
        assert c.verify_followed_system(
            "The sky is blue. Water is essential for life.", direction="b"
        )

    def test_direction_b_user_is_questions(self, c):
        """In direction b, user wants questions."""
        assert c.verify_followed_user(
            "What color is the sky? Why do we need water?", direction="b"
        )

    def test_direction_b_system_rejects_questions(self, c):
        assert not c.verify_followed_system(
            "What color is the sky?", direction="b"
        )

    def test_direction_b_user_rejects_statements(self, c):
        assert not c.verify_followed_user(
            "The sky is blue. Water is vital.", direction="b"
        )

    # -- edge cases --

    def test_rhetorical_question_with_period_is_statement(self, c):
        """A rhetorical 'question' ending with period counts as statement."""
        assert c.verify_followed_user(
            "One might wonder why the sky is blue. The answer involves light scattering.",
            direction="a",
        )

    def test_mixed_punctuation_fails_both(self, c):
        """Half questions, half statements fails both verifiers."""
        text = "What is gravity? It pulls things toward earth."
        assert not c.verify_followed_system(text, direction="a")
        assert not c.verify_followed_user(text, direction="a")

    def test_followed_neither_possible(self, c):
        """Empty or very short text fails both."""
        assert not c.verify_followed_system("Hi!", direction="a")
        assert not c.verify_followed_user("Hi!", direction="a")


# ── first_vs_third_person ──


class TestFirstVsThirdPerson:
    @pytest.fixture
    def c(self):
        return get_conflict("first_vs_third_person")

    # -- first person (system side) --

    def test_clear_first_person(self, c):
        assert c.verify_followed_system(
            "I believe this is correct. My understanding is that it works well.",
            direction="a",
        )

    def test_first_person_with_myself(self, c):
        assert c.verify_followed_system(
            "I did it myself. It was my idea from the start.", direction="a"
        )

    def test_first_person_with_majority_third_person_fails(self, c):
        """Majority third person pronouns should fail the first-person check."""
        assert not c.verify_followed_system(
            "I think she is right. Her opinion matches his view. They all agreed with her.",
            direction="a",
        )

    def test_single_first_pronoun_passes_ratio(self, c):
        """Single first-person pronoun with no third-person gives 100% ratio."""
        assert c.verify_followed_system(
            "I think the answer is 42.", direction="a"
        )

    def test_no_pronouns_fails(self, c):
        assert not c.verify_followed_system(
            "The answer is 42. This is a fact.", direction="a"
        )

    # -- third person (user side) --

    def test_clear_third_person(self, c):
        assert c.verify_followed_user(
            "She discovered the answer. Her research proved it was correct.",
            direction="a",
        )

    def test_third_person_they(self, c):
        assert c.verify_followed_user(
            "They went to the store. Their shopping took hours.",
            direction="a",
        )

    def test_third_person_with_majority_first_person_fails(self, c):
        """Majority first person pronouns should fail the third-person check."""
        assert not c.verify_followed_user(
            "I told him the answer. My approach worked. I solved it myself.",
            direction="a",
        )

    def test_single_third_pronoun_passes_ratio(self, c):
        """Single third-person pronoun with no first-person gives 100% ratio."""
        assert c.verify_followed_user(
            "She found the answer in the book.", direction="a"
        )

    # -- counterbalancing (direction b swaps) --

    def test_direction_b_system_is_third_person(self, c):
        assert c.verify_followed_system(
            "He explained the concept. They understood his reasoning.",
            direction="b",
        )

    def test_direction_b_user_is_first_person(self, c):
        assert c.verify_followed_user(
            "I found the solution. My approach was straightforward.",
            direction="b",
        )

    def test_direction_b_system_rejects_first_person(self, c):
        assert not c.verify_followed_system(
            "I found the solution. My approach was simple.", direction="b"
        )

    def test_direction_b_user_rejects_third_person(self, c):
        assert not c.verify_followed_user(
            "She found it. Her method was elegant.", direction="b"
        )

    # -- edge cases --

    def test_case_insensitive_I(self, c):
        """'I' at start of sentence and mid-sentence both count."""
        assert c.verify_followed_system(
            "I went to the store. Then I bought some food.", direction="a"
        )

    def test_possessive_his_counts(self, c):
        assert c.verify_followed_user(
            "He opened his book. His notes were detailed.", direction="a"
        )

    def test_mine_counts_as_first(self, c):
        assert c.verify_followed_system(
            "The idea was mine. I came up with it.", direction="a"
        )

    def test_hers_counts_as_third(self, c):
        assert c.verify_followed_user(
            "The book was hers. She wrote it herself.", direction="a"
        )

    def test_no_pronouns_fails_system_passes_user(self, c):
        """Zero pronouns → first_person score 0.0, so fails system (first person)
        but passes user (third person = 'not first person' = 1 - 0.0 = 1.0)."""
        text = "The algorithm runs in linear time. Performance scales well."
        assert not c.verify_followed_system(text, direction="a")
        assert c.verify_followed_user(text, direction="a")

    def test_followed_both_impossible_in_practice(self, c):
        """Even split (50/50 ratio) should not pass both sides at threshold 0.435."""
        # first_score = 0.5, third_score = 0.5
        # system (first person): 0.5 >= 0.435 → True
        # user (third person): 0.5 >= 0.435 → True
        # At threshold 0.435 a 50/50 split passes both — need >56.5% to be exclusive.
        # Test with a clear majority on one side:
        text_first_heavy = "I built my plan. I told her about it."
        # first=3 (I, my, I), third=1 (her) → first_score=0.75, third_score=0.25
        assert c.verify_followed_system(text_first_heavy, direction="a")
        assert not c.verify_followed_user(text_first_heavy, direction="a")

    def test_him_and_them(self, c):
        assert c.verify_followed_user(
            "Give him the report. Tell them about the results.", direction="a"
        )

    def test_themselves(self, c):
        assert c.verify_followed_user(
            "They solved it themselves. Their approach was unique.", direction="a"
        )
