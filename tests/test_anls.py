"""ANLS scorer for the DocVQA benchmark (evals/docvqa/score.py)."""

import pytest

from evals.docvqa.score import anls_score, levenshtein, score_answers


def test_levenshtein_basics():
    assert levenshtein("", "") == 0
    assert levenshtein("abc", "abc") == 0
    assert levenshtein("kitten", "sitting") == 3
    assert levenshtein("abc", "") == 3


def test_exact_match_scores_one():
    assert anls_score("42 mg", ["42 mg"]) == 1.0
    assert anls_score("  42 MG ", ["42 mg"]) == 1.0  # case/whitespace insensitive


def test_near_match_above_threshold_keeps_similarity():
    # "monday" vs "mondays": distance 1 over max length 7 -> 1 - 1/7 ~= 0.857
    assert anls_score("monday", ["mondays"]) == pytest.approx(1 - 1 / 7)


def test_below_threshold_scores_zero():
    assert anls_score("completely wrong", ["42 mg"]) == 0.0


def test_max_over_multiple_references():
    assert anls_score("42 mg", ["forty-two", "42 mg"]) == 1.0


def test_score_answers_aggregates():
    answers = [
        {"prediction": "42 mg", "answers": ["42 mg"]},
        {"prediction": "wrong", "answers": ["right"]},
    ]
    result = score_answers(answers)
    assert result["questions"] == 2
    assert result["anls"] == 0.5
    assert result["exact_or_close"] == 1


def test_score_answers_empty():
    assert score_answers([]) == {"questions": 0, "anls": None, "exact_or_close": 0}
