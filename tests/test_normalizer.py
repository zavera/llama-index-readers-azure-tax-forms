"""Unit tests for field normalisation — pure functions, no mocks needed."""
import pytest
from llama_index_readers_azure_tax_forms.normalizer import (
    normalize_key,
    normalize_value,
    normalize_pairs,
)


class TestNormalizeKey:
    def test_strips_trailing_space(self):
        assert normalize_key("Wages/Salary/Tips - HHA ") == "Wages/Salary/Tips - HHA"

    def test_fixes_known_typo_second_read(self):
        assert normalize_key("SeconD Read") == "Second Read"

    def test_fixes_known_typo_student_contribution(self):
        assert normalize_key("Student Conrtirbution") == "Student Contribution"

    def test_replaces_gt_separator(self):
        result = normalize_key("Based Year Academic Expenses> Tuition Paid - HHA")
        assert ">" not in result
        assert result == "Based Year Academic Expenses - Tuition Paid - HHA"

    def test_returns_none_for_blank(self):
        assert normalize_key("   ") is None

    def test_returns_none_for_none(self):
        assert normalize_key(None) is None

    def test_passthrough_for_clean_key(self):
        assert normalize_key("Adjusted Gross Income") == "Adjusted Gross Income"


class TestNormalizeValue:
    def test_strips_quotes_for_known_numeric_key(self):
        assert normalize_value('"75000"', "IM Original Need") == "75000"

    def test_does_not_strip_quotes_for_unknown_key(self):
        assert normalize_value('"some value"', "Unknown Key") == '"some value"'

    def test_strips_whitespace(self):
        assert normalize_value("  42000  ") == "42000"

    def test_returns_none_for_blank(self):
        assert normalize_value("") is None

    def test_returns_none_for_none(self):
        assert normalize_value(None) is None


class TestNormalizePairs:
    def test_drops_pairs_with_blank_key(self):
        pairs = [("  ", "value", 0.9), ("Good Key", "val", 0.8)]
        result = normalize_pairs(pairs)
        assert len(result) == 1
        assert result[0][0] == "Good Key"

    def test_applies_key_and_value_normalisation(self):
        pairs = [("SeconD Read", '"yes"', 0.95)]
        result = normalize_pairs(pairs)
        assert result[0][0] == "Second Read"
        # value is not in the quoted-numeric set → quotes preserved
        assert result[0][1] == '"yes"'

    def test_empty_input_returns_empty(self):
        assert normalize_pairs([]) == []
