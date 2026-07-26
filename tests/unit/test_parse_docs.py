"""
Smoke-тесты для src/mhgb/parse_docs.py.
Тестируют только pure functions — без реальных docx-файлов.
"""

import pytest
from mhgb.parse_docs import parse_russian_date, parse_article_header


class TestParseRussianDate:
    def test_standard_format(self):
        assert parse_russian_date("ред. от 25 февраля 2024 г.") == "2024-02-25"

    def test_no_date_returns_none(self):
        assert parse_russian_date("Статья без даты редакции") is None

    def test_empty_string_returns_none(self):
        assert parse_russian_date("") is None

    def test_january(self):
        assert parse_russian_date("1 января 2024 г.") == "2024-01-01"

    def test_march(self):
        assert parse_russian_date("15 марта 2023 г.") == "2023-03-15"

    def test_december(self):
        assert parse_russian_date("31 декабря 2025 г.") == "2025-12-31"

    def test_may(self):
        assert parse_russian_date("5 мая 2022 г.") == "2022-05-05"

    def test_date_in_middle_of_text(self):
        result = parse_russian_date("Федеральный закон от 12 апреля 2021 г. № 99-ФЗ")
        assert result == "2021-04-12"

    def test_zero_padded_day(self):
        result = parse_russian_date("07 июля 2020 г.")
        assert result == "2020-07-07"

    def test_year_format(self):
        result = parse_russian_date("1 января 2000 г.")
        assert result == "2000-01-01"


class TestParseArticleHeader:
    def test_standard_format(self):
        result = parse_article_header("Статья 261. Гарантии беременной женщине")
        assert result == ("261", "Гарантии беременной женщине")

    def test_dotted_article_number(self):
        result = parse_article_header("Статья 261.1. Дополнительные гарантии при увольнении")
        assert result is not None
        assert result[0] == "261.1"
        assert "Дополнительные" in result[1]

    def test_article_without_title(self):
        result = parse_article_header("Статья 261")
        assert result is not None
        assert result[0] == "261"
        assert result[1] == ""

    def test_genitive_form_stati(self):
        # "Статьи" — родительный падеж, тоже должен распознаваться
        result = parse_article_header("Статьи 81. Расторжение трудового договора")
        assert result is not None

    def test_title_stripped(self):
        result = parse_article_header("Статья 100.  Пробелы в названии  ")
        assert result is not None
        assert result[1] == "Пробелы в названии"

    def test_non_article_returns_none(self):
        assert parse_article_header("Глава 1. Общие положения") is None

    def test_empty_string_returns_none(self):
        assert parse_article_header("") is None

    def test_random_text_returns_none(self):
        assert parse_article_header("Трудовой кодекс Российской Федерации") is None

    def test_article_with_period_in_title(self):
        result = parse_article_header("Статья 392. Сроки обращения в суд за разрешением индивидуального трудового спора")
        assert result is not None
        assert result[0] == "392"
