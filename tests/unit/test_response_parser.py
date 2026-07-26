"""
Тесты для src/mhgb/eval/response_parser.py.
Проверяют:
  - parse_structured_answer: извлечение 3 секций
  - extract_norms_from_text: нормализация упоминаний норм → ТК_261 формат
"""

import pytest
from mhgb.eval.response_parser import extract_norms_from_text, parse_structured_answer


FULL_ANSWER = (
    "ПРИМЕНИМЫЕ СТАТЬИ: ст. 261 ТК РФ, ст. 256 ТК РФ\n"
    "ЦЕПОЧКА РАССУЖДЕНИЙ: Шаг 1 — норма X запрещает. Шаг 2 — норма Y уточняет.\n"
    "ВЫВОД: Увольнение незаконно."
)


class TestParseStructuredAnswer:
    def test_extracts_all_three_sections(self):
        result = parse_structured_answer(FULL_ANSWER)
        assert result["applicable_articles"] == "ст. 261 ТК РФ, ст. 256 ТК РФ"
        assert "Шаг 1" in result["reasoning"]
        assert result["conclusion"] == "Увольнение незаконно."

    def test_missing_section_returns_empty_string(self):
        text = "ПРИМЕНИМЫЕ СТАТЬИ: ст. 261 ТК РФ\nВЫВОД: незаконно"
        result = parse_structured_answer(text)
        assert result["applicable_articles"] == "ст. 261 ТК РФ"
        assert result["reasoning"] == ""
        assert result["conclusion"] == "незаконно"

    def test_all_sections_missing_returns_empty_strings(self):
        result = parse_structured_answer("Просто текст без секций.")
        assert result["applicable_articles"] == ""
        assert result["reasoning"] == ""
        assert result["conclusion"] == ""

    def test_empty_string_returns_empty(self):
        result = parse_structured_answer("")
        assert result == {"applicable_articles": "", "reasoning": "", "conclusion": ""}

    def test_sections_in_different_order(self):
        text = (
            "ВЫВОД: законно\n"
            "ПРИМЕНИМЫЕ СТАТЬИ: ст. 81 ТК РФ\n"
            "ЦЕПОЧКА РАССУЖДЕНИЙ: норма позволяет"
        )
        result = parse_structured_answer(text)
        assert result["conclusion"] == "законно"
        assert result["applicable_articles"] == "ст. 81 ТК РФ"
        assert result["reasoning"] == "норма позволяет"

    def test_case_insensitive_headers(self):
        text = (
            "Применимые Статьи: ст. 1 ГК РФ\n"
            "Цепочка Рассуждений: применяется\n"
            "Вывод: законно"
        )
        result = parse_structured_answer(text)
        assert result["applicable_articles"] == "ст. 1 ГК РФ"
        assert result["reasoning"] == "применяется"
        assert result["conclusion"] == "законно"

    def test_extra_whitespace_in_header(self):
        text = "ПРИМЕНИМЫЕ  СТАТЬИ:  ст. 261 ТК\nВЫВОД: незаконно"
        result = parse_structured_answer(text)
        assert result["applicable_articles"] == "ст. 261 ТК"

    def test_multiline_reasoning_preserved(self):
        text = (
            "ПРИМЕНИМЫЕ СТАТЬИ: ст. 1 ГК\n"
            "ЦЕПОЧКА РАССУЖДЕНИЙ: Шаг 1.\nШаг 2.\nШаг 3.\n"
            "ВЫВОД: итог"
        )
        result = parse_structured_answer(text)
        assert "Шаг 1" in result["reasoning"]
        assert "Шаг 2" in result["reasoning"]
        assert "Шаг 3" in result["reasoning"]

    def test_conclusion_only(self):
        text = "ВЫВОД: нарушение выявлено"
        result = parse_structured_answer(text)
        assert result["conclusion"] == "нарушение выявлено"
        assert result["applicable_articles"] == ""
        assert result["reasoning"] == ""

    def test_content_stripped(self):
        text = "ПРИМЕНИМЫЕ СТАТЬИ:   ст. 1 ГК   \nВЫВОД: ок"
        result = parse_structured_answer(text)
        assert result["applicable_articles"] == "ст. 1 ГК"

    def test_returns_typed_dict_keys(self):
        result = parse_structured_answer(FULL_ANSWER)
        assert set(result.keys()) == {"applicable_articles", "reasoning", "conclusion"}

    def test_full_answer_no_extra_content_in_articles(self):
        result = parse_structured_answer(FULL_ANSWER)
        assert "ЦЕПОЧКА" not in result["applicable_articles"]
        assert "ВЫВОД" not in result["applicable_articles"]

    def test_reasoning_does_not_include_conclusion(self):
        result = parse_structured_answer(FULL_ANSWER)
        assert "ВЫВОД" not in result["reasoning"]


# ---------------------------------------------------------------------------
# TestExtractNormsFromText
# ---------------------------------------------------------------------------

class TestExtractNormsFromText:
    """Тесты извлечения и нормализации упоминаний норм из произвольного текста."""

    def _corpus(self) -> dict:
        return {
            "ТК_261":    {"id": "ТК_261"},
            "ТК_256":    {"id": "ТК_256"},
            "ТК_81":     {"id": "ТК_81"},
            "ТК_392":    {"id": "ТК_392"},
            "ГК_1064":   {"id": "ГК_1064"},
            "ГК_15":     {"id": "ГК_15"},
            "КоАП_12.1": {"id": "КоАП_12.1"},
            "КРФ_37":    {"id": "КРФ_37"},
            "УК_105":    {"id": "УК_105"},
            "НК_75":     {"id": "НК_75"},
        }

    # --- базовые форматы ---

    def test_short_form_with_rf(self):
        result = extract_norms_from_text("ст. 261 ТК РФ запрещает увольнение", self._corpus())
        assert result == {"ТК_261"}

    def test_short_form_without_rf(self):
        result = extract_norms_from_text("применяется ст. 261 ТК", self._corpus())
        assert result == {"ТК_261"}

    def test_no_space_after_dot(self):
        result = extract_norms_from_text("ст.261 ТК РФ", self._corpus())
        assert result == {"ТК_261"}

    # --- дефис-суффикс статей (25.13-1) и диапазоны (254-269) ---

    def _corpus_dash(self) -> dict:
        # corpus сворачивает дефис-статьи в базовые (D5.1): НК_25.13-1 → НК_25.13
        return {
            "НК_25.13":   {"id": "НК_25.13"},
            "НК_105.16":  {"id": "НК_105.16"},
            "НК_1":       {"id": "НК_1"},      # реальная ст.1 НК — не должна ложно всплывать
            "НК_254":     {"id": "НК_254"},
            "НК_269":     {"id": "НК_269"},
            "НК_346.37":  {"id": "НК_346.37"},
            "НК_346.38":  {"id": "НК_346.38"},
            "ЖК_44":      {"id": "ЖК_44"},
            "ЖК_48":      {"id": "ЖК_48"},
        }

    def test_dash_suffix_maps_to_base_norm(self):
        # «25.13-1» — единая статья, corpus хранит базовую НК_25.13
        result = extract_norms_from_text("ст. 25.13-1 НК РФ", self._corpus_dash())
        assert result == {"НК_25.13"}

    def test_dash_suffix_no_phantom_norm_1(self):
        # КРИТИЧНО: суффикс -1 НЕ должен порождать фантом НК_1
        result = extract_norms_from_text("ст. 25.13-1 НК РФ", self._corpus_dash())
        assert "НК_1" not in result

    def test_dash_suffix_various(self):
        result = extract_norms_from_text("ст. 105.16-1 НК РФ", self._corpus_dash())
        assert result == {"НК_105.16"}
        assert "НК_1" not in result

    def test_range_of_integers_preserved(self):
        # диапазон «254-269» → оба конца (не суффикс)
        result = extract_norms_from_text("ст. 254-269 НК РФ", self._corpus_dash())
        assert result == {"НК_254", "НК_269"}

    def test_range_short_articles(self):
        result = extract_norms_from_text("ст. 44-48 ЖК РФ", self._corpus_dash())
        assert result == {"ЖК_44", "ЖК_48"}

    def test_range_of_fractional_articles(self):
        # дробный диапазон 346.37-346.38 → концы (НЕ суффикс: right тоже дробный)
        result = extract_norms_from_text("ст. 346.37-346.38 НК РФ", self._corpus_dash())
        assert result == {"НК_346.37", "НК_346.38"}

    def test_dates_not_parsed_as_norms(self):
        # даты вне NORM_PATTERN (не после «ст.») → НЕ нормы, фикс их не втягивает
        c = self._corpus_dash()
        assert extract_norms_from_text("срок с 2004-06-29 по 2005-01-01", c) == set()
        assert extract_norms_from_text("в 1994-2005 годах", c) == set()
        # дата рядом с реальной нормой — норма извлечена, дата проигнорирована
        r = extract_norms_from_text("ст. 25.13 НК РФ (ред. 2020-11-09)", c)
        assert r == {"НК_25.13"}

    def test_mixed_suffix_range_date_one_text(self):
        # пограничный: все три случая в одной строке
        c = self._corpus_dash()
        text = "ст. 25.13-1 НК РФ и ст. 254-269 НК РФ, ред. 2020-11-09"
        result = extract_norms_from_text(text, c)
        assert result == {"НК_25.13", "НК_254", "НК_269"}  # суффикс→база, диапазон→концы
        assert "НК_1" not in result                          # фантом не появился
        # дата 2020-11-09 не породила норм (2020, 11, 09 не после «ст.»)

    def test_long_law_form_with_rf(self):
        result = extract_norms_from_text(
            "согласно статье 261 Трудового кодекса РФ", self._corpus()
        )
        assert result == {"ТК_261"}

    def test_long_law_form_without_rf(self):
        result = extract_norms_from_text(
            "предусмотрено статьёй 261 Трудового кодекса", self._corpus()
        )
        assert result == {"ТК_261"}

    def test_yo_normalization(self):
        # "статьёй" → "статьей" — должно матчиться
        result = extract_norms_from_text("предусмотрено статьёй 81 ТК РФ", self._corpus())
        assert result == {"ТК_81"}

    def test_case_insensitive(self):
        result = extract_norms_from_text("СТ. 261 ТК РФ", self._corpus())
        assert result == {"ТК_261"}

    # --- списки статей ---

    def test_list_with_i(self):
        result = extract_norms_from_text("ст. 256 и 261 ТК РФ", self._corpus())
        assert result == {"ТК_256", "ТК_261"}

    def test_list_with_comma(self):
        result = extract_norms_from_text("ст. 81, 256 ТК РФ", self._corpus())
        assert result == {"ТК_81", "ТК_256"}

    # --- разные кодексы ---

    def test_gk_short(self):
        result = extract_norms_from_text("ст. 1064 ГК РФ", self._corpus())
        assert result == {"ГК_1064"}

    def test_gk_long_form(self):
        result = extract_norms_from_text(
            "в соответствии со статьёй 15 Гражданского кодекса", self._corpus()
        )
        assert result == {"ГК_15"}

    def test_koap_decimal_article(self):
        result = extract_norms_from_text("ст. 12.1 КоАП РФ", self._corpus())
        assert result == {"КоАП_12.1"}

    def test_constitution(self):
        result = extract_norms_from_text("ст. 37 Конституции РФ", self._corpus())
        assert result == {"КРФ_37"}

    def test_uk(self):
        result = extract_norms_from_text("ст. 105 УК РФ", self._corpus())
        assert result == {"УК_105"}

    def test_nk(self):
        result = extract_norms_from_text("ст. 75 НК РФ", self._corpus())
        assert result == {"НК_75"}

    def test_two_different_laws_in_one_text(self):
        text = "ст. 261 ТК РФ исключает применение ст. 1064 ГК РФ в данной ситуации"
        result = extract_norms_from_text(text, self._corpus())
        assert result == {"ТК_261", "ГК_1064"}

    # --- фильтрация ---

    def test_norm_not_in_corpus_filtered(self):
        # ТК_500 не существует в corpus
        result = extract_norms_from_text("ст. 500 ТК РФ", self._corpus())
        assert result == set()

    def test_empty_corpus_returns_empty(self):
        result = extract_norms_from_text("ст. 261 ТК РФ", {})
        assert result == set()

    # --- граничные случаи ---

    def test_empty_text(self):
        assert extract_norms_from_text("", self._corpus()) == set()

    def test_no_norms_in_text(self):
        result = extract_norms_from_text("Текст без упоминания каких-либо статей.", self._corpus())
        assert result == set()

    def test_embedded_in_long_sentence(self):
        text = (
            "На основании ст. 261 ТК РФ увольнение запрещено, "
            "а ст. 392 ТК устанавливает срок исковой давности."
        )
        result = extract_norms_from_text(text, self._corpus())
        assert result == {"ТК_261", "ТК_392"}

    def test_returns_set_not_list(self):
        result = extract_norms_from_text("ст. 261 ТК РФ", self._corpus())
        assert isinstance(result, set)
