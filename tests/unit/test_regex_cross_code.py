"""
Тесты для межкодексных ссылок в build_graph.extract_explicit_refs.
Проверяем все 10 НПА, полные и короткие формы, ё→е нормализацию.
"""

import pytest
from mhgb.build_graph import extract_explicit_refs


def art(law: str, num: str, paragraphs: list[str]) -> dict:
    return {"id": f"{law}_{num}", "law_short": law, "paragraphs": paragraphs}


def targets(edges: list[tuple]) -> set[str]:
    return {t for _, t in edges}


# ---------------------------------------------------------------------------
# Полные названия НПА в родительном падеже
# ---------------------------------------------------------------------------

class TestFullNameRefs:
    def test_gk_full(self):
        a = art("ТК", "261", ["в соответствии со статьёй 150 Гражданского кодекса"])
        assert "ГК_150" in targets(extract_explicit_refs(a))

    def test_sk_full(self):
        a = art("ГК", "1", ["согласно статье 65 Семейного кодекса РФ"])
        assert "СК_65" in targets(extract_explicit_refs(a))

    def test_zhk_full(self):
        a = art("ГК", "1", ["предусмотренном статьёй 30 Жилищного кодекса"])
        assert "ЖК_30" in targets(extract_explicit_refs(a))

    def test_uk_full(self):
        a = art("КоАП", "3", ["статьёй 161 Уголовного кодекса Российской Федерации"])
        assert "УК_161" in targets(extract_explicit_refs(a))

    def test_nk_full(self):
        a = art("ТК", "136", ["по статье 145 Налогового кодекса РФ"])
        assert "НК_145" in targets(extract_explicit_refs(a))

    def test_zk_full(self):
        a = art("ГК", "1", ["статьями 7 и 8 Земельного кодекса"])
        t = targets(extract_explicit_refs(a))
        assert "ЗК_7" in t and "ЗК_8" in t

    def test_grk_full(self):
        a = art("ГК", "1", ["статьёй 51 Градостроительного кодекса РФ"])
        assert "ГрК_51" in targets(extract_explicit_refs(a))

    def test_koap_full(self):
        a = art("УК", "1", ["предусмотренном статьёй 14.5 Кодекса об административных правонарушениях"])
        assert "КоАП_14.5" in targets(extract_explicit_refs(a))

    def test_krf_full(self):
        a = art("ТК", "1", ["статьёй 37 Конституции Российской Федерации"])
        assert "КРФ_37" in targets(extract_explicit_refs(a))

    def test_krf_short_rf(self):
        a = art("ТК", "1", ["статьёй 46 Конституции РФ"])
        assert "КРФ_46" in targets(extract_explicit_refs(a))


# ---------------------------------------------------------------------------
# Короткие аббревиатуры с «РФ»
# ---------------------------------------------------------------------------

class TestShortWithRF:
    def test_tk_rf(self):
        a = art("ГК", "1", ["статьёй 81 ТК РФ"])
        assert "ТК_81" in targets(extract_explicit_refs(a))

    def test_gk_rf(self):
        a = art("ТК", "1", ["статьёй 807 ГК РФ"])
        assert "ГК_807" in targets(extract_explicit_refs(a))

    def test_uk_rf(self):
        a = art("КоАП", "1", ["статьёй 158 УК РФ"])
        assert "УК_158" in targets(extract_explicit_refs(a))

    def test_nk_rf(self):
        a = art("ТК", "1", ["статьёй 217 НК РФ"])
        assert "НК_217" in targets(extract_explicit_refs(a))

    def test_zk_rf(self):
        a = art("ГК", "1", ["статьёй 22 ЗК РФ"])
        assert "ЗК_22" in targets(extract_explicit_refs(a))

    def test_grk_rf(self):
        a = art("ГК", "1", ["статьёй 49 ГрК РФ"])
        assert "ГрК_49" in targets(extract_explicit_refs(a))

    def test_koap_rf(self):
        a = art("УК", "1", ["статьёй 20.1 КоАП РФ"])
        assert "КоАП_20.1" in targets(extract_explicit_refs(a))


# ---------------------------------------------------------------------------
# Короткие аббревиатуры без «РФ»
# ---------------------------------------------------------------------------

class TestShortWithoutRF:
    def test_tk_bare(self):
        a = art("ГК", "1", ["предусмотрено статьёй 81 ТК"])
        assert "ТК_81" in targets(extract_explicit_refs(a))

    def test_gk_bare(self):
        a = art("ТК", "1", ["статьёй 807 ГК"])
        assert "ГК_807" in targets(extract_explicit_refs(a))

    def test_koap_bare(self):
        a = art("УК", "1", ["статьёй 20.1 КоАП"])
        assert "КоАП_20.1" in targets(extract_explicit_refs(a))

    def test_grk_bare(self):
        a = art("ГК", "1", ["статьёй 49 ГрК"])
        assert "ГрК_49" in targets(extract_explicit_refs(a))


# ---------------------------------------------------------------------------
# Базовое поведение (регрессия)
# ---------------------------------------------------------------------------

class TestBaselineBehavior:
    def test_same_law_nastoyashchego(self):
        a = art("ТК", "261", ["статьёй 256 настоящего кодекса"])
        assert "ТК_256" in targets(extract_explicit_refs(a))

    def test_same_law_default(self):
        a = art("ГК", "1", ["в соответствии со статьёй 150"])
        assert "ГК_150" in targets(extract_explicit_refs(a))

    def test_no_self_reference(self):
        a = art("ТК", "81", ["статьёй 81 настоящего кодекса"])
        assert len(extract_explicit_refs(a)) == 0

    def test_yo_normalization(self):
        a = art("ГК", "1", ["статьёй 150 Гражданского кодекса"])
        t = targets(extract_explicit_refs(a))
        assert "ГК_150" in t

    def test_multiple_articles_in_one_para(self):
        a = art("ТК", "1", ["статьями 81 и 82 ТК РФ"])
        t = targets(extract_explicit_refs(a))
        assert "ТК_81" in t and "ТК_82" in t

    def test_cross_codex_gk_sk(self):
        a = art("ГК", "256", ["статьёй 34 Семейного кодекса"])
        assert "СК_34" in targets(extract_explicit_refs(a))
