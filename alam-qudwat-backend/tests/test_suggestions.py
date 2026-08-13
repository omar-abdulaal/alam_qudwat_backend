from app.services.suggestions import (
    ADULTS_FIRST_SUGGESTION,
    ADULTS_ROLE_SUGGESTIONS,
    KIDS_PREDEFINED_SUGGESTIONS,
    filter_unused,
    predefined_suggestions,
)


def test_kids_predefined_suggestions_are_fixed_regardless_of_categories():
    assert predefined_suggestions("kids", []) == list(KIDS_PREDEFINED_SUGGESTIONS)
    assert predefined_suggestions("kids", ["خليفة", "طبيب"]) == list(KIDS_PREDEFINED_SUGGESTIONS)


def test_adults_with_no_matching_role_gets_only_the_universal_suggestion():
    assert predefined_suggestions("adults", []) == [ADULTS_FIRST_SUGGESTION]
    assert predefined_suggestions("adults", ["not-a-real-role"]) == [ADULTS_FIRST_SUGGESTION]


def test_adults_with_one_role_gets_its_fixed_question():
    assert predefined_suggestions("adults", ["قائد عسكري"]) == [
        ADULTS_FIRST_SUGGESTION,
        "كيف تعامل مع تحديات المعارك والقيادة؟",
    ]
    assert predefined_suggestions("adults", ["عالم"]) == [
        ADULTS_FIRST_SUGGESTION,
        "كيف واجه الصعوبات في رحلته لطلب العلم؟",
    ]
    assert predefined_suggestions("adults", ["تاجر"]) == [
        ADULTS_FIRST_SUGGESTION,
        "كيف أدار تحدياته المالية والاجتماعية؟",
    ]


def test_adults_with_multiple_roles_picks_deterministically_by_priority():
    # "خليفة" outranks "طبيب" and "تاجر" in ADULTS_ROLE_SUGGESTIONS' order.
    result = predefined_suggestions("adults", ["طبيب", "خليفة", "تاجر"])
    assert result == [ADULTS_FIRST_SUGGESTION, ADULTS_ROLE_SUGGESTIONS["خليفة"]]

    # Order of the input list must not matter -- only priority order does.
    result_reordered = predefined_suggestions("adults", ["تاجر", "طبيب", "خليفة"])
    assert result_reordered == result


def test_every_taxonomy_category_has_a_fixed_adult_suggestion():
    # scripts/generate_character_classification.py's CATEGORY_TAXONOMY;
    # duplicated here deliberately (app/ must not import scripts/) -- keep
    # in sync if the taxonomy changes.
    taxonomy = (
        "خليفة", "أمير", "والي", "قائد عسكري", "فارس", "فقيه", "محدث", "مفسر",
        "مقرئ", "قاضٍ", "عالم", "داعية", "معلّم", "كاتب", "شاعر", "أديب",
        "نسابة", "طبيب", "تاجر", "راوية",
    )
    assert set(ADULTS_ROLE_SUGGESTIONS.keys()) == set(taxonomy)
    for question in ADULTS_ROLE_SUGGESTIONS.values():
        assert question.strip()


def test_filter_unused_drops_exact_matches_only():
    suggestions = ["سؤال أ", "سؤال ب", "سؤال ج"]
    assert filter_unused(suggestions, {"سؤال ب"}) == ["سؤال أ", "سؤال ج"]
    assert filter_unused(suggestions, set()) == suggestions
    assert filter_unused(suggestions, {"سؤال غير موجود"}) == suggestions
