"""IOManager and UserInputValidator: terminal prompts, web forms and formatting."""
import pytest

from manager.exceptions import InvalidInputError
from manager.io_manager import IOManager, UserInputValidator
from manager.logic_manager import LogicManager
from tests.conftest import make_profile, make_request

V = UserInputValidator()


def scripted(answers):
    """An IOManager fed from a list instead of the keyboard."""
    answers, shown = iter(answers), []
    io = IOManager(input_fn=lambda prompt: next(answers), output_fn=lambda *a: shown.append(" ".join(map(str, a))))
    return io, shown


# ---- validator ---------------------------------------------------------------------
@pytest.mark.parametrize("raw,ok", [("13", True), ("99", True), ("12", False), ("100", False),
                                    ("-3", False), ("abc", False), ("", False), ("21.5", False)])
def test_age(raw, ok):
    if ok:
        assert V.validate_age(raw) == int(raw)
    else:
        with pytest.raises(InvalidInputError):
            V.validate_age(raw)


@pytest.mark.parametrize("method,good,bad", [
    ("validate_weight", ["20.1", "75", "299.9"], ["20", "300", "-5", "abc", ""]),
    ("validate_height", ["50.1", "180", "249.9"], ["50", "250", "0", "tall", ""])])
def test_weight_and_height_ranges(method, good, bad):
    for raw in good:
        assert getattr(V, method)(raw) == float(raw)
    for raw in bad:
        with pytest.raises(InvalidInputError):
            getattr(V, method)(raw)


@pytest.mark.parametrize("raw,expected", [("M", "male"), ("f", "female"), (" F ", "female")])
def test_gender_accepts_m_or_f(raw, expected):
    assert V.validate_gender(raw) == expected


@pytest.mark.parametrize("raw", ["", "x", "male", "MF"])
def test_gender_rejects_everything_else(raw):
    with pytest.raises(InvalidInputError):
        V.validate_gender(raw)


def test_choice_validators():
    assert V.validate_activity_level("Very Active") == "very active"
    assert V.validate_goal(" GAIN ") == "gain"
    assert V.validate_meal_source("Out") == "out"
    for method, bad in (("validate_activity_level", "athletic"), ("validate_goal", "bulk"),
                        ("validate_meal_source", "takeaway")):
        with pytest.raises(InvalidInputError):
            getattr(V, method)(bad)


@pytest.mark.parametrize("raw,expected", [("", None), ("0", None), ("0.0", None), ("6", 6.0), ("20.5", 20.5)])
def test_budget_values(raw, expected):
    assert V.validate_budget(raw) == expected


@pytest.mark.parametrize("raw", ["5.99", "-1", "lots"])
def test_budget_rejects_low_negative_and_text(raw):
    with pytest.raises(InvalidInputError):
        V.validate_budget(raw)


def test_text_validators():
    assert V.validate_restrictions("allergic to fish") == ["fish"]
    assert V.validate_restrictions("nil") == []
    assert V.validate_meal_preference(" noodles ") == "noodles"
    for method in ("validate_restrictions", "validate_meal_preference"):
        with pytest.raises(InvalidInputError):
            getattr(V, method)("123")
    for method in ("validate_username", "validate_name"):
        with pytest.raises(InvalidInputError):
            getattr(V, method)("   ")
    assert V.parse_list(" chicken , rice,, ") == ["chicken", "rice"]
    assert V.parse_list(None) == []


# ---- terminal prompts --------------------------------------------------------------------
def test_prompt_profile_reasks_until_valid():
    io, shown = scripted(["Tim", "5", "23", "x", "M", "abc", "75", "10", "180", "athlete", "sedentary",
                          "bulk", "gain", "12", "allergic to fish"])
    profile = io.prompt_profile("tim")
    assert (profile.name, profile.age, profile.gender, profile.weight_kg) == ("Tim", 23, "male", 75.0)
    assert (profile.height_cm, profile.activity_level, profile.goal) == (180.0, "sedentary", "gain")
    assert profile.dietary_restrictions == ["fish"]
    assert sum("age between" in line for line in shown) == 1
    assert sum("M or F" in line for line in shown) == 1


def test_prompt_profile_changes_keeps_blank_answers():
    io, _ = scripted(["", "30", "", "", "", "", "lose", ""])
    changes = io.prompt_profile_changes(make_profile())
    assert changes == {"age": 30, "goal": "lose"}


def test_prompt_request_skips_pantry_when_eating_out():
    io, _ = scripted(["noodles", "out", "0"])
    request = io.prompt_request(make_profile())
    assert request.meal_source == "eat_out" and request.pantry_ingredients == [] and request.daily_budget is None


def test_prompt_request_collects_pantry_when_eating_in():
    io, _ = scripted(["noodles", "in", "egg, rice", "25"])
    request = io.prompt_request(make_profile())
    assert request.pantry_ingredients == ["egg", "rice"] and request.daily_budget == 25.0


def test_menu_choice_prints_the_options():
    io, shown = scripted([" 3 "])
    assert io.prompt_menu_choice() == "3"
    assert len(shown) == 5 and shown[0].startswith("1.")


# ---- web forms ------------------------------------------------------------------------------
PROFILE_FORM = {"username": " sam ", "name": "Sam", "age": "30", "gender": "M", "weight_kg": "70",
                "height_cm": "175", "activity_level": "moderately active", "goal": "lose",
                "dietary_restrictions": "no peanuts, nil"}


def test_read_profile_form():
    profile = IOManager().read_profile_form(PROFILE_FORM)
    assert (profile.username, profile.gender, profile.activity_level) == ("sam", "male", "moderate")


def test_read_profile_form_uses_the_signed_in_username():
    assert IOManager().read_profile_form(PROFILE_FORM, "someone").username == "someone"


@pytest.mark.parametrize("field,value", [("age", "abc"), ("gender", ""), ("weight_kg", "5"), ("goal", "bulk"),
                                         ("name", " "), ("username", "")])
def test_read_profile_form_rejects_bad_fields(field, value):
    with pytest.raises(InvalidInputError):
        IOManager().read_profile_form({**PROFILE_FORM, field: value})


def test_read_request_form_defaults_and_pantry_rules():
    io, profile = IOManager(), make_profile()
    both = io.read_request_form({"meal_preference": "rice", "ingredients": "egg, rice", "budget": ""}, profile)
    assert both.meal_source == "both" and both.pantry_ingredients == ["egg", "rice"] and both.daily_budget is None
    out = io.read_request_form({"meal_source": "out", "ingredients": "egg", "budget": "0"}, profile)
    assert out.meal_source == "eat_out" and out.pantry_ingredients == []
    with pytest.raises(InvalidInputError):
        io.read_request_form({"meal_source": "carry-out"}, profile)


# ---- output ----------------------------------------------------------------------------------
def test_format_plan_layout(candidates, request_both):
    plan = LogicManager().build_meal_plan(candidates, request_both)
    text = IOManager().format_plan(plan)
    assert text.startswith("=== Your Meal Plan ===\nWhy this plan: " + plan.explanation)
    assert text.index("Breakfast:") < text.index("Lunch:") < text.index("Dinner:")
    assert "Macros: Protein High (>20-30g) / Carbs High (>30-60g) / Fat High (>10-20g)" in text
    assert "Cost: Budget (<$5)" in text and "Missing meals:" not in text
    assert "Ingredients: 150g chicken breast, 1 cup cooked rice" in text


def test_format_partial_plan_lists_missing_slots(candidates, request_both):
    plan = LogicManager().build_meal_plan(candidates[:1], request_both)
    text = IOManager().format_plan(plan)
    assert "Breakfast: Eggs on Toast (home_cooked)" in text
    assert "Missing meals: lunch, dinner" in text and "Lunch:" not in text


def test_allergen_notice_only_when_eating_out(candidates):
    io, manager = IOManager(), LogicManager()
    home = manager.build_meal_plan(candidates, make_request(make_profile(), meal_source="in"))
    mixed = manager.build_meal_plan(candidates, make_request(make_profile(), meal_source="out"))
    assert io.allergen_notice(home) == "" and IOManager.ALLERGEN_NOTICE not in io.format_plan(home)
    assert io.allergen_notice(mixed) == IOManager.ALLERGEN_NOTICE and IOManager.ALLERGEN_NOTICE in io.format_plan(mixed)


def test_format_profile_hides_internal_fields():
    text = IOManager().format_profile(make_profile(dietary_restrictions=["fish"]))
    assert "Username: nisha" in text and "Dietary restrictions: fish" in text
    assert "meal_preference" not in text.lower()


def test_show_methods_write_to_the_output_function(candidates, request_both):
    io, shown = scripted([])
    plan = LogicManager().build_meal_plan(candidates, request_both)
    io.show_plan(plan)
    io.show_history({"id1": plan})
    io.show_profile(make_profile())
    io.show_shopping_list(LogicManager().build_shopping_list(plan, []))
    text = "\n".join(shown)
    assert "Meal Plan ID: id1" in text and "Here is your profile" in text and "Shopping list:" in text
