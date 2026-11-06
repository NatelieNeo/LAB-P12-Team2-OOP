"""CliApp driven by scripted keyboard input and the offline stub provider."""
from main import CliApp
from manager.io_manager import IOManager
from tests.conftest import make_profile


class QuietSpinner:
    """Stands in for ProgressSpinner so tests print nothing."""
    started = stopped = 0

    def start(self):
        QuietSpinner.started += 1

    def stop(self):
        QuietSpinner.stopped += 1


def run_cli(controller, answers):
    answers, shown = iter(answers), []
    io = IOManager(input_fn=lambda prompt: next(answers), output_fn=lambda *a: shown.append(" ".join(map(str, a))))
    CliApp(controller, io, spinner_factory=QuietSpinner).run()
    return "\n".join(shown)


NEW_USER = ["tim", "Tim", "23", "M", "75", "180", "sedentary", "gain", "nil"]


def test_new_user_creates_profile_makes_a_plan_and_exits(controller_factory, response_text):
    controller, _ = controller_factory(response_text)
    QuietSpinner.started = QuietSpinner.stopped = 0
    out = run_cli(controller, NEW_USER + ["1", "rice dishes", "in", "eggs, rice", "0", "", "5"])
    assert "No existing profile found for username: tim" in out
    assert "=== Your Meal Plan ===" in out and "Shopping list:" in out
    assert "Goodbye!" in out
    assert QuietSpinner.started == 1 and QuietSpinner.stopped == 1
    assert controller.get_profile("tim").name == "Tim"
    assert len(controller.get_history("tim")) == 1


def test_returning_user_is_welcomed_and_can_view_history_and_profile(controller_factory, response_text):
    controller, _ = controller_factory(response_text)
    controller.register_profile(make_profile(username="tim", name="Tim"))
    out = run_cli(controller, ["tim", "2", "", "3", "", "5"])
    assert "Welcome back, Tim!" in out
    assert "No meal plans found for this user." in out
    assert "Username: tim" in out


def test_history_shows_saved_plans(controller_factory, response_text):
    controller, _ = controller_factory(response_text)
    out = run_cli(controller, NEW_USER + ["1", "rice", "both", "eggs", "0", "", "2", "", "5"])
    assert "Meal Plan ID:" in out


def test_update_profile_keeps_blank_fields(controller_factory):
    controller, _ = controller_factory("")
    controller.register_profile(make_profile(username="tim", name="Tim"))
    run_cli(controller, ["tim", "4", "", "40", "", "", "", "", "", "", "", "5"])
    updated = controller.get_profile("tim")
    assert updated.age == 40 and updated.name == "Tim"


def test_failed_ai_call_shows_a_message_and_the_menu_continues(controller_factory):
    controller, _ = controller_factory("not json")
    out = run_cli(controller, NEW_USER + ["1", "rice", "in", "eggs", "0", "", "5"])
    assert "Unable to generate a meal plan" in out and "Goodbye!" in out
    assert controller.get_history("tim") == {}


def test_eating_out_shows_no_shopping_list(controller_factory, response_text):
    controller, _ = controller_factory(response_text)
    out = run_cli(controller, NEW_USER + ["1", "rice", "out", "0", "", "5"])
    assert "=== Your Meal Plan ===" in out and "Shopping list" not in out


def test_unknown_menu_choice_is_handled(controller_factory):
    controller, _ = controller_factory("")
    controller.register_profile(make_profile(username="tim"))
    out = run_cli(controller, ["tim", "9", "5"])
    assert "Please choose a number from 1 to 5." in out
