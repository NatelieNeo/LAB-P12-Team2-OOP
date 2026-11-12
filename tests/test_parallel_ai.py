"""Tests for the threading feature: ParallelAIProvider (manager/ai_parallel.py).

Two AI models are asked at the same time and the first valid answer wins; the
other model is the safety net.

Every provider here is a fake, so nothing needs the network or an API key. Slow
models are simulated with a threading.Event ("gate") that the test opens itself,
so the tests do not depend on timing.

Run from the project root with:  python -m pytest tests/test_parallel_ai.py -v
"""
import json
import os
import sys
import threading
import time

import pytest

# Lets the tests import config and manager/ when pytest is started from any folder.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from manager.ai_handler_manager import AIHandlerManager  # noqa: E402
from manager.ai_manager import AIManager, AIProvider, GeminiProvider  # noqa: E402
from manager.ai_parallel import ParallelAIProvider  # noqa: E402
from manager.controller import AppFactory, CaloraController  # noqa: E402
from manager.data_manager import SqliteDataManager  # noqa: E402
from manager.exceptions import AIConnectionError, AIResponseError, AllProvidersFailedError  # noqa: E402
from manager.job_manager import PlanJobManager  # noqa: E402
from manager.logic_manager import LogicManager, NutritionCalculator  # noqa: E402
from manager.models import MealPlanRequest, UserProfile  # noqa: E402


# ---- sample data ----------------------------------------------------------------------
def make_meal(slot, name, calories, cost, source="home_cooked"):
    """One recommendation with exactly the fields the prompt asks Gemini for."""
    home_cooked = source == "home_cooked"
    return {
        "meal_type": slot, "meal_name": name, "meal_source_type": source,
        "calories": calories, "protein_g": 25, "carbs_g": 40, "fat_g": 15,
        "estimated_cost": cost, "portion_size": "1 serving",
        "pantry_ingredients_used": ["eggs"] if home_cooked else [],
        "ingredients": ["2 eggs", "1 cup cooked rice"] if home_cooked else ["rice", "chicken"],
        "recipe_steps": ["Cook everything thoroughly.", "Serve."] if home_cooked else [],
    }


def make_response():
    """Ten options, in the JSON structure AIHandlerManager expects."""
    return {
        "estimated_nutrition_targets": {
            "calorie_target": 1650, "protein_target_g": 90, "carb_target_g": 185, "fat_target_g": 55},
        "recommendations": [
            make_meal("breakfast", "Eggs on Toast", 400, 2.5),
            make_meal("breakfast", "Boiled Eggs and Toast", 250, 1.2),
            make_meal("breakfast", "Kaya Toast Set", 350, 3.5, source="eat_out"),
            make_meal("lunch", "Chicken Rice", 500, 4.5, source="eat_out"),
            make_meal("lunch", "Garlic Chicken Pasta", 480, 3.0),
            make_meal("lunch", "Sliced Fish Soup", 380, 5.5, source="eat_out"),
            make_meal("lunch", "Vegetable Rice Bowl", 650, 4.0),
            make_meal("dinner", "Chicken and Rice Bowl", 520, 3.5),
            make_meal("dinner", "Yong Tau Foo", 420, 6.0, source="eat_out"),
            make_meal("dinner", "Salmon Rice Bowl", 600, 7.0),
        ],
    }


def make_profile():
    return UserProfile(username="nisha", name="Nisha", age=21, gender="female", weight_kg=52,
                       height_cm=165, activity_level="sedentary", goal="maintain",
                       dietary_restrictions=[])


def make_request(profile):
    return MealPlanRequest(profile, meal_preference="rice dishes", meal_source="both",
                           pantry_ingredients=["eggs", "chicken", "rice"], daily_budget=10.5)


def ask_ai_manager(manager):
    """Calls AIManager the way the controller does."""
    profile = make_profile()
    return manager.get_recommendations(make_request(profile), NutritionCalculator().targets(profile))


# ---- fakes ----------------------------------------------------------------------------
class FakeProvider(AIProvider):
    """A fake AI model. It plays back one outcome per call: a string is returned,
    an Exception is raised (the last outcome repeats).
    If a gate is given, the call blocks until the test opens it."""

    def __init__(self, name, *outcomes, gate=None):
        self._name = name
        self._outcomes = list(outcomes)
        self._gate = gate
        self._lock = threading.Lock()
        self.calls = 0
        self.prompts = []
        self.started = threading.Event()

    @property
    def name(self):
        return self._name

    def generate(self, prompt):
        with self._lock:
            outcome = self._outcomes[min(self.calls, len(self._outcomes) - 1)]
            self.calls += 1
            self.prompts.append(prompt)
        self.started.set()
        if self._gate is not None:
            self._gate.wait(10)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class MeetingProvider(AIProvider):
    """Answers only once another thread is inside generate() at the same time.
    If the providers were called one after another, the barrier would break."""

    def __init__(self, name, barrier, answer):
        self._name = name
        self._barrier = barrier
        self._answer = answer

    @property
    def name(self):
        return self._name

    def generate(self, prompt):
        self._barrier.wait(timeout=5)
        return self._answer


# ---- fixtures -------------------------------------------------------------------------
@pytest.fixture
def response():
    return make_response()


@pytest.fixture
def response_text(response):
    return json.dumps(response)


@pytest.fixture
def valid():
    """The real validator that AppFactory gives to ParallelAIProvider."""
    return AIHandlerManager().is_valid


@pytest.fixture
def gates():
    """Events handed to slow fakes; always opened afterwards so no thread is left waiting."""
    created = []

    def new():
        gate = threading.Event()
        created.append(gate)
        return gate

    yield new
    for gate in created:
        gate.set()


# ---- is_valid (the validator) ---------------------------------------------------------
def test_is_valid_accepts_a_good_response_and_never_raises(response_text):
    handler = AIHandlerManager()
    assert handler.is_valid(response_text) is True
    for bad in ("", None, "not json", "[]", "{}", '{"recommendations": []}'):
        assert handler.is_valid(bad) is False


def test_is_valid_rejects_the_wrong_number_of_recommendations(response):
    response["recommendations"] = response["recommendations"][:3]
    assert AIHandlerManager().is_valid(json.dumps(response)) is False


# ---- set-up ---------------------------------------------------------------------------
def test_at_least_one_provider_is_required():
    with pytest.raises(ValueError):
        ParallelAIProvider([])


def test_name_lists_every_provider():
    parallel = ParallelAIProvider([FakeProvider("a", "x"), FakeProvider("b", "x")])
    assert parallel.name == "parallel(a, b)"


def test_last_winner_is_none_before_the_first_call():
    assert ParallelAIProvider([FakeProvider("a", "x")]).last_winner is None


# ---- the race: first valid response wins ----------------------------------------------
def test_the_fast_valid_answer_wins_without_waiting_for_the_slow_model(response_text, valid, gates):
    slow = FakeProvider("slow", response_text, gate=gates())
    fast = FakeProvider("fast", response_text)
    parallel = ParallelAIProvider([slow, fast], validator=valid, timeout=30)
    started = time.monotonic()
    assert parallel.generate("prompt") == response_text
    assert time.monotonic() - started < 5
    assert parallel.last_winner == "fast"
    assert slow.started.wait(2), "the slow model was never asked"


def test_the_models_really_run_at_the_same_time(response_text, valid):
    barrier = threading.Barrier(2)
    parallel = ParallelAIProvider(
        [MeetingProvider("a", barrier, response_text), MeetingProvider("b", barrier, response_text)],
        validator=valid, timeout=10)
    assert parallel.generate("prompt") == response_text


def test_every_model_is_asked_once_with_the_same_prompt(response_text, valid):
    first, second = FakeProvider("a", response_text), FakeProvider("b", response_text)
    ParallelAIProvider([first, second], validator=valid).generate("plan my meals")
    assert first.started.wait(2) and second.started.wait(2)
    assert first.prompts == ["plan my meals"] and second.prompts == ["plan my meals"]
    assert first.calls == 1 and second.calls == 1


def test_a_single_provider_still_works(response_text, valid):
    only = ParallelAIProvider([FakeProvider("only", response_text)], validator=valid)
    assert only.generate("prompt") == response_text
    assert only.last_winner == "only"


def test_without_a_validator_any_answer_wins():
    parallel = ParallelAIProvider([FakeProvider("a", "anything")])
    assert parallel.generate("prompt") == "anything"


def test_last_winner_follows_the_latest_call(response_text, valid):
    primary = FakeProvider("primary", response_text, AIConnectionError("down"))
    backup = FakeProvider("backup", AIConnectionError("busy"), response_text)
    parallel = ParallelAIProvider([primary, backup], validator=valid)
    parallel.generate("first")
    assert parallel.last_winner == "primary"
    parallel.generate("second")
    assert parallel.last_winner == "backup"


# ---- the safety net -------------------------------------------------------------------
def test_when_the_first_model_fails_the_second_is_the_safety_net(response_text, valid):
    broken = FakeProvider("primary", AIConnectionError("503 unavailable"))
    backup = FakeProvider("backup", response_text)
    parallel = ParallelAIProvider([broken, backup], validator=valid)
    assert parallel.generate("prompt") == response_text
    assert parallel.last_winner == "backup"


def test_an_invalid_answer_does_not_win_the_race(response_text, valid, gates):
    # The garbage arrives first; the valid answer arrives later and must still be chosen.
    release_good = gates()
    garbage = FakeProvider("garbage", "this is not json")
    good = FakeProvider("good", response_text, gate=release_good)
    parallel = ParallelAIProvider([garbage, good], validator=valid, timeout=30)
    threading.Timer(0.2, release_good.set).start()
    assert parallel.generate("prompt") == response_text
    assert parallel.last_winner == "good"


def test_an_unexpected_crash_in_one_model_does_not_stop_the_other(response_text, valid):
    crashing = FakeProvider("crashing", RuntimeError("unexpected bug"))
    backup = FakeProvider("backup", response_text)
    parallel = ParallelAIProvider([crashing, backup], validator=valid)
    assert parallel.generate("prompt") == response_text


def test_a_hung_model_does_not_stop_the_other_from_winning(response_text, valid, gates):
    parallel = ParallelAIProvider(
        [FakeProvider("hung", response_text, gate=gates()), FakeProvider("ok", response_text)],
        validator=valid, timeout=0.5)
    assert parallel.generate("prompt") == response_text
    assert parallel.last_winner == "ok"


# ---- every model fails ----------------------------------------------------------------
def test_when_every_model_has_a_connection_error_the_failures_are_reported(valid):
    parallel = ParallelAIProvider(
        [FakeProvider("one", AIConnectionError("down")), FakeProvider("two", AIConnectionError("quota"))],
        validator=valid)
    with pytest.raises(AllProvidersFailedError) as caught:
        parallel.generate("prompt")
    assert isinstance(caught.value, AIConnectionError)  # so AIManager's existing handler catches it
    assert set(caught.value.failures) == {"one", "two"}
    assert "down" in str(caught.value) and "quota" in str(caught.value)
    assert caught.value.retryable is True
    assert parallel.last_winner is None


def test_when_every_model_answers_badly_it_is_a_response_error(valid):
    # Every model answered, so this is a response problem, not a connection problem.
    parallel = ParallelAIProvider([FakeProvider("one", "nope"), FakeProvider("two", "{}")], validator=valid)
    with pytest.raises(AIResponseError) as caught:
        parallel.generate("prompt")
    assert not isinstance(caught.value, AIConnectionError)
    assert "one" in str(caught.value) and "two" in str(caught.value)


def test_one_connection_error_and_one_bad_answer_is_reported_as_all_failed(valid):
    parallel = ParallelAIProvider(
        [FakeProvider("one", AIConnectionError("down")), FakeProvider("two", "nope")], validator=valid)
    with pytest.raises(AllProvidersFailedError) as caught:
        parallel.generate("prompt")
    assert isinstance(caught.value.failures["one"], AIConnectionError)
    assert isinstance(caught.value.failures["two"], AIResponseError)


def test_models_that_never_answer_time_out(valid, gates):
    parallel = ParallelAIProvider(
        [FakeProvider("hung-a", "x", gate=gates()), FakeProvider("hung-b", "x", gate=gates())],
        validator=valid, timeout=0.2)
    started = time.monotonic()
    with pytest.raises(AllProvidersFailedError) as caught:
        parallel.generate("prompt")
    assert time.monotonic() - started < 5
    assert "did not answer" in str(caught.value)
    assert set(caught.value.failures) == {"hung-a", "hung-b"}


def test_one_failure_and_one_timeout_is_reported_as_all_failed(valid, gates):
    parallel = ParallelAIProvider(
        [FakeProvider("broken", AIConnectionError("down")), FakeProvider("hung", "x", gate=gates())],
        validator=valid, timeout=0.2)
    with pytest.raises(AllProvidersFailedError) as caught:
        parallel.generate("prompt")
    assert "down" in str(caught.value.failures["broken"])
    assert "did not answer" in str(caught.value.failures["hung"])


# ---- the retryable flag ---------------------------------------------------------------
def test_failures_that_can_never_succeed_are_not_retryable(valid):
    missing_key = AIConnectionError("GEMINI_API_KEY was not found.", retryable=False)
    parallel = ParallelAIProvider(
        [FakeProvider("a", missing_key), FakeProvider("b", missing_key)], validator=valid)
    with pytest.raises(AllProvidersFailedError) as caught:
        parallel.generate("prompt")
    assert caught.value.retryable is False


def test_one_retryable_failure_keeps_the_error_retryable(valid):
    missing_key = AIConnectionError("GEMINI_API_KEY was not found.", retryable=False)
    parallel = ParallelAIProvider(
        [FakeProvider("a", missing_key), FakeProvider("b", AIConnectionError("busy"))], validator=valid)
    with pytest.raises(AllProvidersFailedError) as caught:
        parallel.generate("prompt")
    assert caught.value.retryable is True


# ---- substitutability (LSP / OCP): AIManager uses it like a single model ---------------
def test_it_is_a_drop_in_ai_provider(response_text, valid):
    parallel = ParallelAIProvider([FakeProvider("a", response_text)], validator=valid)
    assert isinstance(parallel, AIProvider)
    candidates = ask_ai_manager(AIManager(parallel, retry_delay=0))
    assert len(candidates) == 10


def test_ai_manager_does_not_retry_when_the_safety_net_worked(response_text, valid):
    primary = FakeProvider("primary", AIConnectionError("down"))
    backup = FakeProvider("backup", response_text)
    manager = AIManager(ParallelAIProvider([primary, backup], validator=valid), retry_delay=0)
    assert len(ask_ai_manager(manager)) == 10
    assert primary.calls == 1 and backup.calls == 1


def test_ai_manager_retries_a_parallel_provider_that_failed_everywhere_once(response_text, valid):
    # Attempt 1: both models fail. Attempt 2: the backup recovers.
    primary = FakeProvider("primary", AIConnectionError("down"), AIConnectionError("down"))
    backup = FakeProvider("backup", AIConnectionError("busy"), response_text)
    manager = AIManager(ParallelAIProvider([primary, backup], validator=valid), retry_delay=0)
    assert len(ask_ai_manager(manager)) == 10
    assert backup.calls == 2


def test_ai_manager_gives_up_after_max_retries(valid):
    a = FakeProvider("a", AIConnectionError("down"))
    b = FakeProvider("b", AIConnectionError("down"))
    manager = AIManager(ParallelAIProvider([a, b], validator=valid), max_retries=3, retry_delay=0)
    with pytest.raises(AllProvidersFailedError):
        ask_ai_manager(manager)
    assert a.calls == 3 and b.calls == 3


def test_ai_manager_does_not_retry_when_no_model_can_ever_succeed(valid):
    missing_key = AIConnectionError("GEMINI_API_KEY was not found.", retryable=False)
    a, b = FakeProvider("a", missing_key), FakeProvider("b", missing_key)
    manager = AIManager(ParallelAIProvider([a, b], validator=valid), max_retries=3, retry_delay=0)
    with pytest.raises(AllProvidersFailedError):
        ask_ai_manager(manager)
    assert a.calls == 1 and b.calls == 1


def test_ai_manager_retries_when_every_answer_is_invalid(valid):
    a, b = FakeProvider("a", "nope"), FakeProvider("b", "{}")
    manager = AIManager(ParallelAIProvider([a, b], validator=valid), max_retries=2, retry_delay=0)
    with pytest.raises(AIResponseError):
        ask_ai_manager(manager)
    assert a.calls == 2 and b.calls == 2


def test_a_plan_is_built_when_the_primary_model_is_down(response_text, valid, tmp_path):
    jobs = PlanJobManager(max_workers=2)
    try:
        provider = ParallelAIProvider(
            [FakeProvider("primary", AIConnectionError("503")), FakeProvider("backup", response_text)],
            validator=valid)
        controller = CaloraController(
            AIManager(provider, retry_delay=0), LogicManager(),
            SqliteDataManager(str(tmp_path / "calora.db")), jobs)
        profile = make_profile()
        controller.register_profile(profile)
        plan = controller.create_meal_plan(make_request(profile))
        assert plan.meals and provider.last_winner == "backup"
    finally:
        jobs.shutdown()


# ---- thread safety --------------------------------------------------------------------
def test_many_plans_can_be_requested_at_once(response_text, valid):
    parallel = ParallelAIProvider(
        [FakeProvider("a", response_text), FakeProvider("b", response_text)], validator=valid)
    results, errors = [], []

    def worker():
        try:
            results.append(parallel.generate("prompt"))
        except Exception as error:  # noqa: BLE001
            errors.append(error)

    threads = [threading.Thread(target=worker) for _ in range(12)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    assert not errors and results == [response_text] * 12


# ---- wiring (AppFactory and GeminiProvider) -------------------------------------------
def test_factory_builds_two_gemini_models_in_parallel_by_default(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.setenv("GEMINI_FALLBACK_MODEL", "second-model")
    monkeypatch.delenv("CALORA_PARALLEL_AI", raising=False)
    provider = AppFactory.build_ai_provider()
    assert isinstance(provider, ParallelAIProvider)
    assert config.GEMINI_MODEL in provider.name and "second-model" in provider.name


def test_factory_can_switch_parallel_mode_off(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.setenv("CALORA_PARALLEL_AI", "0")
    assert isinstance(AppFactory.build_ai_provider(), GeminiProvider)


def test_factory_uses_one_model_if_the_fallback_is_the_same_model(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.setenv("GEMINI_FALLBACK_MODEL", config.GEMINI_MODEL)
    monkeypatch.delenv("CALORA_PARALLEL_AI", raising=False)
    assert isinstance(AppFactory.build_ai_provider(), GeminiProvider)


def test_gemini_provider_names_its_model():
    assert GeminiProvider(api_key="x", model="m1").name == "gemini:m1"


def test_gemini_client_creation_is_guarded_by_a_lock():
    provider = GeminiProvider(api_key="x", model="m1")
    assert hasattr(provider._client_lock, "acquire")  # a threading.Lock shared by all threads
