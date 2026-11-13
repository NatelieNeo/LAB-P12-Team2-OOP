"""Offline integration, SDK adapter, safety and concurrency tests for multi-AI."""
import copy
import json
import logging
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import config
from manager.ai_handler_manager import AIHandlerManager
from manager.ai_manager import (AIManager, AIProvider, ClaudeProvider, GeminiProvider,
                                MultiAIRecommendationManager)
from manager.controller import AppFactory, CaloraController
from manager.data_manager import SqliteDataManager
from manager.exceptions import (AIConnectionError, AIResponseError, AllProvidersFailedError,
                                InvalidInputError, NoSuitableMealsError)
from manager.job_manager import PlanJobManager
from manager.logic_manager import LogicManager, NutritionCalculator
from manager.models import AIResponseSet, MealPlanRequest, UserProfile


@pytest.fixture
def request_data():
    profile = UserProfile("test-user", "Test", 24, "female", 60, 165, "sedentary", "maintain")
    return MealPlanRequest(profile, "rice", "both", ["rice"], 15)


def response_data(request, calorie_factor=1.0, cost=3.0, prefix="Rice"):
    targets = NutritionCalculator().targets(request.profile)
    recommendations = []
    for i, slot in enumerate(("breakfast",) * 3 + ("lunch",) * 4 + ("dinner",) * 3):
        recommendations.append({
            "meal_type": slot, "meal_name": f"{prefix} {chr(65 + i)}", "meal_source_type": "home_cooked",
            "calories": targets.per_meal_calories * calorie_factor,
            **targets.per_meal_macros, "estimated_cost": cost, "portion_size": "1 serving",
            "pantry_ingredients_used": ["rice"],
            "ingredients": [{"item": "rice", "quantity": 100, "unit": "g"}],
            "recipe_steps": ["Cook rice for 15 minutes."]})
    return {"estimated_nutrition_targets": {
        "calorie_target": targets.daily_calories, "protein_target_g": targets.daily_macros["protein_g"],
        "carb_target_g": targets.daily_macros["carbs_g"], "fat_target_g": targets.daily_macros["fat_g"]},
        "recommendations": recommendations}


def parsed_set(name, data):
    return AIResponseSet(name, tuple(AIHandlerManager().parse_response(json.dumps(data))), 0, 1)


class FakeProvider(AIProvider):
    def __init__(self, name, *outcomes, barrier=None, gate=None):
        self._name, self.outcomes = name, outcomes
        self.barrier, self.gate = barrier, gate
        self.calls, self.prompts, self.threads = 0, [], []
        self.started = threading.Event()

    @property
    def name(self):
        return self._name

    def generate(self, prompt):
        index = min(self.calls, len(self.outcomes) - 1)
        self.calls += 1
        self.prompts.append(prompt)
        self.threads.append(threading.current_thread().name)
        self.started.set()
        if self.barrier:
            self.barrier.wait(timeout=3)
        if self.gate and not self.gate.wait(timeout=3):
            raise RuntimeError("Test gate was not released")
        result = self.outcomes[index]
        if isinstance(result, Exception):
            raise result
        return result


def collect(providers, request, **kwargs):
    return MultiAIRecommendationManager(providers, retry_delay=0, **kwargs).get_response_sets(
        request, NutritionCalculator().targets(request.profile))


def test_gemini_sdk_request_uses_json_prompt_and_configured_model(request_data):
    client = Mock()
    client.models.generate_content.return_value.text = json.dumps(response_data(request_data))
    provider = GeminiProvider(api_key="fake", model="configured-gemini", client=client)
    output = provider.generate("shared prompt")
    kwargs = client.models.generate_content.call_args.kwargs
    assert kwargs["model"] == "configured-gemini" and kwargs["contents"] == "shared prompt"
    assert kwargs["config"]["response_mime_type"] == "application/json"
    assert len(AIHandlerManager().parse_response(output)) == 10


def test_sdk_clients_have_network_timeouts_no_internal_retries_and_are_closed(monkeypatch):
    from google import genai
    import anthropic
    google_client, claude_client = Mock(), Mock()
    google_client.models.generate_content.return_value.text = "{}"
    claude_client.messages.create.return_value = SimpleNamespace(stop_reason="end_turn", content=[])
    google_factory = Mock(return_value=google_client)
    claude_factory = Mock(return_value=claude_client)
    monkeypatch.setattr(genai, "Client", google_factory)
    monkeypatch.setattr(anthropic, "Anthropic", claude_factory)
    GeminiProvider(api_key="fake", timeout=7).generate("prompt")
    ClaudeProvider(api_key="fake", timeout=7).generate("prompt")
    assert google_factory.call_args.kwargs["http_options"] == {
        "timeout": 7000, "retry_options": {"attempts": 1}}
    assert claude_factory.call_args.kwargs["timeout"] == 7
    assert claude_factory.call_args.kwargs["max_retries"] == 0
    google_client.close.assert_called_once()
    claude_client.close.assert_called_once()


def test_real_sdks_serialize_and_parse_both_adapters_without_network(request_data):
    """Use installed SDKs with in-memory HTTP transports, not mocked SDK methods."""
    import anthropic
    import httpx2
    from google import genai
    text = json.dumps(response_data(request_data))
    requests = []

    def google_response(request):
        requests.append(json.loads(request.content))
        return httpx2.Response(200, json={"candidates": [{"content": {
            "role": "model", "parts": [{"text": text}]}, "finishReason": "STOP"}]})

    def claude_response(request):
        requests.append(json.loads(request.content))
        return httpx2.Response(200, json={"id": "msg_mock", "type": "message", "role": "assistant",
            "model": config.CLAUDE_MODEL, "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1}})

    with httpx2.Client(transport=httpx2.MockTransport(google_response)) as http:
        with genai.Client(api_key="fake", http_options={"httpx_client": http}) as client:
            assert GeminiProvider(api_key="fake", client=client).generate("same prompt") == text
    with httpx2.Client(transport=httpx2.MockTransport(claude_response)) as http:
        with anthropic.Anthropic(api_key="fake", http_client=http, max_retries=0) as client:
            assert ClaudeProvider(api_key="fake", client=client).generate("same prompt") == text
    assert requests[0]["contents"][0]["parts"][0]["text"] == requests[1]["messages"][0]["content"]


@pytest.mark.parametrize("status,retryable", [(401, False), (403, False), (429, True), (503, True)])
@pytest.mark.parametrize("provider_name", ["gemini", "claude"])
def test_sdk_errors_are_sanitised_and_classified(provider_name, status, retryable):
    import anthropic
    import httpx2
    from google.genai import errors
    client = Mock()
    secret = "secret-key-and-private-prompt"
    if provider_name == "gemini":
        client.models.generate_content.side_effect = errors.APIError(status, {"error": {"message": secret}})
        provider = GeminiProvider(api_key="fake", client=client)
    else:
        response = httpx2.Response(status, request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages"))
        client.messages.create.side_effect = anthropic.APIStatusError(secret, response=response, body={})
        provider = ClaudeProvider(api_key="fake", client=client)
    with pytest.raises(AIConnectionError) as caught:
        provider.generate("private prompt")
    assert caught.value.retryable is retryable
    assert secret not in str(caught.value)


def test_gemini_network_timeout_is_a_controlled_error():
    import httpx2
    client = Mock()
    client.models.generate_content.side_effect = httpx2.ReadTimeout("private details")
    with pytest.raises(AIConnectionError) as caught:
        GeminiProvider(api_key="fake", client=client).generate("prompt")
    assert "private details" not in str(caught.value)


def test_claude_sdk_extracts_text_blocks_without_thinking_blocks(request_data):
    client = Mock()
    text = json.dumps(response_data(request_data))
    client.messages.create.return_value = SimpleNamespace(stop_reason="end_turn", content=[
        SimpleNamespace(type="thinking", thinking="ignored"),
        SimpleNamespace(type="text", text=text[:20]), SimpleNamespace(type="text", text=text[20:])])
    provider = ClaudeProvider(api_key="fake", model="configured-claude", client=client)
    assert provider.generate("shared prompt") == text
    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["model"] == "configured-claude"
    assert kwargs["messages"] == [{"role": "user", "content": "shared prompt"}]
    assert len(AIHandlerManager().parse_response(text)) == 10


def test_claude_rejects_truncated_response():
    client = Mock()
    client.messages.create.return_value.stop_reason = "max_tokens"
    with pytest.raises(AIResponseError, match="token"):
        ClaudeProvider(api_key="fake", client=client).generate("prompt")


@pytest.mark.parametrize("provider_class", [GeminiProvider, ClaudeProvider])
def test_missing_api_keys_are_controlled_and_not_retryable(provider_class):
    provider = provider_class(api_key="")
    with pytest.raises(AIConnectionError) as error:
        provider.generate("prompt")
    assert not error.value.retryable


def test_both_calls_overlap_receive_identical_prompts_and_keep_attribution(request_data):
    barrier = threading.Barrier(2)
    text = json.dumps(response_data(request_data))
    providers = [FakeProvider("Gemini", text, barrier=barrier), FakeProvider("Claude", text, barrier=barrier)]
    results = collect(providers, request_data)
    assert {r.provider for r in results} == {"Gemini", "Claude"}
    assert providers[0].prompts == providers[1].prompts
    assert all(p.threads[0].startswith("calora-ai") for p in providers)
    assert providers[0].threads != providers[1].threads
    assert max(r.started_at for r in results) < min(r.completed_at for r in results)
    assert all(len(r.candidates) == config.RECOMMENDATION_COUNT for r in results)


@pytest.mark.parametrize("broken", [0, 1])
@pytest.mark.parametrize("failure", [AIConnectionError("network"), AIResponseError("empty"), "invalid json"])
def test_either_provider_failure_preserves_valid_peer(request_data, broken, failure):
    text = json.dumps(response_data(request_data))
    providers = [FakeProvider("Gemini", text), FakeProvider("Claude", text)]
    providers[broken] = FakeProvider(providers[broken].name, failure)
    results = collect(providers, request_data, max_retries=1)
    assert [r.provider for r in results] == [providers[1 - broken].name]
    assert all(p.calls == 1 for p in providers)


def test_retry_does_not_repeat_successful_provider(request_data):
    text = json.dumps(response_data(request_data))
    a, b = FakeProvider("Gemini", text), FakeProvider("Claude", AIConnectionError("busy"), text)
    assert len(collect([a, b], request_data)) == 2
    assert a.calls == 1 and b.calls == 2


def test_both_fail_produces_controlled_aggregate_without_retrying_auth(request_data):
    providers = [FakeProvider(name, AIConnectionError("auth", retryable=False)) for name in ("Gemini", "Claude")]
    with pytest.raises(AllProvidersFailedError) as caught:
        collect(providers, request_data)
    assert set(caught.value.failures) == {"Gemini", "Claude"}
    assert not caught.value.retryable
    assert all(p.calls == 1 for p in providers)


def test_both_invalid_produces_response_error(request_data):
    with pytest.raises(AIResponseError):
        collect([FakeProvider("a", "{}"), FakeProvider("b", "not json")], request_data, max_retries=1)


def test_slow_provider_times_out_without_discarding_valid_response(request_data):
    gate = threading.Event()
    text = json.dumps(response_data(request_data))
    slow = FakeProvider("Claude", text, gate=gate)
    try:
        results = collect([FakeProvider("Gemini", text), slow], request_data, timeout=0.05)
        assert [r.provider for r in results] == ["Gemini"]
        assert not gate.is_set()  # Returned while the slow call was still waiting.
    finally:
        gate.set()


def test_both_timeouts_are_controlled_and_pending_threads_can_finish(request_data):
    gate = threading.Event()
    providers = [FakeProvider(name, "{}", gate=gate) for name in ("Gemini", "Claude")]
    try:
        with pytest.raises(AllProvidersFailedError) as caught:
            collect(providers, request_data, timeout=0.05)
        assert set(caught.value.failures) == {"Gemini", "Claude"}
    finally:
        gate.set()


@pytest.mark.parametrize("error_type", [RuntimeError, TimeoutError])
def test_programming_errors_are_not_hidden_as_provider_outages(request_data, error_type):
    with pytest.raises(error_type, match="bug"):
        collect([FakeProvider("a", error_type("bug")),
                 FakeProvider("b", json.dumps(response_data(request_data)))], request_data)


def test_best_response_can_arrive_after_the_first(request_data):
    from concurrent.futures import ThreadPoolExecutor
    gate = threading.Event()
    fast = FakeProvider("Gemini", json.dumps(response_data(request_data, calorie_factor=0.3)))
    slow = FakeProvider("Claude", json.dumps(response_data(request_data)), gate=gate)
    executor = ThreadPoolExecutor(max_workers=1)
    collection = executor.submit(collect, [fast, slow], request_data)
    try:
        assert fast.started.wait(2) and slow.started.wait(2)
        assert not collection.done()  # A first valid answer alone must not win.
        gate.set()
        results = collection.result(timeout=3)
        selection = LogicManager().select_response_sets(results, request_data)
        assert selection.response.provider == "Claude"
        assert all(c.meal_name.startswith("Rice") for c in selection.response.candidates)
    finally:
        gate.set()
        executor.shutdown()


def test_scores_are_explainable_deterministic_and_ties_ignore_completion_order(request_data):
    logic = LogicManager()
    data = response_data(request_data)
    a, b = parsed_set("Gemini", data), parsed_set("Claude", data)
    assert logic.score_response(a, request_data).score == pytest.approx(100)
    assert logic.select_response_sets([a, b], request_data).response.provider == logic.select_response_sets(
        [b, a], request_data).response.provider
    assert sum(config.RESPONSE_SCORE_WEIGHTS.values()) == 100


def test_macro_fit_changes_score_with_identical_calories(request_data):
    good = response_data(request_data)
    bad = copy.deepcopy(good)
    for item in bad["recommendations"]:
        item["protein_g"] = item["carbs_g"] = item["fat_g"] = 0
    selection = LogicManager().select_response_sets([parsed_set("a", bad), parsed_set("b", good)], request_data)
    assert selection.response.provider == "b"


@pytest.mark.parametrize("restriction,ingredient", [("no peanuts", "peanut oil"), ("dairy", "milk"),
                                                        ("vegan", "egg"), ("vegetarian", "chicken"),
                                                        ("nut allergy", "almond"),
                                                        ("peanuts", "peanut-free sauce")])
def test_one_restricted_candidate_disqualifies_whole_response(request_data, restriction, ingredient):
    request_data = MealPlanRequest(request_data.profile.updated_with({"dietary_restrictions": restriction}))
    good = response_data(request_data)
    bad = copy.deepcopy(good)
    bad["recommendations"][0]["ingredients"] = [ingredient]
    result = LogicManager().select_response_sets([parsed_set("unsafe", bad), parsed_set("safe", good)], request_data)
    assert result.response.provider == "safe"


def test_both_unsuitable_fail_instead_of_returning_unsafe_plan(request_data):
    request_data = MealPlanRequest(request_data.profile.updated_with({"dietary_restrictions": ["rice"]}))
    data = response_data(request_data)
    with pytest.raises(NoSuitableMealsError):
        LogicManager().select_response_sets([parsed_set("a", data), parsed_set("b", data)], request_data)


def test_complete_set_requires_all_three_meal_slots(request_data):
    data = response_data(request_data)
    for item in data["recommendations"]:
        item["meal_type"] = "lunch"
    assert LogicManager().score_response(parsed_set("a", data), request_data) is None


def test_empty_ingredients_are_not_assumed_safe(request_data):
    data = response_data(request_data)
    data["recommendations"][0]["ingredients"] = []
    assert LogicManager().score_response(parsed_set("a", data), request_data) is None


def test_source_requirement_cannot_be_outweighed(request_data):
    request_data = MealPlanRequest(request_data.profile, meal_source="out")
    assert LogicManager().score_response(parsed_set("home", response_data(request_data)), request_data) is None


def test_advisory_budget_penalises_full_day_cost(request_data):
    logic = LogicManager()
    cheap, expensive = [parsed_set(str(cost), response_data(request_data, cost=cost)) for cost in (3, 10)]
    assert logic.score_response(cheap, request_data).score > logic.score_response(expensive, request_data).score
    assert logic.score_response(expensive, request_data).components["budget"] == 0.5


def test_strict_budget_chooses_feasible_combination_and_rejects_overruns(request_data):
    strict = MealPlanRequest(request_data.profile, daily_budget=6, strict_budget=True)
    data = response_data(strict, cost=10)
    for i in (0, 3, 7):
        data["recommendations"][i]["estimated_cost"] = 2
        data["recommendations"][i]["calories"] *= 0.5
    logic = LogicManager()
    result = logic.score_response(parsed_set("feasible", data), strict)
    assert result.plan.total_cost == 6 and result.plan.within_budget
    assert logic.score_response(parsed_set("over", response_data(strict, cost=3)), strict) is None


def test_strict_budget_does_not_reject_exact_sgd_total_due_to_float_error(request_data):
    strict = MealPlanRequest(request_data.profile, daily_budget=7, strict_budget=True)
    data = response_data(strict, cost=2.2)
    for item in data["recommendations"]:
        if item["meal_type"] == "dinner":
            item["estimated_cost"] = 2.6
    result = LogicManager().score_response(parsed_set("budget", data), strict)
    assert result.plan.total_cost == 7 and result.plan.within_budget


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "invalid"])
def test_invalid_timeout_configuration_is_a_controlled_input_error(monkeypatch, value):
    monkeypatch.setenv("AI_TOTAL_TIMEOUT_SECONDS", value)
    with pytest.raises(InvalidInputError):
        config.get_ai_total_timeout()


def test_duplicates_and_missing_recipes_reduce_quality(request_data):
    data = response_data(request_data)
    duplicate = copy.deepcopy(data)
    for item in duplicate["recommendations"]:
        item["meal_name"] = "Rice rice"
        item["recipe_steps"] = []
    logic = LogicManager()
    assert logic.score_response(parsed_set("a", data), request_data).score > logic.score_response(
        parsed_set("b", duplicate), request_data).score


@pytest.mark.parametrize("value", [True, -1, float("nan"), float("inf"), "123"])
def test_both_providers_use_strict_numeric_validation(request_data, value):
    data = response_data(request_data)
    data["recommendations"][0]["calories"] = value
    with pytest.raises(AIResponseError):
        AIHandlerManager().parse_response(json.dumps(data))


def test_count_and_exact_schema_are_preserved(request_data):
    data = response_data(request_data)
    assert len(AIHandlerManager().parse_response(json.dumps(data))) == 10
    data["recommendations"][0]["unexpected"] = 123
    assert not AIHandlerManager().is_valid(json.dumps(data))
    data = response_data(request_data)
    data["recommendations"].pop()
    assert not AIHandlerManager().is_valid(json.dumps(data))


@pytest.mark.parametrize("mode", ["gemini", "claude", "multi"])
def test_factory_explicit_modes(monkeypatch, mode):
    monkeypatch.setenv("AI_MODE", mode)
    manager = AppFactory.build_ai_manager()
    if mode == "multi":
        assert isinstance(manager, MultiAIRecommendationManager)
    else:
        assert isinstance(manager, AIManager)
        assert isinstance(manager._provider, GeminiProvider if mode == "gemini" else ClaudeProvider)


def test_auto_mode_and_models_are_resolved_lazily(monkeypatch):
    monkeypatch.delenv("AI_MODE", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "fake")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-test")
    monkeypatch.setenv("CLAUDE_MODEL", "claude-test")
    manager = AppFactory.build_ai_manager()
    assert [p.name for p in manager._providers] == ["gemini:gemini-test", "claude:claude-test"]
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    assert config.get_ai_mode() == "gemini"
    monkeypatch.setenv("AI_MODE", "typo")
    with pytest.raises(InvalidInputError):
        AppFactory.build_ai_manager()


def test_controller_saves_only_selected_plan_and_logs_safe_timing(request_data, tmp_path, caplog):
    text = json.dumps(response_data(request_data))
    ai = MultiAIRecommendationManager([FakeProvider("Gemini", text), FakeProvider("Claude", text)])
    jobs = PlanJobManager()
    data = SqliteDataManager(str(tmp_path / "test.db"))
    controller = CaloraController(ai, LogicManager(), data, jobs)
    try:
        controller.register_profile(request_data.profile)
        with caplog.at_level(logging.INFO):
            plan = controller.create_meal_plan(request_data)
        assert len(controller.get_history(request_data.profile.username)) == 1
        assert plan.ai_selection["provider"] in ("Gemini", "Claude")
        assert "Thread started" in caplog.text and "comparison" in caplog.text
        assert request_data.profile.username not in caplog.text
        assert "USER PROFILE" not in caplog.text
    finally:
        jobs.shutdown()


def test_gemini_only_workflow_still_builds_plan(request_data, tmp_path):
    jobs = PlanJobManager()
    controller = CaloraController(AIManager(FakeProvider("Gemini", json.dumps(response_data(request_data)))),
                                  LogicManager(), SqliteDataManager(str(tmp_path / "single.db")), jobs)
    try:
        controller.register_profile(request_data.profile)
        assert controller.create_meal_plan(request_data).is_complete
    finally:
        jobs.shutdown()
