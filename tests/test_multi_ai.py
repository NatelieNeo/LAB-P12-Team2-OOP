"""Offline integration, SDK adapter, safety and concurrency tests for multi-AI."""
import copy
import json
import logging
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import config
from tests.helpers import FakeProvider, response_data
from manager.ai_handler_manager import AIHandlerManager
from manager.ai_manager import (AIManager, ClaudeProvider, GeminiProvider,
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


def parsed_set(name, data):
    return AIResponseSet(name, tuple(AIHandlerManager().parse_response(json.dumps(data))), 0, 1)


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
    assert kwargs["max_tokens"] == config.get_claude_max_output_tokens()
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


def test_three_claude_timeouts_start_gemini_backup_in_the_same_worker(request_data, caplog):
    barrier = threading.Barrier(2)
    text = json.dumps(response_data(request_data))
    primary = FakeProvider("Gemini", text, barrier=barrier)
    claude = FakeProvider("Claude", AIConnectionError("APITimeoutError"))
    backup = FakeProvider("Gemini", text)
    # Only the first attempt joins the barrier; retries must not replay Gemini.
    original_generate = claude.generate

    def first_attempt_overlap(prompt):
        if claude.calls == 0:
            barrier.wait(timeout=5)
        return original_generate(prompt)

    claude.generate = first_attempt_overlap
    with caplog.at_level(logging.INFO):
        results = collect([primary, claude], request_data,
                          fallback_provider=backup, fallback_for="Claude")
    assert primary.calls == 1 and claude.calls == 3 and backup.calls == 1
    assert {result.provider for result in results} == {"Gemini", "Gemini [backup for Claude]"}
    assert primary.prompts == backup.prompts == claude.prompts[:1]
    assert len(set(claude.threads + backup.threads)) == 1
    assert claude.threads[0] != primary.threads[0]
    assert "same worker slot" in caplog.text


def test_backup_overlaps_original_gemini_when_it_is_still_running(request_data):
    barrier = threading.Barrier(2)
    text = json.dumps(response_data(request_data))
    primary = FakeProvider("Gemini", text, barrier=barrier)
    backup = FakeProvider("Gemini", text, barrier=barrier)
    claude = FakeProvider("Claude", AIConnectionError("timeout"))
    results = collect([primary, claude], request_data, fallback_provider=backup, fallback_for="Claude")
    assert claude.calls == 3 and primary.calls == backup.calls == 1
    assert primary.threads[0] != backup.threads[0]
    assert max(result.started_at for result in results) < min(result.completed_at for result in results)


@pytest.mark.parametrize("attempt", [1, 2, 3])
def test_claude_success_within_three_attempts_does_not_start_backup(request_data, attempt):
    text = json.dumps(response_data(request_data))
    claude = FakeProvider("Claude", *([AIConnectionError("timeout")] * (attempt - 1)), text)
    backup = FakeProvider("Gemini", text)
    results = collect([FakeProvider("Gemini", text), claude], request_data,
                      fallback_provider=backup, fallback_for="Claude")
    assert {result.provider for result in results} == {"Gemini", "Claude"}
    assert claude.calls == attempt and backup.calls == 0


@pytest.mark.parametrize("failure", [AIConnectionError("auth", retryable=False), "invalid JSON"])
def test_claude_auth_or_invalid_json_can_use_backup(request_data, failure):
    text = json.dumps(response_data(request_data))
    claude = FakeProvider("Claude", failure)
    backup = FakeProvider("Gemini", text)
    results = collect([FakeProvider("Gemini", text), claude], request_data,
                      fallback_provider=backup, fallback_for="Claude")
    assert claude.calls == (1 if isinstance(failure, AIConnectionError) else 3)
    assert backup.calls == 1
    assert len(results) == 2


def test_failed_backup_preserves_original_gemini_without_repeating_it(request_data):
    text = json.dumps(response_data(request_data))
    primary = FakeProvider("Gemini", text)
    backup = FakeProvider("Gemini", AIConnectionError("timeout"))
    results = collect([primary, FakeProvider("Claude", AIConnectionError("timeout"))], request_data,
                      fallback_provider=backup, fallback_for="Claude")
    assert [result.provider for result in results] == ["Gemini"]
    assert primary.calls == backup.calls == 1


def test_backup_can_win_when_both_original_providers_fail(request_data):
    backup = FakeProvider("Gemini", json.dumps(response_data(request_data)))
    results = collect([FakeProvider("Gemini", AIConnectionError("timeout")),
                       FakeProvider("Claude", AIConnectionError("timeout"))], request_data,
                      fallback_provider=backup, fallback_for="Claude")
    selection = LogicManager().select_response_sets(results, request_data)
    assert selection.response.provider == "Gemini [backup for Claude]"
    assert selection.plan.is_complete


def test_backup_is_compared_by_quality_instead_of_automatically_selected(request_data):
    primary = FakeProvider("Gemini", json.dumps(response_data(request_data)))
    backup = FakeProvider("Gemini", json.dumps(response_data(request_data, calorie_factor=0.3)))
    results = collect([primary, FakeProvider("Claude", AIConnectionError("timeout"))], request_data,
                      fallback_provider=backup, fallback_for="Claude")
    assert LogicManager().select_response_sets(results, request_data).response.provider == "Gemini"


def test_all_primaries_and_backup_failing_is_controlled_and_bounded(request_data):
    primary, claude, backup = [FakeProvider(name, AIConnectionError("timeout"))
                               for name in ("Gemini", "Claude", "Gemini")]
    with pytest.raises(AllProvidersFailedError) as caught:
        collect([primary, claude], request_data, fallback_provider=backup, fallback_for="Claude")
    assert primary.calls == claude.calls == 3 and backup.calls == 1
    assert isinstance(caught.value.failures["Claude"], AllProvidersFailedError)
    assert "Gemini [backup for Claude]" in caught.value.failures["Claude"].failures


def test_invalid_primaries_and_invalid_backup_preserve_response_error(request_data):
    with pytest.raises(AIResponseError):
        collect([FakeProvider("Gemini", "{}"), FakeProvider("Claude", "{}")], request_data,
                fallback_provider=FakeProvider("Gemini", "{}"), fallback_for="Claude")


def test_backup_uses_reserved_budget_and_requests_receive_remaining_time(request_data, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(time, "perf_counter", lambda: clock[0])
    budgets = []
    text = json.dumps(response_data(request_data))

    class BudgetProvider(FakeProvider):
        def generate_with_timeout(self, prompt, timeout):
            budgets.append((self.name, timeout))
            clock[0] += timeout
            return self.generate(prompt)

    claude = BudgetProvider("Claude", AIConnectionError("timeout"))
    backup = BudgetProvider("Gemini", text)
    manager = MultiAIRecommendationManager([FakeProvider("primary", text), claude], timeout=10,
                                           retry_delay=0, fallback_provider=backup,
                                           fallback_for="Claude", fallback_timeout=2)
    result = manager._ask(claude, "same prompt", deadline=110)
    assert budgets == [("Claude", 8), ("Gemini", 2)]
    assert result.provider == "Gemini [backup for Claude]"
    assert claude.calls == backup.calls == 1  # The deadline takes precedence over three attempts.


def test_backup_never_overlaps_the_claude_call_it_replaces(request_data):
    from concurrent.futures import ThreadPoolExecutor
    gate = threading.Event()
    text = json.dumps(response_data(request_data))
    claude = FakeProvider("Claude", AIConnectionError("auth", retryable=False), gate=gate)
    backup = FakeProvider("Gemini", text)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(collect, [FakeProvider("Gemini", text), claude], request_data,
                                 fallback_provider=backup, fallback_for="Claude")
        try:
            assert claude.started.wait(2)
            assert not backup.started.is_set()
            gate.set()
            assert len(future.result(timeout=3)) == 2
        finally:
            gate.set()


def test_late_claude_failure_does_not_launch_backup_after_collection_deadline(request_data):
    gate, finished = threading.Event(), threading.Event()
    text = json.dumps(response_data(request_data))
    claude = FakeProvider("Claude", AIConnectionError("timeout"), gate=gate)
    backup = FakeProvider("Gemini", text)

    class TrackingManager(MultiAIRecommendationManager):
        def _ask(self, provider, prompt, deadline):
            try:
                return super()._ask(provider, prompt, deadline)
            finally:
                if provider is claude:
                    finished.set()

    manager = TrackingManager([FakeProvider("Gemini", text), claude], timeout=0.1,
                              retry_delay=0, fallback_provider=backup, fallback_for="Claude")
    try:
        results = manager.get_response_sets(request_data, NutritionCalculator().targets(request_data.profile))
        assert [result.provider for result in results] == ["Gemini"]
        assert claude.started.is_set() and not finished.is_set()
        assert backup.calls == 0
    finally:
        gate.set()
        assert finished.wait(3)
    assert backup.calls == 0


def test_claude_timeout_is_sanitised_and_retryable():
    import anthropic
    import httpx2
    client = Mock()
    client.messages.create.side_effect = anthropic.APITimeoutError(
        request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages"))
    with pytest.raises(AIConnectionError, match="APITimeoutError") as caught:
        ClaudeProvider(api_key="fake", client=client).generate("prompt")
    assert caught.value.retryable


def test_network_adapters_cap_timeout_by_remaining_budget(monkeypatch):
    import anthropic
    from google import genai
    google_client, claude_client = Mock(), Mock()
    google_client.models.generate_content.return_value.text = "{}"
    claude_client.messages.create.return_value = SimpleNamespace(stop_reason="end_turn", content=[])
    google_factory, claude_factory = Mock(return_value=google_client), Mock(return_value=claude_client)
    monkeypatch.setattr(genai, "Client", google_factory)
    monkeypatch.setattr(anthropic, "Anthropic", claude_factory)
    GeminiProvider(api_key="fake", timeout=90).generate_with_timeout("prompt", 5)
    ClaudeProvider(api_key="fake", timeout=120).generate_with_timeout("prompt", 5)
    assert google_factory.call_args.kwargs["http_options"]["timeout"] == 5000
    assert claude_factory.call_args.kwargs["timeout"] == 5
    assert claude_client.messages.create.call_args.kwargs["timeout"] == 5


def test_factory_backup_and_claude_timeout_settings_are_configurable(monkeypatch):
    monkeypatch.setenv("AI_MODE", "multi")
    monkeypatch.setenv("CLAUDE_GEMINI_BACKUP", "1")
    monkeypatch.setenv("CLAUDE_REQUEST_TIMEOUT_SECONDS", "150")
    monkeypatch.setenv("GEMINI_BACKUP_MODEL", "configured-backup")
    manager = AppFactory.build_ai_manager()
    assert manager._providers[1]._timeout == 150
    assert manager._fallback_for == manager._providers[1].name
    assert manager._fallback_provider.name == "gemini:configured-backup"
    assert manager._fallback_provider is not manager._providers[0]
    monkeypatch.setenv("CLAUDE_GEMINI_BACKUP", "0")
    assert AppFactory.build_ai_manager()._fallback_provider is None


def test_default_backup_reuses_current_gemini_model_not_the_legacy_race_model(monkeypatch):
    monkeypatch.delenv("GEMINI_BACKUP_MODEL", raising=False)
    monkeypatch.setenv("GEMINI_MODEL", "current-model")
    monkeypatch.setenv("GEMINI_FALLBACK_MODEL", "legacy-model")
    assert config.get_gemini_backup_model() == "current-model"


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "invalid", "1.5"])
def test_invalid_claude_output_budget_is_controlled(monkeypatch, value):
    monkeypatch.setenv("CLAUDE_MAX_OUTPUT_TOKENS", value)
    with pytest.raises(InvalidInputError):
        config.get_claude_max_output_tokens()


def test_claude_output_budget_is_independent_of_gemini(monkeypatch):
    monkeypatch.setenv("AI_MAX_OUTPUT_TOKENS", "8192")
    monkeypatch.setenv("CLAUDE_MAX_OUTPUT_TOKENS", "16384")
    assert config.get_max_output_tokens() == 8192
    assert config.get_claude_max_output_tokens() == 16384
