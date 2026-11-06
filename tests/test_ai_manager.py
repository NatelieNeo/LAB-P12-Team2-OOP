"""AIManager, providers and prompt. The live API is never contacted."""
import json
import time

import pytest

from manager.ai_manager import AIManager, AIProvider, GeminiProvider, PromptBuilder, StubAIProvider
from manager.exceptions import AIConnectionError, AIResponseError, CaloraError
from manager.logic_manager import NutritionCalculator
from tests.conftest import make_profile, make_request


@pytest.fixture
def targets():
    return NutritionCalculator().targets(make_profile())


def manager(provider, retries=3):
    return AIManager(provider, max_retries=retries, retry_delay=0)


def test_returns_validated_candidates(response_text, request_both, targets):
    provider = StubAIProvider(response_text)
    candidates = manager(provider).get_recommendations(request_both, targets)
    assert len(candidates) == 10 and provider.call_count == 1


def test_prompt_contains_calculated_targets_and_user_details(request_both, targets):
    prompt = PromptBuilder().build(request_both, targets)
    assert "Name: Nisha" in prompt and "Meal source: both" in prompt
    assert "rice dishes" in prompt and "SGD 10.5" in prompt
    assert f"{targets.daily_calories:.0f} kcal" in prompt
    assert "exactly 10 food recommendations" in prompt
    assert '"recommendations"' in prompt  # the JSON template survived formatting


def test_prompt_handles_missing_values(targets):
    request = make_request(make_profile(), meal_preference="", pantry_ingredients=[], daily_budget=None)
    prompt = PromptBuilder().build(request, targets)
    assert "Dietary restrictions/allergies: none" in prompt
    assert "Available pantry ingredients: none" in prompt


def test_in_and_out_use_the_users_wording(targets):
    prompt = PromptBuilder().build(make_request(make_profile(), meal_source="home_cooked"), targets)
    assert "Meal source: in" in prompt


def test_retries_after_a_malformed_response(response_text, request_both, targets):
    provider = StubAIProvider(["not json", "{}", response_text])
    assert len(manager(provider).get_recommendations(request_both, targets)) == 10
    assert provider.call_count == 3


def test_retries_after_a_connection_error(response_text, request_both, targets):
    provider = StubAIProvider([AIConnectionError("timeout"), response_text])
    assert len(manager(provider).get_recommendations(request_both, targets)) == 10
    assert provider.call_count == 2


def test_gives_up_with_response_error_after_max_attempts(request_both, targets):
    provider = StubAIProvider("not json")
    with pytest.raises(AIResponseError):
        manager(provider).get_recommendations(request_both, targets)
    assert provider.call_count == 3


def test_gives_up_with_connection_error_after_max_attempts(request_both, targets):
    provider = StubAIProvider(AIConnectionError("down"))
    with pytest.raises(AIConnectionError):
        manager(provider, retries=2).get_recommendations(request_both, targets)
    assert provider.call_count == 2


def test_a_configuration_error_is_not_retried(request_both, targets):
    provider = StubAIProvider(AIConnectionError("no key", retryable=False))
    with pytest.raises(AIConnectionError):
        manager(provider).get_recommendations(request_both, targets)
    assert provider.call_count == 1


def test_all_failures_are_calora_errors(request_both, targets):
    with pytest.raises(CaloraError):
        manager(StubAIProvider("")).get_recommendations(request_both, targets)


def test_retry_delay_is_applied_between_attempts(request_both, targets):
    provider = StubAIProvider("not json")
    slow = AIManager(provider, max_retries=3, retry_delay=0.05)
    start = time.monotonic()
    with pytest.raises(AIResponseError):
        slow.get_recommendations(request_both, targets)
    assert time.monotonic() - start >= 0.09


def test_stub_repeats_its_last_response():
    stub = StubAIProvider(["a", "b"])
    assert [stub.generate("x") for _ in range(3)] == ["a", "b", "b"]


def test_gemini_provider_without_a_key_fails_cleanly(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(AIConnectionError) as error:
        GeminiProvider().generate("hello")
    assert error.value.retryable is False and "GEMINI_API_KEY" in str(error.value)


def test_gemini_provider_wraps_sdk_failures():
    class BrokenClient:
        class models:
            @staticmethod
            def generate_content(**kwargs):
                raise TimeoutError("network down")

    provider = GeminiProvider(api_key="test-key")
    provider._client = BrokenClient()
    with pytest.raises(AIConnectionError) as error:
        provider.generate("hello")
    assert error.value.retryable is True and "network down" in str(error.value)


def test_gemini_provider_returns_the_response_text():
    class FakeResponse:
        text = json.dumps({"ok": True})

    class FakeClient:
        class models:
            @staticmethod
            def generate_content(**kwargs):
                assert kwargs["config"]["response_mime_type"] == "application/json"
                return FakeResponse()

    provider = GeminiProvider(api_key="test-key")
    provider._client = FakeClient()
    assert provider.generate("hello") == '{"ok": true}'


def test_api_key_is_private_and_not_in_repr():
    provider = GeminiProvider(api_key="secret-key")
    assert "secret-key" not in repr(vars(provider).get("_model"))
    assert not hasattr(provider, "api_key")


def test_providers_share_one_interface():
    assert issubclass(GeminiProvider, AIProvider) and issubclass(StubAIProvider, AIProvider)
