<!-- Threading (parallel AI) text for README.md, grouped by the README's numbered sections. Paste each part under its heading. -->

> Historical notes for the retained two-Gemini first-valid race. The current
> Gemini + Claude comparison feature and configuration are documented in
> [the project README](../README.md). The new multi mode collects both answers
> and scores them; it does not use first-response selection.

## 1. What changed in Part 2

**Advanced feature: threading (parallel AI models).** `ParallelAIProvider` asks two Gemini models for a meal plan at the same time and uses the first valid answer. If one model is slow, unreachable or returns a badly formatted response, the other is the safety net, so the user waits less and sees fewer failures.

- `ParallelAIProvider` (`manager/ai_parallel.py`) is itself an `AIProvider`, so `AIManager` uses it exactly like a single model and did not need to change (Strategy pattern, open/closed principle).
- It starts one thread per model with a `ThreadPoolExecutor`. Threads suit this job because the work is waiting on network calls, not CPU.
- Each answer is checked with `AIHandlerManager.is_valid`. The first answer that passes is returned immediately; the slower thread is ignored.
- If every model fails, it raises `AllProvidersFailedError` (at least one connection problem or timeout) or `AIResponseError` (every model answered, none validly). `AIManager` then retries as usual.
- `AppFactory.build_ai_provider` (`manager/controller.py`) wires the primary and fallback models together.

## 2. Structure

```
manager/ai_parallel.py         ParallelAIProvider (two models race in threads, first valid answer wins)
manager/exceptions.py          + AllProvidersFailedError
manager/ai_handler_manager.py  + is_valid
manager/controller.py          + AppFactory.build_ai_provider
tests/test_parallel_ai.py      35 tests for the parallel provider
docs/uml/                      fig3 (AI classes), fig4 (exceptions), fig5 (threading), fig6 (parallel sequence)
docs/exception_matrix_parallel_ai.md
```

## 3. Configuration

| Variable | Required | Purpose |
|---|---|---|
| `GEMINI_FALLBACK_MODEL` | No | Second model for the parallel run (default `gemini-2.5-flash`, set in `config.py`; check it is available to your key) |
| `CALORA_PARALLEL_AI` | No | `1` (default) asks both models at once; `0` uses only the primary model |

`PARALLEL_TIMEOUT_SECONDS` in `config.py` (default 60) is the longest one attempt waits for the models.

## 5. Tests (no API key or network needed)

```
python -m pytest tests/test_parallel_ai.py -v
```

35 tests cover the parallel AI provider: the fast model wins, both models really run at the same time, failover when one model errors, answers invalidly, crashes or hangs, every model failing, timeouts, non-retryable errors, retries through `AIManager`, a full plan built with the primary model down, thread safety, and the `AppFactory` wiring. The AI is replaced by fake providers, so no test calls Gemini; slow models are simulated with `threading.Event`, so the tests do not depend on timing.

## 8. Known limitations

- Parallel mode sends every request to two models, so it uses twice the API quota.
- A request still running when the other model wins cannot be cancelled; its answer is ignored.
- Both models use the same API key, so an invalid key or exhausted quota fails both.
