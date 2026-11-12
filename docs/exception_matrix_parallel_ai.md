# Exception matrix: parallel AI (threading feature)

| # | What goes wrong | Exception | Raised in | Handled in | What happens next |
|---|---|---|---|---|---|
| 1 | One model cannot be reached (network, quota, API error) | `AIConnectionError` | `GeminiProvider.generate` (worker thread) | `ParallelAIProvider.generate` | Recorded in `failures` and logged as a warning. The other model's answer is used, so the user notices nothing. |
| 2 | One model answers, but the text is empty, not JSON or the wrong format | `AIResponseError` | `ParallelAIProvider._ask`, after `AIHandlerManager.is_valid` returns `False` | `ParallelAIProvider.generate` | Same as row 1: the other model is the safety net. |
| 3 | One model crashes with an unexpected error | any `Exception` | worker thread | `ParallelAIProvider.generate` | Treated as a failure of that model only; the other model can still win. |
| 4 | One model is still running when the timeout (`PARALLEL_TIMEOUT_SECONDS`, 60 s) ends | recorded as `AIConnectionError` ("did not answer within ...") | `ParallelAIProvider.generate` | `ParallelAIProvider.generate` | Only matters if the other model also failed. The slow thread cannot be cancelled; its result is ignored. |
| 5 | Every model failed, and at least one failure was a connection error or timeout | `AllProvidersFailedError` (an `AIConnectionError`, carries `failures`) | `ParallelAIProvider.generate` | `AIManager.get_recommendations` | Retried up to `MAX_RETRIES` (3) times, `RETRY_DELAY_SECONDS` apart. If still failing, re-raised and the plan job is marked FAILED with the message. |
| 6 | Every model answered, but none validly | `AIResponseError` | `ParallelAIProvider.generate` | `AIManager.get_recommendations` | Retried up to `MAX_RETRIES` times, then re-raised and the plan job is marked FAILED. |
| 7 | No model could ever succeed (for example `GEMINI_API_KEY` is missing) | `AllProvidersFailedError` with `retryable = False` | `GeminiProvider._get_client`, then `ParallelAIProvider.generate` | `AIManager.get_recommendations` | Re-raised immediately with no retry, because retrying cannot help. |
| 8 | `ParallelAIProvider` is built with an empty provider list | `ValueError` | `ParallelAIProvider.__init__` | not handled (programming error) | Cannot happen through `AppFactory`, which always passes two providers. |
| 9 | Parallel mode is off (`CALORA_PARALLEL_AI=0`), or the fallback model is blank or the same as the primary | none | `AppFactory.build_ai_provider` | n/a | The app falls back to the single primary `GeminiProvider`. |

`AllProvidersFailedError.retryable` is `True` when at least one model's failure was retryable.

Every row is covered by a test in `tests/test_parallel_ai.py`. The exception classes are drawn in `docs/uml/fig4_exceptions.mmd`.
