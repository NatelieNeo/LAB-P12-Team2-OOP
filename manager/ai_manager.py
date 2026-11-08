# AI Manager: builds the prompt, calls an AI provider, retries, and hands back candidates
import logging
import os
import time
from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, Union

from config import GEMINI_MODEL, MAX_RETRIES, RETRY_DELAY_SECONDS
# Changed from parse_response() function to AIHandlerManager class
from manager.ai_handler_manager import AIHandlerManager
from manager.exceptions import AIConnectionError, AIResponseError
from manager.models import MealCandidate, MealPlanRequest, NutritionTargets

logger = logging.getLogger(__name__)


# Any AI provider just needs to turn a prompt into response text
# Changed from AIClient to AIProvider
class AIProvider(ABC):

    @abstractmethod
    def generate(self, prompt: str) -> str:
        # should raise AIConnectionError if the AI can't be reached
        pass


# Talks to Google Gemini, API key is kept private
# Changed from GeminiClient to GeminiProvider
class GeminiProvider(AIProvider):

    # Changed from load_dotenv() + required key in __init__ to optional api_key argument
    def __init__(self, api_key: Optional[str] = None, model: str = GEMINI_MODEL):
        self.__api_key = api_key if api_key is not None else os.getenv("GEMINI_API_KEY")
        self._model = model
        self._client = None

    # Changed from creating the client in __init__ to creating it on first use
    def _get_client(self):
        # Changed from raising ValueError to raising AIConnectionError
        if not self.__api_key:
            raise AIConnectionError("GEMINI_API_KEY was not found. Add it to your .env file.", retryable=False)
        if self._client is None:
            try:
                from google import genai
                self._client = genai.Client(api_key=self.__api_key)
            except Exception as error:
                raise AIConnectionError(f"Failed to initialise the Gemini client: {error}", retryable=False) from error
        return self._client

    def generate(self, prompt: str) -> str:
        client = self._get_client()
        # Changed from letting API errors bubble up to raising AIConnectionError
        try:
            response = client.models.generate_content(
                model=self._model,
                contents=prompt,
                # Changed from only response_mime_type to also setting temperature and seed
                config={"response_mime_type": "application/json", "temperature": 1, "seed": 42},
            )
        except Exception as error:
            raise AIConnectionError(f"Gemini request failed: {error}") from error
        return response.text
