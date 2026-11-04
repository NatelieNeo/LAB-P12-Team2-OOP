import os
import logging
from abc import ABC, abstractmethod

from dotenv import load_dotenv
from google import genai

from config import GEMINI_MODEL, MAX_RETRIES
from manager.ai_handler_manager import parse_response
from manager.meal_prompt_builder import MealPromptBuilder

logger = logging.getLogger(__name__)

class AIClient(ABC):

    @abstractmethod
    def generate(self, prompt):
        pass


# Handles talking to the Gemini API
class GeminiClient(AIClient):

    def __init__(self, model=GEMINI_MODEL):
        load_dotenv() # Load environment variables from .env file

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY was not found in the .env file.")

        self.__model = model
        self.__client = genai.Client(api_key=api_key)

    @property
    def model(self):
        return self.__model
    
    def generate(self, prompt):
        response = self.__client.models.generate_content(
            model=self.__model,
            contents=prompt,
            config={
                "response_mime_type": "application/json"
            }
        )
        return response.text


# Builds the prompt, calls the AI and retries if the response is invalid
class AIManager:

    def __init__(self, client, max_retries=MAX_RETRIES):
        self.__client = client  # passed in GeminiClient()
        self.__prompt_builder = MealPromptBuilder()
        self.__max_retries = max_retries

    @property
    def max_retries(self):
        return self.__max_retries

    # Save the raw response so we can check it when debugging
    def __save_raw_response(self, raw_text):
        with open("ai_response.json", "w") as f:
            f.write(raw_text)

    # One call to the AI, returns the validated dict or None
    def __attempt(self, prompt, attempt):
        try:
            logger.info(
                "Calling AI API - attempt %s/%s",
                attempt,
                self.__max_retries
            )

            raw_text = self.__client.generate(prompt)
            self.__save_raw_response(raw_text)

            # convert the response to JSON and validate it
            result = parse_response(raw_text)

            if result is None:
                logger.warning(
                    "Invalid AI response on attempt %s.",
                    attempt
                )
            return result

        except Exception as error:
            logger.error(
                "AI API error on attempt %s: %s",
                attempt,
                error
            )
            return None