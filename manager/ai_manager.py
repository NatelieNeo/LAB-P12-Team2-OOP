import os
from abc import ABC, abstractmethod

from dotenv import load_dotenv
from google import genai

from config import GEMINI_MODEL, MAX_RETRIES
from manager.meal_prompt_builder import MealPromptBuilder


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