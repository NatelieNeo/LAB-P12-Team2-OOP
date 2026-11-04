import os
from abc import ABC, abstractmethod

from dotenv import load_dotenv
from google import genai

from config import GEMINI_MODEL


class AIClient(ABC):

    @abstractmethod
    def generate(self, prompt):
        pass


class GeminiClient(AIClient):

    def __init__(self, model=GEMINI_MODEL):
        load_dotenv() 

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