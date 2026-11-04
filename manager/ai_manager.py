from abc import ABC, abstractmethod

# Any AI client needs a generate() method that returns the raw response text
class AIClient(ABC):

    @abstractmethod
    def generate(self, prompt):
        pass