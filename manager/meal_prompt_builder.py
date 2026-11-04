from abc import ABC, abstractmethod


# Base class for prompt builders, subclasses need to implement build()
class PromptBuilder(ABC):

    @abstractmethod
    def build(self, input_record):
        pass