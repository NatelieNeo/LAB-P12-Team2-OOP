
from abc import ABC, abstractmethod

class InvalidInput(ValueError): #Raised when an input is rejected. Blank is allowed
    def __init__(self, message: str, keep_null: bool =False):
        super().__init__(message)
        self.keep_null = keep_null


class Field(ABC): 
    def __init__(self, name, label, replies: dict | None = None):
        self.name = name
        self.label = label
        self.replies = replies or {}

    def ask(self, allow_null=False):
        while True:
            raw = input(self.label)
            if allow_null and self._is_null(raw):
                return None
            try:
                value = self.convert(raw) #each subclass has its own rules
            except InvalidInput as e:
                print(e)
                allow_null = allow_null and e.keep_null
                continue 
            self._reply(value)  # Great! Hi Bro etc.
            return value          

    def applies_to(self, record: dict) -> bool: #Check if this field applies to the given record
       
        return True  # By default, all fields apply to all records

    

