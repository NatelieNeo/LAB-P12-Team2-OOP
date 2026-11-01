
from abc import ABC 


class Field(ABC): 
    def __init__(self, name, label, replies: dict | None = None):
        self.name = name
        self.label = label
        self.replies = replies or {}