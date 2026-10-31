# we are creating a graph 
# and the first thing you create is a state

import os
#1) typed DICT
from typing import TypedDict #most common approach for creating a state

class State(TypedDict):
    topic: str
    summary: str
    score: int
    
#2)pydantic approach
#it is good at data validation and type checking at runtime
from pydantic import BaseModel, field_validator  

class State(BaseModel):
    topic: str
    summary: str=""
    score: int
    
    @field_validator("score")
    def score_positive(cls, v):
        if v < 0:
            raise ValueError("score must be positive")
        
#python dataclassess
#standard python dataclass but it is used rarely
from dataclasses import dataclass, field
@dataclass
class State:
    topic: str=""
    summary: str=""
    message: list= field(default_factory=list)
    
    from langgraph.graph import MessagesState
    class State(MessagesState):
        user_name: str
        language: str