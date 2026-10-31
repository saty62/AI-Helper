import os
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END

load_dotenv()
llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.1)

def merge_score_dicts(existing: dict, newupdate: dict) -> dict:
    if existing is None:
        return newupdate
    return {** existing, **newupdate}

#create a state for the graph
class AnalyzerState(TypedDict):
    raw_input: str
    safety_score: Annotated[dict[str, int], merge_score_dicts]
    
#nodes
def toxicity_node(state: AnalyzerState) -> dict:
    prompt = ("you are an expert in content moderation. analyze the following text for toxicity."
              "Return a dictionary with the toxicity score as an integer between 0 and 100.\n\n"
              f"Text:\n{state['raw_input']}")
    response = llm.invoke(prompt)
    try:
        score=int(response.content.strip())
    except ValueError:
        score=0
        #return a sub-dictonary under our single state key
    return {"safety_score": {"toxicity": score}}

def copyright_node(state: AnalyzerState) -> dict:
    prompt = ("you are an expert in copyright law. analyze the following text for potential copyright infringement."
              "Return a dictionary with the copyright score as an integer between 0 and 100.\n\n"
              f"Text:\n{state['raw_input']}")
    response = llm.invoke(prompt)
    try:
        score=int(response.content.strip())
    except ValueError:
        score=0
    return {"safety_score": {"copyright": score}}

def culture_node(state: AnalyzerState) -> dict:
    prompt = ("you are an expert in cultural sensitivity. analyze the following text for cultural insensitivity."
              "Return a dictionary with the cultural sensitivity score as an integer between 0 and 100.\n\n"
              f"Text:\n{state['raw_input']}")
    response = llm.invoke(prompt)
    try:
        score=int(response.content.strip())
    except ValueError:
        score=0
    return {"safety_score": {"cultural_sensitivity": score}}

builder = StateGraph(AnalyzerState)
builder.add_node("toxicity", toxicity_node)
builder.add_node("copyright", copyright_node)
builder.add_node("culture", culture_node)

builder.add_edge(START, "toxicity")
builder.add_edge(START, "copyright")        
builder.add_edge(START, "culture")
builder.add_edge("toxicity", END)
builder.add_edge("copyright", END)
builder.add_edge("culture", END)
app=builder.compile()
sample_script = "This is a sample script that needs to be analyzed for toxicity, copyright infringement, and cultural insensitivity."
initial_state = {"raw_input": sample_script, "safety_score": {}}
final_state = app.invoke(initial_state)
print("Final State:", final_state)