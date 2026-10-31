import os
from typing import TypedDict
#Lets create the state first

class pipelinestate(TypedDict):
    raw_input: str
    edited_text: str
    script_text: str
    final_output: str
    
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.7)

def editor_node(state: pipelinestate) -> dict:
    # Use the LLM to edit the raw input text
   prompt = ("you are an expertor. clean up the following raw input."
             "fix any grammatical errors, spelling mistakes, and smooth out the transition flow"
             "while keeping the core message intact. Return only the edited text.\n\n"
             f"Text:\n{state['raw_input']}")
   response = llm.invoke(prompt)
   return {"edited_text": response.content.strip()}

def scriptwriter_node(state: pipelinestate) -> dict:
    # Use the LLM to generate a script based on the edited text
    print("\n stage 2: scriptwriter_node-executing")
    prompt = ("you are an expert scriptwriter. write a script based on the following edited text."
              "Return only the script text.\n\n"
              f"Edited Text:\n{state['edited_text']}")
    response = llm.invoke(prompt)
    return {"script_text": response.content.strip()}

def translator_node(state: pipelinestate) -> dict:
    # Use the LLM to translate the script text into another language
    print("\n stage 3: translator_node-executing")
    prompt = ("you are an expert translator. translate the following script text into Spanish."
              "Return only the translated text.\n\n"
              f"Script Text:\n{state['script_text']}")
    response = llm.invoke(prompt)
    return {"final_output": response.content.strip()}

#now your states and nodes are ready to be used in the graph. you can now create a graph and add these nodes to it.
#edges are very important in the graph. they define the flow of data between nodes. you can create edges between nodes to define the flow of data 
from langgraph.graph import StateGraph, START, END
#CREATE THE GRAPH ADDING THE NODES IN OUR GRAPH
graph = StateGraph(pipelinestate)
graph.add_node("editor", editor_node)
graph.add_node("scriptwriter", scriptwriter_node)
graph.add_node("translator", translator_node)
#add edges to define the flow of data between nodes
graph.add_edge(START, "editor")
graph.add_edge("editor", "scriptwriter")
graph.add_edge("scriptwriter", "translator")
graph.add_edge("translator", END)
#compile the graph
app = graph.compile()
result = app.invoke({
    "raw_input": "This is a sample raw input text that needs to be edited, scripted, and translated into Spanish."
})

print("Final Output:", result["final_output"])