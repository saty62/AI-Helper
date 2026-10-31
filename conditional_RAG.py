import os
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def build_retriver(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    return retriever

base_dir = os.path.dirname(os.path.abspath(__file__))
academic_retriever = build_retriver(os.path.join(base_dir, "academics_handbook.pdf"))
fee_retriever = build_retriver(os.path.join(base_dir, "fee_structure.pdf"))

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.5)
#step 2: state 

class State(TypedDict):
    programme : str
    messages : Annotated[list, add_messages]
    query_type : str
    retrieved_context : str
    
    #step 3: nodes generation
    
def classifier_node(state: State)->dict:
    """Look at the latest user message and decide which path to take."""
    
    last_message = state['messages'][-1].content
    prompt = (
    "Classify the following student query into exactly one category: "
    "'academic', 'fee', or 'general'.\n\n"
    "Use 'academic' for questions about attendance, exams, grading, credits, "
    "promotion, course structure, summer training, or degree requirements.\n"
    "Use 'fee' for questions about tuition, payment, refund, late charges, "
    "scholarships, or any money-related topic.\n"
    "Use 'general' for greetings, casual talk, or anything not related to "
    "the college rules or fee.\n\n"
    f"Query: {last_message}\n\n"
    "Return only one word: academic, fee, or general."
)
    response = llm.invoke(prompt)
    category = response.content.strip().lower()
    #full proof function to ensure the category is one of the expected values
    if "academic" in category:
        category = "academic"
    elif "fee" in category:
        category = "fee"
    else:
        category = "general"
    return {"query_type": category}

def academic_rag_node(state: State)->dict:
    """Retrieve relevant academic information based on the user's query."""
    last_message = state['messages'][-1].content
    retrieved_docs = academic_retriever.invoke(last_message)
    retrieved_texts = [doc.page_content for doc in retrieved_docs]
    retrieved_context = "\n\n".join(retrieved_texts)
    return {"retrieved_context": retrieved_context}

def fee_rag_node(state: State)->dict:
    """Retrieve relevant fee information based on the user's query."""
    last_message = state['messages'][-1].content
    retrieved_docs = fee_retriever.invoke(last_message)
    retrieved_texts = [doc.page_content for doc in retrieved_docs]
    retrieved_context = "\n\n".join(retrieved_texts)
    return {"retrieved_context": retrieved_context}

def general_node(state: State)->dict:
    """Handle general queries without retrieval."""
    return {"retrieved_context": "No specific information retrieved for general queries."}


def response_node(state: State) -> dict:
    """Generates the final answer, personalized using the student's programme."""
    query = state["messages"][-1].content
    programme = state.get("programme", "BCA")
    context = state["retrieved_context"]

    if context == "NO_RETRIEVAL_NEEDED":
        prompt = (
            f"You are a friendly college assistant talking to a {programme} student. "
            f"Answer this question using your own general knowledge:\n\n{query}"
        )
    else:
        prompt = (
            f"You are a college assistant helping a {programme} student. "
            f"Use the following context from the official college documents "
            f"to answer the student's question.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}"
            f"Give a clear, friendly, and precise answer."
        )
    response = llm.invoke(prompt)
    return {"messages": [("ai", response.content.strip())]}
    
    #router function
    
def route_query(state: State):
    """Route the query to the appropriate node based on the query type."""
    query_type = state.get("query_type", "general")
    if query_type == "academic":
        return "academic_rag"
    elif query_type == "fee":
        return "fee_rag"
    else:
        return "general"
    #buildeing the graph
    
graph = StateGraph(State)

graph.add_node("classifier", classifier_node)
graph.add_node("academic_rag", academic_rag_node)
graph.add_node("fee_rag", fee_rag_node)
graph.add_node("general", general_node)
graph.add_node("response", response_node)

#edges
graph.add_edge(START, "classifier")
graph.add_conditional_edges(
    "classifier", route_query
)

graph.add_edge("academic_rag", "response")
graph.add_edge("fee_rag", "response")
graph.add_edge("general", "response")

graph.add_edge("response", END)

app= graph.compile()
# run the code

print("Welcome to the College Assistant! Type 'exit' to quit.")
print("1. BCA")
print("2. BBA")
print("3. B.com(Hons)")

choice = input("\n Enter 1, 2 or 3")
programme_map = {"1": "BCA", "2": "BBA", "3": "B.com(Hons)"}

student_programme = programme_map.get(choice, "BCA")

print(f"\n Great! you have set as a {student_programme} student.")

while True:
    user_input = input("\nYou: ")
    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    result = app.invoke({
        "programme": student_programme,
        "messages": [("human", user_input)]
    })
    print(f"Assistant: {result['messages'][-1].content}")