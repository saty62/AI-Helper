import os
from typing import TypedDict, Annotated

import streamlit as st
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# PAGE CONFIG + STYLING
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="College Assistant",
    page_icon="🎓",
    layout="centered",
)

st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.2rem;
            font-weight: 700;
            background: linear-gradient(90deg, #4f46e5, #06b6d4);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0;
        }
        .sub-title {
            color: #6b7280;
            margin-top: 0;
            margin-bottom: 1rem;
        }
        .badge {
            display: inline-block;
            padding: 2px 10px;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 600;
            color: white;
            margin-bottom: 6px;
        }
        .badge-academic { background: #4f46e5; }
        .badge-fee      { background: #059669; }
        .badge-general  { background: #6b7280; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# BACKEND (same logic as your original code, unchanged)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading models and documents...")
def load_app():
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

    # step 2: state
    class State(TypedDict):
        programme: str
        messages: Annotated[list, add_messages]
        query_type: str
        retrieved_context: str

    # step 3: nodes generation
    def classifier_node(state: State) -> dict:
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
        # full proof function to ensure the category is one of the expected values
        if "academic" in category:
            category = "academic"
        elif "fee" in category:
            category = "fee"
        else:
            category = "general"
        return {"query_type": category}

    def academic_rag_node(state: State) -> dict:
        """Retrieve relevant academic information based on the user's query."""
        last_message = state['messages'][-1].content
        retrieved_docs = academic_retriever.invoke(last_message)
        retrieved_texts = [doc.page_content for doc in retrieved_docs]
        retrieved_context = "\n\n".join(retrieved_texts)
        return {"retrieved_context": retrieved_context}

    def fee_rag_node(state: State) -> dict:
        """Retrieve relevant fee information based on the user's query."""
        last_message = state['messages'][-1].content
        retrieved_docs = fee_retriever.invoke(last_message)
        retrieved_texts = [doc.page_content for doc in retrieved_docs]
        retrieved_context = "\n\n".join(retrieved_texts)
        return {"retrieved_context": retrieved_context}

    def general_node(state: State) -> dict:
        """Handle general queries without retrieval."""
        return {"retrieved_context": "No specific information retrieved for general queries."}

    def response_node(state: State) -> dict:
        """Generates the final answer, personalized using the student's programme."""
        query = state["messages"][-1].content
        programme = state.get("programme", "Unknown")
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

    # router function
    def route_query(state: State):
        """Route the query to the appropriate node based on the query type."""
        query_type = state.get("query_type", "general")
        if query_type == "academic":
            return "academic_rag"
        elif query_type == "fee":
            return "fee_rag"
        else:
            return "general"

    # building the graph
    graph = StateGraph(State)

    graph.add_node("classifier", classifier_node)
    graph.add_node("academic_rag", academic_rag_node)
    graph.add_node("fee_rag", fee_rag_node)
    graph.add_node("general", general_node)
    graph.add_node("response", response_node)

    # edges
    graph.add_edge(START, "classifier")
    graph.add_conditional_edges("classifier", route_query)

    graph.add_edge("academic_rag", "response")
    graph.add_edge("fee_rag", "response")
    graph.add_edge("general", "response")

    graph.add_edge("response", END)

    return graph.compile()


app = load_app()

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
programme_map = {"BCA": "BCA", "BBA": "BBA", "B.Com (Hons)": "B.com(Hons)"}

CATEGORY_LABELS = {
    "academic": "📚 Academic",
    "fee": "💰 Fee",
    "general": "💬 General",
}

# --- Sidebar ---
with st.sidebar:
    st.markdown("## 🎓 College Assistant")
    st.caption("Ask about academic rules and fee details.")
    st.divider()

    selected_label = st.selectbox(
        "Your programme",
        options=list(programme_map.keys()),
        index=0,
    )
    student_programme = programme_map[selected_label]

    st.divider()
    st.markdown("**Try asking:**")
    suggestions = [
        "What is the minimum attendance required?",
        "What is the late fee charge?",
        "How is the CGPA calculated?",
        "Is there a refund policy?",
    ]
    for s in suggestions:
        if st.button(s, use_container_width=True, key=f"sugg_{s}"):
            st.session_state["pending_prompt"] = s

    st.divider()
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state["chat"] = []
        st.rerun()

# --- Header ---
st.markdown('<p class="main-title">🎓 College Assistant</p>', unsafe_allow_html=True)
st.markdown(
    f'<p class="sub-title">Hi! You are set as a <b>{selected_label}</b> student. '
    f"Ask me anything about academics or fees.</p>",
    unsafe_allow_html=True,
)

# --- Chat state ---
if "chat" not in st.session_state:
    st.session_state["chat"] = []

if not st.session_state["chat"]:
    st.info("👋 Welcome! Type a question below or pick a suggestion from the sidebar.")

# --- Render history ---
for msg in st.session_state["chat"]:
    avatar = "🧑‍🎓" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        if msg["role"] == "assistant" and msg.get("category"):
            cat = msg["category"]
            st.markdown(
                f'<span class="badge badge-{cat}">{CATEGORY_LABELS[cat]}</span>',
                unsafe_allow_html=True,
            )
        st.markdown(msg["content"])

# --- Input ---
user_input = st.chat_input("Type your question here...")
if "pending_prompt" in st.session_state:
    user_input = st.session_state.pop("pending_prompt")

if user_input:
    st.session_state["chat"].append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking..."):
            try:
                result = app.invoke({
                    "programme": student_programme,
                    "messages": [("human", user_input)],
                })
                answer = result["messages"][-1].content
                category = result.get("query_type", "general")
            except Exception as e:
                answer = f"⚠️ Something went wrong: {e}"
                category = None

        if category:
            st.markdown(
                f'<span class="badge badge-{category}">{CATEGORY_LABELS[category]}</span>',
                unsafe_allow_html=True,
            )
        st.markdown(answer)

    st.session_state["chat"].append(
        {"role": "assistant", "content": answer, "category": category}
    )