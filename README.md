# AI Helper – Agentic College Assistant

An AI-powered college assistant that helps students get quick answers to academic, fee-related, and general questions through a conversational Streamlit interface.

The application uses **LangGraph** to classify each query and route it to the appropriate workflow. For academic and fee questions, it retrieves relevant passages from separate PDF documents using **Hugging Face sentence embeddings** and **FAISS**, then uses a **Groq-hosted language model** to generate a clear response.

## Features

- **Query classification:** Categorizes incoming questions as academic, fee-related, or general.
- **Academic document search:** Retrieves relevant information from `academics_handbook.pdf`.
- **Fee document search:** Retrieves relevant information from `fee_structure.pdf`.
- **Semantic retrieval:** Uses the `sentence-transformers/all-MiniLM-L6-v2` embedding model with FAISS.
- **General conversation:** Handles greetings and general questions without searching the college PDFs.
- **Student context:** Lets users select a programme (BCA, BBA, or B.Com (Hons)).
- **Chat interface:** Provides suggested questions, conversation history, category labels, and a clear-chat option.

## How It Works

1. A student selects their programme and enters a question.
2. A LangGraph classification node asks the language model to label the query as `academic`, `fee`, or `general`.
3. LangGraph routes the query:
   - **Academic:** Searches the academic handbook.
   - **Fee:** Searches the fee structure document.
   - **General:** Skips PDF retrieval.
4. For document-based questions, the selected PDF has already been loaded, split into chunks, embedded, and indexed in FAISS. The retriever returns the three most relevant chunks.
5. The response node sends the question, programme context, and retrieved text (when applicable) to the Groq model and displays the answer in Streamlit.

## Tech Stack

- **Language:** Python
- **UI:** Streamlit
- **Agent orchestration:** LangGraph
- **LLM integration:** LangChain, ChatGroq
- **Embeddings:** Hugging Face Sentence Transformers
- **Vector store:** FAISS
- **PDF loading:** PyPDF
- **Configuration:** python-dotenv

## Project Structure

```text
AI-Helper/
├── app.py                  # Main Streamlit college assistant
├── academics_handbook.pdf  # Academic rules and information
├── fee_structure.pdf       # College fee information
├── conditional_RAG.py      # Command-line conditional RAG workflow
├── project.py              # Sequential LLM pipeline example
├── states.py               # Examples of LangGraph state definitions
├── iterative_tools.py      # Iterative tool-use example
├── parallel_reducers.py    # Parallel workflow/reducer example
├── requirements.txt        # Python dependencies
└── .env                    # Your local API key (create this file)
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/saty62/AI-Helper.git
cd AI-Helper
```

### 2. Create and activate a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure your Groq API key

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Get an API key from the [Groq Console](https://console.groq.com/). Keep the key private and do not commit the `.env` file.

### 5. Run the Streamlit application

```bash
streamlit run app.py
```

Streamlit will print a local URL in the terminal (typically `http://localhost:8501`). Open that URL in your browser.

## Example Questions

- What is the minimum attendance requirement?
- How is CGPA calculated?
- What is the late fee charge?
- Is there a fee refund policy?
- What are the academic requirements for my programme?

The suggested questions are examples; the answer depends on the contents of the PDF documents included in the repository.

## Configuration Notes

- The application expects `academics_handbook.pdf` and `fee_structure.pdf` in the same directory as `app.py`.
- On first launch, the app loads the embedding model and builds in-memory FAISS indexes from both PDFs. This may take a little time and requires an internet connection if the model is not already cached.
- The chat model is configured in `app.py` as `openai/gpt-oss-120b` through Groq. Confirm that this model is available to your Groq account if you encounter a model-related API error.
- The programme selection currently supports BCA, BBA, and B.Com (Hons). Update `programme_map` in `app.py` to change these options.

## Additional Workflow Examples

This repository also contains smaller LangGraph/LangChain learning examples:

- `conditional_RAG.py`: Runs the academic/fee/general conditional retrieval workflow in the terminal.
- `project.py`: Demonstrates a sequential LLM pipeline for editing text, writing a script, and translating it into Spanish.
- `states.py`: Contains examples of state structures using TypedDict, Pydantic, and dataclasses.
- `iterative_tools.py` and `parallel_reducers.py`: Explore additional workflow patterns.

The main user-facing application is `app.py`.

## Limitations

- Answers to academic and fee questions depend on the content and quality of the supplied PDFs.
- General questions are answered using the language model without college-document retrieval, so they should not be treated as official institutional guidance.
- The FAISS indexes are rebuilt at app startup and are not persisted between runs.

## Future Improvements

- Add source-page references to document-grounded answers.
- Persist vector indexes to avoid rebuilding them on every fresh deployment.
- Add support for more programmes and institutional documents.
- Add evaluation tests for query classification and retrieval quality.

## Author

**Priyadarshan Satyam Bharti**

- GitHub: [@saty62](https://github.com/saty62)
- Repository: [AI-Helper](https://github.com/saty62/AI-Helper)
