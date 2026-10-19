# Document Q&A Chatbot — Local RAG with Mistral 7B

An offline Retrieval-Augmented Generation (RAG) assistant that answers questions from PDF and Word documents. Answers are grounded in the uploaded text, and the language model runs locally, so no document ever leaves the machine.

> Built in 2025 during a remote AI internship at Ovivia AI (Qatar), as the sole developer from requirements to testing. Designed for maintenance technicians who need quick answers from long equipment manuals.

## How it works

```
Document (PDF / DOCX)
      │  text extraction (pdfplumber, docx2txt)
      ▼
Token-based chunking (~200 tokens)
      │  embeddings: Sentence Transformers (all-MiniLM-L6-v2)
      ▼
FAISS vector index
      │  user question → embedded → cosine similarity → top 3–5 chunks
      ▼
Prompt = retrieved context + question
      │  Mistral 7B (quantized, run locally with llama.cpp)
      ▼
Grounded answer
```

## Features

- Upload PDF and Word documents and ask questions in natural language
- Answers use **only** the retrieved context. When the answer isn't in the documents, the assistant replies with a fixed "not available" message instead of guessing
- Fully local inference: no external API calls, no cloud services
- **Admin dashboard:** document upload and management, user management, and a page to review and correct chatbot answers
- User authentication with hashed passwords (Flask-Login, Werkzeug)
- Temporary user uploads are deleted when the session ends

## Tech stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask, Flask-Login, Flask-SQLAlchemy |
| Retrieval | Sentence Transformers (`all-MiniLM-L6-v2`), FAISS |
| Generation | Mistral 7B (quantized) via llama.cpp |
| Document parsing | pdfplumber, docx2txt |
| Database | PostgreSQL |
| Frontend | HTML, CSS, JavaScript, Bootstrap |

## Getting started

**Requirements:** Python 3.8+, PostgreSQL, about 16 GB RAM (a GPU is not required but speeds up generation), and a quantized Mistral 7B model file in GGUF format.

```bash
# 1. Clone the repository
git clone https://github.com/yassintgn18/RAG_System.git
cd RAG_System

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure
#    - Download a quantized Mistral 7B (GGUF) model and set its path in app.py
#    - Set your PostgreSQL connection settings in app.py

# 5. Run
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

## Design decisions and lessons learned

- **Model size matters.** 800M and 1.5B parameter models gave vague or wrong answers. Moving to Mistral 7B was the biggest quality gain.
- **Chunking decides retrieval quality.** Character-based splitting lost context. Token-based chunks of about 200 tokens retrieved far more relevant passages.
- **Prompt grounding reduces hallucination.** A strict context-only prompt with a fixed fallback answer stopped the model from inventing information.
- **Hardware limits.** Running a 7B model locally without a GPU means answers can take around 50 seconds. Quantization made it feasible on consumer hardware.

## Evaluation

Tested manually after every change to the model, embedding model, and chunking strategy, scoring answers on how well they stayed grounded in the document. Estimated accuracy is **~85%**. This was self-evaluated and has not yet been tested with external users or a benchmark dataset.

## Roadmap

- Automated evaluation scripts instead of manual testing
- Faster inference (GPU, or an API-based model as an option)
- Multi-document and multilingual (Arabic) support
- Voice interaction

## Author

**Yassin Touggani** — AI & Computer Science engineering student at ENSAM Casablanca
[LinkedIn](https://www.linkedin.com/in/yassintouggani/) · [GitHub](https://github.com/yassintgn18)
