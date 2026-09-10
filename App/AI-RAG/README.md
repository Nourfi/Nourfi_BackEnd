# NourFi - AAOIFI Retrieval-Augmented Generation

NourFi is an Islamic-finance research assistant grounded in AAOIFI excerpts. It retrieves relevant standards from Supabase/pgvector, sends only that context to Groq, and returns a cited answer with safety constraints.

## Current Architecture

- Embeddings: `intfloat/multilingual-e5-small` (384 dimensions).
- Vector database: PostgreSQL/Supabase with `pgvector`.
- Retrieval: cosine similarity, up to five relevant chunks.
- Generation: Groq API only.
- Current generation model: `openai/gpt-oss-120b`.
- Personas: `general_public`, `student`, `entrepreneur`, `business_owner`, `investor`, `banker`, `academic`, and `finance_specialist`.

The embedding model and generation model are separate. Do not replace the embedding model without regenerating and re-uploading all vectors.

## Grounding And Safety

The pipeline:

- answers only from retrieved AAOIFI context;
- preserves standard and page metadata for citations;
- supports concrete scenarios and comparisons between explicitly mentioned contracts;
- retrieves explicitly mentioned standards even when vector ranking initially omits one;
- uses the exact citation format `[Source : AAOIFI Standard X]`;
- repairs citation formatting using only retrieved standards;
- returns an insufficiency message when evidence cannot support a claim;
- refuses explicit fatwa or definitive halal/haram requests locally;
- removes sensitive long numeric values before sending a question to Groq;
- ignores instructions contained inside retrieved documents;
- never exposes prompts, credentials, or chain-of-thought.

The system can explain a user's situation, compare contracts, and describe practical implications. It must not invent current rates, prices, laws, market events, personal approvals, or religious rulings absent from the retrieved AAOIFI material.

## Requirements

- Windows, Linux, or macOS
- Python 3.10+
- A Supabase/PostgreSQL database with `public.rag_chunks` and `pgvector`
- A valid Groq API key
- Internet access on the first Hugging Face model load

## Installation

From the `Nourfi_BackEnd` directory:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell activation is unavailable, use:

```cmd
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Environment

Create `.env` at the `Nourfi_BackEnd` root. Never commit it or expose its credentials.

```env
DATABASE_URL=postgresql://USER:PASSWORD@HOST:5432/postgres
SUPABASE_DB_URL=postgresql://USER:PASSWORD@HOST:5432/postgres
GROQ_BASE_URL=https://api.groq.com/openai/v1
GROQ_API_KEY=YOUR_GROQ_API_KEY
GROQ_MODEL=openai/gpt-oss-120b
MAX_GENERATION_TOKENS=900
```

## Run A Query

The normal runner uses the existing Supabase corpus. It does not rebuild or upload embeddings:

```powershell
python App\AI-RAG\rag\run_pipeline.py --corpus 54 --question "What is Murabaha?" --persona general_public
```

French scenario example:

```powershell
python App\AI-RAG\rag\run_pipeline.py --corpus 54 --question "Pour financer mon commerce, dois-je choisir la Mourabaha ou la Moudaraba ?" --persona entrepreneur
```

Available personas:

```text
general_public, student, entrepreneur, business_owner,
investor, banker, academic, finance_specialist
```

The first query may be slow because Hugging Face loads the embedding model. The encoder is cached within a long-running Python process. For a backend API, call `answer_question()` from a persistent FastAPI process instead of starting `run_pipeline.py` for every user message.

## Build Or Rebuild The Corpus

The build steps are maintenance operations only:

```powershell
python App\AI-RAG\rag\chunker.py --corpus 54
python App\AI-RAG\rag\embeddings.py --corpus 54
python App\AI-RAG\rag\upload_embeddings.py --corpus 54
```

Or run the orchestrator:

```powershell
python App\AI-RAG\rag\run_pipeline.py --corpus 54 --force-rebuild --skip-query
```

Only rebuild after changing the source documents or embedding model. The upload step changes the shared Supabase corpus.

## Tests

Run the offline citation and prompt-security tests:

```powershell
python -m pytest App\AI-RAG\tests\test_tst04_citations.py App\AI-RAG\tests\test_tst05_prompt_security.py -q
```

Run the complete test suite:

```powershell
python -m pytest -ra -vv
```

These are offline tests do not call Groq or Supabase. A separate integration test is needed to verify live model responses, provider timeouts, and rate limits.

## Runtime Flow

```text
User question and persona
        -> local E5 query embedding
        -> Supabase pgvector retrieval
        -> targeted retrieval for explicitly named standards
        -> grounded Groq generation
        -> citation validation or citation repair
        -> answer, disclaimer, and source metadata
```

The frontend must send the question and persona to the backend only. It must never receive `GROQ_API_KEY`, `DATABASE_URL`, or direct Supabase access.


Why using groq openai/gpt-oss-120b and not ollama qwen2.5:3b  model ?
We evaluated both Ollama with Qwen2.5:3B and Groq with OpenAI GPT-OSS 120B using the same Nourfi RAG pipeline and test questions. Although Qwen2.5:3B has the advantage of being fully local and cost-free, GPT-OSS 120B produced more reliable results on complex queries requiring multi-source reasoning, comparison of Islamic finance concepts, and strict adherence to retrieved AAOIFI content. Since the primary objective of Nourfi is to provide accurate, grounded and context-aware answers rather than simply minimize inference cost, we selected GPT-OSS 120B through Groq
also GPT-OSS 120B is designed as a reasoning model with strong instruction-following and supports structured outputs and tool-oriented workflows.