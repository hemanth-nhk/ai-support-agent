# AI Customer Support & Tabular Analytics Platform (Local-Only)

An enterprise-grade, locally-hosted customer support automation and deterministic tabular analytics platform. Combines rule-based support workflows, local RAG document grounding, and an in-memory DuckDB analytical engine with server-side chart rendering and conversational multi-turn analytical memory.

**No cloud APIs, no external LLM dependencies, zero cloud data leakage.** Inference runs on your own machine via Ollama (`llama3.2:3b` by default), vector search runs in local Chroma, and structured data is queried using in-memory DuckDB.

---

## Architecture

```
User (Browser SPA / REST API / GraphQL)
  │
  ├── REST Endpoints (FastAPI)
  ├── GraphQL Playground (Strawberry)
  └── Single-Page Application (HTML5 / Vanilla CSS / JS)
        │
        ▼
   LangGraph Agent
        │
   Deterministic Intent Classifier (Keyword & regex rules — NOT the LLM)
        │
  ┌─────┴───────────────┬────────────────────────┬────────────────────────┐
  │                     │                        │                        │
Order / Customer      FAQ RAG                Document QA           Tabular Analytics
Workflows             (Chroma Cosine)        (Session Ingestion)   Engine (DuckDB)
  │                     │                        │                        │
  ▼                     ▼                        ▼                        ▼
SQLite Database       Local Chroma Store      Local Uploads        AST Planner & Compiler
(Parameterized)       (Approved Excerpts)    (PDF / DOCX / TXT)   (AnalysisPlan AST)
                                                                          │
                                                                          ▼
                                                                   In-Memory DuckDB
                                                                   (Sandboxed Execution)
                                                                          │
                                                                          ▼
                                                                   Visualization Engine
                                                                   (Matplotlib OO Canvas)
                                                                          │
  ┌───────────────────────────────────────────────────────────────────────┘
  ▼
Guardrails & Schema Envelopes (Pydantic, Fail-Closed)
  │
  ▼
Response Envelope: { response, intent, tool_used, retrieval_count, status, chart, table, pivot }
```

### Core Architecture & Guarantees

1. **Zero Arbitrary SQL:** The LLM *never* generates SQL strings. Queries are compiled from a strictly validated Pydantic AST (`AnalysisPlan`), using double-quote identifier escaping, data-type compatibility checks, and parameterized filters / safe literal formatting.
2. **Zero Arbitrary Python Execution:** Neither the LLM nor the user can inject Python code. Visualization uses an isolated, server-controlled Object-Oriented Matplotlib engine (`Figure`/`Canvas`) with fixed styling tokens.
3. **No Hallucinated Facts or Numbers:**
   - Order and customer details are rendered from deterministic SQLite templates.
   - FAQ responses quote approved excerpts verbatim.
   - Analytical findings are grounded exclusively in executed DuckDB result sets and matrix totals.
4. **Sandboxed DuckDB Runtime:** Queries run in ephemeral in-memory databases with hard resource boundaries (`512MB RAM`, `2 threads`, `5.0-second` hardware cancellation timers, row-output clamps).
5. **Server-Verified Signed Session Identity:** Prevents session spoofing and cross-session data exfiltration. Session identity is cryptographically signed using HMAC-SHA256 (`SESSION_SECRET_KEY`) and resolved exclusively from `app_session_token` HTTP-only SameSite cookies or `X-Session-Token` headers. Client-provided body/query `session_id` values are never trusted as identity. Conversation ownership is immutable across `PATCH` updates, and `/chat` analytical execution verifies ownership before any database query or tool executes.
6. **Fail-Closed Semantics:** Any unhandled exception, schema mismatch, or path violation fails closed to a safe error envelope without state corruption.

---

## Implemented Capabilities (Phases 1–9)

### 1. Customer Support Workflows
- **Order Lookup:** Instant order tracking (`ORDER_STATUS`) by ID using parameterized SQLite queries.
- **Customer Order History:** Lookup of past orders gated by a demo API key header (`X-API-Key`).
- **Entity Recovery & Ambiguity Handling:** Conversational recovery when an order ID is missing or multiple customer records match.
- **Out-of-Scope Protection:** Immediate, zero-LLM refusal for unrelated requests.

### 2. Document RAG (FAQ Knowledge Base)
- **Local Vector Search:** Persistent Chroma store (`chromadb==1.5.9`) configured with explicit cosine space (`hnsw:space: cosine`).
- **Sentence-Transformers Embeddings:** Local embedding model (`all-MiniLM-L6-v2`).
- **Threshold-Gated Citations:** Cosine distance cutoff (default `0.42` calibrated for `all-MiniLM-L6-v2`). Queries exceeding distance threshold fall back to a safe "insufficient information" response without guessing.

### 3. Ad-Hoc Session Document QA
- **Multi-Format Ingestion:** Session-isolated upload of `.pdf`, `.docx`, and `.txt` files.
- **DOCX Structure Extraction:** Deterministic traversal of section headers, body paragraphs, tables (with pipe-delimited cell representations and horizontally merged cell deduplication), and section footers.
- **Chunking & Storage:** Ingested into session-isolated Chroma collections.
- **Prompt-Injection Defense:** Untrusted document contents are wrapped in rigid boundary delimiters; LLM is instructed to treat text strictly as passive data.
- **Explicit OCR Boundary:** Scanned or image-only PDFs lacking embedded text streams fail closed safely with a validation error; local offline text extraction intentionally avoids heavy OCR dependencies.

### 4. Tabular Dataset Ingestion & Fast Profiling
- **Supported Formats:** `.csv`, `.xlsx`, and `.json`.
- **Automated Profiling:** Extracts column data types, null counts, distinct cardinality, numerical summaries (mean, min, max, stddev), and temporal boundaries.
- **File Validation:** Size limits (default `50MB`), format checking, and path-traversal prevention.

### 5. Deterministic DuckDB Analytics Engine
- **AST Compiler:** Translates natural-language intent into an `AnalysisPlan` model, compiled into DuckDB SQL.
- **Aggregations & Grouping:** `sum`, `avg`, `count`, `count_distinct`, `min`, `max`, `median`, `stddev`, and `percentile_cont`.
- **Temporal Bucketing:** Date truncation (`year`, `quarter`, `month`, `day`).
- **Parameterized Filtering:** `==`, `!=`, `>`, `>=`, `<`, `<=`, `in`, `not_in`, `contains`, `starts_with`, `between`, `is_null`, `is_not_null`.

### 6. Advanced Analytics & Statistical Operators
- **Rolling Windows:** Moving averages and moving sums with configurable window sizes and partition dimensions.
- **Cumulative Metrics:** Running sums, running totals, and running averages over temporal dimensions.
- **Quantile Segmentation:** NTILE bucketing (quartiles, deciles, percentiles) for distribution analysis.
- **Z-Score Anomaly Detection:** Mean and standard deviation computation with parameterized threshold filtering (e.g., $z > 3.0$) and zero-variance guards.
- **Period Comparisons:** Month-over-month (MoM) and year-over-year (YoY) percentage growth and delta comparisons.
- **Derived Metrics & Ratios:** Safe percentage and ratio calculations with division-by-zero protection.
- **Correlation:** Deterministic Pearson correlation matrix calculation between numerical columns.

### 7. Multi-Turn Analytical Memory (Conversational AST Mutations)
- **Session-Persisted Plan:** `active_analysis_plan` is tracked per session across user turns.
- **Incremental Mutations:** Modifies plans incrementally without restarting analysis:
  - `ADD_FILTER` / `CHANGE_FILTER` / `REMOVE_FILTER`
  - `CHANGE_GROUP_BY` / `ADD_GROUP_BY` / `REMOVE_GROUP_BY`
  - `CHANGE_METRIC` / `CHANGE_SORT` / `CHANGE_LIMIT`
  - `ADD_ROLLING` / `ADD_CUMULATIVE` / `TOGGLE_CHART` / `RESET_ANALYSIS`
- **Fail-Closed Rollback:** Invalid column names or uncompilable mutations preserve the previous valid plan.
- **Dataset Isolation:** Switching `dataset_id` automatically purges previous analytical memory to prevent cross-dataset contamination.

### 8. 2D Cross-Tab / Pivot Analysis & Heatmap Visualization (Phase 9)
- **Pivot AST (`PivotSpec` & `PivotResult`):** Strict Pydantic models for 2D matrix transformations with bounded row/column cardinalities.
- **Deterministic DuckDB PIVOT:** Dynamic SQL pivoting with escaped identifiers, safe literal filters, and bounded `LIMIT`.
- **Matrix Normalization:** Outputs structured row headers, column headers, 2D numerical values (`list[list[Optional[float]]]`), and marginal row, column, and grand totals.
- **Server-Side Heatmap Rendering:** Matplotlib OO Canvas heatmap with `"YlGnBu"` colormap, annotated cells with contrast thresholding, colorbars, and rotated labels.
- **Conversational Mutations:** `TOGGLE_HEATMAP` and `CHANGE_PIVOT_COLUMNS` follow-up support.
- **Grounded Matrix Findings:** Generates factual summaries referencing matrix dimensions, the leading cell value, and the grand total.

### 9. Web Application Dashboard
- **Single-Page Application (`static/`):** Full chat interface with localStorage thread history.
- **Document & Dataset Chips:** Visual indicators for active attachments.
- **Interactive Tables:** Paginated result tables for analytical responses.
- **Chart Viewer:** Inline chart cards with fullscreen modal zoom and PNG download.

---

## Tech Stack

| Component | Technology | Role |
| :--- | :--- | :--- |
| **API Framework** | FastAPI (REST) + Strawberry (GraphQL) | High-performance API routing and shared service layer |
| **Agent Framework** | LangGraph | Deterministic DAG state graph and conversation lifecycle |
| **Inference Engine** | Ollama (`llama3.2:3b`) | Local-only conversational LLM (used strictly for non-factual text) |
| **Analytics Kernel** | DuckDB (`duckdb==1.5.6`) | In-memory analytical SQL processing engine |
| **Vector Database** | ChromaDB (`chromadb==1.5.9`) | Local persistent vector search with cosine distance metric |
| **Embeddings** | sentence-transformers (`all-MiniLM-L6-v2`) | Local offline sentence embeddings |
| **Data Processing** | pandas + openpyxl + pypdf + python-docx | Dataframe manipulation, Excel, PDF, and Word parsing |
| **Visualization** | Matplotlib (`matplotlib==3.11.2`) | Deterministic Object-Oriented server-side PNG chart rendering |
| **Validation** | Pydantic v2 | Strict schema validation, AST enforcement, and fail-closed gates |
| **Testing** | pytest + httpx + playwright | Automated unit, regression, integration, and E2E testing |

---

## Project Structure

```
ai-support-agent/
├── app/
│   ├── main.py                     FastAPI entrypoint, mounting REST, GraphQL, static UI
│   ├── config.py                   Pydantic Settings (loads .env)
│   ├── api/
│   │   ├── rest.py                 REST endpoints (/chat, /charts/*, /orders/*, etc.)
│   │   ├── conversations.py        Conversation lifecycle and dataset binding endpoints
│   │   ├── session_security.py     HMAC-SHA256 session token verification & CSRF defense
│   │   ├── graphql.py              Strawberry GraphQL schema and resolvers
│   │   ├── schemas.py              Pydantic API request/response models
│   │   └── auth.py                 Demo-level API key authorization gate
│   ├── agent/
│   │   ├── state.py                LangGraph AgentState definition
│   │   ├── router.py               Deterministic intent classification
│   │   ├── nodes.py                Graph execution nodes (support, RAG, analytics)
│   │   ├── graph.py                Compiled LangGraph state workflow
│   │   ├── guardrails.py           Response envelope validation (fail-closed)
│   │   └── llm_client.py           Ollama client wrapper
│   ├── analytics/
│   │   ├── schemas.py              AnalysisPlan, PivotSpec, PivotResult, mutations
│   │   ├── compiler.py             Deterministic DuckDB SQL compiler (standard + PIVOT)
│   │   ├── executor.py             Sandboxed in-memory DuckDB query runner
│   │   ├── planner.py              Natural-language intent planner and grounded findings
│   │   ├── mutator.py              Deterministic AST mutation engine (Phase 8 & 9)
│   │   ├── visualizer.py           Matplotlib chart engine (bar, line, scatter, hist, heatmap)
│   │   ├── chart_style.py          Centralized color palette, geometry, and design tokens
│   │   ├── profiler.py             Fast dataset profiling engine
│   │   └── dataset_service.py      Dataset lifecycle, disk storage, and session isolation
│   ├── services/
│   │   ├── order_service.py        Order querying and customer management
│   │   ├── order_policy_service.py Order return/refund policy evaluation & cross-referencing
│   │   ├── conversation_service.py Multi-turn conversation management & persistence
│   │   └── document_service.py     Session document ingestion and retrieval
│   ├── repositories/
│   │   ├── order_repository.py     Raw SQLite database queries
│   │   ├── customer_repository.py  Customer record queries
│   │   └── conversation_repository.py SQLite conversation metadata queries
│   ├── database/
│   │   ├── connection.py           SQLite thread-safe connection factory
│   │   ├── models.py               SQLite schema definitions
│   │   └── seed.py                 Idempotent demo dataset seeder
│   └── rag/
│       ├── ingestion.py            FAQ document loading, chunking, and Chroma persistence
│       ├── retriever.py            Threshold-enforced cosine vector search
│       └── embeddings.py           sentence-transformers model wrapper
├── static/                         Frontend Single-Page Dashboard (HTML, CSS, JS)
├── data/
│   └── faq/                        Static customer policy documents (shipping, returns, etc.)
├── evaluation/                     Empirical benchmark & evaluation suites
│   └── reliability/                Real-World File Reliability & Adversarial Benchmark (104 cases)
├── tests/                          Comprehensive test suite (458 tests)
│   ├── test_session_spoofing_prevention.py Phase 13 Session Security & Anti-Spoofing tests (14 tests)
│   ├── test_analysis_pivot.py      Phase 9 Pivot & Heatmap test suite (21 tests)
│   ├── test_analysis_multiturn.py  Phase 8 Multi-Turn Memory test suite (19 tests)
│   ├── test_top_n_analytics.py     Deterministic Top-N / Ranking test suite (32 tests)
│   ├── test_analysis_v3.py         Phase 7 Advanced Analytics test suite
│   ├── test_analysis_v2.py         Phase 6 Statistical Operators test suite
│   ├── test_analysis_compiler.py   Phase 3 Deterministic Compiler test suite (31 tests)
│   ├── test_visualization.py       Phase 4 Visualization Engine test suite (11 tests)
│   ├── test_datasets.py            Phase 2 Dataset Ingestion & Profiling test suite (37 tests)
│   ├── test_documents.py           Session Document QA & extraction test suite
│   ├── test_document_query.py      Session Document retrieval test suite
│   ├── test_chat_endpoint.py       End-to-end /chat API integration tests
│   └── ...                         Agent, RAG, GraphQL, and security tests
├── pytest.ini                      Pytest configuration (scoped to tests/)
├── requirements.txt                Fully pinned production dependencies
├── .env.example                    Example configuration template
└── README.md                       Project documentation
```

---

## Setup & Quickstart

### Prerequisites
- Python 3.11+
- [Ollama](https://ollama.com/) installed and running locally

### Installation (Windows / PowerShell)

```powershell
# 1. Clone the repository
git clone https://github.com/hemanth-nhk/ai-support-agent.git
cd ai-support-agent

# 2. Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\activate

# 3. Install pinned dependencies
pip install -r requirements.txt

# 4. Configure environment
copy .env.example .env

# 5. Seed demo database and build local FAQ vector store
python -m app.database.seed
python -m app.rag.ingestion

# 6. Pull local Ollama model
ollama pull llama3.2:3b

# 7. Start the development server
uvicorn app.main:app --reload
```

### Installation (Linux / macOS)

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python3 -m app.database.seed
python3 -m app.rag.ingestion
ollama pull llama3.2:3b
uvicorn app.main:app --reload
```

Visit the application in your browser:
- **Interactive UI Dashboard:** `http://127.0.0.1:8000/app`
- **Interactive Swagger Docs:** `http://127.0.0.1:8000/docs`
- **GraphQL Playground:** `http://127.0.0.1:8000/graphql`

---

## Verification & Test Suite

The project includes an exhaustive automated test suite covering unit contracts, deterministic SQL compilation, security injection defenses, session isolation, multi-turn memory, and browser E2E flows, alongside an empirical reliability and adversarial stress benchmark.

```powershell
# Run the complete core regression suite (458 tests)
pytest -v --tb=short

# Run Phase 13 session security & access-control suite (14 tests)
pytest tests/test_session_spoofing_prevention.py -v

# Run Phase 9 cross-tab pivot & heatmap tests (21 tests)
pytest tests/test_analysis_pivot.py -v

# Run the 35-Case End-to-End Evaluation Suite (40 cases)
python evaluation/eval_35_cases.py
```

**Core Regression Status:** **`458 passed, 0 failures, 0 errors, 0 regressions`**.
**Evaluation Status:** **`40/40 passed (100% pass rate; 100% Reliability, 100% Support, 100% Analytics, 100% Document)`**.
**Reliability Benchmark Status:** **`104/104 passed (100% extraction fidelity, 100% prompt-injection resistance, 0.0% cross-session leakage)`**.

---

## Implemented Scope vs. Future Work

To maintain architectural integrity, scope boundaries are strictly maintained:

### Implemented (Phases 1–9 + Reliability Remediation)
- Local-only support workflow routing and customer order queries.
- Chroma vector RAG with cosine thresholding and verbatim quotes.
- Ingestion, chunking, and deterministic QA of session-uploaded `.pdf`, `.docx`, and `.txt` documents.
- DOCX full-structure extraction covering section headers, body paragraphs, table rows/cells, and footers.
- Fast tabular profiling and validation of `.csv`, `.xlsx`, and `.json` datasets.
- Deterministic DuckDB SQL AST compiler with double-quoted escaping and parameter binding.
- Advanced statistical analytics (rolling windows, cumulative metrics, quantile NTILEs, z-score anomaly detection, MoM growth, ratios, correlation).
- Server-side Matplotlib chart rendering (`bar`, `line`, `scatter`, `histogram`, `heatmap`).
- Multi-turn conversational memory with AST mutations and fail-closed rollback.
- Phase 9 2D cross-tab pivot tables with marginal totals and bounded heatmaps.
- Disambiguated dual-attachment routing between tabular analysis and document queries.

### Excluded / Future Scope (Explicit Non-Goals for This Milestone)
- **Scanned Image-Only PDF OCR:** Extraction relies on embedded text streams; offline text parser deliberately omits heavy OCR engines (e.g. Tesseract) to preserve lightweight, deterministic offline execution.
- **Cross-Source Joint Querying across Dataset and Document:** Queries route deterministically to either tabular SQL or document RAG; simultaneous multi-hop joins across structured and unstructured data are classified as unsupported.
- **Multi-Table Relational JOINs across files:** Analysis is scoped to single uploaded tabular datasets.
- **Predictive Machine Learning / Auto-regressive Forecasting:** Analytical queries compute exact empirical statistics, not synthetic ML extrapolations.
- **Arbitrary LLM-Generated Python / SQL (ReAct code interpreter loops):** Explicitly avoided to guarantee deterministic safety.
- **Cloud Database Connectors (Snowflake, BigQuery):** Local in-memory DuckDB kernel only.

---

## Author

Hemanth ([@hemanth-nhk](https://github.com/hemanth-nhk)) — B.Tech Computer Science (AI & ML), MIST (JNTUH), Hyderabad.
