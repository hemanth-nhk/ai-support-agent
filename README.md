# AI Customer Support Agent (Local-Only)

A locally-run AI customer support system: FastAPI REST + Strawberry GraphQL, a deterministic LangGraph agent, RAG over local FAQ documents, and a SQLite order database. **No cloud APIs, no API keys for the LLM** -- the only LLM inference is via Ollama running on your own machine. (A separate, unrelated `SUPPORT_API_KEY` gates one sensitive endpoint -- see Guardrails below.)

## Architecture

```
User
 |
 |-- REST (FastAPI)
 `-- GraphQL (Strawberry)
       |
       v
  LangGraph Agent
       |
  Deterministic Intent Classifier (keyword/pattern rules -- NOT the LLM)
       |
 +-----+--------------+----------------+
 |     |              |                |
Order  Customer      FAQ (RAG)      General / Out-of-scope
Tool   Orders Tool    Tool
 |     |              |
 +-----+-- SQLite     `-- Chroma (local vector store, cosine distance)
       |                      |
       `-- Service layer -----'
                |
       Guardrails (Pydantic validation, fail-closed)
                |
        Response envelope: {response, intent, tool_used, retrieval_count, status}
                |
                v
              User
```

**Key design decision:** the LLM (Ollama) is used *only* for general conversational replies (greetings, "what can you help with") that carry no factual/business content. Order status, customer order lists, and FAQ answers are built from **deterministic templates and verbatim approved text** -- the LLM never touches them, and can therefore never fabricate a detail in these paths. Routing is 100% deterministic keyword/pattern matching (`app/agent/router.py`), not LLM-based classification.

## Guardrails (enforced, not aspirational)

- **True no-fabrication guarantee on facts.** Order/customer responses are built with an f-string template directly from SQLite fields. FAQ responses return the approved document excerpt verbatim. The LLM is never invoked on these paths at all -- not "asked not to invent," structurally unable to.
- FAQ retrieval uses a similarity threshold on **cosine distance**, with the Chroma collection explicitly configured for cosine space (`metadata={"hnsw:space": "cosine"}` in `app/rag/ingestion.py`) -- without this, Chroma's default L2 distance would make the threshold meaningless. A weak/no match returns a static "not enough information" message.
- Out-of-scope requests are refused via keyword detection before reaching the LLM.
- Every `/chat` response is validated by Pydantic (`app/agent/guardrails.py`) against a fixed schema and an allow-list of valid `status`/`tool_used` values. Any malformed state fails closed to `status: "error"`.
- No chain-of-thought is ever exposed. The API returns exactly 5 fields: `response`, `intent`, `tool_used`, `retrieval_count`, `status`.
- `GET /customers/{id}/orders` and the GraphQL `customerOrders` field require an `X-API-Key` header matching `SUPPORT_API_KEY` in `.env`. **This is demo-level protection only** -- a shared key that gates the whole endpoint, not real per-customer authentication/ownership checks. Replace before any non-demo use. See `app/api/auth.py`.
- `python -m app.database.seed` is idempotent by default -- it will not wipe existing data unless you pass `--force`.

## Tech Stack

- **API:** FastAPI (REST), Strawberry (GraphQL) -- both call the same service layer, no duplicated logic
- **Agent:** LangGraph (deterministic routing, not LLM-based)
- **LLM:** Ollama, local (`llama3.2:3b` by default) via `langchain-ollama` -- used only for non-factual general replies
- **RAG:** sentence-transformers embeddings + local persistent Chroma (`chromadb==1.5.9`, cosine distance)
- **Database:** SQLite
- **Validation:** Pydantic
- **Testing:** pytest -- **55 tests, all passing** as of the last verified run, including RAG mechanics, guardrail semantic consistency, multi-turn entity recovery, and DATA_QUERY v2 parameter search suite.

## Project Structure

```
app/
|-- main.py                 FastAPI app, mounts REST + GraphQL
|-- config.py                 Settings (from .env)
|-- logging_config.py
|-- api/
|   |-- rest.py               REST endpoints incl. POST /chat, protected customer-orders route
|   |-- graphql.py              Strawberry schema, reuses services, protected customerOrders field
|   |-- auth.py                  Demo-level API key check (app/api/auth.py)
|   `-- schemas.py                Pydantic request/response models
|-- agent/
|   |-- state.py               LangGraph state definition
|   |-- router.py               Deterministic intent classification
|   |-- nodes.py                 Graph nodes -- deterministic templates for order/customer/FAQ, LLM only in general_node
|   |-- graph.py                  Graph assembly + compiled entrypoint
|   |-- llm_client.py              Ollama wrapper (text-only, no structured output)
|   `-- guardrails.py               Response envelope validation, fail-closed
|-- tools/
|   `-- order_tool.py           LangGraph-facing wrapper over order_service
|-- services/
|   `-- order_service.py         Business logic (shared by REST/GraphQL/agent)
|-- repositories/
|   |-- order_repository.py       Raw SQLite access
|   `-- customer_repository.py
|-- database/
|   |-- connection.py
|   |-- models.py                  Schema DDL
|   `-- seed.py                     Idempotent demo data (--force to reset)
`-- rag/
    |-- ingestion.py                Load + chunk + embed + persist to Chroma (cosine space configured)
    |-- embeddings.py                sentence-transformers wrapper
    `-- retriever.py                  Threshold-enforced retrieval, injectable for testing

data/faq/         shipping.txt, returns.txt, refunds.txt, cancellation.txt (DEMO DATA)
tests/            test_api.py, test_graphql.py, test_agent.py, test_tools.py, test_chat_endpoint.py, test_rag.py
```

## Setup (Windows)

See **WINDOWS_SETUP_CHECKLIST.md** for the full step-by-step, including the API key setup and manual RAG verification. Short version:

```powershell
git clone https://github.com/hemanth-nhk/ai-support-agent.git
cd ai-support-agent
py -3.11 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
REM edit .env and set SUPPORT_API_KEY to any value
python -m app.database.seed
python -m app.rag.ingestion
ollama pull llama3.2:3b
uvicorn app.main:app --reload
```

Visit:
- REST docs: `http://127.0.0.1:8000/docs`
- GraphQL playground: `http://127.0.0.1:8000/graphql`

## Running Tests

```powershell
pytest tests/ -v
```

35 tests pass without Ollama running and without a real embedding model (RAG tests use a deterministic stub -- see `tests/test_rag.py`). To validate real embedding quality and live Ollama phrasing, use the manual checks in the Windows checklist.

## API Reference

### `POST /chat`
```json
{ "message": "Where is order 1001?" }
```
```json
{
  "response": "Order 1001 (Laptop) is currently: Shipped, estimated delivery 2026-09-15.",
  "intent": "ORDER_STATUS",
  "tool_used": "get_order_status",
  "retrieval_count": 1,
  "status": "ok"
}
```
Note this response is a deterministic template, not LLM output -- see Guardrails above.

### `GET /orders/{order_id}`
No auth required (order-only lookup, no customer-identifying aggregation).

### `GET /customers/{customer_id}/orders`
**Requires `X-API-Key` header.** Returns 401 without it, 503 if `SUPPORT_API_KEY` isn't configured server-side.

### `POST /graphql`
```graphql
query {
  order(id: 1001) { id product status estimatedDelivery }
}
```
`customerOrders` requires the same `X-API-Key` header.

## Known Limitations / Honest Notes

- Demo data only (`app/database/seed.py`, `data/faq/*.txt`) -- replace before any real use.
- The `SUPPORT_API_KEY` gate is a single shared secret, not per-customer authentication. A real deployment needs session/JWT-based auth with an ownership check (caller may only see *their own* orders).
- `llama3.2:3b` is a small model, but this only affects the wording quality of general/non-factual replies now -- it has no path to fabricating an order or policy detail.
- The RAG similarity threshold (0.35) is a starting point calibrated on the cosine-distance metric (now correctly configured), not empirically tuned against the real embedding model -- validate it against your own test queries and adjust in `.env` if needed.
- `GET /orders/{order_id}` is intentionally left unauthenticated (single-order lookup by ID, no aggregation across a customer's history) -- reassess this if order IDs become guessable/sequential in a real deployment.

## Author

Hemanth ([@hemanth-nhk](https://github.com/hemanth-nhk)) -- final-year B.Tech CS (AI & ML), MIST (JNTUH), Hyderabad
