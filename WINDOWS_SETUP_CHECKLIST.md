# Windows Setup & Verification Checklist

This project was built and **35/35 automated tests verified** in a Linux sandbox (LLM mocked where it's actually used -- order/customer/FAQ paths no longer touch the LLM at all, see FINAL_BUILD_BRIEF.md). RAG mechanics (cosine distance config, threshold filtering) are tested with a deterministic stub embedding, proving the code paths work. What genuinely could **not** be tested in that sandbox: (1) real Ollama inference, (2) real sentence-transformers embedding quality on real FAQ text. Everything else -- SQLite, FastAPI, GraphQL, routing, guardrails, the API-key auth gate, cosine distance configuration -- is proven working, not just written.

## 1. Prerequisites

- [ ] **Python 3.11+ installed (3.13 should now work too — see note below).**
- [ ] Git installed
- [ ] [Ollama for Windows](https://ollama.com/download) installed

**Note on a Python 3.13 issue found in review:** an earlier version of this project pinned `chromadb==0.5.23`, which depends on `chroma-hnswlib` -- a package with no prebuilt Windows wheel for Python 3.13, causing a failing source build. This is fixed by upgrading to `chromadb==1.5.9`, which dropped that dependency entirely (verified via `pip show chromadb`). Python 3.11 remains the most-tested path here, but 3.13 should now install cleanly. If you hit any wheel-build error on 3.13 despite this fix, fall back to 3.11:
```powershell
py -3.11 -m venv venv
```

## 2. Install Ollama and pull the model

```powershell
ollama --version
ollama pull llama3.2:3b
```
If `ollama --version` fails, install Ollama first -- do not substitute a cloud LLM; that breaks the local-only design.
```powershell
ollama list
```
You should see `llama3.2:3b` in the output.

## 3. Project setup

```powershell
cd ai-support-agent
py -3.11 -m venv venv
venv\Scripts\activate
```
Confirm the right interpreter: `python --version`

## 4. Install dependencies

```powershell
pip install -r requirements.txt
```

**Watch for this:** `sentence-transformers` depends on `torch`. If you don't have an NVIDIA GPU and want a smaller/faster install, run this *before* the main install:
```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
```
Then `pip install -r requirements.txt` -- pip will see torch is already satisfied and skip the larger GPU build.

## 5. Configure environment

```powershell
copy .env.example .env
```
Open `.env` and set `SUPPORT_API_KEY` to any value you'll remember (it's used in step 10 to test the protected customer-orders lookup). Everything else can stay at its default for local testing.

## 6. Seed the database

```powershell
python -m app.database.seed
```
Expected output: `Seeded 3 customers and 4 orders (deterministic demo data).`

Note: seeding is now idempotent by default -- running it again with existing data is a safe no-op. Use `python -m app.database.seed --force` if you actually want to wipe and reset to demo data.

## 7. Build the FAQ vector index (the main unverified piece)

```powershell
python -m app.rag.ingestion
```
Expected output something like: `Indexed 14 FAQ chunks into Chroma collection 'faq_documents'.`

**First run downloads the embedding model** (`all-MiniLM-L6-v2`, ~80MB) from Hugging Face -- needs internet once, then works fully offline.

If this step errors, most likely causes:
- `sentence-transformers`/`torch` didn't install correctly -> re-check step 4
- No internet on first run -> the model download needs one connection the first time only

## 8. Run the automated test suite

```powershell
pytest tests/ -v
```
Expected: **35 passed**. None of these require Ollama running (order/customer/FAQ responses are fully deterministic now) or a real embedding model (RAG tests use a stub). This confirms the codebase itself is intact on your machine -- it does not yet confirm real embedding quality; that's step 10d.

## 9. Start the app

```powershell
uvicorn app.main:app --reload
```

## 10. Manually verify the parts that couldn't be tested in the sandbox

Open a **second** PowerShell window (keep `uvicorn` running in the first):

### 10a. Ollama is reachable
```powershell
curl http://localhost:11434/api/tags
```
Should list your installed models. If this fails, start Ollama (`ollama serve`, or check it's running as a background service).

### 10b. A found order returns a deterministic, data-exact response (no LLM involved)
```powershell
curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" -d "{\"message\": \"Where is order 1001?\"}"
```
Expected: `status: "ok"`, `tool_used: "get_order_status"`, and `response` reading exactly `"Order 1001 (Laptop) is currently: Shipped, estimated delivery 2026-09-15."` -- this is a template, not LLM phrasing, by design (see FINAL_BUILD_BRIEF.md on the fabrication-guarantee fix). This request should return almost instantly since Ollama is never called.

### 10c. A missing order fails closed
```powershell
curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" -d "{\"message\": \"Where is order 9999?\"}"
```
Expected: `status: "not_found"`, `retrieval_count: 0`.

### 10d. FAQ retrieval actually works with real embeddings — the genuinely unverified piece
```powershell
curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" -d "{\"message\": \"How long does standard shipping take?\"}"
```
Expected: `status: "ok"`, `tool_used: "retrieve_faq"`, `retrieval_count` >= 1, and `response` containing the verbatim FAQ text mentioning "3-5 business days" -- FAQ answers are returned as approved excerpts, not LLM paraphrase, so the exact wording from `data/faq/shipping.txt` should appear.

**If retrieval_count is 0 for a question that should clearly match:** the similarity threshold (`RAG_SIMILARITY_THRESHOLD=0.35` in `.env`) needs calibration for how the real embedding model scores your queries -- the cosine-space bug that made this threshold meaningless has been fixed (collections are now explicitly configured for cosine distance), but the *value* itself is still a starting estimate, not empirically tuned. Try raising it in steps (0.5, 0.7, 0.9) and re-testing. See the "Known Limitations" section in README.md.

### 10e. Out-of-scope refusal still works live, instantly
```powershell
curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" -d "{\"message\": \"write me a python script\"}"
```
Expected: `status: "out_of_scope"`, near-instant response (no LLM call).

### 10f. Customer-orders lookup requires the API key (privacy fix)
Without a key -- should be rejected:
```powershell
curl http://127.0.0.1:8000/customers/1/orders
```
Expected: `401 Unauthorized`.

With the key you set in `.env`:
```powershell
curl http://127.0.0.1:8000/customers/1/orders -H "X-API-Key: your-key-here"
```
Expected: `200 OK` with the order list.

Remember: this is a demo-level shared-key gate, not real per-customer authentication -- see the docstring in `app/api/auth.py` and the README before using this for anything beyond local demoing.

### 10g. GraphQL works, including the protected field
Open `http://127.0.0.1:8000/graphql` in a browser:
```graphql
query {
  order(id: 1001) { id product status estimatedDelivery }
}
```
Then try `customerOrders` without an `X-API-Key` header -- it should return a GraphQL error. Add the header via a GraphQL client that supports custom headers to confirm it succeeds with the right key.

## 11. If everything above checks out

You have a fully verified, genuinely local AI customer support agent with a real no-fabrication guarantee on order/FAQ data. Move on to committing and pushing to GitHub (see the separate git push guide) whenever you're ready.

## Quick troubleshooting reference

| Symptom | Likely cause |
|---|---|
| `/chat` for a greeting hangs then errors | Ollama not running, or wrong `OLLAMA_BASE_URL` in `.env` -- this only affects general_node now |
| FAQ questions always return "not enough information" | Similarity threshold too strict for real embeddings -- raise `RAG_SIMILARITY_THRESHOLD` |
| `pip install` fails or is extremely slow/huge | torch pulling GPU/CUDA packages -- use the CPU-only index in step 4 |
| Import errors on startup | Re-run `pip install -r requirements.txt` inside the activated venv -- check `(venv)` shows in your prompt |
| Order lookups return 404 for orders you expect to exist | Re-run `python -m app.database.seed --force` -- the DB may be empty or stale |
| `/customers/{id}/orders` returns 401 even with a key | Confirm `SUPPORT_API_KEY` in `.env` matches the `X-API-Key` header exactly, and that you restarted `uvicorn` after editing `.env` |
| `/customers/{id}/orders` returns 503 | `SUPPORT_API_KEY` is empty/unset in `.env` -- the endpoint fails closed rather than allowing open access |
