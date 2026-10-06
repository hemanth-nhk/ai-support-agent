# Final Build Brief — AI Customer Support Agent (as actually built)

This documents the specification that was actually implemented, after several corrections made during development. Keep this alongside the code as the authoritative record of *why* things are built the way they are — useful for explaining the project in an interview.

## What changed from the original plan, and why

1. **LLM: Ollama (local) instead of a cloud provider.**
   Original drafts assumed a cloud LLM. This was corrected to Ollama (`llama3.2:3b`) running locally, because the project's goal is a genuinely local-only stack — using a cloud API while claiming "local-only" would be a contradiction worth catching before an interview, not after.

2. **The LLM never owns structured output.**
   Small local models are less reliable at consistent JSON/structured output than large cloud models. Rather than asking Ollama to produce `{intent, tool_used, status}` and validating/retrying, the design eliminates the risk entirely: intent classification is deterministic (keyword/pattern matching in `app/agent/router.py`), and the LLM is only ever asked to phrase natural-language text around data the application already decided on. This means a bad or malformed LLM response can only degrade *wording*, never *correctness* or *routing*.

3. **Deterministic routing, not LLM-based classification.**
   Because the LLM doesn't own routing, `classify_intent()` uses explicit regex/keyword rules. This is intentional, not a limitation: it makes the system reproducible (same input always yields the same route) and unit-testable without needing a running LLM at all.

4. **Pydantic validation with fail-closed behavior.**
   Every `/chat` response passes through `build_validated_envelope()` before reaching the caller. If any field is malformed or unexpected, the system returns a fixed `ERROR_ENVELOPE` (`status: "error"`) rather than propagating a guess. This was added specifically so that *even if* a future change introduced model-generated structured output, a malformed payload could never leak through as a false "ok" response.

5. **Static templates for "not found," not LLM fallback.**
   When SQLite has no matching order, or RAG retrieval doesn't clear the similarity threshold, the LLM is **not invoked at all** for that turn — a pre-written template is returned directly. This was a deliberate late addition: it closes the door on the LLM ever "helpfully" filling a gap with invented information, because it's never given the opportunity to try.

6. **Similarity threshold on RAG retrieval.**
   Retrieved FAQ chunks are only used if their distance score clears `RAG_SIMILARITY_THRESHOLD` (default `0.35`, configurable via `.env`). A weak match is treated identically to no match. This value is explicitly flagged as a starting estimate needing empirical validation — not asserted as correct.

7. **Ollama existence check instead of an unconditional `ollama pull`.**
   The build process checks `ollama --version` and that the target model is available before proceeding, and stops with an explicit install/pull command if not — rather than assuming a pull has already happened, or silently falling back to a cloud model if it's missing.

## Corrections from an independent code review (round 2)

A second, independent review of the built code caught several real gaps that the first pass had missed. Each is fixed in code, not just noted:

8. **The LLM could still fabricate wording even on the "grounded" order/FAQ paths.**
   The original design (point 5 above) kept the LLM out of the loop *when data was missing*, but still handed found order/FAQ data to Ollama to paraphrase, trusting the instruction "don't add anything not in the data." A model can still violate that instruction. **Fix:** order and customer-orders responses are now built with a deterministic string template directly from the database fields (`_format_order_response()` in `app/agent/nodes.py`). FAQ responses now return the approved document excerpt **verbatim**, not an LLM paraphrase. Ollama is only invoked in `general_node`, which carries no factual/business content by design — a bad response there can be off-topic, never fabricated fact. This is a structural guarantee, not a prompted one.

9. **The RAG similarity threshold was being compared against the wrong distance metric.**
   `retriever.py` treated the threshold as a cosine-distance cutoff, but `ingestion.py` never configured the Chroma collection to actually use cosine space — Chroma's default is L2 (squared Euclidean), a completely different scale. The `0.35` threshold was therefore meaningless against whatever L2 scores were actually being returned. **Fix:** `ingestion.py` now explicitly creates the collection with `metadata={"hnsw:space": "cosine"}`. Verified with a real (non-mocked) test using a deterministic word-overlap embedding: a genuinely matching query scored `0.42` cosine distance, a mismatched query scored `1.0` — a real, usable separation for a threshold to discriminate on.

10. **Python 3.13 + the original `chromadb==0.5.23` pin failed to install on Windows.**
    That version depends on `chroma-hnswlib`, which has no prebuilt wheel for Python 3.13 on Windows and falls back to a source build that fails without Visual C++ build tools. **Fix:** upgraded to `chromadb==1.5.9`, confirmed (via `pip show chromadb`) to no longer depend on `chroma-hnswlib` at all. Python 3.11 remains the most-tested path, but the root cause of the 3.13 failure is removed, not just worked around.

11. **`GET /customers/{id}/orders` (and the GraphQL equivalent) had no access control.**
    Anyone who could guess or enumerate a customer ID could pull that customer's full order history. **Fix:** added a demo-level shared API key gate (`app/api/auth.py`, `require_api_key` dependency on the REST route, an equivalent check in the GraphQL resolver reading the same `X-API-Key` header). This is explicitly documented as *not* real authentication — a real deployment needs per-customer identity and an ownership check, not a single shared secret that grants equal access to every customer's data. The endpoint fails closed (503) if no key is configured server-side, rather than silently allowing open access.

12. **`seed()` destructively wiped all data on every call.**
    Fine for a disposable demo DB, but a real footgun if ever run against a database holding anything else. **Fix:** `seed()` is now idempotent by default — it checks for existing rows and no-ops if found, only wiping and reseeding when called with `force=True` (or `--force` from the command line).

13. **The advertised test count didn't actually cover RAG mechanics.**
    The original 26 tests mocked the LLM everywhere and never exercised ingestion or retrieval at all. **Fix:** `ingestion.py` and `retriever.py` were made to accept an injectable `persist_dir` / `embedding_fn` / `threshold`, and `tests/test_rag.py` uses a deterministic stub embedding function to test real ingestion, real Chroma add/query round-trips, the cosine-space configuration (regression test for point 9), a known-match query being accepted, and a known-miss query being rejected at a strict threshold — all without needing network access or a torch install. This proves the retrieval *mechanics* are correct; it does not prove the real sentence-transformers model's embedding *quality* on real FAQ text — that still needs the manual check in `WINDOWS_SETUP_CHECKLIST.md` step 10d.

## Architecture (final)

```
User → REST (FastAPI) / GraphQL (Strawberry)
         │
         ▼
    LangGraph Agent
         │
    classify_node (deterministic — app/agent/router.py)
         │
    conditional routing
         │
 ┌───────┼────────────┬─────────────────┐
 │       │             │                 │
order_   customer_   faq_node        general_node /
status_  orders_     (RAG, cosine +   out_of_scope_node
node     node        threshold-       (only node that
(templ.) (templ.)    enforced,        calls the LLM)
 │       │            verbatim
 │       │            excerpt)
 └───────┴── SQLite    │
   (auth-gated on          └── Chroma (local, persistent, cosine space)
    customer-orders)                │
         │                          │
         └── Service layer ─────────┘
                  │
       guardrails.build_validated_envelope()
       (Pydantic, fail-closed on any malformed state)
                  │
       {response, intent, tool_used, retrieval_count, status}
                  │
                  ▼
                User
```

## Non-negotiable rules (enforced in code, not just documented)

| Rule | Enforced in |
|---|---|
| Never invent order/policy data | `app/agent/nodes.py` — deterministic templates / verbatim excerpts; LLM never invoked on these paths at all, found or not |
| LLM never controls structured fields | `app/agent/llm_client.py` docstring + `nodes.py` design — LLM output only ever fills `general_node`'s `response` string |
| Fail closed on malformed state | `app/agent/guardrails.py` — Pydantic validation, `ERROR_ENVELOPE` fallback |
| No chain-of-thought exposed | Response schema in `app/api/schemas.py` — exactly 5 fields, nothing else |
| Layered architecture, no duplicated logic | `app/services/order_service.py` used identically by REST, GraphQL, and the order tool |
| No secrets in source | `.env.example` committed, `.env` gitignored |
| Local-only, no cloud LLM | `app/agent/llm_client.py` — Ollama only, `OLLAMA_BASE_URL` defaults to `localhost:11434` |
| Correct distance metric for RAG threshold | `app/rag/ingestion.py` — explicit `hnsw:space: cosine` collection config |
| Access control on sensitive lookup | `app/api/auth.py`, `app/api/rest.py`, `app/api/graphql.py` — demo-level shared key, fails closed if unconfigured |
| Non-destructive seeding | `app/database/seed.py` — idempotent unless `force=True` |

14. **Structured Private Data Retrieval v2 (`DATA_QUERY`):**
    Enables parameterized multi-field search over orders (by customer, product, status, date range, 24-hr time range, and amount bounds). Follows the strict pipeline:
    `Natural Query -> Deterministic Intent Routing -> X-API-Key Auth -> LLM Structured Parameter Extraction (JSON only, no markdown, extra forbidden) -> Pydantic Validation (rejecting invalid dates, cross-midnight times, inverted ranges) -> Customer Entity Resolution (parameterized SQL, candidate names on ambiguity) -> Deterministic Parameterized SQL Search (bound parameters only, LIMIT 20, COUNT(*) total_matches) -> Deterministic Response Formatting (zero LLM phrasing)`.
    *Qualification Note:* The malformed-LLM-output handling is test-covered via deterministic mock in `test_malformed_llm_json_needs_clarification_no_db_search`, not live-model verified.

## Verification status (as of last build session)

- **55/55 automated tests passing** — REST (8, incl. auth), GraphQL (5, incl. auth), agent routing (11, incl. ambiguity & DATA_QUERY), tools (4), `/chat` endpoint (7, incl. auth & multi-turn session), RAG mechanics (5), DATA_QUERY v2 suite (15).
- **Evaluation suite:** 100% of benchmark targets met (`evaluation/run_eval.py`).
- **Verified live in this session:** Live Ollama inference with `llama3.2:3b`, SQLite seed/queries, REST, GraphQL, guardrail semantic consistency, multi-turn entity recovery, and parameterized order search.


## What this project demonstrates (for interviews)

- Designing an agentic system where an LLM's failure mode is bounded by architecture, not by hoping the model behaves — and iterating that guarantee further under review pressure (paraphrase-with-instructions → structurally excluding the LLM from factual paths entirely) is the single most defensible design decision in this project if asked to justify it
- REST + GraphQL on a shared service layer with zero duplicated business logic
- Grounded RAG with a correctly-configured distance metric and an explicit, tunable rejection threshold instead of always returning *something*
- LangGraph state management and conditional routing
- Layered architecture (Tool → Service → Repository → DB) as a deliberate, explainable choice
- Test-driven verification of guardrails, not just happy-path testing — including regression tests written specifically because a review caught a bug (the cosine-space test exists *because* that bug existed once)
- Treating "the LLM is only mocked in tests" as a coverage gap worth closing, not accepting it as good enough — the RAG stub-embedding tests exist for exactly this reason
- Responding to external review findings by fixing root causes (chromadb version, cosine config, LLM's role in the architecture) rather than patching symptoms
