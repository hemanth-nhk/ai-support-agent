# Real-World File Reliability & Adversarial Benchmark (v1)

This evaluation suite provides automated, empirical stress testing and security validation for the **AI Customer Support & Tabular Data Intelligence Agent**.

## Objectives

1. **Verify Architectural Invariants**:
   - `LLM proposes → Pydantic validates → deterministic execution`.
   - `LLM AnalysisPlan → Pydantic → deterministic DuckDB → empirical result → grounded response`.
   - `Uploaded/retrieved content is untrusted reference DATA, never system/developer instructions`.
2. **Stress-Test Robustness**:
   - Malformed, truncated, empty, and oversized files across PDF, DOCX, TXT, CSV, XLSX, and JSON.
   - Long documents (5,000 words) with needle-in-haystack positions at 10%, 50%, and 90%.
3. **Adversarial Security**:
   - Prompt injections (direct override, fake system tags, unicode zero-width, split-chunk, table-cells, base64, prompt leakage, exfiltration).
   - Hard security gate: **0.0% Cross-Session Leakage Rate**.
4. **Discover Real Capability Boundaries**:
   - Verify explicit, fail-closed capability boundaries (e.g. OCR-less scanned PDFs, cross-source joint reasoning) without speculative hallucination.

## Benchmark Matrix Structure

The suite comprises 104 test cases organized into:

- `test_ingestion.py`: Ingestion & backend boundary enforcement (13 cases).
- `test_extraction.py`: Fact extraction fidelity across PDF, DOCX, and TXT (15 cases).
- `test_retrieval.py`: Retrieval precision, recall, needles, and distractors (15 cases).
- `test_refusal.py`: Unsupported questions, hallucination probes, and contradictions (12 cases).
- `test_grounding.py`: Source citation and attribution (2 cases).
- `test_prompt_injection.py`: Adversarial injections and untrusted context framing (15 cases).
- `test_isolation.py`: Cross-session isolation and security gates (10 cases).
- `test_edge_cases.py`: Resource limits, file spoofing, and long-document needles (10 cases).
- `test_dataset_files.py`: CSV/XLSX/JSON profiling and dual-attachment routing (12 cases).

## Running the Benchmark

### Dedicated Benchmark Runner
```powershell
.\venv\Scripts\python evaluation/reliability/run_benchmark.py
```

### Pytest Runner
```powershell
.\venv\Scripts\pytest evaluation/reliability -q
```

### Regression Baseline Separation
The standard regression suite is configured via `pytest.ini` (`testpaths = tests`) to ensure `pytest -q` runs only the 245 core unit and integration tests.

## Reports & Artifacts
Results are automatically generated under:
- `evaluation/reliability/results/latest.json` (Machine-readable)
- `evaluation/reliability/results/latest.md` (Human-readable Markdown summary)
