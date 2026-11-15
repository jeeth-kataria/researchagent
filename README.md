# Paper Research Team — Research Desk

Examine **one English research-paper PDF**: understand its proposal, inspect the
reported evidence, and see what remains unclear, unsupported, or missing.
The existing Research Desk interface uses plain HTML/CSS/JavaScript and a small
Python backend. It does not search other literature or establish global novelty.

The motivation came from Jeeth's ACL ARR submission: reassuring LLM feedback
missed ambiguities human reviewers later identified. That motivates a separate
critical stage; it does **not** demonstrate that this app detects those ambiguities
reliably or is more accurate than one model call.

## Run locally

Use Python 3.12. Keep your existing `.env`; copy the example only on first setup.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
# First setup only, if .env does not exist:
[ -f .env ] || cp .env.example .env
# Edit .env locally: set GROQ_API_KEY and GROQ_MODEL.
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

Open **http://localhost:8000**. `python app.py` also launches the same FastAPI app
through Uvicorn (without reload). Windows activation: `.venv\Scripts\activate`.
Environment variables override `.env`; restart after changes. Keys belong only
in the server environment, never browser code, reports, chat, or Git.

```sh
python -m pytest -q
# Optional frontend syntax checks (Node is not needed to run the application):
node --check public/app.js
node --check tests/frontend_contract.cjs
```

## The fixed workflow

```mermaid
flowchart LR
    S([START]) --> R[Paper Reader] --> A[Evidence Analyst]
    A --> W[Report Writer: draft] --> C[Critical Reviewer: final] --> E([END])
```

| Role | Responsibility and direct state context |
| --- | --- |
| Paper Reader | Inspect sources; explain problem, approach, contributions; separate author claims from interpretations |
| Evidence Analyst | Receive reader findings; inspect datasets, baselines, metrics, conditions, results and missing details |
| Report Writer | Receive reader + analyst findings; produce the paper-explanation draft, optionally checking sources |
| Critical Reviewer | Receive both findings + draft; inspect sources and critique both the paper and its representation in the draft |

All four roles use **one configured Groq-hosted model**. LangGraph defines fixed
edges and a typed shared state; no supervisor, correction loop, repeated review,
or writer invocation after review exists. A failed stage marks the run incomplete;
later nodes skip execution without making model calls.

The result shows the **paper explanation (unchanged draft)**, the **final critical
review**, source passages, and remaining uncertainties. Review concerns name their
`target` (`paper` or `report`), category, concern, evidence IDs, and suggested
clarification. Report concerns also copy an exact `affected_statement` from the
draft so the UI can identify the disputed text. Missing-information concerns may
have no evidence IDs. No concern is silently applied as a rewrite or treated as
scientific approval. A failed reviewer leaves a visibly unreviewed draft.

## Two retrieval tools

- `search_paper(query)`: case-insensitive keyword overlap over exact passages;
  returns up to four matches, ranked by overlap then physical page/offset.
- `get_evidence(evidence_id)`: return one exact stored passage with page and offsets.

Both tools are permitted for all four roles. Reader, analyst, and reviewer must
receive **nonempty source evidence** before finalizing; an empty search alone does
not satisfy this check. Writer may finish from shared findings without tools.

`run_agent` sends role instructions, direct state context and explicit function
schemas to Groq. The model selects tools; Python validates the whole batch's names,
arguments and call IDs, executes a fixed dispatch table, then sends ID-matched
`tool` replies for the next request. No model-generated code is executed. Final
JSON is validated locally; no strict `response_format` is combined with tool calls.
Malformed JSON, unknown IDs, invalid pages or source slices stop the run visibly.

Python assigns deterministic IDs such as `E0001` to exact slices of at most 800
characters, using **physical PDF pages**, including covers. References like
`[E0001 · PDF page 1]` are validated against those slices. They establish an exact
source address, **not** semantic support. Keyword retrieval can miss paraphrases
and is not embedding-based search. There is no vector database or external search.

## Configuration and bounds

| Variable | Default / behavior |
| --- | --- |
| `GROQ_API_KEY` | Required server-side; local `.env` or hosting environment |
| `GROQ_MODEL` | `openai/gpt-oss-20b`; same model for every role |
| `GROQ_MAX_OUTPUT_TOKENS` | 2048 per request; allowed 128–8192 |
| `MAX_UPLOAD_MB` | 4 MiB; web boundary caps at 4 |
| `MAX_TEXT_CHARS` | 20,000 extracted characters; rejects rather than truncates |
| `MAX_MODEL_REQUESTS_PER_VISIT` | 3; allowed 1–3 |
| `MAX_TOOL_CALLS_PER_VISIT` | 2; allowed 1–2 |
| `MAX_TOTAL_MODEL_REQUESTS` | 12; configurable 1–24 for compatibility with older `.env` files |
| `PORT` | 8000 for `python app.py`; the Uvicorn CLI uses `--port` |

Each role is visited once. Four visits × three requests = **at most 12 model
requests**, even when an older total setting is 24. Four visits × two tools =
**at most 8 tool executions**. A lower total cap stops the run sooner. A successful
minimal path needs seven requests and three tools: source request + final JSON
for reader, analyst and reviewer, and one writer response. One response requesting
two tools counts as one model request and two tool executions. Counters increment
before invocation and include failed attempts. The final request disables tools;
insufficient budgets produce incomplete results rather than extra requests.

The graph has a cooperative 240-second deadline after extraction; model calls
use remaining time, capped at 45 seconds. This does not interrupt all PDF parsing
or provide a hard process-level deadline. SDK retries are disabled. Provider errors
use safe messages; no retry, provider switch, paid upgrade or billing setup occurs.
Free account quotas can still interrupt a run; all public users share the server
key's allowance. Consult your account and the [Groq rate-limit documentation](https://console.groq.com/docs/rate-limits).
The configured model is retained from the existing project; current account/model
availability and live inference were not reverified in this change.

## Files and technology

| File / dependency | Purpose and tradeoff |
| --- | --- |
| `app.py`, `api/research.py` / FastAPI | ASGI endpoints, validated uploaded form data, safe errors and NDJSON stream; the Vercel entrypoint exports the same app |
| Uvicorn | Local ASGI server, replacing the manual standard-library server; Vercel uses its own Python runtime |
| python-multipart | Form/file parsing used by FastAPI; replaces the custom email-parser code |
| `public/` | Existing responsive Research Desk design, safe DOM text, sources/activity, tab storage and Markdown export; custom JS instead of Streamlit |
| `pdf_reader.py` / pypdf | In-memory embedded-text extraction and exact passage preparation; a layout-aware parser would help difficult tables |
| `tools.py` | Two schemas, allowlists, argument validation and fixed Python dispatch |
| `agents.py` | Four prompts, JSON/evidence validators and one bounded agent/tool loop |
| `workflow.py` / LangGraph | Shared state, four fixed nodes/edges, deterministic output formatting; plain functions would also work for a sequence this small |
| `llm.py` / Groq SDK | One provider helper, timeouts, disabled retries, safe errors and redaction; Groq hosts the model, it is not itself the model |
| python-dotenv | Local configuration convenience; deployment uses ordinary environment variables |
| `tests/` / pytest + HTTPX | FastAPI TestClient and mocked graph/HTTP tests; optional Node frontend-contract and real-socket smoke checks |

LangGraph supplies execution/state management; the code does not use LangChain
model wrappers or retrieval chains. Supporting packages may be transitive dependencies.
Retrieval augments responses in the broad sense, but there is no conventional
embedding-based RAG pipeline. Four roles separate responsibilities as a design
choice, not proof of improved accuracy.

## HTTP requests and streaming

FastAPI defines `GET /`, `/app.js`, `/styles.css`, and `GET`/`POST /api/research`.
Only the three intended public files are served locally, without a root-directory
static mount. API configuration returns key presence and limits, never the key.
`UploadFile` and `Form` handle one PDF and one question. Invalid fields, malformed
forms and PDF/configuration errors return readable `{"error": "..."}` responses
before streaming; the existing frontend needs no changes.

The endpoint strips/limits the question to 2,000 characters and reads **at most
upload-limit + one byte**, checking the actual bytes rather than trusting MIME,
filename or Content-Length. All parsed upload files close before extraction,
including rejected duplicate/extra uploads. Multipart parsing may spool files to
temporary disk **before** endpoint validation: this read bound is not a network
request-body limit. The hosted payload ceiling still applies.

PDF extraction runs in a worker thread. `InvestigationResponse`, a small
`StreamingResponse` subclass, consumes the existing synchronous NDJSON iterator
through Starlette's thread pool and closes it on disconnect. Blocking graph/model
execution does not run on the async event loop. No queue, job service, WebSocket,
SSE, token streaming or async Groq rewrite was added. A synchronous call already
running cannot be instantly cancelled; cleanup waits for the current worker step.

NDJSON is **one JSON object per line in an HTTP response**, not a replacement for
HTTP. The unchanged `extraction`, `state`, `event`, `finished` messages let the UI
show real progress and preserve partial results. Failures after streaming begins
use incomplete state messages; HTTP 200 alone does not establish success. Migration
to ASGI does not establish higher performance or make synchronous model calls async.

## Limits, privacy and hosting

Unreadable, encrypted, empty, apparently scanned or oversized PDFs are rejected.
No OCR or English-language detection exists. Tables, equations, columns and mixed
scanned pages can extract poorly. Inspect the Extracted text view. Real but
irrelevant passages can pass validation; limited review can miss important issues.
PDF/tool content is treated as untrusted, but prompt instructions do not guarantee
injection resistance. This public prototype has no authentication or shared
per-user abuse controls; per-run bounds do not prevent aggregate quota abuse.

After multipart processing, PDF bytes/registries live in request memory. Selected passages/question go to Groq.
The browser keeps received pages, findings and events in tab `sessionStorage` under
`research-desk.run.v2`. Refresh restores data without rerunning; interrupted runs
become incomplete. Old v1 records are ignored. New investigation clears the current
record. There is no server database, durable resume or background worker.

The existing Vercel static/Python hosting structure is intentionally retained.
Official documentation checked for this migration confirms [ASGI `app` exports in
file-based Python functions](https://vercel.com/docs/functions/runtimes/python/api-directory).
`api/research.py` now exports `from app import app`, the FastAPI instance.
`vercel.json` keeps `framework: null` (the [Other preset](https://vercel.com/docs/project-configuration/vercel-json#framework)),
`outputDirectory: public`, and the `api/research.py` function settings. Thus static
files keep their URLs and the file-based function keeps `/api/research`; no catch-all
rewrite, new frontend build or framework-preset migration is needed. Python
packages come from `requirements.txt`, with Python 3.12 selected by `.python-version`.
The 300-second function setting, security headers and bundle exclusions remain.
The [Vercel payload ceiling](https://vercel.com/docs/functions/limitations#request-body-size)
is 4.5 MB; the application's PDF cap remains 4 MiB.

Set the key/model in the server environment, never public files. `.env` and local
study notes remain excluded by `.vercelignore` and function-bundle exclusions.
In a publishing checkout, locally exclude `/local-docs/` through `.git/info/exclude`
before staging; private study notes are not part of the public source upload.
Migration verification made no deployment, production-setting change or paid-service
change. ASGI import/routing and
configuration were checked locally; a Vercel build/deployment and current live
behavior remain unverified. Existing aliases may serve the previous implementation.

## Verification for this revision

Python **3.12.2**; pinned direct versions: pypdf **6.21.4**, LangGraph **1.2.20**,
Groq SDK **1.7.0**, python-dotenv **1.2.4**, pytest **9.2.0**, FastAPI **0.136.1**,
Uvicorn **0.54.0**, python-multipart **0.0.33**, HTTPX **0.28.1**. Also tested with
Starlette **1.10.0**, Pydantic **2.14.2**, AnyIO **4.16.0**. The added FastAPI
package was installed from the existing local package cache; dependency compatibility
checks passed. Requirements pin direct dependencies, not every transitive package.

- **120 tests passed; one real-socket test skipped** because this sandbox denies
  binding local sockets. All provider responses were mocked; core workflow/tool/PDF
  tests were preserved.
- HTTP tests use FastAPI's TestClient/HTTPX. Six existing frontend-contract cases
  consume the same NDJSON payloads using a minimal DOM substitute.
- Tests cover bounded reads, file closure, safe validation errors, private-path
  exclusion, fixed graph behavior, partial results, thread-pool execution and
  disconnect cleanup. Both JavaScript syntax checks passed.
- `python app.py` completed Uvicorn/ASGI startup and shutdown, but socket binding
  failed with permission denied. In-process smoke checks passed. TestClient buffers
  responses, so incremental network/browser delivery was **not** verified here.
  `python -m pytest -q -rs tests/test_smoke.py` attempts that check on an unrestricted
  machine using a real HTTP connection and a deliberately blocked mocked model.
- Agent/core and frontend files were unchanged. No live Groq requests, accuracy
  benchmarks, real-browser QA, Vercel build or deployment was performed.
