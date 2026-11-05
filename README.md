# Paper Research Team

Four specialist roles investigate **one uploaded English research-paper PDF**:
Paper investigator → Methods analyst → Critical reviewer → Report writer.
The model chooses local Python tools through Groq function calling. LangGraph
routes at most one targeted correction. Results include evidence/page references,
review history, actual tool activity, and a Markdown download.

## Setup and run

Use Python **3.12**. From the project directory:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
# Edit .env locally: set GROQ_API_KEY to your own free-plan Groq key.
python -m streamlit run app.py
```

On Windows, activate with `.venv\Scripts\activate`. Obtain your key from the
[Groq console](https://console.groq.com/keys); never put it in chat or Git.
Restart Streamlit after editing `.env`. Shell environment variables take precedence
 over `.env`, then Streamlit secrets supply missing settings.

```sh
python -m pytest -q
```

| Variable | Default / allowed maximum |
| --- | --- |
| `GROQ_API_KEY` | Required, server-side only |
| `GROQ_MODEL` | `openai/gpt-oss-20b`, shared by all four roles |
| `GROQ_MAX_OUTPUT_TOKENS` | 2048 per request; allowed 128–8192 |
| `MAX_UPLOAD_MB` | 10 MiB |
| `MAX_TEXT_CHARS` | 20,000 extracted characters, before page labels |
| `MAX_MODEL_REQUESTS_PER_VISIT` | 3; configurable downward |
| `MAX_TOOL_CALLS_PER_VISIT` | 2; configurable downward |
| `MAX_TOTAL_MODEL_REQUESTS` | 24; configurable downward |

A correction cycle is hard-capped at one. Very low budgets intentionally produce
incomplete runs. Update `.streamlit/config.toml` too when changing the default
upload size. Questions are limited to 2,000 characters.

## Workflow and shared state

```mermaid
flowchart TD
    S([START]) --> I[Paper investigator]
    I --> M[Methods analyst]
    M --> R[Critical reviewer]
    R -->|accepted or correction already used| W[Report writer]
    R -->|investigator issue; first correction| I
    R -->|methods issue; first correction| M
    W --> E([END])
```

Every node also ends the run if a request, validation, or budget check fails.
`ResearchState` carries pages, the evidence registry, current findings, review
history, correction count, report, execution events, and counters. Nodes update
this dictionary; conditional edges inspect it. There is no fifth LLM supervisor.
The correction count increments **before** routing back. Investigator correction
clears both agents' current findings; methods correction clears only methods.
Earlier versions remain in the correction history. The second reviewer pass
always proceeds to the writer, preserving unresolved issues. The writer receives
no additional reviewer pass.

## Function calling and evidence

`llm.call_model` is the only SDK request helper. `agents.run_agent` sends role
instructions, relevant state, and permitted function schemas. It reads the model's
tool requests, validates names/arguments, runs ordinary Python functions, appends
an assistant tool-call message and matching `role="tool"` replies using the exact
call IDs, then requests the next response. It does not execute a scripted tool
sequence. Final JSON findings/decisions are validated locally.

| Tool | Purpose | Roles |
| --- | --- | --- |
| `list_pages()` | Physical pages and short previews | Investigator |
| `search_paper(query)` | Keyword overlap, up to four passages | Investigator, Methods |
| `read_page(page_number)` | Page text and evidence references | Investigator, Methods, Reviewer |
| `get_evidence(evidence_id)` | Exact source passage and page | Methods, Reviewer, Writer |
| `check_quote(page_number, quote)` | Whitespace-normalized quote existence | Reviewer |
| `get_findings()` | Current findings and latest review | Writer |

Python assigns IDs such as `E0001` to exact slices of extracted text, recording
page and offsets. Models cannot register evidence. Citations such as
`[E0001 · PDF page 2]` are generated from validated IDs. Bounds, source slices,
and canonical ID/page pairs are checked; **semantic support is not guaranteed**.

Tool outputs are bounded: passages ≤800 characters, four search matches, first
40 page previews, and first 4,000 characters per page read. Partial page reads
are explicitly marked and list remaining evidence IDs; search/lookup can retrieve
other passages. PDF extraction itself is never silently truncated. Quote checks
collapse Unicode whitespace runs to single spaces and strip ends; they remain
case-sensitive, without punctuation or Unicode-composition normalization.

## Technology and limits

Streamlit handles the complete UI; pypdf extracts text; LangGraph expresses shared
state/conditional edges; the official Groq SDK handles inference and function
calling; python-dotenv loads local settings; pytest tests offline behavior. Other
logic uses the standard library. No top-level LangChain wrappers are used, although
LangGraph installs supporting packages. Keyword search keeps one short paper
simple; embeddings, vector search, RAG, databases, and web search are unnecessary
for this version. Keyword matching can miss paraphrases.

Checked **2026-11-05**: Groq lists [GPT-OSS 20B with function calling](https://console.groq.com/docs/model/openai/gpt-oss-20b)
and [free-plan limits](https://console.groq.com/docs/rate-limits) of 30 requests/minute,
1,000/day, 8,000 tokens/minute, and 200,000/day; account limits may differ. This is
Groq-hosted inference, not the OpenAI API. Groq's [strict structured outputs](https://console.groq.com/docs/structured-outputs)
do not support tool use, so this loop does not combine `response_format` with
tools. It requests JSON through instructions and rejects malformed output.

Use a **free-plan account without billing enabled**. The app cannot establish your
account tier or enforce account-wide spending settings. There are no automatic
retries, provider switches, or paid upgrades. All attempted requests count,
including failures and follow-ups. Multiple tools in one response count separately;
an over-budget batch is rejected before execution. The last available model
request disables tools to reserve a final response. The graph permits at most
seven visits (21 requests at three per visit), within the additional 24-request
ceiling. Quotas can still interrupt runs; partial findings remain inspectable and
downloadable, clearly marked incomplete. A new click starts a new run using quota.

No OCR or language detection is included. English is an input requirement.
Scanned, encrypted, damaged, empty, and oversized files are rejected. Mixed scans,
columns, equations, figures, and tables can lose meaning; inspect the preview.
Selected passages/question go to Groq. Files are processed in memory. PDF/tool
content is untrusted; prompts address injection but cannot guarantee resistance.
Events show operations and explicit findings, never private model reasoning.
Credential-looking values and the configured key are redacted in views/exports.
Agent agreement is not scientific verification. Follow-ups are proposals; no
external literature search or global novelty assessment occurs.

## Files and verification

`app.py`: UI, settings, session state, export. `pdf_reader.py`: extraction/passages.
`tools.py`: schemas, validation, six tools. `agents.py`: prompts, JSON validation,
bounded loop. `workflow.py`: typed state, graph, correction routing, report rendering.
`llm.py`: Groq client/helper and credential redaction. `tests/`: PDF, tools, graph tests.

Tested on **Python 3.12.2 / macOS**, with direct versions pinned in
`requirements.txt`: Streamlit 1.66.1, pypdf 6.21.4, LangGraph 1.2.20, Groq 1.7.0,
python-dotenv 1.2.4, pytest 9.2.0.

- **63 tests passed** with mocked SDK responses and actual compiled graph routing.
- All 68 installed packages passed dependency compatibility checking.
- Streamlit AppTest startup and a supplementary full UI smoke check passed,
  including the real SDK with offline HTTP responses, streamed events, downloads,
  persistent results, invalid uploads, and incomplete-run views.
- A headless server launch was attempted; sandbox socket restrictions blocked it.
  Browser serving, live Groq behavior/model quality, and deployment remain unverified.
  An existing environment credential was not used because its account tier was unconfirmed.

## Free hosting (manual)

Use [Streamlit Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).
Current official instructions support maintained Python versions and default to
3.12. Manually put the source in your GitHub repository, choose its branch and
`app.py` entrypoint, and select Python **3.12** in Advanced settings. The root
`requirements.txt` supplies dependencies; no system services are required.

Add these [deployment secrets](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management):

```toml
GROQ_API_KEY = "your-own-free-plan-key"
GROQ_MODEL = "openai/gpt-oss-20b"
GROQ_MAX_OUTPUT_TOKENS = 2048
```

Other variables can also be set there. Never commit `.env`,
`.streamlit/secrets.toml`, uploaded PDFs, or generated reports. Public visitors
share the key's quota; Community Cloud has resource limits and no uptime guarantee.

Private study notes under `local-docs/` are not part of this repository. Keep any
local copy excluded through `.git/info/exclude`, rather than shared ignore rules.
Deployment is manual; running the application does not publish or deploy it.
