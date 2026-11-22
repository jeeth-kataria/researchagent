# Reproducible paper-review comparison

The target is **12 real papers** with independently established issues. There is now
**one pinned public pilot case with two reference issues; 0 real papers are scored**. The default custom manifest is empty. Two original
synthetic fixtures exercise the mechanics. They are not a research-paper benchmark.
No private ACL ARR paper or reviewer feedback is included or inferred.

## Application upgrade boundary

The application now uses BM25 retrieval, layout extraction, a separately configurable
critic and optional targeted revision. No new evaluation run or label was added for
these changes. This CLI retains its existing same-model, no-revision comparison;
archived cases default to plain extraction so exact reference anchors remain valid.
Do not attribute earlier artifacts to this upgraded configuration or claim a measured
BM25/revision improvement. Application coverage limits were not expanded.

## Run the guided public pilot

From the project root with the existing virtual environment and `.env`:

```sh
.venv/bin/python evaluate.py pilot --live --max-calls 13 \
  --out artifacts/public-evaluation
```

This downloads the pinned paper/review/licence, verifies their hashes and reference
quotes, runs both approaches, saves raw outputs and blank annotations, and asks you
to assess each paper concern. It shows the reference criteria and exact cited source
passages. Use `u` for uncertain findings, `q` to save and stop. It never supplies
AI-generated correctness labels or attests on your behalf. After your explicit `YES`,
it writes a `.scores.json` file. The pilot defaults to an output cap of 1,536
tokens for both arms and 30 seconds between request starts. Set `--output-tokens`
and `--request-interval` explicitly to override. Spacing is not retrying; measured
elapsed time includes these waits. It does not guarantee quota availability, and
the workflow still has its cooperative 240-second deadline. The 13-request ceiling
is a count limit, not a promise every request fits the time/token budget. A completed run is not a scored paper until both
arms have completed and every unique paper concern is adjudicated.

The case is **A New Formula for Vietnamese Text Readability Assessment**, submission
68 in [PeerRead's ACL 2017 archive](https://github.com/allenai/PeerRead/tree/9bb37751781a900cee9e74ec3105997732c8e8e5/data/acl_2017).
The collection is CC BY 4.0; attribution: Kang et al., *A Dataset of Peer Reviews
(PeerRead)*, NAACL 2018. The registry pins the repository revision, PDF/review/licence
hashes, review index, quoted comment, matching criterion and source anchors. Its two
reference concerns are missing baseline comparisons and unclear Vietnamese word
segmentation. They are reviewer concerns, not unquestionable factual errors.

Version evidence is the archive's paired submission ID plus a content cross-check:
the review's quoted 81% practical-use claim appears in that draft. This is not a live
revision-history audit. No private ARR feedback is used. Unclear-licence ICLR
candidates and known revised-version candidates were excluded from this pilot.

The complete PDF contains 24,203 extracted characters before per-page stripping.
It exceeds the default web cap, so the case explicitly records an **evaluation-only
30,000-character cap**; both arms receive the same untruncated extraction. The web
configuration is unchanged. This pilot tests the backend with that override; it
does not establish that the deployed default upload settings accept this paper.
Raw PDFs, reviews, prompts, judgements and results remain under ignored `artifacts/`.

Preparation and inference can be separated, and guided assessment can be resumed:

```sh
python evaluate.py prepare-public --out artifacts/public-evaluation
python evaluate.py plan --cases artifacts/public-evaluation/cases.json --case-limit 1
python evaluate.py run --cases artifacts/public-evaluation/cases.json \
  --case-limit 1 --max-calls 13 --live --out artifacts/public-evaluation/runs
python evaluate.py review --run PATH_TO_RUN.json --annotations PATH_TO_ANNOTATIONS.json
```

`run` creates the annotations file automatically and stops with nonzero exit status
on an incomplete arm. `review` saves after every answer and validates run/finding
fingerprints. Resume the existing run to adjudicate; starting `pilot` again explicitly
starts another quota-consuming investigation, not an inference resume.

**Observed diagnostics:** the original Python run failed at DNS. Standalone
network-enabled curl subsequently authenticated successfully, found the configured
model active and completed a 97-token `OK` inference. The initial full-paper
baseline received HTTP 413: requested 8,555 tokens versus an observed account
limit of 8,000. Removing JSON-escaped PDF lines reduced overhead without dropping
source text; the request still exceeded the limit at 8,068. With a 1,536 output cap,
the full-paper baseline completed (7,839 prompt + 281 completion tokens reported).
These provider accounting figures are observations, not a universal formula for
predicting admission under the limit.

The paired workflow stopped after two requests because its reader omitted required
finding fields. Validation preserved the partial run. The prompt now explicitly
requires `evidence_ids: []` for uncited missing-information findings, and narrows
absence claims to inspected passages when coverage is partial. The later prompt and
pacing changes have offline tests; no completed four-role real-paper comparison or
human score is claimed. [Sanitized diagnostic record](groq_diagnostic.json) retains
settings, code hashes and observed outcomes. Raw artifacts remain private. No billing
or provider changes were made.

## Prepare cases and estimate calls

```sh
python evaluate.py plan
python evaluate.py prepare-synthetic
python evaluate.py plan --cases artifacts/evaluation/cases.json --case-limit 1
```

One pair costs at most **13 model requests**: baseline 1, workflow at most 12.
Twelve pairs cost at most **156 requests**, before any explicitly started new run.
The runner requires both `--live` and an explicit `--max-calls` sufficient for the
printed estimate. It stops the batch on an incomplete arm, including quota failure,
and saves partial output. It never extrapolates or retries.

```sh
# Consumes existing provider quota. Review the plan first.
python evaluate.py run --cases artifacts/evaluation/cases.json \
  --case-limit 1 --max-calls 13 --live --out artifacts/evaluation-runs
```

Both arms use `GROQ_MODEL`, the same PDF bytes/hash, question, output-token limit
and final concern schema. The single-call baseline gets **all extracted passages**
for inputs accepted under each case's recorded limit (20,000 by default; the public pilot explicitly uses 30,000), not a reduced keyword
sample. It has no tools. The workflow has bounded section packets, two retrieval
tools and up to 12 requests. Its source access and call budget therefore differ;
results cannot isolate the causal effect of having four roles.

Each saved run contains the source/version identifiers, model and settings, code
hashes, dependency versions, outgoing prompt/tool conversations, outputs, confirmed
source delivery, request/tool counts, provider token counts where returned, and
actual per-arm elapsed seconds from PDF extraction through output. Timing excludes
upload transit to the application. Missing token usage is null, not a guessed zero.
Results are stochastic; preserving inputs does not guarantee identical reruns.
Generated PDFs, raw outputs and annotations stay under ignored `artifacts/`.

## Add genuine cases

Create a local manifest under `artifacts/` using this format. Do not commit private
PDFs/reviews. Each real case needs permission/licence and a curator-confirmed exact
version match between the PDF and its reference reviews.

```json
{
  "target_real_cases": 12,
  "cases": [{
    "id": "replace-with-real-case-id",
    "kind": "real",
    "paper_identifier": "DOI, OpenReview note ID, or stable identifier",
    "version": "Exact revision/date and permanent PDF URL",
    "pdf": "paper.pdf",
    "pdf_sha256": "SHA-256 of those exact PDF bytes",
    "source": "Paper and review URLs, review date/revision identifiers",
    "permission": "Licence or explicit permission for this use",
    "question": "Focused review task",
    "known_issues": [{
      "id": "issue-1",
      "category": "ambiguity",
      "description": "Specific independently documented issue",
      "provenance": "Exact review URL/identifier and quoted comment",
      "version_match_confirmed": true,
      "match_criteria": "What a finding must identify to count as a match",
      "passages": [{"page": 2, "quote": "Exact extracted source text"}]
    }]
  }]
}
```

Use categories `ambiguity`, `missing_detail`, `unsupported_claim`, or
`subjective_suggestion`. Review comments are evidence of a concern, not automatic
proof of a factual error. Absence claims can have no passage anchor; record their
scope and matching criterion. The loader verifies PDF hashes and supplied passage
quotes. Version provenance, permission and independent labels still require curator
judgement. One public pilot pair is prepared; reaching the 12-case target still requires
11 more permitted, version-matched pairs and criteria, completed model runs and
human assessment. No larger corpus or completed score is implied.

## Human assessment and scoring

```sh
python evaluate.py annotate --run artifacts/evaluation-runs/RUN.json \
  > artifacts/annotations.json
# A human fills in assessor/rationale/verdicts and explicitly attests the assessment.
python evaluate.py score --run artifacts/evaluation-runs/RUN.json \
  --annotations artifacts/annotations.json > artifacts/scores.json
```

Annotations bind to the run ID and a fingerprint of each concern. Set top-level
`assessor` and `human_attested: true` only after human inspection. Each judged finding
needs `assessor`, `rationale` and a verdict:

- `matches_known_issue`: include existing `issue_ids` satisfying the recorded criteria.
- `false_alarm`: human-determined unsupported concern, not merely absent from labels.
- `additional_valid`: useful concern outside an incomplete reference list.
- `duplicate`: identify an earlier `duplicate_of` finding; exact normalized duplicates
  are also counted mechanically without double credit.
- `unjudged`: no correctness claim or automatic false-alarm penalty.

Only paper-target concerns enter this comparison; report-target concerns are counted
separately. Distinct known-issue hits, known issues not matched, false alarms,
duplicates, unjudged findings and additional valid concerns retain raw denominators.
Recall/missed counts require a completed, fully adjudicated arm; otherwise they are
null. Precision uses judged unique paper findings only. Synthetic scores never count
as real-paper scores. There is no LLM judge or automatic semantic label matching.

A meaningful résumé accuracy comparison requires completed real runs **and** human
adjudication. The implemented runner and coverage ledger can be described without
claiming an accuracy improvement.

## Observed operational smoke

[observed_smoke.json](observed_smoke.json) records a completed production run on the
original synthetic comparison fixture, including the exact PDF hash, code revision,
output, source receipts and provider usage. It used four requests and zero tools
because full source packets were supplied. Its HTTP elapsed time is one actual
observation, not a benchmark. Two earlier incomplete attempts are recorded too.
No human adjudication or real-paper accuracy score is assigned to this smoke.
