# Reproducible paper-review comparison

The target is **12 real papers** with independently established issues. The current
real-case manifest contains **0 cases; 0 real papers are scored**. Two original
synthetic fixtures exercise the mechanics. They are not a research-paper benchmark.
No private ACL ARR paper or reviewer feedback is included or inferred.

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
for accepted inputs (up to 20,000 extracted characters), not a reduced keyword
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
judgement. Missing inputs are the 10–20 version-matched papers/reviews, permissions,
reference-issue criteria and human assessment. No real corpus has been fabricated.

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
