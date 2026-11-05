"""Streamlit UI: one explicit start action, with persistent complete/partial results."""

import hashlib
import json
import os
import re
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langgraph.config import get_stream_writer

from agents import Limits
from llm import RunError, make_client, sanitize
from pdf_reader import PDFError, extract_pdf, labeled_text
from workflow import build_graph, initial_state

load_dotenv(Path(__file__).with_name(".env"))
st.set_page_config(page_title="Paper Research Team", page_icon="📄", layout="centered")


def setting(name: str, default: str = "") -> str:
    """Shell environment/.env takes precedence over Streamlit secrets."""
    if name in os.environ:
        return os.environ[name].strip()
    try:
        return str(st.secrets.get(name, default)).strip()
    except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
        return default


def integer_setting(name: str, default: int) -> int:
    try:
        result = int(setting(name, str(default)))
        if result < 1:
            raise ValueError
        return result
    except ValueError:
        raise RunError(f"{name} must be a positive integer.") from None


def markdown_download(result: dict) -> str:
    """Export validated findings and source passages, including incomplete-run status."""
    state = result["state"]
    parts = ["# Paper Research Team — run record",
             f"PDF: {result['filename']}\n\nSHA-256: {result['digest']}",
             f"Model via Groq: {result['model']}\n\nStatus: {state['status']}",
             f"Question: {state['question']}",
             f"Model requests: {state['model_calls']} · Tool executions: {state['tool_calls']}"]
    parts.append(state["final_report"] if state["status"] == "complete"
                 else "## Incomplete run\n" + state["error"] + "\nNo completed report was produced.")
    for key, title in [("investigator_findings", "Investigator findings"),
                       ("methods_findings", "Methods findings"),
                       ("reviews", "Reviewer history"), ("correction_history", "Correction history")]:
        parts.extend([f"## {title}", "```json\n" + json.dumps(state[key], indent=2, ensure_ascii=False) + "\n```"])
    referenced = set(re.findall(r"\bE\d{4,}\b", "\n".join(parts)))
    parts.append("## Referenced source passages")
    for evidence_id in sorted(referenced):
        if evidence_id in state["evidence"]:
            entry = state["evidence"][evidence_id]
            parts.extend([f"### {evidence_id} · PDF page {entry['page']}", entry["text"]])
    parts.append("Only the uploaded paper was investigated. No external literature search or "
                 "global novelty assessment was performed. Valid references and matching quotes "
                 "do not guarantee semantic correctness. Agent agreement is not scientific verification.")
    return "\n\n".join(parts)


st.title("Paper Research Team")
st.write("Four specialist agents investigate one paper, assess its methods, review the findings, "
         "and write a report. They may make one targeted correction.")
st.caption("English PDFs with embedded text only. No OCR or external-paper search. "
           "On Start investigation, your question and selected paper passages are sent to Groq.")
try:
    max_upload = integer_setting("MAX_UPLOAD_MB", 10)
    max_chars = integer_setting("MAX_TEXT_CHARS", 20_000)
    limits = Limits(
        model_per_visit=integer_setting("MAX_MODEL_REQUESTS_PER_VISIT", 3),
        tools_per_visit=integer_setting("MAX_TOOL_CALLS_PER_VISIT", 2),
        total_model_calls=integer_setting("MAX_TOTAL_MODEL_REQUESTS", 24),
        output_tokens=integer_setting("GROQ_MAX_OUTPUT_TOKENS", 2048),
    )
except RunError as error:
    st.error(str(error))
    st.stop()

upload = st.file_uploader("Upload one research-paper PDF", type=["pdf"], max_upload_size=max_upload)
question = st.text_area("Question about the paper", max_chars=2000,
    placeholder="Explain this paper’s contribution, assess its experimental evidence, identify "
                "its stated limitations, and suggest follow-up research questions.")
model = setting("GROQ_MODEL", "openai/gpt-oss-20b")
st.caption(f"Upload limit: {max_upload} MB · Extracted text: {max_chars:,} characters · "
           f"Groq model: {model}")
st.caption(f"Per visit: {limits.model_per_visit} model requests, {limits.tools_per_visit} tools. "
           f"Run limit: {limits.total_model_calls} model requests and one correction cycle. "
           "A run can exhaust free-tier quotas.")

if st.button("Start investigation", type="primary"):
    if upload is None or not question.strip():
        st.error("Upload a PDF and enter a question before starting.")
    else:
        status = st.empty()
        try:
            status.info("Extracting PDF text…")
            data = upload.getvalue()
            pages = extract_pdf(data, max_upload, max_chars)
            st.session_state["extraction"] = {"filename": upload.name, "text": labeled_text(pages)}
            state = initial_state(question, pages)
            key = setting("GROQ_API_KEY")
            result = {"filename": upload.name, "digest": hashlib.sha256(data).hexdigest(),
                      "model": model, "state": state}
            try:
                with make_client(key) as client:
                    # Custom events stream to the main UI thread; no Streamlit calls in graph nodes.
                    graph = build_graph(client, model, limits,
                                        on_event=lambda event: get_stream_writer()(event))
                    for mode, update in graph.stream(state, stream_mode=["updates", "custom"]):
                        if mode == "custom":
                            detail = update.get("tool", update["kind"].replace("_", " "))
                            status.info(f"{update['agent'].capitalize()}: {detail}")
                        elif mode == "updates":
                            result["state"] = next(iter(update.values()))
            except RunError as error:
                result["state"]["status"] = "incomplete"
                result["state"]["error"] = str(error)
            except Exception:
                result["state"]["status"] = "incomplete"
                result["state"]["error"] = "Unexpected workflow error. Available findings are preserved."
            # Views/exports redact known credentials; the raw registry is used only during execution.
            st.session_state["result"] = sanitize(result, key)
            status.empty()
        except (PDFError, RunError) as error:
            status.empty()
            st.error(str(error))

if "extraction" in st.session_state:
    extraction = st.session_state["extraction"]
    with st.expander(f"PDF extraction preview — {extraction['filename']}"):
        st.text(sanitize(extraction["text"], setting("GROQ_API_KEY")))
        st.caption("Physical PDF page numbers include cover pages. Check extraction order and "
                   "missing tables, figures, formulas, or scanned pages. No text is silently truncated.")

if "result" in st.session_state:
    result = sanitize(st.session_state["result"], setting("GROQ_API_KEY"))
    state = result["state"]
    st.divider()
    st.subheader("Investigation results")
    st.write("PDF: " + result["filename"])
    st.caption(f"SHA-256: {result['digest']} · Model via Groq: {result['model']}")
    st.write("Question: " + state["question"])
    st.caption("These results belong to the PDF/question above. Changing inputs does not rerun "
               "the team. An invalid new upload leaves the previous result visible.")
    st.write(f"Model requests: {state['model_calls']} · Tool executions: {state['tool_calls']} · "
             f"Correction cycles: {state['correction_count']}/1")
    if state["status"] == "complete":
        st.success("Investigation completed. Agent agreement is not scientific verification.")
    else:
        st.error("Incomplete run: " + state["error"])
        st.info("Available findings and tool results are preserved below. No completed report was produced.")
    for field, title in [("investigator_findings", "Paper investigator"),
                         ("methods_findings", "Methods analyst"),
                         ("review", "Critical reviewer — latest decision"),
                         ("writer_findings", "Report writer — structured findings")]:
        with st.expander(title):
            st.json(state[field])
    with st.expander("Reviewer decisions and correction history"):
        st.json({"reviews": state["reviews"], "corrections": state["correction_history"]})
    with st.expander("Execution events — tool calls, arguments, and source passages"):
        for event in state["events"]:
            st.write(f"{event['agent']}: {event['kind'].replace('_', ' ')}")
            if event["kind"] == "tool_executed":
                st.code(event["tool"], language=None)
                st.json(event["arguments"])
                st.json(event["result"])
            elif "message" in event:
                st.write(event["message"])
        st.caption("Operational events and explicit findings only; private model reasoning is not recorded. "
                   "Credential-looking strings are redacted in views and downloads.")
    if state["final_report"] and state["status"] == "complete":
        st.subheader("Final report")
        st.markdown(state["final_report"])
    st.caption("References are checked against exact extracted passages and physical pages. "
               "These checks cannot prove semantic support. Keyword search can miss paraphrases. "
               "The writer's report receives no additional reviewer pass.")
    st.download_button("Download Markdown", markdown_download(result),
                       file_name="paper-research-report.md", mime="text/markdown", on_click="ignore")
