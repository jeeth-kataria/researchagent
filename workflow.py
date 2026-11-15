"""Four fixed LangGraph stages; the final review never rewrites the draft."""

import re
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from agents import DRAFT_TOPICS, Limits, emit, run_agent
from llm import RunError
from pdf_reader import prepare_passages


class ResearchState(TypedDict):
    question: str
    pages: dict[int, str]
    evidence: dict[str, dict]
    reader_findings: dict
    analyst_findings: dict
    report_draft: dict
    critical_review: dict
    final_output: str
    events: list[dict]
    visits: list[dict]
    model_calls: int
    tool_calls: int
    status: str
    error: str
    deadline: float


def initial_state(question: str, pages: dict[int, str]) -> ResearchState:
    if not question.strip() or len(question) > 2000 or not pages:
        raise RunError("Provide a question (at most 2000 characters) and extracted PDF pages.")
    return ResearchState(question=question.strip(), pages=dict(pages),
        evidence=prepare_passages(pages), reader_findings={}, analyst_findings={},
        report_draft={}, critical_review={}, final_output="", events=[], visits=[],
        model_calls=0, tool_calls=0, status="running", error="", deadline=0.0)


def citations(ids: list[str], state: ResearchState) -> str:
    return " ".join(f"[{eid} · PDF page {state['evidence'][eid]['page']}]" for eid in ids)


def render_output(state: ResearchState) -> str:
    """Ordinary formatting after the reviewer, including inspectable partial runs."""
    parts = ["# Paper Research Team", "## Question", state["question"]]
    if state["status"] != "complete":
        parts += ["## Incomplete investigation", state["error"],
                  "The draft did not receive a completed final critical review."]
    draft = state["report_draft"]
    if draft:
        parts += ["## Paper explanation — report draft",
                  "This draft is not automatically corrected or scientifically verified."]
        for topic in DRAFT_TOPICS:
            parts.append(f"### {topic}")
            for finding in draft["findings"]:
                if finding["topic"] == topic:
                    parts.append(f"**{finding['kind'].replace('_', ' ')}:** "
                                 + finding["statement"] + " "
                                 + citations(finding["evidence_ids"], state))
    else:
        parts.append("No completed draft is available.")
    review = state["critical_review"]
    parts.append("## Final critical review")
    if review:
        parts.append(review["overall_assessment"])
        for concern in review["concerns"]:
            parts.append(f"### {concern['target'].upper()} concern — {concern['category']}")
            if concern["affected_statement"]:
                parts.append("**Affected draft statement:** " + concern["affected_statement"])
            parts.append(concern["concern"] + " " + citations(concern["evidence_ids"], state))
            parts.append("**Suggested clarification:** " + concern["suggested_clarification"])
        if not review["concerns"]:
            parts.append("No concrete concerns were returned after limited inspection; this is not scientific approval.")
    else:
        parts.append("Not completed. Treat the available explanation as an unreviewed draft.")
    parts.append("## Remaining uncertainties")
    uncertainties = list(dict.fromkeys(item for output in (
        state["reader_findings"], state["analyst_findings"], draft, review)
        for item in output.get("uncertainties", [])))
    parts.extend(f"- {item}" for item in uncertainties)
    if not uncertainties:
        parts.append("No additional uncertainties were listed; this does not prove completeness.")
    # Include partial findings too, so a failed writer still yields a useful export.
    for title, findings in (("Paper Reader findings", state["reader_findings"]),
                            ("Evidence Analyst findings", state["analyst_findings"])):
        if findings:
            parts.append(f"## {title}")
            for item in findings["findings"]:
                parts.append(f"- {item['statement']} " + citations(item["evidence_ids"], state))
    referenced = set(re.findall(r"\bE\d{4,}\b", "\n".join(parts)))
    parts.append("## Referenced source passages")
    for eid in sorted(referenced & state["evidence"].keys()):
        source = state["evidence"][eid]
        # Blockquotes prevent a passage's headings from looking like report sections.
        quote = "\n".join("> " + line for line in source["text"].splitlines())
        parts.append(f"### {eid} · PDF page {source['page']}\n\n{quote}")
    parts.append("Only the uploaded paper was examined. No external literature search or global "
                 "novelty assessment occurred. All four roles share one model. Exact references "
                 "do not guarantee semantic support. Role agreement is not scientific verification. "
                 "No agent rewrites the draft after the final review.")
    return "\n\n".join(parts)


def build_graph(client, model: str, limits: Limits | None = None, on_event=None):
    limits = limits or Limits()
    if not model.strip():
        raise RunError("GROQ_MODEL must not be empty.")
    output_fields = {"reader": "reader_findings", "analyst": "analyst_findings",
                     "writer": "report_draft", "reviewer": "critical_review"}

    def node(role):
        def visit(state: ResearchState):
            # Fixed edges still advance after failure, but later nodes do no work.
            # No subsequent model call is permitted once a run is incomplete.
            if state["status"] == "incomplete":
                return state
            try:
                state[output_fields[role]] = run_agent(role, state, client, model, limits, on_event)
                if role == "reviewer":
                    state["status"] = "complete"
                    state["final_output"] = render_output(state)
            except Exception as error:
                state["status"] = "incomplete"
                state["error"] = str(error) if isinstance(error, RunError) else (
                    "Unexpected execution error. Available findings are preserved; check installation/configuration.")
                if state["visits"]:
                    state["visits"][-1]["status"] = "incomplete"
                emit(state, role, "incomplete", {"message": state["error"]}, on_event)
                state["final_output"] = render_output(state)
            return state
        return visit

    graph = StateGraph(ResearchState)
    roles = ("reader", "analyst", "writer", "reviewer")
    for role in roles:
        graph.add_node(role, node(role))
    graph.add_edge(START, "reader")
    graph.add_edge("reader", "analyst")
    graph.add_edge("analyst", "writer")
    graph.add_edge("writer", "reviewer")
    graph.add_edge("reviewer", END)
    return graph.compile()
