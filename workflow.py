"""The typed LangGraph, one targeted correction, and deterministic report rendering."""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from agents import Limits, emit, run_agent
from llm import RunError
from pdf_reader import prepare_passages


class ResearchState(TypedDict):
    question: str
    pages: dict[int, str]
    evidence: dict[str, dict]
    investigator_findings: dict
    methods_findings: dict
    review: dict
    reviews: list[dict]
    correction_count: int
    correction_history: list[dict]
    final_report: str
    writer_findings: dict
    events: list[dict]
    visits: list[dict]
    model_calls: int
    tool_calls: int
    status: str
    error: str
    next_agent: str
    deadline: float


def initial_state(question: str, pages: dict[int, str]) -> ResearchState:
    if not question.strip() or len(question) > 2000 or not pages:
        raise RunError("Provide a question (at most 2000 characters) and extracted PDF pages.")
    return ResearchState(question=question.strip(), pages=dict(pages),
        evidence=prepare_passages(pages), investigator_findings={}, methods_findings={},
        review={}, reviews=[], correction_count=0, correction_history=[], final_report="",
        writer_findings={}, events=[], visits=[], model_calls=0, tool_calls=0,
        status="running", error="", next_agent="", deadline=0.0)


def citations(ids: list[str], state: ResearchState) -> str:
    return " ".join(f"[{eid} · PDF page {state['evidence'][eid]['page']}]" for eid in ids)


def render_report(output: dict, state: ResearchState) -> str:
    """Only validated IDs are rendered; latest concerns cannot be omitted by the writer."""
    parts = ["# Paper Research Team", "## Question", state["question"]]
    for topic in ("Answer", "Contributions", "Methods and evidence", "Limitations"):
        parts.append(f"## {topic}")
        for finding in output["findings"]:
            if finding["topic"] == topic:
                kind = finding["kind"].replace("_", " ")
                parts.append(f"**{kind.capitalize()}:** {finding['statement']} "
                             + citations(finding["evidence_ids"], state))
    parts.append("## Follow-up questions — proposals, not established findings")
    parts.extend(f"- {question}" for question in output["follow_up_questions"])
    if not output["follow_up_questions"]:
        parts.append("No follow-up proposals were produced.")
    parts.append("## Remaining uncertainties")
    uncertainties = list(dict.fromkeys(output["uncertainties"] + state["review"]["uncertainties"]
        + state["investigator_findings"]["uncertainties"] + state["methods_findings"]["uncertainties"]))
    parts.extend(f"- {item}" for item in uncertainties)
    if not uncertainties:
        parts.append("No additional uncertainties were listed by the agents; this is not proof of completeness.")
    if state["review"]["issues"]:
        heading = ("## Unresolved reviewer issues — correction limit reached"
                   if state["review"]["correction_required"] else "## Reviewer concerns")
        parts.append(heading)
        for issue in state["review"]["issues"]:
            parts.append(f"- {issue['statement']} " + citations(issue["evidence_ids"], state))
    parts.append("Only the uploaded paper was investigated. No other literature was searched "
                 "and global novelty was not established. All four roles use the same model. "
                 "Agent agreement, valid citations, and matching quotes do not constitute scientific "
                 "verification. The writer's report has no additional reviewer pass.")
    return "\n\n".join(parts)


def build_graph(client, model: str, limits: Limits | None = None, on_event=None):
    limits = limits or Limits()
    if not model.strip():
        raise RunError("GROQ_MODEL must not be empty.")

    def node(role):
        def visit(state: ResearchState):
            try:
                output = run_agent(role, state, client, model, limits, on_event)
                if role in {"investigator", "methods"}:
                    state[f"{role}_findings"] = output
                elif role == "reviewer":
                    state["review"] = output
                    state["reviews"] = state["reviews"] + [output]
                    if output["correction_required"] and state["correction_count"] == 0:
                        # Persist the increment before the conditional edge routes backwards.
                        state["correction_count"] = 1
                        target = output["target"]
                        state["next_agent"] = target
                        state["correction_history"] = [{
                            "target": target, "review": output,
                            "previous_investigator": state["investigator_findings"],
                            "previous_methods": state["methods_findings"],
                        }]
                        state["methods_findings"] = {}
                        if target == "investigator":
                            state["investigator_findings"] = {}
                        emit(state, role, "correction_scheduled", {"target": target}, on_event)
                    else:
                        state["next_agent"] = "writer"
                else:
                    state["writer_findings"] = output
                    state["final_report"] = render_report(output, state)
                    state["status"] = "complete"
            except RunError as error:
                state["status"] = "incomplete"
                state["error"] = str(error)
                if state["visits"]:
                    state["visits"][-1]["status"] = "incomplete"
                emit(state, role, "incomplete", {"message": str(error)}, on_event)
            except Exception:
                # Raw SDK/parser errors can contain document content; preserve safe partial state.
                state["status"] = "incomplete"
                state["error"] = "Unexpected execution error. Available findings are preserved; check installation/configuration."
                if state["visits"]:
                    state["visits"][-1]["status"] = "incomplete"
                emit(state, role, "incomplete", {"message": state["error"]}, on_event)
            return state
        return visit

    graph = StateGraph(ResearchState)
    for role in ("investigator", "methods", "reviewer", "writer"):
        graph.add_node(role, node(role))
    graph.add_edge(START, "investigator")
    graph.add_conditional_edges("investigator", lambda state: END if state["status"] == "incomplete" else "methods",
                                {END: END, "methods": "methods"})
    graph.add_conditional_edges("methods", lambda state: END if state["status"] == "incomplete" else "reviewer",
                                {END: END, "reviewer": "reviewer"})
    graph.add_conditional_edges("reviewer", lambda state: END if state["status"] == "incomplete" else state["next_agent"],
                                {END: END, "investigator": "investigator", "methods": "methods", "writer": "writer"})
    graph.add_edge("writer", END)
    return graph.compile()
