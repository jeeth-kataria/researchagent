"""Six ordinary document tools, explicit schemas, and an allowlisted dispatcher."""

import copy
import re


class ToolError(ValueError):
    """Invalid tool name, arguments, or source reference."""


ALLOWED_TOOLS = {
    "investigator": ("list_pages", "search_paper", "read_page"),
    "methods": ("search_paper", "read_page", "get_evidence"),
    "reviewer": ("read_page", "get_evidence", "check_quote"),
    "writer": ("get_findings", "get_evidence"),
}

# The registry and page dictionary are injected by the application, never the model.
ARGUMENTS = {
    "list_pages": {},
    "search_paper": {"query": {"type": "string", "minLength": 1, "maxLength": 160}},
    "read_page": {"page_number": {"type": "integer", "minimum": 1}},
    "get_evidence": {"evidence_id": {"type": "string", "pattern": "^E[0-9]{4,}$"}},
    "check_quote": {"page_number": {"type": "integer", "minimum": 1},
                    "quote": {"type": "string", "minLength": 1, "maxLength": 800}},
    "get_findings": {},
}
DESCRIPTIONS = {
    "list_pages": "List physical page numbers and short previews (first 40 pages).",
    "search_paper": "Keyword search; return up to four exact passages with evidence IDs and pages. May miss paraphrases.",
    "read_page": "Read a physical PDF page (first 4000 characters); return visible evidence and remaining IDs if partial.",
    "get_evidence": "Look up one exact stored source passage and its physical PDF page.",
    "check_quote": "Check a quote on one page after collapsing whitespace. Existence is not semantic support.",
    "get_findings": "Read current investigator/methods findings, latest review, and correction count.",
}


def tool_definitions(role: str) -> list[dict]:
    return [{"type": "function", "function": {
        "name": name, "description": DESCRIPTIONS[name], "parameters": {
            "type": "object", "properties": copy.deepcopy(ARGUMENTS[name]),
            "required": list(ARGUMENTS[name]), "additionalProperties": False,
        }}} for name in ALLOWED_TOOLS[role]]


def validate_arguments(role: str, name: str, arguments: dict, state: dict) -> None:
    """Validate independently of the model/provider's schema enforcement."""
    if name not in ALLOWED_TOOLS.get(role, ()):
        raise ToolError("The agent requested a tool outside its allowlist.")
    schema = ARGUMENTS[name]
    if not isinstance(arguments, dict) or set(arguments) != set(schema):
        raise ToolError("Tool arguments have missing or unexpected fields.")
    for key, rules in schema.items():
        value = arguments[key]
        if rules["type"] == "integer":
            if type(value) is not int or value not in state["pages"]:
                raise ToolError("Tool page_number must be an existing physical PDF page.")
        elif not isinstance(value, str) or not value.strip():
            raise ToolError("Tool string arguments must be nonempty strings.")
        elif len(value) > rules.get("maxLength", 100):
            raise ToolError("A tool string argument exceeds its length limit.")
    if name == "get_evidence" and arguments["evidence_id"] not in state["evidence"]:
        raise ToolError("The requested evidence ID does not exist.")


def list_pages(state: dict) -> dict:
    pages = list(state["pages"].items())
    return {"pages": [{"page": number, "preview": text[:100]} for number, text in pages[:40]],
            "total_pages": len(pages), "partial_listing": len(pages) > 40}


def search_paper(state: dict, query: str) -> dict:
    keywords = set(re.findall(r"[a-z0-9]+", query.casefold()))
    if not keywords:
        raise ToolError("Search query must contain a keyword with letters or digits.")
    ranked = []
    for evidence in state["evidence"].values():
        words = set(re.findall(r"[a-z0-9]+", evidence["text"].casefold()))
        score = len(keywords & words)
        if score:
            ranked.append((score, evidence))
    ranked.sort(key=lambda pair: (-pair[0], pair[1]["page"], pair[1]["start"]))
    return {"matches": [copy.deepcopy(item) for _, item in ranked[:4]],
            "total_matches": len(ranked), "method": "case-insensitive keyword overlap"}


def read_page(state: dict, page_number: int) -> dict:
    if type(page_number) is not int or page_number not in state["pages"]:
        raise ToolError("That physical PDF page does not exist.")
    text = state["pages"][page_number]
    entries = [entry for entry in state["evidence"].values() if entry["page"] == page_number]
    return {"page": page_number, "text": text[:4000], "total_characters": len(text),
            "partial_page": len(text) > 4000,
            "evidence": [copy.deepcopy(entry) for entry in entries if entry["end"] <= 4000],
            "remaining_evidence_ids": [entry["evidence_id"] for entry in entries if entry["end"] > 4000]}


def get_evidence(state: dict, evidence_id: str) -> dict:
    if evidence_id not in state["evidence"]:
        raise ToolError("The requested evidence ID does not exist.")
    return copy.deepcopy(state["evidence"][evidence_id])


def check_quote(state: dict, page_number: int, quote: str) -> dict:
    if type(page_number) is not int or page_number not in state["pages"]:
        raise ToolError("That physical PDF page does not exist.")
    normalized = " ".join(quote.split())
    if not normalized:
        raise ToolError("Quote must not be empty.")
    return {"page": page_number,
            "found": normalized in " ".join(state["pages"][page_number].split()),
            "normalization": "Collapse all Unicode whitespace runs to one space and strip ends; case-sensitive.",
            "note": "Quote existence does not establish semantic support."}


def get_findings(state: dict) -> dict:
    return copy.deepcopy({"investigator": state["investigator_findings"],
                          "methods": state["methods_findings"], "review": state["review"],
                          "correction_count": state["correction_count"]})


def execute_tool(role: str, name: str, arguments: dict, state: dict):
    validate_arguments(role, name, arguments, state)
    functions = {"list_pages": list_pages, "search_paper": search_paper,
                 "read_page": read_page, "get_evidence": get_evidence,
                 "check_quote": check_quote, "get_findings": get_findings}
    return functions[name](state, **arguments)
