"""Two ordinary retrieval tools, explicit schemas, and an allowlisted dispatcher."""

import copy
import re
import math
from collections import Counter


class ToolError(ValueError):
    """Invalid tool name, arguments, or source reference."""


ALLOWED_TOOLS = {
    role: ("search_paper", "get_evidence")
    for role in ("reader", "analyst", "writer", "reviewer")
}

# State is injected by Python, never supplied by the model.
ARGUMENTS = {
    "search_paper": {"query": {"type": "string", "minLength": 1, "maxLength": 160}},
    "get_evidence": {"evidence_id": {"type": "string", "pattern": "^E[0-9]{4,}$"}},
}
DESCRIPTIONS = {
    "search_paper": "BM25 lexical search; return up to four exact passages with evidence IDs and pages. May miss paraphrases.",
    "get_evidence": "Look up one exact stored source passage and its physical PDF page.",
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
        if not isinstance(value, str) or not value.strip():
            raise ToolError("Tool string arguments must be nonempty strings.")
        elif len(value) > rules.get("maxLength", 100):
            raise ToolError("A tool string argument exceeds its length limit.")
    if name == "get_evidence" and arguments["evidence_id"] not in state["evidence"]:
        raise ToolError("The requested evidence ID does not exist.")


def search_paper(state: dict, query: str) -> dict:
    """Okapi BM25 (k1=1.5, b=.75); lexical ranking, not semantic retrieval."""
    tokenize = lambda text: re.findall(r"[a-z0-9]+", text.casefold())
    keywords = set(tokenize(query))
    if not keywords:
        raise ToolError("Search query must contain a keyword with letters or digits.")
    corpus = [(entry, Counter(tokenize(entry['text']))) for entry in state['evidence'].values()]
    average = sum(sum(words.values()) for _, words in corpus) / max(1, len(corpus))
    ranked = []
    for entry, words in corpus:
        score = 0.0
        length = sum(words.values())
        for word in keywords & words.keys():
            frequency = sum(word in document for _, document in corpus)
            inverse = math.log(1 + (len(corpus) - frequency + .5) / (frequency + .5))
            count = words[word]
            score += inverse * count * 2.5 / (count + 1.5 * (.25 + .75 * length / max(1, average)))
        if score:
            ranked.append((score, entry))
    ranked.sort(key=lambda pair: (-pair[0], pair[1]['page'], pair[1]['start']))
    return {'matches': [copy.deepcopy(entry) for _, entry in ranked[:4]],
            'total_matches': len(ranked), 'method': 'BM25 (k1=1.5, b=0.75)'}


def get_evidence(state: dict, evidence_id: str) -> dict:
    if evidence_id not in state["evidence"]:
        raise ToolError("The requested evidence ID does not exist.")
    return copy.deepcopy(state["evidence"][evidence_id])


def execute_tool(role: str, name: str, arguments: dict, state: dict):
    validate_arguments(role, name, arguments, state)
    functions = {"search_paper": search_paper, "get_evidence": get_evidence}
    return functions[name](state, **arguments)
