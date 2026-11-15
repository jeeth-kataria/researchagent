"""Only two bounded, model-callable retrieval functions are exposed."""
import pytest

from tools import ToolError, execute_tool, get_evidence, search_paper, tool_definitions
from workflow import initial_state


def paper():
    return initial_state('What supports the result?', {
        1: 'Our sorting study uses a small dataset. The result used one machine.',
        2: 'The baseline was insertion sort. Accuracy was not measured.'})


def test_keyword_search_and_exact_evidence():
    state = paper()
    match = search_paper(state, 'BASELINE insertion')['matches'][0]
    assert match['page'] == 2
    assert match == get_evidence(state, match['evidence_id'])
    assert match['text'] == state['pages'][2][match['start']:match['end']]
    match['text'] = 'changed copy'
    assert get_evidence(state, 'E0002')['text'] == state['pages'][2]
    assert search_paper(state, 'computer')['matches'] == []  # No paraphrase matching.


@pytest.mark.parametrize('role', ['reader', 'analyst', 'writer', 'reviewer'])
def test_only_two_tool_schemas_and_dispatch_entries(role):
    definitions = tool_definitions(role)
    assert {item['function']['name'] for item in definitions} == {'search_paper', 'get_evidence'}
    for item in definitions:
        schema = item['function']['parameters']
        assert schema['additionalProperties'] is False
        assert schema['required'] == list(schema['properties'])
        assert 'state' not in schema['properties']
    assert execute_tool(role, 'get_evidence', {'evidence_id': 'E0001'}, paper())['page'] == 1
    for obsolete in ['list_pages', 'read_page', 'check_quote', 'get_findings', 'eval']:
        with pytest.raises(ToolError, match='allowlist'):
            execute_tool(role, obsolete, {}, paper())


@pytest.mark.parametrize('role,name,args', [
    ('unknown', 'search_paper', {'query': 'sort'}),
    ('reader', 'search_paper', {}),
    ('reader', 'search_paper', {'query': 'sort', 'extra': 1}),
    ('analyst', 'search_paper', {'query': ''}),
    ('analyst', 'search_paper', {'query': 1}),
    ('analyst', 'search_paper', {'query': 'x' * 161}),
    ('analyst', 'search_paper', {'query': '!!!'}),
    ('reviewer', 'get_evidence', {'evidence_id': 'E9999'}),
    ('reviewer', 'get_evidence', {'evidence_id': True}),
    ('writer', 'get_evidence', {'evidence_id': 'E0001', 'page_number': 99}),
])
def test_arguments_and_unknown_evidence(role, name, args):
    with pytest.raises(ToolError):
        execute_tool(role, name, args, paper())


def test_search_bounds_ranking_and_tie_order():
    state = initial_state('Analyze.', {1: 'sort ' * 1500, 2: 'sort baseline'})
    results = search_paper(state, 'sort baseline')
    assert results['matches'][0]['page'] == 2
    assert len(results['matches']) == 4
    assert results['total_matches'] > 4
    assert [entry['start'] for entry in results['matches'][1:]] == [0, 800, 1600]
    assert all(len(entry['text']) <= 800 for entry in results['matches'])
