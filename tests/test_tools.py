import pytest

from tools import (ToolError, check_quote, execute_tool, get_evidence, list_pages,
                   read_page, search_paper, tool_definitions)
from workflow import initial_state


def paper():
    return initial_state('What evidence supports the result?', {
        1: 'Our sorting study uses a small dataset.\nThe result was measured on one machine.',
        2: 'The baseline was insertion sort. Accuracy was not measured.\nMore tests are needed.',
    })


def test_keyword_search_and_exact_evidence():
    state = paper()
    result = search_paper(state, 'BASELINE insertion')
    match = result['matches'][0]
    assert match['page'] == 2
    assert match == get_evidence(state, match['evidence_id'])
    assert match['text'] == state['pages'][2]
    assert search_paper(state, 'nonexistentkeyword')['matches'] == []
    # Deliberately lexical, with no claim of semantic matching.
    assert search_paper(state, 'computer')['matches'] == []
    assert len(list_pages(state)['pages']) == 2
    assert read_page(state, 1)['evidence'][0]['evidence_id'] == 'E0001'


def test_quote_whitespace_normalization_is_case_sensitive():
    state = paper()
    assert check_quote(state, 1, 'dataset.   \n The result')['found'] is True
    assert check_quote(state, 1, 'the result')['found'] is False
    assert check_quote(state, 2, 'invented quote')['found'] is False


@pytest.mark.parametrize('page', [0, -1, 3, True, '1'])
def test_invalid_pages(page):
    with pytest.raises(ToolError):
        read_page(paper(), page)
    with pytest.raises(ToolError):
        check_quote(paper(), page, 'quote')


@pytest.mark.parametrize('role,name,args', [
    ('writer', 'read_page', {'page_number': 1}),
    ('investigator', 'get_evidence', {'evidence_id': 'E0001'}),
    ('reviewer', 'search_paper', {'query': 'sort'}),
    ('investigator', 'delete_file', {}),
    ('methods', 'read_page', {'page_number': True}),
    ('methods', 'read_page', {'page_number': 99}),
    ('methods', 'read_page', {'page_number': 1, 'api_key': 'do-not-log'}),
    ('methods', 'read_page', {}),
    ('methods', 'search_paper', {'query': ''}),
    ('methods', 'search_paper', {'query': 'x' * 161}),
    ('methods', 'search_paper', {'query': '!!!'}),
    ('methods', 'get_evidence', {'evidence_id': 'E9999'}),
    ('reviewer', 'check_quote', {'page_number': 1, 'quote': ' '}),
])
def test_allowlists_and_argument_validation(role, name, args):
    with pytest.raises(ToolError):
        execute_tool(role, name, args, paper())


def test_tool_schemas_have_no_implicit_state_argument():
    definitions = tool_definitions('reviewer')
    assert {tool['function']['name'] for tool in definitions} == {'read_page', 'get_evidence', 'check_quote'}
    for definition in definitions:
        schema = definition['function']['parameters']
        assert schema['additionalProperties'] is False
        assert 'state' not in schema['properties']


def test_large_page_and_search_results_are_bounded_and_explicit():
    state = initial_state('Analyze.', {1: 'sort ' * 1500})
    page = read_page(state, 1)
    assert page['partial_page'] is True
    assert len(page['text']) == 4000
    assert page['remaining_evidence_ids']
    search = search_paper(state, 'sort')
    assert len(search['matches']) == 4
    assert all(len(entry['text']) <= 800 for entry in search['matches'])
