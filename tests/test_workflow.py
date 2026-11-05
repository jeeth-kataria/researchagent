"""Scripted SDK responses exercise the real compiled graph and tool dispatcher."""

import json
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from agents import Limits
from llm import RunError, sanitize
from workflow import build_graph, initial_state


def source_state():
    return initial_state('Explain the contribution and experimental limits.', {
        1: 'A sorting comparison is the contribution. The problem is slow sorting.',
        2: 'Experiments use one machine and insertion sort as baseline. More tests are needed.'})


def response(value=None, calls=None, finish=None):
    message = NS(content=json.dumps(value) if isinstance(value, dict) else value, tool_calls=calls)
    return NS(choices=[NS(message=message, finish_reason=finish or ('tool_calls' if calls else 'stop'))])


def tool(name, args, call_id='chosen-call'):
    return NS(id=call_id, type='function', function=NS(name=name, arguments=json.dumps(args)))


def findings(statement='Original finding', evidence_id='E0001', writer=False):
    result = {'findings': [{'topic': 'Contribution', 'statement': statement,
                          'kind': 'author_statement', 'evidence_ids': [evidence_id]}],
              'uncertainties': ['Only the supplied passages were inspected.']}
    if writer:
        result['findings'] = [dict(result['findings'][0], topic=topic) for topic in
                              ['Answer', 'Contributions', 'Methods and evidence', 'Limitations']]
        result['follow_up_questions'] = ['Could a larger experiment test this result?']
    return result


def review(target=None, issue='Correct the unsupported comparison.'):
    return {'correction_required': target is not None, 'target': target,
            'issues': [{'statement': issue, 'evidence_ids': ['E0002']}] if target else [],
            'uncertainties': ['Limited experimental coverage.']}


def visit(role, output=None):
    name, args = {
        'investigator': ('search_paper', {'query': 'sorting contribution'}),
        'methods': ('read_page', {'page_number': 2}),
        'reviewer': ('get_evidence', {'evidence_id': 'E0002'}),
        'writer': ('get_findings', {}),
    }[role]
    if output is None:
        output = review() if role == 'reviewer' else findings(
            evidence_id='E0002' if role == 'methods' else 'E0001', writer=role == 'writer')
    return [response(calls=[tool(name, args)]), response(output)]


def run(outputs, limits=None):
    client = Mock()
    client.api_key = 'mock-only'
    client.chat.completions.create.side_effect = outputs
    result = build_graph(client, 'openai/gpt-oss-20b', limits).invoke(source_state())
    return result, client


def approval():
    return sum((visit(role) for role in ['investigator', 'methods', 'reviewer', 'writer']), [])


def test_approval_route_and_function_call_message_pairing():
    state, client = run(approval())
    assert state['status'] == 'complete'
    assert [v['agent'] for v in state['visits']] == ['investigator', 'methods', 'reviewer', 'writer']
    assert state['model_calls'] == client.chat.completions.create.call_count == 8
    assert state['tool_calls'] == 4
    assert state['correction_count'] == 0
    assert '[E0001 · PDF page 1]' in state['final_report']
    second = client.chat.completions.create.call_args_list[1].kwargs
    assistant, reply = second['messages'][-2:]
    assert assistant['role'] == 'assistant'
    assert assistant['tool_calls'][0]['id'] == reply['tool_call_id'] == 'chosen-call'
    assert reply['role'] == 'tool'
    assert json.loads(reply['content'])['matches'][0]['evidence_id'] == 'E0001'
    assert all('response_format' not in call.kwargs for call in client.chat.completions.create.call_args_list)
    assert all(call.kwargs['model'] == 'openai/gpt-oss-20b' for call in client.chat.completions.create.call_args_list)
    # Initial prompts contain context, not the full original paper.
    initial_prompt = client.chat.completions.create.call_args_list[0].kwargs['messages'][1]['content']
    assert 'A sorting comparison is the contribution.' not in initial_prompt
    assert 'reasoning' not in json.dumps(state['events'])


@pytest.mark.parametrize('target,expected', [
    ('investigator', ['investigator', 'methods', 'reviewer', 'investigator', 'methods', 'reviewer', 'writer']),
    ('methods', ['investigator', 'methods', 'reviewer', 'methods', 'reviewer', 'writer']),
])
def test_correction_routes_replace_stale_findings(target, expected):
    outputs = visit('investigator') + visit('methods') + visit('reviewer', review(target))
    if target == 'investigator':
        outputs += visit('investigator', findings('Corrected investigator conclusion'))
    outputs += visit('methods', findings('Fresh methods conclusion', 'E0002'))
    outputs += visit('reviewer') + visit('writer')
    state, client = run(outputs)
    assert state['status'] == 'complete'
    assert [v['agent'] for v in state['visits']] == expected
    assert state['correction_count'] == 1
    assert state['correction_history'][0]['target'] == target
    assert state['correction_history'][0]['previous_methods']['findings'][0]['statement'] == 'Original finding'
    assert state['methods_findings']['findings'][0]['statement'] == 'Fresh methods conclusion'
    if target == 'investigator':
        assert state['investigator_findings']['findings'][0]['statement'] == 'Corrected investigator conclusion'
    # Before the corrective visit, the count is already incremented.
    corrected_context = json.loads(client.chat.completions.create.call_args_list[6].kwargs['messages'][1]['content'])
    assert corrected_context['correction_count'] == 1
    assert state['model_calls'] == len(expected) * 2


def test_second_reviewer_request_never_starts_second_correction():
    outputs = (visit('investigator') + visit('methods') + visit('reviewer', review('methods'))
               + visit('methods', findings('Corrected method', 'E0002'))
               + visit('reviewer', review('investigator', 'Still unresolved: unsupported claim.'))
               + visit('writer'))
    state, _ = run(outputs)
    assert state['status'] == 'complete'
    assert len(state['reviews']) == 2
    assert state['correction_count'] == 1
    assert len(state['visits']) == 6
    assert state['visits'][-1]['agent'] == 'writer'
    assert 'Still unresolved: unsupported claim.' in state['final_report']
    assert 'correction limit reached' in state['final_report']


def test_two_tools_in_one_response_count_separately_and_pair_correctly():
    outputs = [response(calls=[tool('list_pages', {}, 'pages'),
                              tool('read_page', {'page_number': 1}, 'read')]), response(findings())]
    outputs += visit('methods') + visit('reviewer') + visit('writer')
    state, client = run(outputs)
    assert state['status'] == 'complete'
    assert state['tool_calls'] == 5
    second = client.chat.completions.create.call_args_list[1].kwargs
    assert second['tool_choice'] == 'none'
    assert [m['tool_call_id'] for m in second['messages'][-2:]] == ['pages', 'read']
    assert state['visits'][0]['tool_calls'] == 2


def test_per_visit_budget_stops_without_extra_requests():
    outputs = [response(calls=[tool('read_page', {'page_number': 1}, 'first')]),
               response(calls=[tool('read_page', {'page_number': 2}, 'second')]),
               response(calls=[tool('list_pages', {}, 'forbidden-third')])]
    state, client = run(outputs)
    assert state['status'] == 'incomplete'
    assert state['final_report'] == ''
    assert state['model_calls'] == client.chat.completions.create.call_count == 3
    assert state['tool_calls'] == 2
    assert 'budget' in state['error']
    assert client.chat.completions.create.call_args.kwargs['tool_choice'] == 'none'


def test_oversized_tool_batch_is_rejected_before_execution():
    state, _ = run([response(calls=[tool('list_pages', {}, str(n)) for n in range(3)])])
    assert state['status'] == 'incomplete'
    assert state['model_calls'] == 1
    assert state['tool_calls'] == 0


def test_total_budget_preserves_completed_findings():
    state, client = run(approval(), Limits(total_model_calls=4))
    assert state['status'] == 'incomplete'
    assert 'Total model-request budget exhausted' in state['error']
    assert state['model_calls'] == client.chat.completions.create.call_count == 4
    assert state['investigator_findings'] and state['methods_findings']
    assert state['final_report'] == ''


@pytest.mark.parametrize('kwargs', [{'model_per_visit': 4}, {'tools_per_visit': 3},
                                  {'total_model_calls': 25}, {'tools_per_visit': True},
                                  {'output_tokens': 100}, {'total_model_calls': 0}])
def test_configurable_limits_cannot_exceed_hard_caps(kwargs):
    with pytest.raises(RunError):
        Limits(**kwargs)


@pytest.mark.parametrize('decision', [
    'not JSON', '[]',
    {'correction_required': 'false', 'target': None, 'issues': [], 'uncertainties': []},
    {'correction_required': True, 'target': 'writer', 'issues': [], 'uncertainties': []},
    {'correction_required': False, 'target': 'methods', 'issues': [], 'uncertainties': []},
    {'correction_required': True, 'target': 'methods', 'issues': [], 'uncertainties': []},
    {'correction_required': False, 'target': None, 'issues': [], 'uncertainties': [], 'extra': 1},
    '{"correction_required": false, "correction_required": true}',
])
def test_malformed_review_is_never_approved(decision):
    state, client = run(visit('investigator') + visit('methods') + visit('reviewer', decision))
    assert state['status'] == 'incomplete'
    assert state['final_report'] == ''
    assert state['model_calls'] == client.chat.completions.create.call_count == 6
    assert state['visits'][-1]['agent'] == 'reviewer'
    assert state['investigator_findings']


@pytest.mark.parametrize('writer_output', [
    findings('Claim with invalid source.', 'E9999', writer=True),
    findings('A claim on [PDF page 99].', writer=True),
    findings('A mismatched citation [E0001 · PDF page 2].', writer=True),
])
def test_final_report_rejects_nonexistent_sources_or_pages(writer_output):
    state, _ = run(visit('investigator') + visit('methods') + visit('reviewer')
                   + visit('writer', writer_output))
    assert state['status'] == 'incomplete'
    assert state['final_report'] == ''
    assert state['review']


def test_disallowed_tool_and_arguments_never_appear_in_trace():
    state, _ = run([response(calls=[tool('get_findings', {'api_key': 'gsk_should_not_display'})])])
    assert state['status'] == 'incomplete'
    assert state['tool_calls'] == 0
    assert 'gsk_should_not_display' not in json.dumps(state['events'])


def test_final_without_source_inspection_is_incomplete():
    state, _ = run([response(findings())])
    assert state['status'] == 'incomplete'
    assert 'inspect original source' in state['error']
    assert state['model_calls'] == 1


def test_incomplete_model_output_stops_without_retry():
    state, client = run([response('partial', finish='length')])
    assert state['status'] == 'incomplete'
    assert 'truncated' in state['error']
    assert client.chat.completions.create.call_count == 1


def test_redaction_applies_recursively():
    assert sanitize({'arguments': ['gsk_abc123', 'custom-secret']}, 'custom-secret') == {
        'arguments': ['[REDACTED]', '[REDACTED]']}


def test_one_request_visit_budget_cannot_silently_add_follow_up():
    state, client = run(approval(), Limits(model_per_visit=1))
    assert state['status'] == 'incomplete'
    assert state['model_calls'] == client.chat.completions.create.call_count == 1
    assert state['tool_calls'] == 0


def test_duplicate_call_ids_are_rejected_before_any_tool_runs():
    state, _ = run([response(calls=[tool('read_page', {'page_number': 1}),
                                  tool('read_page', {'page_number': 2})])])
    assert state['status'] == 'incomplete'
    assert state['tool_calls'] == 0
    assert 'duplicate tool-call ID' in state['error']


def test_rate_limit_is_safe_and_preserves_partial_results():
    import groq
    import httpx

    error = groq.RateLimitError('raw-provider-body-must-not-display',
        response=httpx.Response(429, request=httpx.Request('POST', 'https://example.invalid')),
        body={})
    state, client = run(visit('investigator') + [error])
    assert state['status'] == 'incomplete'
    assert state['investigator_findings']
    assert state['model_calls'] == client.chat.completions.create.call_count == 3
    assert 'quota/rate limit' in state['error']
    assert 'raw-provider-body' not in json.dumps(state)


def test_client_factory_disables_sdk_retries(monkeypatch):
    import llm

    constructor = Mock()
    monkeypatch.setattr(llm.groq, 'Groq', constructor)
    llm.make_client('mock-only')
    constructor.assert_called_once_with(api_key='mock-only', max_retries=0, timeout=60.0)
