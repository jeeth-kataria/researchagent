"""Mock the SDK, not the graph: exercise every real node and the actual tools."""
import json
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from agents import DRAFT_TOPICS, Limits
from llm import RunError, sanitize
from workflow import build_graph, initial_state

ROLES = ['reader', 'analyst', 'writer', 'reviewer']


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
              'uncertainties': ['Only selected passages were inspected.']}
    if writer:
        result['findings'] = [dict(result['findings'][0], topic=topic) for topic in DRAFT_TOPICS]
    return result


def review(target=None, issue='The comparison goes beyond the reported experiment.'):
    concerns = [] if target is None else [{
        'target': target, 'category': 'unsupported comparison', 'concern': issue,
        'evidence_ids': ['E0002'], 'suggested_clarification': 'Limit the conclusion to the reported experiment.',
        'affected_statement': 'Original finding' if target == 'report' else None}]
    return {'overall_assessment': 'Limited inspection; no concrete concerns found.' if not concerns else
            'Specific concerns remain; the explanation is not corrected.',
            'concerns': concerns, 'uncertainties': ['Limited experimental coverage.']}


def visit(role, output=None):
    if output is None:
        output = review() if role == 'reviewer' else findings(
            evidence_id='E0002' if role == 'analyst' else 'E0001', writer=role == 'writer')
    if role == 'writer':  # Writer can finish using direct state, without a tool.
        return [response(output)]
    name, args = ('search_paper', {'query': 'sorting contribution'}) if role == 'reader' else (
        'get_evidence', {'evidence_id': 'E0002'})
    return [response(calls=[tool(name, args)]), response(output)]


def run(outputs, limits=None, state=None):
    client = Mock()
    client.api_key = 'mock-only'
    client.chat.completions.create.side_effect = outputs
    result = build_graph(client, 'openai/gpt-oss-20b', limits).invoke(state or source_state())
    return result, client


def complete_run(review_output=None):
    return sum((visit(role, review_output if role == 'reviewer' else None) for role in ROLES), [])


def test_fixed_graph_four_roles_reviewer_last_and_tool_message_pairing():
    state, client = run(complete_run())
    assert state['status'] == 'complete'
    assert [v['agent'] for v in state['visits']] == ROLES
    assert state['model_calls'] == client.chat.completions.create.call_count == 7
    assert state['tool_calls'] == 3
    assert '[E0001 · PDF page 1]' in state['final_output']
    assert not ({'correction_count', 'correction_history', 'reviews', 'next_agent', 'review'} & state.keys())
    second = client.chat.completions.create.call_args_list[1].kwargs
    assistant, reply = second['messages'][-2:]
    assert assistant['tool_calls'][0]['id'] == reply['tool_call_id'] == 'chosen-call'
    assert assistant['role'] == 'assistant' and reply['role'] == 'tool'
    assert json.loads(reply['content'])['matches'][0]['evidence_id'] == 'E0001'
    for call in client.chat.completions.create.call_args_list:
        assert 'response_format' not in call.kwargs
        assert call.kwargs['model'] == 'openai/gpt-oss-20b'
        assert {item['function']['name'] for item in call.kwargs['tools']} == {'search_paper', 'get_evidence'}
    assert 'reading_packet' in json.loads(second['messages'][1]['content'])
    assert state['coverage']['reader']['evidence_ids']
    assert 'reasoning' not in json.dumps(state['events'])
    edges = {(edge.source, edge.target) for edge in build_graph(client, 'model').get_graph().edges}
    assert edges == set(zip(['__start__', *ROLES], [*ROLES, '__end__']))


def test_previous_findings_and_draft_pass_directly_in_state():
    state, client = run(complete_run())
    calls = client.chat.completions.create.call_args_list
    analyst = json.loads(calls[2].kwargs['messages'][1]['content'])
    writer = json.loads(calls[4].kwargs['messages'][1]['content'])
    reviewer = json.loads(calls[5].kwargs['messages'][1]['content'])
    assert analyst['reader_findings'] == state['reader_findings']
    assert writer['reader_findings'] == reviewer['reader_findings'] == state['reader_findings']
    assert writer['analyst_findings'] == reviewer['analyst_findings'] == state['analyst_findings']
    assert reviewer['report_draft'] == state['report_draft']
    assert reviewer['question'] == state['question']
    assert not any('pages' in context or 'evidence' in context for context in (analyst, writer, reviewer))


@pytest.mark.parametrize('target', ['paper', 'report'])
def test_final_concerns_preserved_without_rewrite(target):
    state, client = run(complete_run(review(target)))
    assert state['status'] == 'complete'
    assert state['critical_review'] == review(target)
    assert state['report_draft'] == findings(writer=True)  # No post-review rewrite.
    assert state['model_calls'] == client.chat.completions.create.call_count == 7
    for text in [f'{target.upper()} concern', review(target)['concerns'][0]['concern'],
                 'Suggested clarification', 'Referenced source passages', 'Limited experimental coverage.']:
        assert text in state['final_output']
    if target == 'report':
        assert 'Affected draft statement:** Original finding' in state['final_output']


def test_missing_information_concern_does_not_need_fake_evidence():
    output = review('paper')
    output['concerns'][0]['evidence_ids'] = []
    state, _ = run(complete_run(output))
    assert state['status'] == 'complete'


@pytest.mark.parametrize('role', ['reader', 'analyst', 'reviewer'])
def test_empty_search_alone_cannot_satisfy_required_inspection(role):
    prefix = sum((visit(previous) for previous in ROLES[:ROLES.index(role)]), [])
    output = review() if role == 'reviewer' else findings()
    state, _ = run(prefix + [response(calls=[tool('search_paper', {'query': 'zzznomatch'})]), response(output)])
    assert state['status'] == 'incomplete'
    assert 'inspect original source' in state['error']
    assert state['visits'][-1]['agent'] == role
    assert not state['critical_review']
    if role == 'reviewer':
        assert state['report_draft']
        assert 'did not receive a completed final critical review' in state['final_output']


def test_empty_search_can_be_followed_by_nonempty_source_within_budget():
    state, _ = run([response(calls=[tool('search_paper', {'query': 'zzznomatch'}, 'empty')]),
                   response(calls=[tool('get_evidence', {'evidence_id': 'E0001'}, 'source')]),
                   response(findings())] + visit('analyst') + visit('writer') + visit('reviewer'))
    assert state['status'] == 'complete'
    assert state['model_calls'] == 8 and state['tool_calls'] == 4


def test_two_tools_count_separately_and_pair_correctly():
    state, client = run([response(calls=[tool('search_paper', {'query': 'sorting'}, 'search'),
                       tool('get_evidence', {'evidence_id': 'E0001'}, 'read')]), response(findings())]
                       + visit('analyst') + visit('writer') + visit('reviewer'))
    assert state['status'] == 'complete' and state['tool_calls'] == 4
    request = client.chat.completions.create.call_args_list[1].kwargs
    assert request['tool_choice'] == 'none'
    assert [message['tool_call_id'] for message in request['messages'][-2:]] == ['search', 'read']


def test_maximum_path_is_twelve_requests_eight_tools_no_extra_visits():
    outputs = []
    for role in ROLES:
        outputs += [response(calls=[tool('get_evidence', {'evidence_id': 'E0001'}, 'first')]),
                    response(calls=[tool('get_evidence', {'evidence_id': 'E0002'}, 'second')]),
                    response(review() if role == 'reviewer' else findings(writer=role == 'writer'))]
    state, client = run(outputs)
    assert state['status'] == 'complete'
    assert state['model_calls'] == client.chat.completions.create.call_count == 12
    assert state['tool_calls'] == 8


def test_per_visit_budget_stops_without_extra_requests():
    outputs = [response(calls=[tool('get_evidence', {'evidence_id': 'E0001'}, str(n))]) for n in range(3)]
    state, client = run(outputs)
    assert state['status'] == 'incomplete'
    assert state['model_calls'] == client.chat.completions.create.call_count == 3
    assert state['tool_calls'] == 2
    assert 'budget' in state['error']
    assert client.chat.completions.create.call_args.kwargs['tool_choice'] == 'none'


def test_oversized_tool_batch_is_rejected_before_execution():
    state, _ = run([response(calls=[tool('search_paper', {'query': 'sort'}, str(n)) for n in range(3)])])
    assert state['status'] == 'incomplete'
    assert state['tool_calls'] == 0


def test_total_budget_preserves_findings_and_draft():
    state, client = run(complete_run(), Limits(total_model_calls=5))
    assert state['status'] == 'incomplete'
    assert 'Total model-request budget exhausted' in state['error']
    assert state['model_calls'] == client.chat.completions.create.call_count == 5
    assert state['reader_findings'] and state['analyst_findings'] and state['report_draft']
    assert not state['critical_review']
    assert 'unreviewed draft' in state['final_output']


@pytest.mark.parametrize('kwargs', [{'model_per_visit': 4}, {'tools_per_visit': 3},
    {'total_model_calls': 25}, {'tools_per_visit': True}, {'output_tokens': 100}, {'total_model_calls': 0}])
def test_limits_cannot_exceed_hard_caps(kwargs):
    with pytest.raises(RunError):
        Limits(**kwargs)


@pytest.mark.parametrize('decision', [
    'not JSON', '[]', {'overall_assessment': 'Fine', 'concerns': False, 'uncertainties': []},
    dict(review(), extra=1), {'overall_assessment': '', 'concerns': [], 'uncertainties': []},
    '{"concerns": [], "concerns": []}',
    dict(review(), concerns=[dict(review('report')['concerns'][0], target='writer')]),
    dict(review(), concerns=[dict(review('report')['concerns'][0], affected_statement='invented draft statement')]),
    dict(review(), concerns=[dict(review('paper')['concerns'][0], affected_statement='not null')]),
    dict(review(), concerns=[dict(review('paper')['concerns'][0], evidence_ids=['E9999'])]),
])
def test_malformed_review_preserves_draft_but_cannot_complete(decision):
    state, client = run(complete_run(decision))
    assert state['status'] == 'incomplete'
    assert state['report_draft'] and not state['critical_review']
    assert state['model_calls'] == client.chat.completions.create.call_count == 7
    assert state['visits'][-1]['agent'] == 'reviewer'
    assert 'did not receive a completed final critical review' in state['final_output']


@pytest.mark.parametrize('writer_output', [
    findings('Invalid source.', 'E9999', writer=True),
    findings('A claim on [PDF page 99].', writer=True),
    findings('A mismatched citation [E0001 · PDF page 2].', writer=True),
])
def test_draft_rejects_unknown_evidence_or_invalid_pages(writer_output):
    state, _ = run(visit('reader') + visit('analyst') + visit('writer', writer_output))
    assert state['status'] == 'incomplete' and not state['report_draft']
    assert state['visits'][-1]['agent'] == 'writer'


def test_evidence_must_match_exact_extracted_source():
    state = source_state()
    state['evidence']['E0001']['text'] = 'fabricated passage'
    result, _ = run([response(calls=[tool('get_evidence', {'evidence_id': 'E0001'})])], state=state)
    assert result['status'] == 'incomplete'
    assert 'does not match' in result['error']


@pytest.mark.parametrize('name,args', [
    ('get_findings', {}), ('read_page', {'page_number': 1}), ('search_paper', {'query': 'sort', 'key': 'secret'}),
    ('get_evidence', {'evidence_id': 'E9999'})])
def test_disallowed_tools_and_arguments_are_not_executed_or_logged(name, args):
    state, _ = run([response(calls=[tool(name, args)])])
    assert state['status'] == 'incomplete' and state['tool_calls'] == 0
    assert not any(event['kind'] == 'tool_executed' for event in state['events'])


def test_final_without_source_is_incomplete():
    state, _ = run([response(findings())])
    assert state['status'] == 'incomplete' and 'inspect original source' in state['error']


def test_truncated_response_stops_without_retry():
    state, client = run([response('partial', finish='length')])
    assert state['status'] == 'incomplete' and 'truncated' in state['error']
    assert client.chat.completions.create.call_count == 1


def test_redaction_is_recursive():
    assert sanitize({'arguments': ['gsk_abc123', 'custom-secret']}, 'custom-secret') == {
        'arguments': ['[REDACTED]', '[REDACTED]']}


def test_one_request_visit_limit_does_not_add_follow_up():
    state, client = run(complete_run(), Limits(model_per_visit=1))
    assert state['status'] == 'incomplete' and state['tool_calls'] == 0
    assert client.chat.completions.create.call_count == 1


def test_duplicate_tool_ids_rejected_before_execution():
    state, _ = run([response(calls=[tool('search_paper', {'query': 'sort'}),
                                  tool('get_evidence', {'evidence_id': 'E0001'})])])
    assert state['status'] == 'incomplete' and state['tool_calls'] == 0
    assert 'duplicate tool-call ID' in state['error']


@pytest.mark.parametrize('kind', ['rate', 'connection', 'authentication', 'model'])
def test_provider_errors_are_safe_and_preserve_draft(kind):
    import groq
    import httpx
    request = httpx.Request('POST', 'https://example.invalid')
    error = groq.APIConnectionError(request=request) if kind == 'connection' else {
        'rate': groq.RateLimitError, 'authentication': groq.AuthenticationError,
        'model': groq.BadRequestError}[kind]('raw-provider-body-must-not-display',
            response=httpx.Response(429 if kind == 'rate' else 400, request=request), body={})
    state, client = run(visit('reader') + visit('analyst') + visit('writer') + [error])
    assert state['status'] == 'incomplete'
    assert state['report_draft'] and not state['critical_review']
    assert state['model_calls'] == client.chat.completions.create.call_count == 6
    assert 'raw-provider-body' not in json.dumps(state)
    assert 'completed final critical review' in state['final_output']


def test_client_factory_disables_sdk_retries(monkeypatch):
    import llm
    constructor = Mock()
    monkeypatch.setattr(llm.groq, 'Groq', constructor)
    llm.make_client('mock-only')
    constructor.assert_called_once_with(api_key='mock-only', max_retries=0, timeout=60.0)


def test_runtime_tool_failure_preserves_draft(monkeypatch):
    import agents
    original = agents.execute_tool
    def fail_reviewer(role, name, arguments, state):
        if role == 'reviewer':
            from tools import ToolError
            raise ToolError('Source lookup failed. Run incomplete.')
        return original(role, name, arguments, state)
    monkeypatch.setattr(agents, 'execute_tool', fail_reviewer)
    result, client = run(complete_run())
    assert result['status'] == 'incomplete' and result['report_draft']
    assert result['critical_review'] == {}
    assert client.chat.completions.create.call_count == 6
    assert 'Source lookup failed' in result['error']
