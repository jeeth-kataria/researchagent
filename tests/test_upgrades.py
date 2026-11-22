"""Offline behavior checks for retrieval, targeted revision and HTTP safeguards."""
import asyncio
import copy
import json
from io import BytesIO
from pathlib import Path
import runpy
from types import SimpleNamespace as NS
from unittest.mock import Mock

import anyio
import groq
import httpx
import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

import app
from agents import Limits
from llm import RunError, call_model, make_client
from pdf_reader import extract_pdf, prepare_passages
from tools import search_paper
from workflow import initial_state, build_graph

H = runpy.run_path(str(Path(__file__).with_name('test_workflow.py')))
HTTP = runpy.run_path(str(Path(__file__).with_name('test_app.py')))


def graph(responses, **kwargs):
    client = Mock(api_key='mock-only')
    client.chat.completions.create.side_effect = responses
    result = build_graph(client, 'openai/gpt-oss-20b', **kwargs).invoke(H['source_state']())
    return result, client


def test_bm25_rare_term_and_length_rank_without_changing_evidence():
    state = initial_state('Explain experiments.', {
        1: 'dataset ' + 'unrelated material ' * 25 + 'ImageNet',
        2: 'The ImageNet dataset experiment reports accuracy.',
        3: 'dataset generic discussion', 4: 'dataset overview'})
    original = copy.deepcopy(state['evidence'])
    result = search_paper(state, 'dataset ImageNet')
    assert result['method'].startswith('BM25')
    assert result['matches'][0]['page'] == 2
    # Old overlap rank tied both two-term passages and selected physical page 1.
    import re
    old = sorted(state['evidence'].values(), key=lambda e: (-len({'dataset','imagenet'} & set(re.findall(r'[a-z0-9]+', e['text'].lower()))), e['page'], e['start']))
    assert old[0]['page'] == 1
    assert state['evidence'] == original
    for match in result['matches']:
        assert match == original[match['evidence_id']]
    assert search_paper(state, 'paraphrase-unseen')['matches'] == []


def test_layout_extraction_preserves_table_columns_and_exact_offsets():
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = DictionaryObject({NameObject('/Type'): NameObject('/Font'),
        NameObject('/Subtype'): NameObject('/Type1'), NameObject('/BaseFont'): NameObject('/Courier')})
    page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'): DictionaryObject({NameObject('/F1'): font})})
    stream = DecodedStreamObject()
    commands = []
    for y, left, right in [(730, 'Method / dataset', 'Accuracy'),
                            (710, 'Baseline ImageNet', '0.80'),
                            (690, 'Proposed ImageNet', '0.85')]:
        for x, text in [(40, left), (270, right)]:
            commands.append(f'BT /F1 12 Tf 1 0 0 1 {x} {y} Tm ({text}) Tj ET')
    stream.set_data('\n'.join(commands).encode())
    page[NameObject('/Contents')] = writer._add_object(stream)
    out = BytesIO(); writer.write(out)
    pages = extract_pdf(out.getvalue(), mode='layout')
    lines = pages[1].splitlines()
    assert any('Baseline ImageNet' in line and '0.80' in line for line in lines)
    assert any('Proposed ImageNet' in line and '0.85' in line for line in lines)
    for item in prepare_passages(pages, 60).values():
        assert pages[item['page']][item['start']:item['end']] == item['text']
    with pytest.raises(ValueError, match='mode'):
        extract_pdf(out.getvalue(), mode='ocr')


def repair_fixture():
    valid = {'findings': [{'topic': 'Inspection', 'statement': 'A detail was not found in inspected passages.',
        'kind': 'missing_information', 'evidence_ids': []}], 'uncertainties': []}
    broken = copy.deepcopy(valid)
    del broken['findings'][0]['evidence_ids']
    return broken, valid


def test_format_reask_is_logged_and_preserves_content_within_visit_budget():
    broken, valid = repair_fixture()
    responses = [H['response'](broken), H['response'](valid)] + H['visit']('analyst') + H['visit']('writer') + H['visit']('reviewer')
    result, client = graph(responses)
    assert result['status'] == 'complete' and result['format_repairs'] == 1
    assert result['reader_findings'] == valid
    assert any(event['kind'] == 'format_repair' for event in result['events'])
    second = client.chat.completions.create.call_args_list[1].kwargs
    assert 'tools' not in second  # Repair cannot inspect new sources or take actions.
    assert result['visits'][0]['model_calls'] == 2
    assert result['model_calls'] == 7


def test_format_reask_cannot_change_claims_or_repeat():
    broken, valid = repair_fixture()
    changed = copy.deepcopy(valid)
    changed['findings'][0]['statement'] = 'An invented different finding.'
    state, client = graph([H['response'](broken), H['response'](changed)])
    assert state['status'] == 'incomplete' and 'changed content' in state['error']
    assert client.chat.completions.create.call_count == 2
    state, client = graph([H['response'](broken), H['response'](broken)])
    assert state['status'] == 'incomplete' and state['format_repairs'] == 1
    assert client.chat.completions.create.call_count == 2
    state, client = graph([H['response'](broken)], limits=Limits(model_per_visit=1))
    assert state['status'] == 'incomplete' and state['format_repairs'] == 0
    assert client.chat.completions.create.call_count == 1


def revision_responses(index=1, evidence='E0002'):
    draft = H['findings'](writer=True)
    for i, finding in enumerate(draft['findings']):
        finding['statement'] = f'Original finding number {i}'
    review = H['review']('report')
    review['concerns'][0]['affected_statement'] = draft['findings'][1]['statement']
    patches = {'patches': [{'finding_index': index, 'statement': 'The comparison is limited to this experiment.',
        'kind': 'interpretation', 'evidence_ids': [evidence]}]}
    responses = H['visit']('reader') + H['visit']('analyst') + H['visit']('writer', draft) + H['visit']('reviewer', review) + [H['response'](patches)]
    return responses, draft, review


def test_one_revision_preserves_original_review_and_unflagged_statements():
    responses, draft, review = revision_responses()
    state, client = graph(responses, enable_revision=True, reviewer_model='openai/gpt-oss-120b', limits=Limits(total_model_calls=13))
    assert state['status'] == state['revision_status'] == 'complete'
    assert state['report_draft'] == draft and state['critical_review'] == review
    for i in [0, 2, 3]:
        assert state['report_revision']['findings'][i] == draft['findings'][i]
    assert state['report_revision']['findings'][1]['statement'] != draft['findings'][1]['statement']
    calls = client.chat.completions.create.call_args_list
    assert calls[-2].kwargs['model'] == 'openai/gpt-oss-120b'
    assert calls[-1].kwargs['model'] == 'openai/gpt-oss-20b' and 'tools' not in calls[-1].kwargs
    assert len(calls) == 8 and len(state['visits']) == 5
    assert 'not reviewed again' in state['final_output']
    assert review['concerns'][0]['concern'] in state['final_output']
    assert 'Original finding number 1' in state['final_output']


@pytest.mark.parametrize('index,evidence', [(0, 'E0002'), (1, 'E9999')])
def test_invalid_revision_keeps_review_and_draft(index, evidence):
    responses, draft, review = revision_responses(index, evidence)
    state, client = graph(responses, enable_revision=True)
    assert state['status'] == 'incomplete' and state['revision_status'] == 'incomplete'
    assert state['report_draft'] == draft and state['critical_review'] == review
    assert state['report_revision'] == {}
    assert client.chat.completions.create.call_count == 8
    assert 'Critical review completed' in state['final_output']


def test_revision_budget_and_paper_only_concerns_do_not_start_extra_calls():
    responses, draft, review = revision_responses()
    state, client = graph(responses, enable_revision=True, limits=Limits(total_model_calls=7))
    assert state['status'] == 'incomplete' and 'budget' in state['error']
    assert client.chat.completions.create.call_count == 7
    state, client = graph(H['complete_run'](H['review']('paper')), enable_revision=True)
    assert state['status'] == 'complete' and not state['report_revision']
    assert client.chat.completions.create.call_count == 7


def test_authentication_fails_before_inference_and_token_is_not_public(monkeypatch):
    monkeypatch.setenv('GROQ_API_KEY', 'mock-only')
    monkeypatch.setenv('RESEARCH_ACCESS_TOKEN', 'private-access-secret')
    factory = Mock()
    monkeypatch.setattr(app, 'make_client', factory)
    body, content_type = HTTP['multipart'](HTTP['PDF_FIXTURE']())
    response = HTTP['request']('POST', body=body, content_type=content_type)
    assert response['status'] == '401 Unauthorized'
    assert factory.call_count == 0
    public = HTTP['request']()
    assert json.loads(public['body'])['auth_required'] is True
    assert b'private-access-secret' not in public['body']
    assert b'mock-only' not in public['body']


def test_rate_limit_is_bounded_and_expires(monkeypatch):
    app.RATE_BUCKETS.clear()
    clock = [100.0]
    monkeypatch.setattr(app.time, 'monotonic', lambda: clock[0])
    assert app.allow_request('client-a', 1)
    assert not app.allow_request('client-a', 1)
    assert app.allow_request('client-b', 1)
    clock[0] += 60
    assert app.allow_request('client-a', 1)
    assert 'client-a' not in app.RATE_BUCKETS
    app.RATE_BUCKETS.clear()


def test_metrics_unknown_usage_and_list_price_are_not_actual_billing():
    state = initial_state('Explain.', {1: 'An original short paper with text.'})
    state.update(model_calls=1, elapsed_seconds=2, usage=[{'model':'model', 'prompt_tokens':100, 'completion_tokens':20, 'total_tokens':120}],
        prices={'model': {'input':1, 'output':2}})
    metrics = app.run_metrics(state)
    assert metrics['estimated_list_price_usd'] == .00014
    assert metrics['total_tokens'] == 120 and metrics['elapsed_seconds'] == 2
    state['model_calls'] = 2
    metrics = app.run_metrics(state)
    assert metrics['total_tokens'] is None and metrics['estimated_list_price_usd'] is None
    assert metrics['reported_total_tokens'] == 120 and metrics['unknown_usage_requests'] == 1


def test_disconnect_cancels_inflight_async_groq_http_request(monkeypatch):
    """Actual SDK + async mock transport; no network/quota or buffered-client claim."""
    monkeypatch.setenv('GROQ_API_KEY', 'mock-only')
    actual = groq.AsyncGroq
    records = {}
    async def exercise():
        started, cancelled = anyio.Event(), anyio.Event()
        async def transport(request):
            started.set()
            try:
                await anyio.sleep_forever()
            finally:
                cancelled.set()
        def factory(**options):
            return actual(**options, http_client=httpx.AsyncClient(transport=httpx.MockTransport(transport)))
        monkeypatch.setattr(app.groq, 'AsyncGroq', factory)
        def iterator():
            try:
                with make_client('mock-only') as client:
                    call_model(client, 'openai/gpt-oss-20b', [{'role':'user','content':'test'}], [], 128, False)
                records['unexpected_completion'] = True
            except RunError as error:
                records['error'] = str(error)
            finally:
                records['closed'] = True
            yield b'not sent after cancellation'
        async def receive():
            await started.wait()
            return {'type':'http.disconnect'}
        async def send(message):
            assert message['type'] != 'http.response.body'
        with anyio.fail_after(3):
            await app.InvestigationResponse(iterator())({'type':'http'}, receive, send)
        assert cancelled.is_set() and records['closed']
        assert 'cancelled' in records['error'] and not records.get('unexpected_completion')
    asyncio.run(exercise())


def test_format_reask_is_global_and_cannot_move_existing_field_values():
    broken, valid = repair_fixture()
    # After one reader repair, an analyst format failure cannot trigger another.
    responses = [H['response'](broken), H['response'](valid), H['response'](broken)]
    state, client = graph(responses)
    assert state['status'] == 'incomplete' and state['format_repairs'] == 1
    assert client.chat.completions.create.call_count == 3
    wrong = copy.deepcopy(valid)
    wrong['findings'][0]['statement'], wrong['findings'][0]['topic'] = wrong['findings'][0]['topic'], wrong['findings'][0]['statement']
    state, client = graph([H['response'](broken), H['response'](wrong)])
    assert state['status'] == 'incomplete' and 'changed content' in state['error']


def test_authenticated_http_revision_survives_frontend_and_markdown(tmp_path, monkeypatch):
    import shutil
    import subprocess
    monkeypatch.setenv('GROQ_API_KEY', 'mock-only')
    monkeypatch.setenv('RESEARCH_ACCESS_TOKEN', 'private-access-secret')
    monkeypatch.setenv('MAX_TOTAL_MODEL_REQUESTS', '13')
    monkeypatch.setenv('MAX_REQUESTS_PER_MINUTE', '1000')
    app.RATE_BUCKETS.clear()
    client = Mock(api_key='mock-only')
    client.__enter__ = Mock(return_value=client)
    client.__exit__ = Mock(return_value=False)
    patches = {'patches':[{'finding_index':0, 'statement':'The conclusion is limited to this experiment.',
        'kind':'interpretation', 'evidence_ids':['E0002']}]}
    client.chat.completions.create.side_effect = H['complete_run'](H['review']('report')) + [H['response'](patches)]
    monkeypatch.setattr(app, 'make_client', lambda _: client)
    with HTTP['TestClient'](app.app) as browser:
        response = browser.post('/api/research', headers={'Authorization':'Bearer private-access-secret'},
            files={'paper':('paper.pdf', HTTP['PDF_FIXTURE'](), 'application/pdf')},
            data={'question':'Explain the evidence.', 'revise_report':'true'})
    assert response.status_code == 200
    messages = [json.loads(line) for line in response.content.splitlines()]
    state = [m['state'] for m in messages if m['type'] == 'state'][-1]
    assert state['status'] == 'complete' and state['report_revision']
    assert any(m.get('state',{}).get('revision_status') == 'pending' and m['state']['status'] == 'running' for m in messages)
    assert b'private-access-secret' not in response.content
    assert b'mock-only' not in response.content
    assert messages[-1] == {'type':'finished', 'status':'complete'}
    node = shutil.which('node')
    if node:
        fixture = tmp_path / 'revision.json'
        fixture.write_text(json.dumps({'messages':messages, 'status':'complete', 'target':'report', 'revision':True}))
        checked = subprocess.run([node, 'tests/frontend_contract.cjs', str(fixture), 'public/app.js'], capture_output=True, text=True)
        assert checked.returncode == 0, checked.stdout + checked.stderr
    app.RATE_BUCKETS.clear()


def test_http_rate_limit_prevents_second_inference(monkeypatch):
    monkeypatch.setenv('GROQ_API_KEY', 'mock-only')
    monkeypatch.setenv('RESEARCH_ACCESS_TOKEN', '')
    monkeypatch.setenv('MAX_REQUESTS_PER_MINUTE', '1')
    app.RATE_BUCKETS.clear()
    client = Mock(api_key='mock-only')
    client.__enter__ = Mock(return_value=client)
    client.__exit__ = Mock(return_value=False)
    client.chat.completions.create.side_effect = H['complete_run']()
    factory = Mock(return_value=client)
    monkeypatch.setattr(app, 'make_client', factory)
    body, content_type = HTTP['multipart'](HTTP['PDF_FIXTURE']())
    assert HTTP['request']('POST', body=body, content_type=content_type)['status'] == '200 OK'
    assert HTTP['request']('POST', body=body, content_type=content_type)['status'] == '429 Too Many Requests'
    assert factory.call_count == 1
    app.RATE_BUCKETS.clear()


def test_full_http_graph_uses_cancellable_async_sdk_adapter(monkeypatch):
    """Successful SDK parsing and real graph through the bridge, not an SDK-shaped mock."""
    monkeypatch.setenv('GROQ_API_KEY', 'mock-only')
    monkeypatch.setenv('RESEARCH_ACCESS_TOKEN', '')
    monkeypatch.setenv('MAX_REQUESTS_PER_MINUTE', '1000')
    monkeypatch.setenv('GROQ_REVIEWER_MODEL', 'openai/gpt-oss-120b')
    monkeypatch.setenv('TOKEN_PRICES_JSON', json.dumps({model: {'input':1, 'output':2} for model in ['openai/gpt-oss-20b', 'openai/gpt-oss-120b']}))
    app.RATE_BUCKETS.clear()
    actual = groq.AsyncGroq
    responses = iter(H['complete_run']())
    requests = []
    async def transport(request):
        data = json.loads(request.content)
        requests.append(data)
        response = next(responses)
        choice = response.choices[0]
        message = {'role':'assistant', 'content':choice.message.content}
        if choice.message.tool_calls:
            message['tool_calls'] = [{'id':c.id, 'type':'function',
                'function':{'name':c.function.name, 'arguments':c.function.arguments}}
                for c in choice.message.tool_calls]
        return httpx.Response(200, json={'id':'mock-response', 'object':'chat.completion',
            'created':0, 'model':data['model'], 'choices':[{'index':0, 'message':message,
            'finish_reason':choice.finish_reason}],
            'usage':{'prompt_tokens':10, 'completion_tokens':5, 'total_tokens':15}})
    def factory(**options):
        return actual(**options, http_client=httpx.AsyncClient(transport=httpx.MockTransport(transport)))
    monkeypatch.setattr(app.groq, 'AsyncGroq', factory)
    body, content_type = HTTP['multipart'](HTTP['PDF_FIXTURE']())
    response = HTTP['request']('POST', body=body, content_type=content_type)
    assert response['status'] == '200 OK'
    messages = [json.loads(line) for line in response['body'].splitlines()]
    state = [m['state'] for m in messages if m['type']=='state'][-1]
    assert messages[-1] == {'type':'finished', 'status':'complete'}
    assert len(requests) == 7 and state['metrics']['total_tokens'] == 105
    assert state['models']['reviewer'] == 'openai/gpt-oss-120b'
    assert state['metrics']['estimated_list_price_usd'] == .00014
    assert 'prices' not in state
    assert b'mock-only' not in response['body']
    app.RATE_BUCKETS.clear()
