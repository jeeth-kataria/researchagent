"""Exercise the actual FastAPI boundary and graph without network calls."""

from http import HTTPStatus
from fastapi.testclient import TestClient
import json
import runpy
from pathlib import Path
from unittest.mock import Mock

import pytest

import app
from agents import Limits
from workflow import initial_state, build_graph

HELPERS = runpy.run_path(str(Path(__file__).with_name('test_workflow.py')))
PDF_FIXTURE = runpy.run_path(str(Path(__file__).with_name('test_pdf_reader.py')))['original_pdf'].__wrapped__


def request(method='GET', path='/api/research', body=b'', content_type='', extra=None):
    headers = {'Content-Type': content_type} if content_type else {}
    headers.update(extra or {})
    # TestClient buffers bodies: these tests establish the NDJSON contract,
    # not incremental delivery over a real socket.
    with TestClient(app.app) as client:
        response = client.request(method, path, content=body, headers=headers)
    return {'status': f'{response.status_code} {HTTPStatus(response.status_code).phrase}',
            'headers': response.headers, 'body': response.content}


def multipart(pdf, question='Explain the contribution and limitations.'):
    boundary = 'research-desk-test-boundary'
    data = (f'--{boundary}\r\nContent-Disposition: form-data; name="question"\r\n\r\n{question}\r\n'
            f'--{boundary}\r\nContent-Disposition: form-data; name="paper"; filename="study.pdf"\r\n'
            'Content-Type: application/pdf\r\n\r\n').encode() + pdf + f'\r\n--{boundary}--\r\n'.encode()
    return data, f'multipart/form-data; boundary={boundary}'


@pytest.fixture(autouse=True)
def environment(monkeypatch):
    monkeypatch.setenv('MAX_REQUESTS_PER_MINUTE', '1000')
    app.RATE_BUCKETS.clear()
    monkeypatch.setenv('RESEARCH_ACCESS_TOKEN', '')
    monkeypatch.setenv('GROQ_API_KEY', 'mock-private-key')
    monkeypatch.setenv('GROQ_MODEL', 'openai/gpt-oss-20b')
    monkeypatch.setenv('MAX_UPLOAD_MB', '4')
    monkeypatch.setenv('MAX_TOTAL_MODEL_REQUESTS', '12')


def test_health_exposes_configuration_but_never_the_key():
    result = request()
    assert result['status'] == '200 OK'
    data = json.loads(result['body'])
    assert data['ready'] is True
    assert data['max_upload_mb'] == 4
    assert b'mock-private-key' not in result['body']
    assert result['headers']['Cache-Control'] == 'no-store'


def test_missing_credentials_do_not_start_a_run(monkeypatch):
    monkeypatch.setenv('GROQ_API_KEY', '')
    body, content_type = multipart(PDF_FIXTURE())
    result = request('POST', body=body, content_type=content_type)
    assert result['status'] == '503 Service Unavailable'
    assert 'key' in json.loads(result['body'])['error']


@pytest.mark.parametrize('pdf,question', [(b'not a PDF', 'Explain.'), (b'', 'Explain.'), (b'%PDF-broken', '')])
def test_invalid_uploads_return_errors_before_model_calls(pdf, question, monkeypatch):
    client_factory = Mock()
    monkeypatch.setattr(app, 'make_client', client_factory)
    body, content_type = multipart(pdf, question)
    result = request('POST', body=body, content_type=content_type)
    assert result['status'] == '400 Bad Request'
    client_factory.assert_not_called()


def test_oversized_upload_checks_bytes_not_content_length(monkeypatch):
    client_factory = Mock()
    monkeypatch.setattr(app, 'make_client', client_factory)
    monkeypatch.setenv('MAX_UPLOAD_MB', '1')
    body, content_type = multipart(b'%PDF-' + b'x' * (1024 * 1024))
    result = request('POST', body=body, content_type=content_type, extra={'Content-Length': '1'})
    assert result['status'] == '400 Bad Request'
    assert 'upload limit' in json.loads(result['body'])['error']
    client_factory.assert_not_called()


def test_static_paths_cannot_read_secrets():
    assert request(path='/.env')['status'] == '404 Not Found'
    assert request(path='/../.env')['status'] == '404 Not Found'
    assert request(path='/')['headers']['Content-Type'].startswith('text/html')
    assert request(path='/app.js')['headers']['Content-Type'].startswith('text/javascript')


def test_stream_runs_actual_graph_and_preserves_pairing_and_sources(monkeypatch):
    client = Mock()
    client.api_key = 'mock-private-key'
    client.__enter__ = Mock(return_value=client)
    client.__exit__ = Mock(return_value=False)
    client.chat.completions.create.side_effect = HELPERS['complete_run'](HELPERS['review']('report'))
    monkeypatch.setattr(app, 'make_client', lambda _: client)
    body, content_type = multipart(PDF_FIXTURE())
    response = request('POST', body=body, content_type=content_type)
    assert response['headers']['Content-Type'].startswith('application/x-ndjson')
    events = [json.loads(line) for line in response['body'].splitlines()]
    assert events[0]['type'] == 'extraction'
    assert list(events[0]['pages']) == ['1', '2']
    assert events[0]['evidence']['E0001']['page'] == 1
    snapshots = [event['state'] for event in events if event['type'] == 'state']
    assert snapshots[-1]['status'] == 'complete'
    assert snapshots[-1]['model_calls'] == client.chat.completions.create.call_count == 7
    assert 'pages' not in snapshots[-1] and 'events' not in snapshots[-1]
    assert len([event for event in events if event['type'] == 'event' and event['event']['kind'] == 'tool_executed']) == 3
    assert snapshots[-1]['report_draft'] and snapshots[-1]['critical_review']
    assert 'Affected draft statement' in snapshots[-1]['final_output']
    assert snapshots[-1]['critical_review']['concerns'][0]['concern'] in snapshots[-1]['final_output']
    assert not ({'correction_count', 'reviews', 'correction_history', 'next_agent'} & snapshots[-1].keys())
    assert events[-1] == {'type': 'finished', 'status': 'complete'}
    assert b'mock-private-key' not in response['body']
    assert all(call.kwargs['timeout'] <= 45 for call in client.chat.completions.create.call_args_list)


def test_deadline_stops_before_spending_another_request():
    state = initial_state('Explain.', {1: 'A small original paper.'})
    state['deadline'] = 1.0  # Already expired monotonic deadline.
    client = Mock()
    result = build_graph(client, 'openai/gpt-oss-20b', Limits()).invoke(state)
    assert result['status'] == 'incomplete'
    assert 'hosting time budget' in result['error']
    assert result['model_calls'] == 0
    client.chat.completions.create.assert_not_called()


def test_bad_budget_does_not_leak_provider_or_env_values(monkeypatch):
    monkeypatch.setenv('MAX_TOTAL_MODEL_REQUESTS', '99')
    response = request()
    assert response['status'] == '400 Bad Request'
    assert b'mock-private-key' not in response['body']


def test_reviewer_failure_stream_preserves_draft_and_labels_incomplete(monkeypatch):
    client = Mock()
    client.api_key = 'mock-private-key'
    client.__enter__ = Mock(return_value=client)
    client.__exit__ = Mock(return_value=False)
    client.chat.completions.create.side_effect = HELPERS['complete_run']('not JSON')
    monkeypatch.setattr(app, 'make_client', lambda _: client)
    body, content_type = multipart(PDF_FIXTURE())
    result = request('POST', body=body, content_type=content_type)
    messages = [json.loads(line) for line in result['body'].splitlines()]
    final = [message['state'] for message in messages if message['type'] == 'state'][-1]
    assert final['status'] == 'incomplete' and final['report_draft']
    assert final['critical_review'] == {}
    assert 'did not receive a completed final critical review' in final['final_output']
    assert messages[-1] == {'type': 'finished', 'status': 'incomplete'}


def test_client_cleanup_failure_cannot_discard_completed_review(monkeypatch):
    client = Mock()
    client.api_key = 'mock-private-key'
    client.__enter__ = Mock(return_value=client)
    client.__exit__ = Mock(side_effect=RuntimeError('private cleanup error'))
    client.chat.completions.create.side_effect = HELPERS['complete_run'](HELPERS['review']('report'))
    monkeypatch.setattr(app, 'make_client', lambda _: client)
    body, content_type = multipart(PDF_FIXTURE())
    result = request('POST', body=body, content_type=content_type)
    messages = [json.loads(line) for line in result['body'].splitlines()]
    final = [message['state'] for message in messages if message['type'] == 'state'][-1]
    assert final['status'] == 'complete'
    assert final['critical_review']['concerns'][0]['concern'] in final['final_output']
    assert 'Affected draft statement' in final['final_output']
    assert messages[-1] == {'type': 'finished', 'status': 'complete'}
    assert client.chat.completions.create.call_count == 7
    assert b'private cleanup error' not in result['body']


@pytest.mark.parametrize('case', ['missing_paper', 'missing_question', 'blank_question',
    'long_question', 'extra_field', 'duplicate_paper', 'duplicate_question', 'json_body', 'malformed_form'])
def test_form_validation_is_readable_and_never_calls_provider(case, monkeypatch):
    factory = Mock()
    monkeypatch.setattr(app, 'make_client', factory)
    files = [('paper', ('study.pdf', PDF_FIXTURE(), 'application/pdf'))]
    data = {'question': 'Explain.'}
    if case == 'missing_paper':
        files = []
    elif case == 'missing_question':
        data = {}
    elif case == 'blank_question':
        data['question'] = '   '
    elif case == 'long_question':
        data['question'] = 'x' * 2001
    elif case == 'extra_field':
        data['extra'] = 'mock-private-key'
    elif case == 'duplicate_paper':
        files += [('paper', ('other.pdf', PDF_FIXTURE(), 'application/pdf'))]
    elif case == 'duplicate_question':
        files += [('question', (None, 'Another question'))]
    with TestClient(app.app) as client:
        if case == 'json_body':
            result = client.post('/api/research', json={'question': 'mock-private-key'})
        elif case == 'malformed_form':
            result = client.post('/api/research', content=b'not multipart',
                                 headers={'Content-Type': 'multipart/form-data'})
        else:
            result = client.post('/api/research', files=files, data=data)
    assert result.status_code == 400
    assert isinstance(result.json()['error'], str)
    assert 'mock-private-key' not in result.text
    assert result.headers['cache-control'] == 'no-store'
    factory.assert_not_called()


@pytest.mark.parametrize('case', ['valid', 'oversized', 'missing_question', 'extra_file', 'missing_key', 'invalid_pdf'])
def test_endpoint_read_bound_and_upload_resource_cleanup(case, monkeypatch):
    import starlette.datastructures
    import starlette.formparsers
    created = []
    reads = []
    original_spool = starlette.formparsers.SpooledTemporaryFile
    original_read = starlette.datastructures.UploadFile.read
    def tracked_spool(*args, **kwargs):
        file = original_spool(*args, **kwargs)
        created.append(file)
        return file
    async def tracked_read(upload, size=-1):
        reads.append(size)
        return await original_read(upload, size)
    monkeypatch.setattr(starlette.formparsers, 'SpooledTemporaryFile', tracked_spool)
    monkeypatch.setattr(starlette.datastructures.UploadFile, 'read', tracked_read)
    monkeypatch.setenv('MAX_UPLOAD_MB', '1')
    factory = Mock()
    client_mock = Mock()
    client_mock.api_key = 'mock-private-key'
    client_mock.__enter__ = Mock(return_value=client_mock)
    client_mock.__exit__ = Mock(return_value=False)
    client_mock.chat.completions.create.side_effect = HELPERS['complete_run']()
    factory.return_value = client_mock
    monkeypatch.setattr(app, 'make_client', factory)
    pdf = b'x' * (1024 * 1024 + 100) if case == 'oversized' else b'not PDF' if case == 'invalid_pdf' else PDF_FIXTURE()
    files = [('paper', ('study.pdf', pdf, 'application/pdf'))]
    if case == 'extra_file':
        files.append(('extra', ('extra.pdf', PDF_FIXTURE(), 'application/pdf')))
    if case == 'missing_key':
        monkeypatch.setenv('GROQ_API_KEY', '')
    with TestClient(app.app) as client:
        result = client.post('/api/research', files=files,
                             data={} if case == 'missing_question' else {'question': 'Explain.'})
    assert result.status_code == (200 if case == 'valid' else 503 if case == 'missing_key' else 400)
    assert created and all(file.closed for file in created)
    assert reads == ([1024 * 1024 + 1] if case in ('valid', 'oversized', 'invalid_pdf') else [])
    if case != 'valid':
        factory.assert_not_called()


@pytest.mark.parametrize('path', ['/.env', '/.git/config', '/local-docs/INTERVIEW_READY.md',
    '/app.py', '/api/research.py', '/requirements.txt', '/public/../.env', '/docs', '/openapi.json'])
def test_only_intended_public_paths_are_served(path):
    response = request(path=path)
    assert response['status'] == '404 Not Found'
    assert json.loads(response['body']) == {'error': 'Not found.'}


def test_empty_model_is_rejected_before_streaming(monkeypatch):
    factory = Mock()
    monkeypatch.setattr(app, 'make_client', factory)
    monkeypatch.setenv('GROQ_MODEL', ' ')
    body, content_type = multipart(PDF_FIXTURE())
    result = request('POST', body=body, content_type=content_type)
    assert result['status'] == '400 Bad Request'
    assert 'GROQ_MODEL' in json.loads(result['body'])['error']
    assert not result['headers']['Content-Type'].startswith('application/x-ndjson')
    factory.assert_not_called()


def test_pdf_text_limit_fails_before_streaming(monkeypatch):
    factory = Mock()
    monkeypatch.setattr(app, 'make_client', factory)
    monkeypatch.setenv('MAX_TEXT_CHARS', '50')
    body, content_type = multipart(PDF_FIXTURE())
    result = request('POST', body=body, content_type=content_type)
    assert result['status'] == '400 Bad Request'
    assert 'not been truncated' in json.loads(result['body'])['error']
    factory.assert_not_called()


def test_pdf_and_sync_stream_are_off_the_async_event_loop(monkeypatch):
    import asyncio
    import threading
    import httpx
    identifiers = {}
    original_extract = app.extract_pdf
    original_updates = app.ndjson_updates
    def extract(*args):
        identifiers['pdf'] = threading.get_ident()
        return original_extract(*args)
    def updates(*args):
        identifiers['stream'] = threading.get_ident()
        yield from original_updates(*args)
    monkeypatch.setattr(app, 'extract_pdf', extract)
    monkeypatch.setattr(app, 'ndjson_updates', updates)
    client = Mock()
    client.api_key = 'mock-private-key'
    client.__enter__ = Mock(return_value=client)
    client.__exit__ = Mock(return_value=False)
    client.chat.completions.create.side_effect = HELPERS['complete_run']()
    monkeypatch.setattr(app, 'make_client', lambda _: client)
    async def run():
        identifiers['event_loop'] = threading.get_ident()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app.app), base_url='http://test') as http:
            result = await http.post('/api/research', files={'paper': ('study.pdf', PDF_FIXTURE())},
                                     data={'question': 'Explain.'})
            assert result.status_code == 200
            assert json.loads(result.content.splitlines()[-1])['status'] == 'complete'
    asyncio.run(run())
    assert identifiers['pdf'] != identifiers['event_loop']
    assert identifiers['stream'] != identifiers['event_loop']
    assert client.chat.completions.create.call_count == 7


def test_disconnect_closes_stream_without_advancing_it():
    import asyncio
    import threading
    from starlette.requests import ClientDisconnect
    records = []
    def iterator():
        try:
            records.append(('step', threading.get_ident()))
            yield b'{"type":"extraction"}\n'
            records.append(('unexpected_next_step', threading.get_ident()))
        finally:
            records.append(('closed', threading.get_ident()))
    async def run():
        loop_thread = threading.get_ident()
        async def receive():
            return {'type': 'http.disconnect'}
        async def send(message):
            if message['type'] == 'http.response.body':
                raise OSError('Disconnected')
        await app.InvestigationResponse(iterator())(
            {'type': 'http', 'asgi': {'spec_version': '2.4'}}, receive, send)
        assert all(identifier != loop_thread for _, identifier in records)
    asyncio.run(run())
    assert not any(kind == 'unexpected_next_step' for kind, _ in records)
    if records:
        assert records[-1][0] == 'closed'


def test_vercel_entrypoint_exports_same_asgi_app_with_existing_routes():
    from api.research import app as deployed_app
    assert deployed_app is app.app
    config = json.loads((app.ROOT / 'vercel.json').read_text())
    assert config['framework'] is None  # Keep file-based routing rather than auto framework detection.
    assert config['outputDirectory'] == 'public'
    assert config['functions']['api/research.py']['maxDuration'] == 300
    assert 'local-docs/**' in config['functions']['api/research.py']['excludeFiles']
    with TestClient(deployed_app) as client:
        assert client.get('/api/research').status_code == 200
        assert client.get('/styles.css').status_code == 200
