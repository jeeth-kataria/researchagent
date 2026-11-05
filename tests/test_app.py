"""Exercise the actual WSGI boundary and graph without network calls."""

from io import BytesIO
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
    environ = {'REQUEST_METHOD': method, 'PATH_INFO': path,
               'CONTENT_LENGTH': str(len(body)), 'CONTENT_TYPE': content_type,
               'wsgi.input': BytesIO(body)}
    environ.update(extra or {})
    response = {}
    def start(status, headers):
        response.update(status=status, headers=dict(headers))
    response['body'] = b''.join(app.application(environ, start))
    return response


def multipart(pdf, question='Explain the contribution and limitations.'):
    boundary = 'research-desk-test-boundary'
    data = (f'--{boundary}\r\nContent-Disposition: form-data; name="question"\r\n\r\n{question}\r\n'
            f'--{boundary}\r\nContent-Disposition: form-data; name="paper"; filename="study.pdf"\r\n'
            'Content-Type: application/pdf\r\n\r\n').encode() + pdf + f'\r\n--{boundary}--\r\n'.encode()
    return data, f'multipart/form-data; boundary={boundary}'


@pytest.fixture(autouse=True)
def environment(monkeypatch):
    monkeypatch.setenv('GROQ_API_KEY', 'mock-private-key')
    monkeypatch.setenv('GROQ_MODEL', 'openai/gpt-oss-20b')
    monkeypatch.setenv('MAX_UPLOAD_MB', '4')


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


def test_oversized_body_is_rejected_before_reading():
    result = request('POST', extra={'CONTENT_LENGTH': str(5 * 1024 * 1024)})
    assert result['status'] == '400 Bad Request'


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
    client.chat.completions.create.side_effect = HELPERS['approval']()
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
    assert snapshots[-1]['model_calls'] == client.chat.completions.create.call_count == 8
    assert 'pages' not in snapshots[-1] and 'events' not in snapshots[-1]
    assert len([event for event in events if event['type'] == 'event' and event['event']['kind'] == 'tool_executed']) == 4
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
