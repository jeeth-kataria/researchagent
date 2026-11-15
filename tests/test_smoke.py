"""Optional real-socket smoke test; provider calls are always mocked."""
import json
import runpy
import socket
import threading
import time
from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest
import uvicorn

import app

HELPERS = runpy.run_path(str(Path(__file__).with_name('test_app.py')))


def test_real_http_delivers_ndjson_before_model_completion(monkeypatch):
    listener = socket.socket()
    try:
        listener.bind(('127.0.0.1', 0))
    except OSError as error:
        listener.close()
        pytest.skip(f'Local socket binding is unavailable: {error.strerror}')
    listener.listen()
    port = listener.getsockname()[1]
    entered = threading.Event()
    release = threading.Event()
    monkeypatch.setenv('GROQ_API_KEY', 'mock-private-key')
    monkeypatch.setenv('GROQ_MODEL', 'openai/gpt-oss-20b')
    monkeypatch.setenv('MAX_TOTAL_MODEL_REQUESTS', '12')
    client = Mock()
    client.api_key = 'mock-private-key'
    client.__enter__ = Mock(return_value=client)
    client.__exit__ = Mock(return_value=False)
    responses = iter(HELPERS['HELPERS']['complete_run']())
    def provider(**kwargs):
        entered.set()
        if not release.wait(5):
            raise AssertionError('Smoke-test provider gate timed out')
        return next(responses)
    client.chat.completions.create.side_effect = provider
    monkeypatch.setattr(app, 'make_client', lambda _: client)
    server = uvicorn.Server(uvicorn.Config(app.app, host='127.0.0.1', port=port,
                                          log_level='error', access_log=False))
    thread = threading.Thread(target=server.run, kwargs={'sockets': [listener]}, daemon=True)
    thread.start()
    try:
        for _ in range(100):
            if server.started:
                break
            if not thread.is_alive():
                pytest.fail('Uvicorn exited before startup')
            time.sleep(0.01)
        assert server.started
        base = f'http://127.0.0.1:{port}'
        with httpx.Client(base_url=base, timeout=5) as http:
            assert http.get('/').status_code == 200
            assert http.get('/app.js').status_code == 200
            with http.stream('POST', '/api/research',
                files={'paper': ('study.pdf', HELPERS['PDF_FIXTURE'](), 'application/pdf')},
                data={'question': 'Explain the contribution.'}) as response:
                assert response.status_code == 200
                assert response.headers['content-type'].startswith('application/x-ndjson')
                lines = response.iter_lines()
                first = json.loads(next(lines))
                assert first['type'] == 'extraction'  # Received before releasing provider.
                assert entered.wait(2)
                # A second real HTTP request succeeds while the model is blocked.
                assert http.get('/api/research', timeout=2).status_code == 200
                assert not release.is_set()
                release.set()
                rest = [json.loads(line) for line in lines]
                assert rest[-1] == {'type': 'finished', 'status': 'complete'}
                assert client.chat.completions.create.call_count == 7
    finally:
        release.set()
        server.should_exit = True
        thread.join(timeout=6)
        listener.close()
        assert not thread.is_alive(), 'Uvicorn did not shut down'
