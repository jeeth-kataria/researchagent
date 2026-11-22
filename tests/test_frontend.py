"""Feed actual HTTP graph streams into the frontend without browser dependencies."""
import json
from pathlib import Path
import runpy
import shutil
import subprocess
from unittest.mock import Mock

import pytest

import app

ROOT = Path(__file__).resolve().parents[1]
HELPERS = runpy.run_path(str(ROOT / 'tests/test_app.py'))
GRAPH = HELPERS['HELPERS']


@pytest.mark.parametrize('scenario', ['no_concerns', 'paper', 'report', 'review_failure', 'interrupted', 'cleanup_failure'])
def test_http_frontend_export_and_storage_contract(scenario, tmp_path, monkeypatch):
    node = shutil.which('node')
    if not node:
        pytest.skip('Node is optional for frontend contract checks; Python runtime does not require it.')
    monkeypatch.setenv('MAX_REQUESTS_PER_MINUTE', '1000')
    monkeypatch.setenv('RESEARCH_ACCESS_TOKEN', '')
    monkeypatch.setenv('GROQ_API_KEY', 'mock-private-key')
    monkeypatch.setenv('MAX_TOTAL_MODEL_REQUESTS', '12')
    client = Mock()
    client.api_key = 'mock-private-key'
    client.__enter__ = Mock(return_value=client)
    client.__exit__ = Mock(return_value=False)
    if scenario == 'cleanup_failure':
        client.__exit__ = Mock(side_effect=RuntimeError('private cleanup error'))
    target = 'report' if scenario == 'cleanup_failure' else scenario if scenario in ('paper', 'report') else None
    output = 'not JSON' if scenario == 'review_failure' else GRAPH['review'](target)
    client.chat.completions.create.side_effect = GRAPH['complete_run'](output)
    monkeypatch.setattr(app, 'make_client', lambda _: client)
    body, content_type = HELPERS['multipart'](HELPERS['PDF_FIXTURE']())
    result = HELPERS['request']('POST', body=body, content_type=content_type)
    messages = [json.loads(line) for line in result['body'].splitlines()]
    status = 'incomplete' if scenario == 'review_failure' else 'complete'
    if scenario == 'interrupted':
        # A real writer snapshot, before any reviewer result or final formatter.
        end = next(index for index, message in enumerate(messages)
                   if message['type'] == 'state' and message['state']['report_draft'])
        messages = messages[:end + 1]
        status = 'running'
    fixture = tmp_path / 'http-stream.json'
    fixture.write_text(json.dumps({'messages': messages, 'status': status, 'target': target}))
    checked = subprocess.run([node, str(ROOT / 'tests/frontend_contract.cjs'), str(fixture),
                              str(ROOT / 'public/app.js')], capture_output=True, text=True)
    assert checked.returncode == 0, checked.stdout + checked.stderr
