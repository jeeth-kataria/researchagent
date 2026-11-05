"""Research Desk: standard-library HTTP boundary for the existing Python graph.

Run `python app.py` locally. Vercel imports the same WSGI application through
api/research.py. Each investigation lives in one streamed HTTP request.
"""

import hashlib
import json
import os
import time
from email import policy
from email.parser import BytesParser
from pathlib import Path
from wsgiref.simple_server import make_server

from dotenv import load_dotenv
from langgraph.config import get_stream_writer

from agents import Limits
from llm import RunError, make_client, sanitize
from pdf_reader import PDFError, extract_pdf
from workflow import build_graph, initial_state

ROOT = Path(__file__).parent
if not os.getenv('VERCEL'):
    load_dotenv(ROOT / '.env')
STATIC = {'/': ('index.html', 'text/html; charset=utf-8'),
          '/styles.css': ('styles.css', 'text/css; charset=utf-8'),
          '/app.js': ('app.js', 'text/javascript; charset=utf-8')}


def settings() -> dict:
    def number(name, default):
        try:
            value = int(os.getenv(name, str(default)))
            if value < 1:
                raise ValueError
            return value
        except ValueError:
            raise RunError(f'{name} must be a positive integer.') from None
    return {
        'key': os.getenv('GROQ_API_KEY', '').strip(),
        'model': os.getenv('GROQ_MODEL', 'openai/gpt-oss-20b').strip(),
        # Four MiB plus form fields fits below Vercel's 4.5 MB body ceiling.
        'max_upload_mb': min(number('MAX_UPLOAD_MB', 4), 4),
        'max_text_chars': number('MAX_TEXT_CHARS', 20_000),
        'limits': Limits(model_per_visit=number('MAX_MODEL_REQUESTS_PER_VISIT', 3),
                         tools_per_visit=number('MAX_TOOL_CALLS_PER_VISIT', 2),
                         total_model_calls=number('MAX_TOTAL_MODEL_REQUESTS', 24),
                         output_tokens=number('GROQ_MAX_OUTPUT_TOKENS', 2048)),
    }


def parse_upload(environ, maximum_bytes: int) -> tuple[bytes, str, str]:
    """Parse one bounded multipart request without a third-party upload package."""
    try:
        length = int(environ.get('CONTENT_LENGTH', '0'))
    except ValueError:
        raise RunError('Invalid upload length.') from None
    if not 0 < length <= maximum_bytes + 16_384:
        raise RunError('Upload one PDF no larger than 4 MB, plus a short question.')
    content_type = environ.get('CONTENT_TYPE', '')
    if not content_type.startswith('multipart/form-data;') or '\r' in content_type or '\n' in content_type:
        raise RunError('Expected a PDF and question in a multipart form.')
    body = environ['wsgi.input'].read(length)
    if len(body) != length:
        raise RunError('The upload did not finish. Please select the PDF again.')
    message = BytesParser(policy=policy.default).parsebytes(
        f'Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n'.encode() + body)
    if not message.is_multipart():
        raise RunError('The upload form could not be read.')
    fields = {}
    for part in message.iter_parts():
        name = part.get_param('name', header='content-disposition')
        if name not in {'paper', 'question'} or name in fields or part.is_multipart():
            raise RunError('Upload exactly one PDF and one question.')
        fields[name] = part
    if set(fields) != {'paper', 'question'}:
        raise RunError('Choose a PDF and enter a question before starting.')
    data = fields['paper'].get_payload(decode=True) or b''
    if len(data) > maximum_bytes:
        raise RunError('The PDF exceeds the configured upload limit (at most 4 MB).')
    try:
        question = (fields['question'].get_payload(decode=True) or b'').decode('utf-8').strip()
    except UnicodeError:
        raise RunError('The question must be readable text.') from None
    if not question or len(question) > 2000:
        raise RunError('Enter a question of at most 2,000 characters.')
    filename = (fields['paper'].get_filename() or 'paper.pdf').replace('\\', '/').split('/')[-1][:180]
    filename = ''.join(character for character in filename if character.isprintable()) or 'paper.pdf'
    return data, filename, question


def public_state(state: dict, key: str) -> dict:
    """Pages/registry are sent once; events are streamed separately, never reasoning."""
    return sanitize({name: value for name, value in state.items()
                     if name not in {'pages', 'evidence', 'events', 'deadline'}}, key)


def investigate(data: bytes, filename: str, question: str, pages: dict, config: dict):
    """Yield real graph events and validated snapshots; keep failures inspectable."""
    state = initial_state(question, pages)
    state['deadline'] = time.monotonic() + 240
    meta = {'filename': filename, 'digest': hashlib.sha256(data).hexdigest(),
            'question': question, 'model': config['model']}
    yield {'type': 'extraction', 'meta': sanitize(meta, config['key']),
           'pages': sanitize(pages, config['key']), 'evidence': sanitize(state['evidence'], config['key'])}
    yield {'type': 'state', 'state': public_state(state, config['key'])}
    try:
        with make_client(config['key']) as client:
            graph = build_graph(client, config['model'], config['limits'],
                                on_event=lambda event: get_stream_writer()(event))
            for mode, update in graph.stream(state, stream_mode=['updates', 'custom']):
                if mode == 'custom':
                    yield {'type': 'event', 'event': sanitize(update, config['key'])}
                else:
                    state = next(iter(update.values()))
                    yield {'type': 'state', 'state': public_state(state, config['key'])}
    except GeneratorExit:
        raise  # A disconnected browser must not cause a retry/new investigation.
    except Exception:
        state['status'] = 'incomplete'
        state['error'] = 'The investigation could not finish. Available findings are preserved; no retry was made.'
        state['final_report'] = ''
        yield {'type': 'state', 'state': public_state(state, config['key'])}
    yield {'type': 'finished', 'status': state['status']}


def application(environ, start_response):
    """Serve only explicit static paths and one same-origin research endpoint."""
    def response(status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode()
        start_response(status, [('Content-Type', 'application/json; charset=utf-8'),
                                 ('Cache-Control', 'no-store'), ('X-Content-Type-Options', 'nosniff')])
        return [body]
    path, method = environ.get('PATH_INFO', '/'), environ.get('REQUEST_METHOD', 'GET')
    if path in STATIC and method == 'GET':
        filename, content_type = STATIC[path]
        start_response('200 OK', [('Content-Type', content_type), ('X-Content-Type-Options', 'nosniff')])
        return [(ROOT / 'public' / filename).read_bytes()]
    if path != '/api/research':
        return response('404 Not Found', {'error': 'Not found.'})
    if method not in {'GET', 'POST'}:
        return response('405 Method Not Allowed', {'error': 'Use GET for configuration or POST to investigate.'})
    try:
        config = settings()
        if method == 'GET':
            return response('200 OK', {'ready': bool(config['key']), 'model': config['model'],
                'max_upload_mb': config['max_upload_mb'], 'max_text_chars': config['max_text_chars']})
        if not config['key']:
            return response('503 Service Unavailable', {'error': 'The server needs a Groq API key before investigations can start.'})
        data, filename, question = parse_upload(environ, config['max_upload_mb'] * 1024 * 1024)
        pages = extract_pdf(data, config['max_upload_mb'], config['max_text_chars'])
    except (RunError, PDFError) as error:
        return response('400 Bad Request', {'error': str(error)})
    start_response('200 OK', [('Content-Type', 'application/x-ndjson; charset=utf-8'),
                              ('Cache-Control', 'no-store'), ('X-Content-Type-Options', 'nosniff'),
                              ('X-Accel-Buffering', 'no')])
    return ((json.dumps(event, ensure_ascii=False) + '\n').encode()
            for event in investigate(data, filename, question, pages, config))


if __name__ == '__main__':
    port = int(os.getenv('PORT', '8000'))
    print(f'Research Desk: http://localhost:{port}')
    with make_server('127.0.0.1', port, application) as server:
        server.serve_forever()
