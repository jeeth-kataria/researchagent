"""FastAPI boundary with NDJSON streaming and cancellable provider requests.

Run `python app.py` or `uvicorn app:app --reload --host 127.0.0.1 --port 8000`.
Vercel imports the same ASGI app through api/research.py.
"""

import hashlib
import asyncio
import hmac
import threading
import logging
from types import SimpleNamespace
from collections import deque
import json
import os
import time
from pathlib import Path
from typing import Annotated

import anyio
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from langgraph.config import get_stream_writer
from starlette.concurrency import run_in_threadpool, iterate_in_threadpool
from starlette.exceptions import HTTPException

from agents import Limits
from llm import RunError, make_client, sanitize, HTTP_CLIENT
import groq
from pdf_reader import PDFError, extract_pdf
from workflow import build_graph, initial_state
from reading import coverage_summary

ROOT = Path(__file__).parent
if not os.getenv('VERCEL'):
    load_dotenv(ROOT / '.env')
STATIC = {'/': ('index.html', 'text/html'),
          '/styles.css': ('styles.css', 'text/css'),
          '/app.js': ('app.js', 'text/javascript')}
API_HEADERS = {'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff'}
RATE_LOCK = threading.Lock()
RATE_BUCKETS = {}
app = FastAPI(title='Research Desk', docs_url=None, redoc_url=None, openapi_url=None)


def settings() -> dict:
    def number(name, default):
        try:
            value = int(os.getenv(name, str(default)))
            if value < 1:
                raise ValueError
            return value
        except ValueError:
            raise RunError(f'{name} must be a positive integer.') from None
    try:
        prices = json.loads(os.getenv('TOKEN_PRICES_JSON', '{}'))
        if not isinstance(prices, dict):
            raise ValueError
    except ValueError:
        raise RunError('TOKEN_PRICES_JSON must be a JSON object of model price settings.') from None
    model = os.getenv('GROQ_MODEL', 'openai/gpt-oss-20b').strip()
    if not model:
        raise RunError('GROQ_MODEL must not be empty.')
    return {
        'key': os.getenv('GROQ_API_KEY', '').strip(),
        'model': model,
        'reviewer_model': os.getenv('GROQ_REVIEWER_MODEL', 'openai/gpt-oss-120b').strip() or model,
        'access_token': os.getenv('RESEARCH_ACCESS_TOKEN', '').strip(),
        'rate_limit': number('MAX_REQUESTS_PER_MINUTE', 3),
        'prices': prices,
        'pdf_mode': os.getenv('PDF_EXTRACTION_MODE', 'layout'),
        # Four MiB plus form fields fits below Vercel's 4.5 MB body ceiling.
        'max_upload_mb': min(number('MAX_UPLOAD_MB', 4), 4),
        'max_text_chars': number('MAX_TEXT_CHARS', 20_000),
        'limits': Limits(model_per_visit=number('MAX_MODEL_REQUESTS_PER_VISIT', 3),
                         tools_per_visit=number('MAX_TOOL_CALLS_PER_VISIT', 2),
                         total_model_calls=number('MAX_TOTAL_MODEL_REQUESTS', 13),
                         output_tokens=number('GROQ_MAX_OUTPUT_TOKENS', 2048),
                         source_chars=number('MAX_SOURCE_CHARS_PER_ROLE', 4800)),
    }


def public_state(state: dict, key: str) -> dict:
    """Pages/registry are sent once; events are streamed separately, never reasoning."""
    state['metrics'] = run_metrics(state)
    return sanitize({**{name: value for name, value in state.items()
                     if name not in {'pages', 'evidence', 'events', 'deadline', 'requests', 'started_at', 'prices'}},
                     'coverage_summary': coverage_summary(state)}, key)


def investigate(data: bytes, filename: str, question: str, pages: dict, config: dict):
    """Yield real graph events and validated snapshots; keep failures inspectable."""
    state = initial_state(question, pages)
    state['deadline'] = time.monotonic() + 240
    state['prices'] = config.get('prices', {})
    meta = {'filename': filename, 'digest': hashlib.sha256(data).hexdigest(),
            'question': question, 'model': config['model'],
            'reviewer_model': config.get('reviewer_model', config['model']), 'extraction_mode': config.get('pdf_mode', 'layout')}
    yield {'type': 'extraction', 'meta': sanitize(meta, config['key']),
           'pages': sanitize(pages, config['key']), 'evidence': sanitize(state['evidence'], config['key'])}
    yield {'type': 'state', 'state': public_state(state, config['key'])}
    try:
        with make_client(config['key']) as client:
            graph = build_graph(client, config['model'], config['limits'],
                                on_event=lambda event: get_stream_writer()(event),
                                reviewer_model=config.get('reviewer_model'), enable_revision=config.get('enable_revision', False))
            for mode, update in graph.stream(state, stream_mode=['updates', 'custom']):
                if mode == 'custom':
                    yield {'type': 'event', 'event': sanitize(update, config['key'])}
                else:
                    state = next(iter(update.values()))
                    yield {'type': 'state', 'state': public_state(state, config['key'])}
    except GeneratorExit:
        raise  # A disconnected browser must not cause a retry/new investigation.
    except Exception:
        # Closing the SDK client can fail after the final state was streamed.
        # Cleanup must not erase an already completed draft and critical review.
        if state['status'] != 'complete':
            state['status'] = 'incomplete'
            state['error'] = 'The investigation could not finish. Available findings are preserved; no retry was made.'
            state['final_output'] = ''
            yield {'type': 'state', 'state': public_state(state, config['key'])}
    logging.getLogger('uvicorn.error').info('research_run %s', json.dumps(run_metrics(state)))
    yield {'type': 'finished', 'status': state['status']}


def allow_request(address, limit):
    """Best-effort per-instance throttle; shared access tokens work across instances."""
    key = hashlib.sha256(address.encode()).hexdigest()
    now = time.monotonic()
    with RATE_LOCK:
        for stale in list(RATE_BUCKETS):
            if not RATE_BUCKETS[stale] or now - RATE_BUCKETS[stale][-1] >= 60:
                del RATE_BUCKETS[stale]
        if key not in RATE_BUCKETS and len(RATE_BUCKETS) >= 1024:
            return False
        bucket = RATE_BUCKETS.setdefault(key, deque())
        while bucket and now - bucket[0] >= 60:
            bucket.popleft()
        if len(bucket) >= limit:
            return False
        bucket.append(now)
        return True


def run_metrics(state):
    records = state.get('usage', [])
    totals = {name: sum(item[name] for item in records) if records and all(
        type(item.get(name)) is int for item in records) and len(records) == state.get('model_calls', 0) else None
        for name in ('prompt_tokens', 'completion_tokens', 'total_tokens')}
    prices = state.get('prices', {})
    cost = 0.0
    priced = bool(records)
    for item in records:
        rates = prices.get(item.get('model'), {}) if isinstance(prices, dict) else {}
        try:
            input_rate, output_rate = float(rates['input']), float(rates['output'])
            if not 0 <= input_rate < 1000 or not 0 <= output_rate < 1000:
                raise ValueError
            cost += (item['prompt_tokens'] * input_rate + item['completion_tokens'] * output_rate) / 1_000_000
        except (KeyError, TypeError, ValueError):
            priced = False
    error = state.get('error', '')
    failure = next((kind for kind, terms in {
        'cancelled': ('cancelled',), 'quota': ('quota', 'rate limit'), 'request_size': ('oversized',),
        'credentials': ('credentials',), 'connection': ('connect', 'timed out'), 'budget': ('budget',), 'model_output': ('JSON', 'finding', 'evidence', 'fields')
    }.items() if any(term.casefold() in error.casefold() for term in terms)), 'other' if error else None)
    return {**totals, 'run_id': state.get('run_id'), 'reported_total_tokens': sum(item['total_tokens'] for item in records if type(item.get('total_tokens')) is int),
        'unknown_usage_requests': max(0, state.get('model_calls', 0) - len(records)), 'model_requests': state.get('model_calls', 0), 'tool_executions': state.get('tool_calls', 0),
        'elapsed_seconds': state.get('elapsed_seconds', 0), 'failure_reason': failure,
        'format_reasks': state.get('format_repairs', 0), 'revision_status': state.get('revision_status', 'not_requested'),
        'estimated_list_price_usd': round(cost, 8) if priced and len(records) == state.get('model_calls', 0) else None,
        'cost_note': 'Configured list-price estimate, not actual billing; missing rates/usage remain unknown.'}


def json_response(payload: dict, status_code: int = 200) -> JSONResponse:
    return JSONResponse(payload, status_code=status_code, headers=API_HEADERS)


@app.exception_handler(RunError)
@app.exception_handler(PDFError)
async def validation_error(request: Request, error: ValueError):
    return json_response({'error': str(error)}, 400)


@app.exception_handler(RequestValidationError)
async def form_error(request: Request, error: RequestValidationError):
    # Never echo validation inputs: they can contain document text or secrets.
    return json_response({'error': 'Choose exactly one PDF and enter a nonempty question '
                          'of at most 2,000 characters.'}, 400)


@app.exception_handler(HTTPException)
async def http_error(request: Request, error: HTTPException):
    messages = {400: 'The upload form could not be read.',
                404: 'Not found.', 405: 'This HTTP method is not supported.',
                413: 'The upload exceeds the allowed request size.'}
    return json_response({'error': messages.get(error.status_code, 'The request could not be processed.')},
                         error.status_code)


@app.get('/', include_in_schema=False)
@app.get('/styles.css', include_in_schema=False)
@app.get('/app.js', include_in_schema=False)
def public_file(request: Request):
    # Explicit paths only. No project-root mount or catch-all can shadow the API.
    filename, media_type = STATIC[request.url.path]
    return FileResponse(ROOT / 'public' / filename, media_type=media_type,
                        headers={'X-Content-Type-Options': 'nosniff'})


@app.get('/api/research')
def configuration():
    config = settings()
    return json_response(sanitize({'ready': bool(config['key']), 'model': config['model'],
        'max_upload_mb': config['max_upload_mb'], 'max_text_chars': config['max_text_chars'],
        'reviewer_model': config['reviewer_model'], 'auth_required': bool(config['access_token']),
        'retrieval': 'BM25', 'pdf_mode': config['pdf_mode']}, config['key']))


def ndjson_updates(data, filename, question, pages, config):
    # Closing this wrapper also closes the underlying graph/client generator.
    events = investigate(data, filename, question, pages, config)
    try:
        for event in events:
            yield (json.dumps(event, ensure_ascii=False) + '\n').encode()
    finally:
        events.close()


class InvestigationResponse(StreamingResponse):
    """Stream graph work off-loop and cancel the async provider await on disconnect."""

    def __init__(self, iterator, key=None):
        self.iterator = iterator
        self.key = key
        super().__init__(iterator, media_type='application/x-ndjson',
                         headers={**API_HEADERS, 'X-Accel-Buffering': 'no'})

    async def __call__(self, scope, receive, send):
        """Cancel the outgoing async HTTP request when the browser disconnects.

        LangGraph stays synchronous in worker threads. A small SDK-shaped adapter
        submits only the provider awaitable onto this ASGI loop. Cancelling a local
        HTTP request cannot guarantee Groq stops remote computation or quota use.
        """
        loop = asyncio.get_running_loop()
        stopped = threading.Event()
        active = set()
        key = self.key or settings()['key']
        async with groq.AsyncGroq(api_key=key, max_retries=0, timeout=45) as provider:
            async def create(**options):
                if stopped.is_set():
                    raise RunError('Investigation cancelled; no further model requests were started.')
                with anyio.CancelScope() as call_scope:
                    active.add(call_scope)
                    try:
                        return await provider.chat.completions.create(**options)
                    finally:
                        active.discard(call_scope)
                raise RunError('Investigation cancelled; the outgoing provider request was closed.')
            def sync_create(**options):
                return asyncio.run_coroutine_threadsafe(create(**options), loop).result()
            adapter = SimpleNamespace(api_key=key, chat=SimpleNamespace(
                completions=SimpleNamespace(create=sync_create)))
            token = HTTP_CLIENT.set(adapter)
            async def disconnected():
                while True:
                    message = await receive()
                    if message['type'] == 'http.disconnect':
                        logging.getLogger('uvicorn.error').info('research_request_disconnected')
                        stopped.set()
                        for call_scope in list(active):
                            call_scope.cancel()
                        return
            try:
                async with anyio.create_task_group() as group:
                    group.start_soon(disconnected)
                    try:
                        await send({'type': 'http.response.start', 'status': self.status_code, 'headers': self.raw_headers})
                        async for chunk in iterate_in_threadpool(self.iterator):
                            if stopped.is_set():
                                break
                            await send({'type': 'http.response.body', 'body': chunk, 'more_body': True})
                        if not stopped.is_set():
                            await send({'type': 'http.response.body', 'body': b'', 'more_body': False})
                    finally:
                        stopped.set()
                        for call_scope in list(active):
                            call_scope.cancel()
                        group.cancel_scope.cancel()
            finally:
                HTTP_CLIENT.reset(token)
                with anyio.CancelScope(shield=True):
                    await run_in_threadpool(self.iterator.close)


@app.post('/api/research')
async def start_investigation(request: Request, paper: Annotated[UploadFile, File()],
                              question: Annotated[str, Form()]):
    try:
        config = settings()
        token = config.get('access_token', '')
        provided = request.headers.get('authorization', '')
        if token and not hmac.compare_digest(provided.encode(), ('Bearer ' + token).encode()):
            return json_response({'error': 'Enter the research desk access token.'}, 401)
        if not config['key']:
            return json_response({'error': 'The server needs a Groq API key before investigations can start.'}, 503)
        form = await request.form()  # Already parsed/cached by FastAPI's File/Form handling.
        names = [name for name, _ in form.multi_items()]
        if sorted(name for name in names if name != 'revise_report') != ['paper', 'question'] or names.count('revise_report') > 1:
            raise RunError('Upload exactly one PDF and one question.')
        revise = form.get('revise_report', 'false')
        if revise not in ('true', 'false'):
            raise RunError('Revision choice must be true or false.')
        config['enable_revision'] = revise == 'true'
        question = question.strip()
        if not question or len(question) > 2000:
            raise RunError('Enter a question of at most 2,000 characters.')
        maximum = config['max_upload_mb'] * 1024 * 1024
        # This bounds the endpoint read, not network receipt or multipart spooling.
        data = await paper.read(maximum + 1)
        if len(data) > maximum:
            raise RunError('The PDF exceeds the configured upload limit (at most 4 MB).')
        filename = (paper.filename or 'paper.pdf').replace('\\', '/').split('/')[-1][:180]
        filename = ''.join(char for char in filename if char.isprintable()) or 'paper.pdf'
    finally:
        # Close every parsed upload, including rejected duplicate/extra files.
        # FastAPI also closes form resources on its validation-error paths.
        await request.close()
    pages = await run_in_threadpool(extract_pdf, data, config['max_upload_mb'], config['max_text_chars'], config['pdf_mode'])
    if not allow_request(request.client.host if request.client else 'unknown', config['rate_limit']):
        return json_response({'error': 'Too many investigations from this client address. Wait one minute.'}, 429)
    return InvestigationResponse(ndjson_updates(data, filename, question, pages, config), key=config['key'])


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=int(os.getenv('PORT', '8000')))
