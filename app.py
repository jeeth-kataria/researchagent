"""FastAPI HTTP boundary; the synchronous paper workflow is unchanged.

Run `python app.py` or `uvicorn app:app --reload --host 127.0.0.1 --port 8000`.
Vercel imports the same ASGI app through api/research.py.
"""

import hashlib
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
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException

from agents import Limits
from llm import RunError, make_client, sanitize
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
    model = os.getenv('GROQ_MODEL', 'openai/gpt-oss-20b').strip()
    if not model:
        raise RunError('GROQ_MODEL must not be empty.')
    return {
        'key': os.getenv('GROQ_API_KEY', '').strip(),
        'model': model,
        # Four MiB plus form fields fits below Vercel's 4.5 MB body ceiling.
        'max_upload_mb': min(number('MAX_UPLOAD_MB', 4), 4),
        'max_text_chars': number('MAX_TEXT_CHARS', 20_000),
        'limits': Limits(model_per_visit=number('MAX_MODEL_REQUESTS_PER_VISIT', 3),
                         tools_per_visit=number('MAX_TOOL_CALLS_PER_VISIT', 2),
                         total_model_calls=number('MAX_TOTAL_MODEL_REQUESTS', 12),
                         output_tokens=number('GROQ_MAX_OUTPUT_TOKENS', 2048),
                         source_chars=number('MAX_SOURCE_CHARS_PER_ROLE', 4800)),
    }


def public_state(state: dict, key: str) -> dict:
    """Pages/registry are sent once; events are streamed separately, never reasoning."""
    return sanitize({**{name: value for name, value in state.items()
                     if name not in {'pages', 'evidence', 'events', 'deadline', 'requests', 'started_at'}},
                     'coverage_summary': coverage_summary(state)}, key)


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
        # Closing the SDK client can fail after the final state was streamed.
        # Cleanup must not erase an already completed draft and critical review.
        if state['status'] != 'complete':
            state['status'] = 'incomplete'
            state['error'] = 'The investigation could not finish. Available findings are preserved; no retry was made.'
            state['final_output'] = ''
            yield {'type': 'state', 'state': public_state(state, config['key'])}
    yield {'type': 'finished', 'status': state['status']}


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
        'max_upload_mb': config['max_upload_mb'], 'max_text_chars': config['max_text_chars']}, config['key']))


def ndjson_updates(data, filename, question, pages, config):
    # Closing this wrapper also closes the underlying graph/client generator.
    events = investigate(data, filename, question, pages, config)
    try:
        for event in events:
            yield (json.dumps(event, ensure_ascii=False) + '\n').encode()
    finally:
        events.close()


class InvestigationResponse(StreamingResponse):
    """Stream a sync iterator in Starlette's pool and close it on disconnect."""

    def __init__(self, iterator):
        self.iterator = iterator
        super().__init__(iterator, media_type='application/x-ndjson',
                         headers={**API_HEADERS, 'X-Accel-Buffering': 'no'})

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            # Cancellation does not instantly interrupt a synchronous provider call.
            # Wait for that worker step before closing; never start another run.
            with anyio.CancelScope(shield=True):
                await run_in_threadpool(self.iterator.close)


@app.post('/api/research')
async def start_investigation(request: Request, paper: Annotated[UploadFile, File()],
                              question: Annotated[str, Form()]):
    try:
        config = settings()
        if not config['key']:
            return json_response({'error': 'The server needs a Groq API key before investigations can start.'}, 503)
        form = await request.form()  # Already parsed/cached by FastAPI's File/Form handling.
        if sorted(name for name, _ in form.multi_items()) != ['paper', 'question']:
            raise RunError('Upload exactly one PDF and one question.')
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
    pages = await run_in_threadpool(extract_pdf, data, config['max_upload_mb'], config['max_text_chars'])
    return InvestigationResponse(ndjson_updates(data, filename, question, pages, config))


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=int(os.getenv('PORT', '8000')))
