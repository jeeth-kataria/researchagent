"""One Groq client factory and one request helper; no implicit retries."""

import re
from contextvars import ContextVar
from contextlib import nullcontext

import groq


HTTP_CLIENT = ContextVar('request_groq_client', default=None)


class RunError(ValueError):
    """A safe, user-facing reason why a run could not finish."""


def sanitize(value, secret: str = ""):
    """Redact credentials in views/events without printing raw provider messages."""
    if isinstance(value, str):
        if secret:
            value = value.replace(secret, "[REDACTED]")
        return re.sub(r"gsk_[A-Za-z0-9_-]+", "[REDACTED]", value)
    if isinstance(value, list):
        return [sanitize(item, secret) for item in value]
    if isinstance(value, dict):
        return {key: sanitize(item, secret) for key, item in value.items()}
    return value


def make_client(api_key: str, http_client=None) -> groq.Groq:
    if not api_key.strip():
        raise RunError("GROQ_API_KEY is missing. Set it in your local .env or deployment environment.")
    options = {} if http_client is None else {'http_client': http_client}
    override = HTTP_CLIENT.get()
    if override is not None and http_client is None:
        return nullcontext(override)
    return groq.Groq(api_key=api_key, max_retries=0, timeout=60.0, **options)


def call_model(client, model: str, messages: list[dict], tool_definitions: list[dict],
               max_tokens: int, allow_tools: bool = True, timeout: float | None = None, usage_log: list | None = None, on_response=None):
    """Return an SDK message, including tool calls. Every invocation is one request."""
    options = {}
    if timeout is not None:
        options['timeout'] = timeout
    if model in {"openai/gpt-oss-20b", "openai/gpt-oss-120b"}:
        options.update(reasoning_effort="low", include_reasoning=False)
    if tool_definitions:
        options.update(tools=tool_definitions, tool_choice="auto" if allow_tools else "none")
    try:
        response = client.chat.completions.create(
            model=model, messages=messages,
            max_completion_tokens=max_tokens, **options,
        )
    except groq.AuthenticationError:
        raise RunError("Groq rejected the credentials. Check the server-side GROQ_API_KEY.") from None
    except groq.RateLimitError:
        raise RunError("Groq quota/rate limit reached. This run is incomplete. Wait before "
                       "explicitly starting another run; no automatic retry was made.") from None
    except groq.APIConnectionError:
        raise RunError("Could not connect to Groq, or the request timed out. No retry was made.") from None
    except groq.APIStatusError as error:
        if error.status_code == 413:
            raise RunError("Groq rejected an oversized model request. The paper/context and "
                           "requested output may exceed this account's token limit. Use a shorter "
                           "paper or reduce the configured context/output limits. Waiting alone "
                           "does not shrink this request; no retry was made.") from None
        raise RunError("Groq rejected the request. Check GROQ_MODEL availability, tool-calling "
                       "support, permissions, input size, and provider status.") from None
    if on_response is not None:
        on_response()  # The provider responded even if final JSON/finish validation later fails.
    if usage_log is not None:
        usage = getattr(response, 'usage', None)
        usage_log.append({name: getattr(usage, name, None) for name in
                          ('prompt_tokens', 'completion_tokens', 'total_tokens')} | {'model': model})
    if not response.choices:
        raise RunError("Groq returned no choices.")
    choice = response.choices[0]
    if choice.finish_reason not in {"stop", "tool_calls"}:
        raise RunError("The model response was incomplete or truncated. Use a shorter question "
                       "or adjust GROQ_MAX_OUTPUT_TOKENS; no automatic retry was made.")
    return choice.message
