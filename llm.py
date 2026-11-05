"""One Groq client factory and one request helper; no implicit retries."""

import re

import groq


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


def make_client(api_key: str) -> groq.Groq:
    if not api_key.strip():
        raise RunError("GROQ_API_KEY is missing. Set it in your local .env or deployment environment.")
    return groq.Groq(api_key=api_key, max_retries=0, timeout=60.0)


def call_model(client, model: str, messages: list[dict], tool_definitions: list[dict],
               max_tokens: int, allow_tools: bool = True, timeout: float | None = None):
    """Return an SDK message, including tool calls. Every invocation is one request."""
    options = {}
    if timeout is not None:
        options['timeout'] = timeout
    if model in {"openai/gpt-oss-20b", "openai/gpt-oss-120b"}:
        options.update(reasoning_effort="low", include_reasoning=False)
    try:
        response = client.chat.completions.create(
            model=model, messages=messages, tools=tool_definitions,
            tool_choice="auto" if allow_tools else "none",
            max_completion_tokens=max_tokens, **options,
        )
    except groq.AuthenticationError:
        raise RunError("Groq rejected the credentials. Check the server-side GROQ_API_KEY.") from None
    except groq.RateLimitError:
        raise RunError("Groq quota/rate limit reached. This run is incomplete. Wait before "
                       "explicitly starting another run; no automatic retry was made.") from None
    except groq.APIConnectionError:
        raise RunError("Could not connect to Groq, or the request timed out. No retry was made.") from None
    except groq.APIStatusError:
        raise RunError("Groq rejected the request. Check GROQ_MODEL availability, tool-calling "
                       "support, permissions, input size, and provider status.") from None
    if not response.choices:
        raise RunError("Groq returned no choices.")
    choice = response.choices[0]
    if choice.finish_reason not in {"stop", "tool_calls"}:
        raise RunError("The model response was incomplete or truncated. Use a shorter question "
                       "or adjust GROQ_MAX_OUTPUT_TOKENS; no automatic retry was made.")
    return choice.message
