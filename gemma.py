"""
gemma.py -- Gemma-3-27B, served through OpenRouter (DeepInfra).

Google deprecated `gemma-3-27b-it` on AI Studio, so the original `google.genai`
call path no longer works. Gemma 3 is open-weight and still hosted, so we reach
the same model rather than substituting a newer one.

Most OpenRouter hosts serve fp8-quantised weights, DeepInfra included; Google's
full-precision endpoint is gone. The provider is pinned with fallbacks disabled
so precision is held constant across every condition.
"""


import os

import backoff
from dotenv import find_dotenv, load_dotenv
from openai import OpenAI, RateLimitError, APIStatusError, APIConnectionError

load_dotenv(find_dotenv(usecwd=True), override=True)

key = os.getenv("OPENROUTER_API_KEY")
if not key:
    raise ValueError("OPENROUTER_API_KEY not found in .env file")

client = OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1")

MODEL = "google/gemma-3-27b-it"

# Exactly one provider, no fallbacks. Do not add a `quantizations` filter:
# combined with a single-entry `order` it can empty the candidate set, which
# OpenRouter reports as a bare 404 rather than a filter error.
PROVIDER_ROUTING = {
    "order": ["deepinfra"],
    "allow_fallbacks": False,
}


# Retry only on transient failures. NOT bare APIStatusError -- that also covers
# 400 (bad request), 402 (out of credit) and 404, none of which get better by
# retrying. Provider pinning (allow_fallbacks=False) means OpenRouter cannot
# route around an upstream limit for us, so we absorb it here instead.
def _is_transient(exc):
    if isinstance(exc, (RateLimitError, APIConnectionError)):
        return True
    code = getattr(exc, "status_code", None)
    return code in (408, 409, 425, 429, 500, 502, 503, 504, 529)


RETRYABLE = (RateLimitError, APIStatusError, APIConnectionError)


@backoff.on_exception(backoff.expo, RETRYABLE, max_time=900, max_tries=8,
                      giveup=lambda e: not _is_transient(e))
def get_completion_with_backoff(**kwargs):
    """Get a completion from OpenRouter with backoff on rate limits."""
    return client.chat.completions.create(**kwargs)


def get_response_from_gemini(prompt, max_tokens=512, temperature=0.2):
    """
    Interact with Gemma-3-27B and generate a response.

    Name kept as-is: agent.py imports `get_response_from_gemini`, so the
    published pipeline keeps working unchanged.

    max_tokens=512 and temperature=0.2 match the published study. Gemma is not
    a reasoning model, so there is no hidden reasoning budget competing for
    these tokens.
    """
    response = get_completion_with_backoff(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
        extra_body={"provider": PROVIDER_ROUTING},
    )
    # OpenRouter reports which backend actually served the call. Worth checking
    # on the first run that it matches the pin.
    served_by = getattr(response, "provider", None)
    if served_by and served_by.lower() not in (
        p.lower() for p in PROVIDER_ROUTING["order"]
    ):
        print(f"[gemma] WARNING: served by {served_by!r}, not the pinned provider")
    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    print("OpenRouter API key loaded successfully.")
    print(f"model    : {MODEL}")
    print(f"provider : {PROVIDER_ROUTING['order'][0]} (fallbacks disabled)")

    reply = get_response_from_gemini("Name three countries that start with K?")
    print("\nGemma response:", reply)
