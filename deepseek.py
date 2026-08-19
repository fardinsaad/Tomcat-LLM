"""
deepseek.py -- DeepSeek-R1, served through OpenRouter (Novita).

`deepseek-reasoner` was retired on 2026-07-24. The suggested migration,
`deepseek-v4-flash`, is a different model, so we stay on R1 -- MIT-licensed and
still hosted -- to keep results comparable with the published study. The CSVs
are dated 2-3 May 2025, predating R1-0528, so this is the original R1.

Novita is the only live R1 host and serves fp8-quantised weights; DeepSeek's
own full-precision endpoint no longer exists. The provider is pinned with
fallbacks disabled so precision is held constant across every condition.
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

MODEL = "deepseek/deepseek-r1"
PROVIDER_ROUTING = {"order": ["novita"], "allow_fallbacks": False}

SYSTEM_PROMPT = (
    "You are an intelligent assistant helping a human in a collaborative game "
    "to collect a gem the human desires. Your task is to interpret the "
    "instruction provided by the human and generate an appropriate response "
    "enabling the human retrieve their desired gem."
)

# The published study used max_tokens=512, which on DeepSeek's own endpoint
# bounded the ANSWER only -- reasoning had a separate allowance. Here reasoning
# and answer share one budget, so a small cap truncates mid-reasoning and
# returns an empty answer at full cost. 16000 is Novita's ceiling for this
# model; the answer itself stays ~600 tokens, so the headroom is spent on
# reasoning rather than on longer output.
MAX_TOKENS = 16000
TEMPERATURE = 0.2          # matches the published study
VERBOSE = True             # per-call usage; set False for long sweeps

# Reasoning is left at the model's default. R1 always reasons -- there is no
# toggle for it -- and the trace is returned alongside the answer in a separate
# `reasoning` field, never mixed into `content`. Capturing it costs nothing and
# is useful for the qualitative analysis.
LAST_REASONING = {}        # prompt-index -> reasoning trace, if a caller wants it


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


@backoff.on_exception(backoff.expo, RETRYABLE, max_time=300, max_tries=6,
                      giveup=lambda e: not _is_transient(e))
def get_completion_with_backoff(**kwargs):
    return client.chat.completions.create(**kwargs)


def get_response_R1(prompt, max_tokens=MAX_TOKENS, temperature=TEMPERATURE):
    """Interact with DeepSeek-R1 and generate a response."""
    response = get_completion_with_backoff(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        stream=False,
        extra_body={"provider": PROVIDER_ROUTING},
    )

    choice = response.choices[0]
    LAST_REASONING["trace"] = getattr(choice.message, "reasoning", None)

    if VERBOSE:
        u = response.usage
        detail = getattr(u, "completion_tokens_details", None)
        reasoning_tokens = getattr(detail, "reasoning_tokens", None) if detail else None
        print(f"[R1] prompt={u.prompt_tokens} completion={u.completion_tokens} "
              f"reasoning={reasoning_tokens} finish={choice.finish_reason} "
              f"provider={getattr(response, 'provider', '?')}")

    if choice.finish_reason == "length" or not choice.message.content:
        raise RuntimeError(
            f"DeepSeek-R1 hit the {max_tokens}-token limit before producing a "
            f"final answer. This is Novita's ceiling for the model, so the "
            f"prompt itself needs shortening rather than the cap raising."
        )
    return choice.message.content.strip()


if __name__ == "__main__":
    print("OpenRouter API key loaded successfully.")
    print(f"model    : {MODEL}")
    print(f"provider : {PROVIDER_ROUTING['order'][0]} (fallbacks disabled)")

    reply = get_response_R1("Name a country that starts with K?")
    print("\nDeepSeek-R1 response:", reply)
