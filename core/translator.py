"""
core/translator.py
==================

Element-wise translation helper used by ``TranslationEngine`` in
``document_pipeline.py``. The 4B-class models we run on are sensitive to
context bleed-through, so every paragraph and every table cell goes
through its own self-contained API call.

Backends
--------
Selected by the ``LLM_BACKEND`` environment variable (defaults to
``geniex`` — see below). Switching backends is one env var — no code
edits — so you can flip between cloud and local inference without
redeploying.

    geniex     Local GenieX server (`geniex serve <model> --host ...`).
               This is the persistent HTTP server — NOT `geniex infer`,
               which reloads the model from scratch on every call (~12s
               of "loading model..." per call, unusable for per-block
               translation of a whole document).
               Env: GENIEX_URL   (default ``http://127.0.0.1:18183``)
                    GENIEX_MODEL (default ``qualcomm/Qwen3-4B-Instruct-2507``)

    openai     OpenAI Chat Completions API.
               Env: OPENAI_API_KEY, OPENAI_MODEL (default ``gpt-4o-mini``),
                    OPENAI_BASE_URL (optional; lets you point at a
                    compatible local server like vllm or LM Studio).

    ollama     Ollama HTTP API (fully local, free, offline).
               Env: OLLAMA_URL    (default ``http://localhost:11434``)
                    OLLAMA_MODEL  (default ``qwen2.5:7b-instruct``)

    nexa       Original Nexa local server. Currently disabled by default
               because its SDK is throwing ``Invalid license`` — kept here
               so a fix on the Nexa side is one env-var switch away.
               Env: NEXA_API_URL  (default ``http://127.0.0.1:18181/v1/chat/completions``)
                    NEXA_MODEL    (default ``NexaAI/Qwen3-4B-Instruct-2507-npu``)

API
---
The public surface (``translate_text``, ``translate_batch``, ``_call_llm``)
is unchanged so ``document_pipeline.py`` and any other caller keep
working without edits.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Callable

import requests

_log = logging.getLogger(__name__)


# ── Load .env if python-dotenv is installed (requirements.txt has it) ────────

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


# ── Translation direction map ────────────────────────────────────────────────

TRANSLATION_DIRECTIONS = {
    "vi->en": ("Vietnamese", "English"),
    "en->vi": ("English",    "Vietnamese"),
}


# ── Backend implementations ──────────────────────────────────────────────────
#
# Each backend is a callable with the same signature:
#
#     def call(prompt: str, system: str, temperature: float) -> str
#
# Returning ``""`` on failure means the caller falls back to the original
# (un-translated) text instead of crashing the pipeline.

def _call_openai(prompt: str, system: str, temperature: float) -> str:
    api_key  = os.environ.get("OPENAI_API_KEY", "").strip()
    base_url = os.environ.get("OPENAI_BASE_URL", "").strip() or None
    model    = os.environ.get("OPENAI_MODEL", "gpt-4o-mini").strip()
    if not api_key:
        _log.error(
            "LLM_BACKEND=openai but OPENAI_API_KEY is not set. "
            "Add it to .env or your shell environment."
        )
        return ""
    try:
        # Lazy import so the openai dep is only required when this
        # backend is actually selected.
        from openai import OpenAI
    except ImportError:
        _log.error(
            "openai package is missing. Install with: pip install openai"
        )
        return ""

    client = OpenAI(api_key=api_key, base_url=base_url) if base_url \
             else OpenAI(api_key=api_key)
    try:
        resp = client.chat.completions.create(
            model       = model,
            messages    = [
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt},
            ],
            temperature = temperature,
            timeout     = 60,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        _log.error(f"OpenAI API error: {e}")
        return ""


def _call_ollama(prompt: str, system: str, temperature: float) -> str:
    base_url = os.environ.get("OLLAMA_URL",   "http://localhost:11434").rstrip("/")
    model    = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b-instruct")
    payload = {
        "model":   model,
        "stream":  False,
        "options": {"temperature": temperature},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ],
    }
    try:
        resp = requests.post(f"{base_url}/api/chat", json=payload, timeout=180)
        resp.raise_for_status()
        data = resp.json()
        # Ollama returns {"message": {"role": ..., "content": "..."}, ...}
        return (data.get("message", {}).get("content") or "").strip()
    except Exception as e:
        _log.error(f"Ollama API error ({base_url}, model={model}): {e}")
        return ""


def _call_nexa(prompt: str, system: str, temperature: float) -> str:
    """Kept for compatibility. Currently broken upstream
    (SDKError: Invalid license). Flip ``LLM_BACKEND=nexa`` to use it
    once the license issue is resolved."""
    url   = os.environ.get(
        "NEXA_API_URL", "http://127.0.0.1:18181/v1/chat/completions",
    )
    model = os.environ.get(
        "NEXA_MODEL",   "NexaAI/Qwen3-4B-Instruct-2507-npu",
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ],
        "temperature": temperature,
    }
    try:
        resp = requests.post(url, json=payload, timeout=300)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        _log.error(f"Nexa API error: {e}")
        return ""


# ══════════════════════════════════════════════════════════════════════════
# GenieX — persistent local server
# ══════════════════════════════════════════════════════════════════════════
#
# IMPORTANT: this talks to `geniex serve <model> --host ...`, the
# long-running HTTP server, NOT `geniex infer`. `infer` re-loads the
# whole model from disk/NPU on every single invocation (~12s of
# "loading model..." spam per call in the probe log) — calling that
# once per paragraph/cell would make a multi-page document take hours.
# `serve` loads the model once and answers requests over HTTP, which is
# what we want for per-block translation.
#
# geniex's HTTP surface isn't fully known from the probe alone (only
# the swagger UI/docs endpoints were seen, not the actual chat route).
# Rather than hardcode one guess and silently produce empty
# translations if it's wrong, we try the shapes most local-inference
# servers use, in order, and log clearly which one worked (once) or
# that all of them failed (with the swagger URL to check by hand).
#
#   1. OpenAI-compatible:  POST {base}/v1/chat/completions
#        {"model", "messages":[{"role","content"}, ...], "temperature"}
#        -> data["choices"][0]["message"]["content"]
#   2. Ollama-style:       POST {base}/api/chat
#        {"model", "messages":[...], "options": {"temperature"}}
#        -> data["message"]["content"]
#   3. Plain generate:     POST {base}/generate
#        {"prompt", "system", "temperature"}
#        -> data["text"] or data["response"] or data["output"]
#
# If your actual geniex swagger (http://<host>/docs/swagger.yaml) uses a
# different path/shape, add it as a 4th candidate in _GENIEX_ATTEMPTS —
# everything else in this file is unaffected.

_GENIEX_LOGGED_WORKING_ROUTE: str | None = None


def _geniex_try_openai_style(base_url: str, model: str, prompt: str,
                              system: str, temperature: float):
    url = f"{base_url}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ],
        "temperature": temperature,
    }
    resp = requests.post(url, json=payload, timeout=180)
    resp.raise_for_status()
    data = resp.json()
    return url, (data["choices"][0]["message"]["content"] or "").strip()


def _geniex_try_ollama_style(base_url: str, model: str, prompt: str,
                             system: str, temperature: float):
    url = f"{base_url}/api/chat"
    payload = {
        "model":  model,
        "stream": False,
        "options": {"temperature": temperature},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ],
    }
    resp = requests.post(url, json=payload, timeout=180)
    resp.raise_for_status()
    data = resp.json()
    return url, (data.get("message", {}).get("content") or "").strip()


def _geniex_try_generate_style(base_url: str, model: str, prompt: str,
                               system: str, temperature: float):
    url = f"{base_url}/generate"
    payload = {
        "model":       model,
        "prompt":      prompt,
        "system":      system,
        "temperature": temperature,
    }
    resp = requests.post(url, json=payload, timeout=180)
    resp.raise_for_status()
    data = resp.json()
    text = data.get("text") or data.get("response") or data.get("output") or ""
    return url, text.strip()


_GENIEX_ATTEMPTS = (
    _geniex_try_openai_style,
    _geniex_try_ollama_style,
    _geniex_try_generate_style,
)


def _call_geniex(prompt: str, system: str, temperature: float) -> str:
    global _GENIEX_LOGGED_WORKING_ROUTE

    base_url = os.environ.get("GENIEX_URL", "http://127.0.0.1:18183").rstrip("/")
    model    = os.environ.get("GENIEX_MODEL", "qualcomm/Qwen3-4B-Instruct-2507")

    # Once we've confirmed a working route this run, don't keep probing
    # the others on every call — jump straight to what worked.
    attempts = _GENIEX_ATTEMPTS
    if _GENIEX_LOGGED_WORKING_ROUTE == "openai":
        attempts = (_geniex_try_openai_style,)
    elif _GENIEX_LOGGED_WORKING_ROUTE == "ollama":
        attempts = (_geniex_try_ollama_style,)
    elif _GENIEX_LOGGED_WORKING_ROUTE == "generate":
        attempts = (_geniex_try_generate_style,)

    errors = []
    for fn in attempts:
        try:
            url, text = fn(base_url, model, prompt, system, temperature)
        except Exception as e:
            errors.append(f"{fn.__name__}: {e}")
            continue
        if text:
            if _GENIEX_LOGGED_WORKING_ROUTE is None:
                tag = {
                    _geniex_try_openai_style:   "openai",
                    _geniex_try_ollama_style:   "ollama",
                    _geniex_try_generate_style: "generate",
                }[fn]
                _GENIEX_LOGGED_WORKING_ROUTE = tag
                _log.info(f"GenieX: using '{tag}'-style route at {url}")
            return text

    _log.error(
        f"GenieX: all known route shapes failed against {base_url} "
        f"(model={model}). Tried: {'; '.join(errors) if errors else 'no responses'}. "
        f"Check the actual API at {base_url}/docs/swagger.yaml and add a "
        "matching _geniex_try_*_style() function in translator.py if none "
        "of the built-in shapes match."
    )
    return ""


_BACKENDS: dict[str, Callable[[str, str, float], str]] = {
    "geniex": _call_geniex,
    "openai": _call_openai,
    "ollama": _call_ollama,
    "nexa":   _call_nexa,
}


def _selected_backend() -> tuple[str, Callable[[str, str, float], str]]:
    # Default is now geniex — the local persistent server you're
    # running via `geniex serve`. Override with LLM_BACKEND=openai/
    # ollama/nexa if you need one of the others.
    name = os.environ.get("LLM_BACKEND", "geniex").lower().strip()
    if name not in _BACKENDS:
        _log.warning(
            f"Unknown LLM_BACKEND={name!r}; falling back to 'geniex'. "
            f"Valid options: {sorted(_BACKENDS.keys())}"
        )
        name = "geniex"
    return name, _BACKENDS[name]


# Cache the backend choice so we only log "using X" once per process.
_BACKEND_LOGGED = False


def _call_llm(prompt: str, system: str, temperature: float = 0.1) -> str:
    """Public entrypoint used by ``TranslationEngine`` and ``_fix_ocr``.

    Routes the call to whichever backend is selected by ``LLM_BACKEND``.
    Always returns a string — empty on failure so callers can fall back
    to the original untranslated text.
    """
    global _BACKEND_LOGGED
    name, fn = _selected_backend()
    if not _BACKEND_LOGGED:
        _log.info(f"LLM backend: {name}")
        _BACKEND_LOGGED = True
    return fn(prompt, system, temperature)


# ── Public translation API (unchanged signatures) ────────────────────────────

def translate_text(text: str, direction: str = "vi->en") -> str:
    if not text or not text.strip():
        return text

    src, tgt = TRANSLATION_DIRECTIONS.get(direction, ("Vietnamese", "English"))

    system = (
        f"You are a professional {src}-{tgt} translator. "
        "Translate the text accurately. "
        "Only output the translated text, without quotes, explanations, "
        "or the original text."
    )
    prompt = f"Translate this text to {tgt}:\n{text}"

    res = _call_llm(prompt, system)
    return res if res else text


def translate_batch(texts: list[str], direction: str = "vi->en") -> list[str]:
    """Per-element translation. Kept as a loop so a 4B-class model
    doesn't lose track of the order on long inputs.
    """
    _log.info(f"Bắt đầu dịch {len(texts)} phân đoạn ({direction})…")
    results: list[str] = []
    t0 = time.time()
    for i, text in enumerate(texts):
        if i % 10 == 0 and i > 0:
            elapsed = time.time() - t0
            rate = i / max(elapsed, 0.001)
            _log.info(f"  → {i}/{len(texts)}  ({rate:.1f} items/s)")
        results.append(translate_text(text, direction=direction))
    return results