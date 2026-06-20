"""Concrete OCR vendor adapters (page image -> transcription text).

`invoke(item)` reads `item['image']` (an absolute path to a page image) and
returns the bench-adapters result dict: `raw_output`/`text`, `latency_ms`,
`cost_usd`, and `mode` (`prompted_vlm` | `native_ocr`); or `{"non_attempt": ...}`
for an uncharged refusal/truncation. API keys are read from the environment via
`_env` (never hardcoded); a vendor whose key/binary is unset raises
`VendorUnavailable` so the harness skips it cleanly.

Two shapes:
  * prompted VLMs (claude/gpt/gemini) subclass `_VisionOCRAdapter` — they share
    image encoding, the truncation-retry loop, refusal mapping, cost, and result
    assembly, implementing only `_transcribe(...)` per vendor;
  * native OCR engines (tesseract/mistral/deepseek) are small standalone invokes.

Heavy SDKs (pytesseract, anthropic) are imported lazily inside `invoke`/`_transcribe`
so importing this module never hard-fails when an optional engine is absent.
"""
from __future__ import annotations

import base64
import io
import os
import time
from dataclasses import dataclass
from typing import Any

import httpx

from bench_adapters.ocr import pricing
from bench_adapters.registry import Adapter, register

# The single, version-pinned transcription instruction shared by every PROMPTED
# vision-LLM adapter. Pinning one prompt across vendors removes prompt phrasing as
# a confound; it is recorded in the run manifest. Native-OCR engines ignore it.
OCR_PROMPT = (
    "Transcribe all text in this image exactly as it appears, preserving reading "
    "order and line breaks. Output only the transcription — no commentary, no "
    "preamble, and no markdown code fences."
)

DEFAULT_MAX_EDGE = 1568  # downscale long edge before sending (Sonnet 4.6 caps ~1568px)
_MAX_TOKEN_TRIES = (8192, 16384)  # truncation guard: retry larger once, then non_attempt


class VendorUnavailable(Exception):
    """Raised when a vendor cannot be queried (missing key, missing binary, ...)."""


class RateLimited(Exception):
    """Transient failure (429 / 5xx / timeout) that persisted past retries.

    The runner must NOT checkpoint a page that raised this — it stays *undone* so a
    later resume retries it, rather than being permanently skipped. Distinct from a
    terminal miss (refused / truncated / empty), which IS recorded.
    """


def _env(*names: str) -> str:
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return ""


def _need(value: str, vendor: str) -> str:
    if not value:
        raise VendorUnavailable(f"{vendor} key unset")
    return value


def _load_image(path: str, *, max_edge: int | None = None):
    """Open an image as RGB, optionally downscaling the long edge. Lazy PIL import."""
    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover
        raise VendorUnavailable(f"Pillow unavailable: {exc}") from exc
    if not path or not os.path.exists(path):
        raise VendorUnavailable(f"image not found: {path!r}")
    img = Image.open(path).convert("RGB")
    if max_edge:
        w, h = img.size
        scale = max_edge / max(w, h)
        if scale < 1.0:
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))
    return img


def _image_b64(path: str, *, max_edge: int = DEFAULT_MAX_EDGE) -> tuple[str, str]:
    """Return (base64 PNG, mime) for a downscaled page image."""
    img = _load_image(path, max_edge=max_edge)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii"), "image/png"


def _post(url: str, body: dict[str, Any], headers: dict[str, str], timeout: float = 120.0,
          max_retries: int = 4) -> dict:
    """POST JSON with Retry-After-aware backoff on 429/5xx/timeout.

    Raises `RateLimited` once retries are exhausted so the caller leaves the page
    undone (resumable) rather than recording a bogus miss.
    """
    delay = 2.0
    for attempt in range(max_retries + 1):
        try:
            r = httpx.post(url, json=body, headers=headers, timeout=timeout)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            if attempt == max_retries:
                raise RateLimited(f"transport error after {max_retries} retries: {exc}") from exc
            time.sleep(delay)
            delay = min(delay * 2, 30.0)
            continue
        if r.status_code == 429 or r.status_code >= 500:
            if attempt == max_retries:
                raise RateLimited(f"HTTP {r.status_code} after {max_retries} retries")
            ra = r.headers.get("retry-after", "")
            wait = float(ra) if ra.replace(".", "", 1).isdigit() else delay
            time.sleep(min(wait, 60.0))
            delay = min(delay * 2, 30.0)
            continue
        r.raise_for_status()
        return r.json()
    raise RateLimited("unreachable")  # pragma: no cover


def _ok(text: str, latency_ms: float, cost: float, mode: str) -> dict[str, Any]:
    return {"raw_output": text, "text": text, "latency_ms": latency_ms,
            "cost_usd": cost, "mode": mode}


def _tesseract_version() -> str:
    try:
        import pytesseract
        return f"tesseract-{pytesseract.get_tesseract_version()}"
    except Exception:
        return "tesseract"


@register("tesseract")
class TesseractAdapter(Adapter):
    """Local Tesseract — the deterministic, free regression sentinel (no API key).

    Expected to be stable and to LOSE most slices; a drift in its anchor-set
    pass-rate (CUSUM) flags harness drift vs vendor drift. Requires the system
    `tesseract` binary (`brew install tesseract`) + `pytesseract`.
    """

    vendor = "tesseract"
    model_version = _tesseract_version()
    is_sentinel = True
    mode = "native_ocr"

    def invoke(self, item: dict[str, Any]) -> dict[str, Any]:
        try:
            import pytesseract
        except Exception as exc:
            raise VendorUnavailable(f"pytesseract unavailable: {exc}") from exc
        img = _load_image(str(item.get("image", "")))  # full res for tesseract
        t0 = time.monotonic()
        try:
            text = pytesseract.image_to_string(img)
        except pytesseract.TesseractNotFoundError as exc:
            raise VendorUnavailable(f"tesseract binary not found: {exc}") from exc
        return _ok(text, (time.monotonic() - t0) * 1000.0, 0.0, self.mode)


# --- prompted vision-LLM adapters ----------------------------------------------
# Model ids are env-overridable so model drift never needs a code change; the
# resolved id is recorded in the run manifest (AdapterSpec.version).
@dataclass
class _VLM:
    """One vision-LLM transcription attempt."""

    text: str
    in_tok: int
    out_tok: int
    truncated: bool = False
    refused: bool = False


class _VisionOCRAdapter(Adapter):
    """Base for prompted vision-LLM OCR adapters.

    Subclasses set `vendor`, `model_version`, `env_names`, and implement
    `_transcribe(key, b64, mime, max_tokens) -> _VLM` (raising `RateLimited` on
    transient errors). This base owns the shared flow: validate key, encode image,
    run the truncation-retry loop, map refusals, price, and assemble the result.
    """

    is_sentinel = False
    mode = "prompted_vlm"
    env_names: tuple[str, ...] = ()

    def _transcribe(self, key: str, b64: str, mime: str, max_tokens: int) -> _VLM:
        raise NotImplementedError

    def invoke(self, item: dict[str, Any]) -> dict[str, Any]:
        key = _need(_env(*self.env_names), self.spec.name)
        b64, mime = _image_b64(str(item.get("image", "")))
        for max_tok in _MAX_TOKEN_TRIES:
            t0 = time.monotonic()
            r = self._transcribe(key, b64, mime, max_tok)
            latency = (time.monotonic() - t0) * 1000.0
            if r.refused:
                return {"non_attempt": "refused", "latency_ms": latency}
            if r.truncated:
                continue  # retry once with a larger cap
            cost = pricing.token_cost(self.model_version, r.in_tok, r.out_tok)
            return _ok(r.text, latency, cost, self.mode)
        return {"non_attempt": "truncated"}


@register("claude-sonnet-ocr")
class ClaudeOCR(_VisionOCRAdapter):
    """Anthropic Claude (vision) via the official `anthropic` SDK."""

    vendor = "anthropic"
    model_version = os.environ.get("CLAUDE_OCR_MODEL", "claude-sonnet-4-6")
    env_names = ("ANTHROPIC_API_KEY",)

    def _transcribe(self, key: str, b64: str, mime: str, max_tokens: int) -> _VLM:
        try:
            import anthropic
        except Exception as exc:
            raise VendorUnavailable(f"anthropic SDK unavailable: {exc}") from exc
        client = anthropic.Anthropic(api_key=key, max_retries=4)  # SDK backs off on 429/5xx
        transient = (anthropic.RateLimitError, anthropic.APITimeoutError,
                     anthropic.APIConnectionError, anthropic.InternalServerError)
        try:
            resp = client.messages.create(
                model=self.model_version, max_tokens=max_tokens,
                thinking={"type": "disabled"},
                messages=[{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": mime, "data": b64}},
                    {"type": "text", "text": OCR_PROMPT},
                ]}],
            )
        except transient as exc:  # leave page undone for resume
            raise RateLimited(f"anthropic transient: {exc!r}"[:200]) from exc
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        return _VLM(text, resp.usage.input_tokens, resp.usage.output_tokens,
                    truncated=resp.stop_reason == "max_tokens",
                    refused=resp.stop_reason == "refusal")


@register("gpt-ocr")
class GptOCR(_VisionOCRAdapter):
    """OpenAI GPT-4o vision via the chat-completions REST API."""

    vendor = "openai"
    model_version = os.environ.get("GPT_OCR_MODEL", "gpt-4o")
    env_names = ("OPENAI_API_KEY",)

    def _transcribe(self, key: str, b64: str, mime: str, max_tokens: int) -> _VLM:
        body = {"model": self.model_version, "max_tokens": max_tokens, "temperature": 0,
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": OCR_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                ]}]}
        d = _post("https://api.openai.com/v1/chat/completions", body,
                  {"Authorization": f"Bearer {key}"})
        ch = d["choices"][0]
        u = d.get("usage", {})
        return _VLM(ch["message"].get("content") or "",
                    u.get("prompt_tokens", 0), u.get("completion_tokens", 0),
                    truncated=ch.get("finish_reason") == "length")


@register("gemini-ocr")
class GeminiOCR(_VisionOCRAdapter):
    """Google Gemini vision via the generateContent REST API."""

    vendor = "google"
    model_version = os.environ.get("GEMINI_OCR_MODEL", "gemini-2.5-pro")
    env_names = ("GEMINI_API_KEY", "GOOGLE_API_KEY")

    def _transcribe(self, key: str, b64: str, mime: str, max_tokens: int) -> _VLM:
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"{self.model_version}:generateContent")
        body = {"contents": [{"parts": [
                    {"text": OCR_PROMPT},
                    {"inline_data": {"mime_type": mime, "data": b64}}]}],
                "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0}}
        d = _post(url, body, {"x-goog-api-key": key})
        cand = (d.get("candidates") or [{}])[0]
        text = "".join(p.get("text", "") for p in cand.get("content", {}).get("parts", []))
        um = d.get("usageMetadata", {})
        finish = cand.get("finishReason")
        return _VLM(text, um.get("promptTokenCount", 0), um.get("candidatesTokenCount", 0),
                    truncated=finish == "MAX_TOKENS",
                    refused=finish in ("SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST"))


# --- native dedicated-OCR adapters ---------------------------------------------
@register("mistral-ocr")
class MistralOCR(Adapter):
    """Mistral dedicated OCR API (document -> markdown) — native OCR, no prompt."""

    vendor = "mistral"
    model_version = os.environ.get("MISTRAL_OCR_MODEL", "mistral-ocr-latest")
    is_sentinel = False
    mode = "native_ocr"

    def invoke(self, item: dict[str, Any]) -> dict[str, Any]:
        key = _need(_env("MISTRAL_API_KEY"), "mistral-ocr")
        b64, mime = _image_b64(str(item.get("image", "")))
        body = {"model": self.model_version,
                "document": {"type": "image_url", "image_url": f"data:{mime};base64,{b64}"}}
        t0 = time.monotonic()
        d = _post("https://api.mistral.ai/v1/ocr", body, {"Authorization": f"Bearer {key}"})
        pages = d.get("pages", [])
        text = "\n\n".join(p.get("markdown", "") for p in pages)
        cost = pricing.page_cost(self.model_version, max(1, len(pages)))
        return _ok(text, (time.monotonic() - t0) * 1000.0, cost, self.mode)


@register("deepseek-ocr")
class DeepSeekOCR(Adapter):
    """DeepSeek-OCR — self-host only (no hosted vision API as of 2026-06).

    Set `DEEPSEEK_OCR_BASE_URL` to a vLLM/Ollama OpenAI-compatible endpoint to
    enable it; otherwise it is skipped (`VendorUnavailable`), uncharged.
    """

    vendor = "deepseek"
    model_version = os.environ.get("DEEPSEEK_OCR_MODEL", "deepseek-ocr")
    is_sentinel = False
    mode = "native_ocr"

    def invoke(self, item: dict[str, Any]) -> dict[str, Any]:
        base = _env("DEEPSEEK_OCR_BASE_URL")
        if not base:
            raise VendorUnavailable(
                "deepseek-ocr is self-host only; set DEEPSEEK_OCR_BASE_URL to a vLLM/Ollama "
                "OpenAI-compatible endpoint"
            )
        key = _env("DEEPSEEK_API_KEY") or "EMPTY"
        b64, mime = _image_b64(str(item.get("image", "")))
        body = {"model": self.model_version, "temperature": 0,
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": OCR_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                ]}]}
        t0 = time.monotonic()
        d = _post(base.rstrip("/") + "/chat/completions", body,
                  {"Authorization": f"Bearer {key}"}, timeout=180.0)
        text = d["choices"][0]["message"].get("content") or ""
        return _ok(text, (time.monotonic() - t0) * 1000.0, 0.0, self.mode)
