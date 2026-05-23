"""Smoke tests for start-frame parsing and the ref-audio cache.

These tests exercise the pure-Python validation layer — no model, no GPU, no
network. They're meant to catch wire-protocol regressions cheaply.
"""
from __future__ import annotations

import base64
import hashlib
import io
import wave

import pytest

from qwen3_tts_streaming.server import (
    MAX_TEXT_CHARS,
    _BadRequest,
    _RefAudioCache,
    _RefNotCached,
    _parse_start_frame,
)


def _make_wav(seconds: float = 0.1, rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        n = int(seconds * rate)
        w.writeframes(b"\x00\x00" * n)
    return buf.getvalue()


def _start(text: str, **extra) -> str:
    import json

    payload = {"type": "start", "text": text}
    payload.update(extra)
    return json.dumps(payload)


@pytest.fixture
def cache() -> _RefAudioCache:
    return _RefAudioCache(capacity=8)


# ---- happy paths -----------------------------------------------------------

def test_full_ref_bytes_roundtrip(cache):
    wav = _make_wav()
    b64 = base64.b64encode(wav).decode("ascii")
    text, path, ref_text, lang = _parse_start_frame(
        _start("Hello there.", ref_audio_b64=b64, ref_text="ref", language="English"),
        cache,
    )
    assert text == "Hello there."
    assert ref_text == "ref"
    assert lang == "English"
    assert path.endswith(".wav")
    # Cache is now populated; the same hash should resolve without bytes.
    h = hashlib.sha256(wav).hexdigest()
    cached = cache.get(h)
    assert cached is not None
    assert cached[0] == path


def test_hash_only_resume_after_cache_warm(cache):
    wav = _make_wav()
    b64 = base64.b64encode(wav).decode("ascii")
    # Warm the cache.
    _, path1, _, _ = _parse_start_frame(
        _start("Chunk one.", ref_audio_b64=b64, ref_text="ref"), cache
    )
    h = hashlib.sha256(wav).hexdigest()
    # Subsequent chunk sends hash only.
    _, path2, ref_text, _ = _parse_start_frame(
        _start("Chunk two.", ref_audio_sha256=h), cache
    )
    assert path2 == path1, "hash hit must reuse the original temp file path"
    assert ref_text == "ref", "ref_text should fall back to cached value when omitted"


def test_language_default_is_auto(cache):
    wav = _make_wav()
    b64 = base64.b64encode(wav).decode("ascii")
    _, _, _, lang = _parse_start_frame(_start("Hi.", ref_audio_b64=b64), cache)
    assert lang == "Auto"


def test_unknown_language_passes_through(cache):
    wav = _make_wav()
    b64 = base64.b64encode(wav).decode("ascii")
    _, _, _, lang = _parse_start_frame(
        _start("Hi.", ref_audio_b64=b64, language="Klingon"), cache,
    )
    assert lang == "Klingon"


# ---- bad_request paths ----------------------------------------------------

def test_missing_text(cache):
    wav = _make_wav()
    b64 = base64.b64encode(wav).decode("ascii")
    import json
    with pytest.raises(_BadRequest, match="text is required"):
        _parse_start_frame(
            json.dumps({"type": "start", "ref_audio_b64": b64}), cache,
        )


def test_empty_text(cache):
    wav = _make_wav()
    b64 = base64.b64encode(wav).decode("ascii")
    with pytest.raises(_BadRequest, match="non-empty"):
        _parse_start_frame(_start("   ", ref_audio_b64=b64), cache)


def test_text_too_long(cache):
    wav = _make_wav()
    b64 = base64.b64encode(wav).decode("ascii")
    too_long = "a" * (MAX_TEXT_CHARS + 1)
    with pytest.raises(_BadRequest, match="too long"):
        _parse_start_frame(_start(too_long, ref_audio_b64=b64), cache)


def test_missing_ref_audio_and_hash(cache):
    with pytest.raises(_BadRequest, match="requires ref_audio_b64"):
        _parse_start_frame(_start("Hi."), cache)


def test_invalid_base64(cache):
    with pytest.raises(_BadRequest, match="base64 decode failed"):
        _parse_start_frame(_start("Hi.", ref_audio_b64="!!! not valid base64 !!!"), cache)


def test_unparseable_audio(cache):
    # Random bytes are neither WAV nor any other format soundfile recognizes.
    b64 = base64.b64encode(b"not an audio file at all xyz123").decode("ascii")
    with pytest.raises(_BadRequest, match="unsupported audio format"):
        _parse_start_frame(_start("Hi.", ref_audio_b64=b64), cache)


def test_hash_mismatch_with_bytes(cache):
    wav = _make_wav()
    b64 = base64.b64encode(wav).decode("ascii")
    wrong_hash = "0" * 64
    with pytest.raises(_BadRequest, match="does not match"):
        _parse_start_frame(
            _start("Hi.", ref_audio_b64=b64, ref_audio_sha256=wrong_hash), cache,
        )


def test_bad_hash_format(cache):
    with pytest.raises(_BadRequest, match="64-char hex"):
        _parse_start_frame(_start("Hi.", ref_audio_sha256="too-short"), cache)


def test_non_object_frame(cache):
    with pytest.raises(_BadRequest, match="must be a JSON object"):
        _parse_start_frame('"just a string"', cache)


def test_wrong_type_field(cache):
    wav = _make_wav()
    b64 = base64.b64encode(wav).decode("ascii")
    import json
    with pytest.raises(_BadRequest, match="must be \\{'type':'start'"):
        _parse_start_frame(
            json.dumps({"type": "synthesize", "text": "Hi.", "ref_audio_b64": b64}), cache,
        )


# ---- ref_not_cached -------------------------------------------------------

def test_unknown_hash_raises_ref_not_cached(cache):
    h = "a" * 64
    with pytest.raises(_RefNotCached):
        _parse_start_frame(_start("Hi.", ref_audio_sha256=h), cache)


# ---- cache LRU eviction ---------------------------------------------------

def test_lru_eviction_drops_oldest():
    cache = _RefAudioCache(capacity=2)
    wav1 = _make_wav(seconds=0.1)
    wav2 = _make_wav(seconds=0.2)
    wav3 = _make_wav(seconds=0.3)
    h1 = cache.put(hashlib.sha256(wav1).hexdigest(), wav1, "r1")
    h2 = cache.put(hashlib.sha256(wav2).hexdigest(), wav2, "r2")
    # Touch h1 so h2 becomes the LRU.
    cache.get(hashlib.sha256(wav1).hexdigest())
    cache.put(hashlib.sha256(wav3).hexdigest(), wav3, "r3")
    assert cache.get(hashlib.sha256(wav1).hexdigest()) is not None
    assert cache.get(hashlib.sha256(wav2).hexdigest()) is None, "wav2 should have been evicted"
    assert cache.get(hashlib.sha256(wav3).hexdigest()) is not None
