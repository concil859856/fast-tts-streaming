"""Voice-clone streaming WebSocket server for faster-qwen3-tts.

Implements the spec at vocence_website/dashboard-backend/docs/tts_server_spec.md:

  GET  /healthz                     -> JSON {status, model_id, sample_rate, inflight, cap, dev_stub}
  WS   /v1/voice-clone/stream       -> bearer-auth'd; one `start` JSON frame in, then
                                       40-ms PCM16LE @ 24 kHz mono binary frames out,
                                       terminated by `{"type":"end"}` (success) or
                                       `{"type":"error","code":...}` (failure).

Designed for one RTX 4090 running Qwen3-TTS-12Hz-1.7B-Base:
  * Single-GPU lock serializes the model; `cap` controls WS connection slots that
    queue politely above that.
  * True mid-generation cancel: WS close -> threading.Event -> generator.close()
    at the next codec-chunk boundary (~120 ms wall on chunk_size=8).
  * SHA-256 ref-audio cache: clients may send `ref_audio_sha256` instead of the
    full ~540 KB base64 payload on chunks 2..N of the same voice.
  * `inflight` decrement is structural via @asynccontextmanager — every close
    path (normal end, error, client close, RST) frees the slot exactly once.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import collections
import hashlib
import io
import json
import logging
import os
import sys
import tempfile
import threading
import time
import wave
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
from aiohttp import WSCloseCode, WSMsgType, web

from faster_qwen3_tts import FasterQwen3TTS

_log = logging.getLogger("qwen3_tts_streaming")

# --- Spec-mandated output format -------------------------------------------
SAMPLE_RATE = 24_000
FRAME_MS = 40
SAMPLES_PER_FRAME = SAMPLE_RATE * FRAME_MS // 1000   # 960
BYTES_PER_FRAME = SAMPLES_PER_FRAME * 2              # 1920 (int16 LE)

# --- Spec languages (unknown values pass through to the model) -------------
_KNOWN_LANGUAGES = {
    "Auto", "English", "Chinese", "Japanese", "Korean", "Spanish", "French",
    "German", "Portuguese", "Italian", "Russian", "Arabic",
}


# --- Tunable knobs (env-overridable) ---------------------------------------
def _env_int(key: str, default: int) -> int:
    v = os.environ.get(key)
    try:
        return int(v) if v is not None and v.strip() != "" else default
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    v = os.environ.get(key)
    try:
        return float(v) if v is not None and v.strip() != "" else default
    except ValueError:
        return default


# Default warmup ref audio ships with the repo at samples/joker.{wav,txt}.
# This lets the server boot with hot CUDA graphs out of the box — no .env
# fiddling required. Override via --warmup-ref-audio / QWEN3_TTS_WARMUP_REF_AUDIO
# only if you want a different reference voice for the warmup pass.
_SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"
DEFAULT_WARMUP_REF_AUDIO = str(_SAMPLES_DIR / "joker.wav")
DEFAULT_WARMUP_REF_TEXT_PATH = _SAMPLES_DIR / "joker.txt"


def _load_default_warmup_ref_text() -> str:
    try:
        return DEFAULT_WARMUP_REF_TEXT_PATH.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


MAX_TEXT_CHARS = _env_int("QWEN3_TTS_MAX_CHARS", 1000)
DEFAULT_CHUNK_SIZE = _env_int("QWEN3_TTS_CHUNK_SIZE", 8)
DEFAULT_CAP = _env_int("QWEN3_TTS_CAP", 4)
REF_CACHE_SIZE = _env_int("QWEN3_TTS_REF_CACHE_SIZE", 100)
IDLE_START_TIMEOUT_S = _env_float("QWEN3_TTS_IDLE_TIMEOUT_S", 10.0)
SYNTH_HARD_TIMEOUT_S = _env_float("QWEN3_TTS_SYNTH_TIMEOUT_S", 45.0)
MAX_REF_AUDIO_BYTES = _env_int("QWEN3_TTS_MAX_REF_BYTES", 10 * 1024 * 1024)  # 10 MB

# Inference defaults — match faster-qwen3-tts/examples/openai_server.py.
TEMPERATURE = _env_float("QWEN3_TTS_TEMPERATURE", 0.9)
TOP_K = _env_int("QWEN3_TTS_TOP_K", 50)
TOP_P = _env_float("QWEN3_TTS_TOP_P", 1.0)
REPETITION_PENALTY = _env_float("QWEN3_TTS_REPETITION_PENALTY", 1.05)
MAX_NEW_TOKENS = _env_int("QWEN3_TTS_MAX_NEW_TOKENS", 2048)


# ---------------------------------------------------------------------------
# Reference-audio cache
# ---------------------------------------------------------------------------
# Maps sha256(ref_audio_bytes) -> (path_on_disk, ref_text). Reusing the same
# path string across calls for one voice lets the model's internal
# _voice_prompt_cache (keyed on str(ref_audio)) hit, skipping the speech
# tokenizer encode on every chunk after the first.

class _RefAudioCache:
    def __init__(self, capacity: int) -> None:
        self._capacity = max(1, capacity)
        self._lock = threading.Lock()
        self._items: "OrderedDict[str, tuple[str, str]]" = OrderedDict()

    def get(self, key: str) -> tuple[str, str] | None:
        with self._lock:
            entry = self._items.get(key)
            if entry is None:
                return None
            self._items.move_to_end(key)
            return entry

    def put(self, key: str, audio_bytes: bytes, ref_text: str, *, suffix: str = ".wav") -> str:
        with self._lock:
            existing = self._items.get(key)
            if existing is not None:
                path, _ = existing
                self._items[key] = (path, ref_text)
                self._items.move_to_end(key)
                return path

            fd, path = tempfile.mkstemp(prefix=f"qwen3tts_{key[:12]}_", suffix=suffix)
            try:
                with os.fdopen(fd, "wb") as f:
                    f.write(audio_bytes)
            except Exception:
                try:
                    os.unlink(path)
                except OSError:
                    pass
                raise

            self._items[key] = (path, ref_text)
            while len(self._items) > self._capacity:
                _, (old_path, _) = self._items.popitem(last=False)
                try:
                    os.unlink(old_path)
                except OSError:
                    pass
            return path


# ---------------------------------------------------------------------------
# Inflight tracking (structural — every close path decrements once)
# ---------------------------------------------------------------------------

class _InflightTracker:
    def __init__(self, cap: int) -> None:
        self._cap = cap
        self._count = 0
        self._lock = asyncio.Lock()

    @property
    def cap(self) -> int:
        return self._cap

    @property
    def inflight(self) -> int:
        return self._count

    @asynccontextmanager
    async def slot(self):
        async with self._lock:
            if self._count >= self._cap:
                raise _ServerBusy()
            self._count += 1
        try:
            yield
        finally:
            async with self._lock:
                self._count = max(0, self._count - 1)


class _ServerBusy(Exception):
    pass


# ---------------------------------------------------------------------------
# Metrics — exposed via GET /metrics for the Vocence ops dashboard to scrape.
# Monotonic counters (backend computes per-minute deltas) plus a sliding-window
# duration percentile sample. All thread-safe; updated from the WS handler
# (asyncio thread) and read from the metrics endpoint.
# ---------------------------------------------------------------------------

class _Metrics:
    def __init__(self, recent_window: int = 1000) -> None:
        self._lock = threading.Lock()
        self.start_ts = time.time()
        self.requests_total = 0
        self.requests_ok = 0
        self.requests_err: dict[str, int] = {}
        self.duration_ms_sum = 0.0
        self.duration_ms_count = 0
        self.recent_durations_ms: collections.deque[float] = collections.deque(maxlen=recent_window)
        self.bytes_sent_total = 0
        self.audio_ms_total = 0  # total synthesized audio length (informational)

    def record_success(self, duration_ms: float, *, bytes_sent: int = 0, audio_ms: int = 0) -> None:
        with self._lock:
            self.requests_total += 1
            self.requests_ok += 1
            self.duration_ms_sum += duration_ms
            self.duration_ms_count += 1
            self.recent_durations_ms.append(duration_ms)
            self.bytes_sent_total += bytes_sent
            self.audio_ms_total += audio_ms

    def record_error(self, code: str, duration_ms: float = 0.0) -> None:
        with self._lock:
            self.requests_total += 1
            self.requests_err[code] = self.requests_err.get(code, 0) + 1
            if duration_ms > 0:
                self.duration_ms_sum += duration_ms
                self.duration_ms_count += 1
                self.recent_durations_ms.append(duration_ms)

    def snapshot(self) -> dict:
        with self._lock:
            durations = sorted(self.recent_durations_ms)
            n = len(durations)

            def pct(p: float) -> float:
                if n == 0:
                    return 0.0
                idx = min(n - 1, int(p * n))
                return durations[idx]

            return {
                "uptime_seconds": int(time.time() - self.start_ts),
                "requests_total": self.requests_total,
                "requests_ok": self.requests_ok,
                "requests_err": dict(self.requests_err),
                "duration_ms_sum": self.duration_ms_sum,
                "duration_ms_count": self.duration_ms_count,
                "duration_ms_avg": (self.duration_ms_sum / self.duration_ms_count) if self.duration_ms_count else 0.0,
                "duration_ms_p50": pct(0.50),
                "duration_ms_p95": pct(0.95),
                "duration_ms_p99": pct(0.99),
                "bytes_sent_total": self.bytes_sent_total,
                "audio_ms_total": self.audio_ms_total,
            }


# ---------------------------------------------------------------------------
# Synthesis driver — model runs in a thread, ships PCM into an asyncio queue
# ---------------------------------------------------------------------------

class _AudioFrameStream:
    """Slices variable-length float32 chunks from the model into fixed-size
    40 ms / 1920-byte PCM16LE frames. The tail of one yield is carried to the
    next yield rather than zero-padded mid-stream (which would cause audible
    clicks). Only the very last partial frame at end-of-utterance is padded."""

    def __init__(self) -> None:
        self._buf = bytearray()

    def feed_float32(self, samples: np.ndarray) -> list[bytes]:
        if samples.size == 0:
            return []
        if samples.ndim > 1:
            samples = samples.reshape(-1)
        pcm = np.clip(samples * 32768.0, -32768.0, 32767.0).astype(np.int16).tobytes()
        self._buf.extend(pcm)
        out: list[bytes] = []
        while len(self._buf) >= BYTES_PER_FRAME:
            out.append(bytes(self._buf[:BYTES_PER_FRAME]))
            del self._buf[:BYTES_PER_FRAME]
        return out

    def flush(self) -> bytes | None:
        if not self._buf:
            return None
        if len(self._buf) < BYTES_PER_FRAME:
            self._buf.extend(b"\x00" * (BYTES_PER_FRAME - len(self._buf)))
        frame = bytes(self._buf[:BYTES_PER_FRAME])
        self._buf.clear()
        return frame


class _SynthesisRunner:
    """Daemon thread driving `generate_voice_clone_streaming`. Pushes either
    PCM frame bytes or sentinels onto an asyncio queue. Setting `cancel_event`
    causes the generator to be closed at the next yield boundary."""

    _DONE = object()
    _CANCELLED = object()

    def __init__(
        self,
        model: FasterQwen3TTS,
        gpu_lock: threading.Lock,
        *,
        text: str,
        ref_path: str,
        ref_text: str,
        language: str,
        chunk_size: int,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self._model = model
        self._gpu_lock = gpu_lock
        self._text = text
        self._ref_path = ref_path
        self._ref_text = ref_text
        self._language = language
        self._chunk_size = chunk_size
        self._loop = loop
        self.cancel_event = threading.Event()
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=64)
        self.duration_samples = 0
        self.error: BaseException | None = None
        self._frame_stream = _AudioFrameStream()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="qwen3tts-synth", daemon=True)
        self._thread.start()

    def _put_threadsafe(self, item: Any) -> None:
        try:
            self._loop.call_soon_threadsafe(self.queue.put_nowait, item)
        except RuntimeError:
            pass  # loop closed during shutdown

    def _run(self) -> None:
        gen = None
        try:
            with self._gpu_lock:
                if self.cancel_event.is_set():
                    self._put_threadsafe(self._CANCELLED)
                    return

                gen = self._model.generate_voice_clone_streaming(
                    text=self._text,
                    language=self._language,
                    ref_audio=self._ref_path,
                    ref_text=self._ref_text,
                    chunk_size=self._chunk_size,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=TEMPERATURE,
                    top_k=TOP_K,
                    top_p=TOP_P,
                    do_sample=True,
                    repetition_penalty=REPETITION_PENALTY,
                    append_silence=True,
                    non_streaming_mode=False,
                )

                for audio_chunk, _sr, _timing in gen:
                    if self.cancel_event.is_set():
                        try:
                            gen.close()
                        except Exception:
                            pass
                        self._put_threadsafe(self._CANCELLED)
                        return

                    if audio_chunk is None or audio_chunk.size == 0:
                        continue
                    self.duration_samples += int(audio_chunk.size)
                    for frame in self._frame_stream.feed_float32(audio_chunk):
                        if self.cancel_event.is_set():
                            self._put_threadsafe(self._CANCELLED)
                            return
                        self._put_threadsafe(frame)

                tail = self._frame_stream.flush()
                if tail is not None and not self.cancel_event.is_set():
                    self._put_threadsafe(tail)

        except BaseException as exc:  # noqa: BLE001 — preserve for caller
            self.error = exc
            try:
                if gen is not None:
                    gen.close()
            except Exception:
                pass
        finally:
            self._put_threadsafe(self._DONE)


# ---------------------------------------------------------------------------
# Request validation
# ---------------------------------------------------------------------------

class _BadRequest(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class _RefNotCached(Exception):
    pass


def _validate_audio(audio_bytes: bytes) -> str:
    """Sanity-check that the bytes decode as audio (WAV/OPUS/FLAC/OGG/MP3/…).
    Returns a filename suffix appropriate for the detected format so the
    on-disk cache file matches the content (some loaders rely on the
    extension). Raises _BadRequest if the bytes are unparseable.

    The spec wording suggests WAV-only, but in practice the backend may
    send `.opus` (CDN-hosted sample voices like design-aria are opus), so
    we accept anything libsndfile can read. The model's
    `_load_ref_audio_with_silence` uses soundfile under the hood, so any
    format soundfile reads is fine for synthesis."""
    if not audio_bytes:
        raise _BadRequest("ref_audio_b64: empty payload")

    # Fast path: WAV. Cheaper than soundfile.info for the common case.
    if audio_bytes[:4] == b"RIFF":
        try:
            with wave.open(io.BytesIO(audio_bytes), "rb") as w:
                if w.getnframes() <= 0:
                    raise _BadRequest("ref_audio_b64: empty WAV")
                if w.getframerate() < 8000 or w.getframerate() > 48000:
                    raise _BadRequest("ref_audio_b64: sample rate out of range (8-48 kHz)")
        except wave.Error as e:
            raise _BadRequest(f"ref_audio_b64: malformed WAV ({e})") from None
        return ".wav"

    # Detect by magic bytes — covers the formats the model accepts.
    head = audio_bytes[:4]
    if head[:4] == b"OggS":
        suffix = ".opus"  # opus is the common Ogg payload from our CDN
    elif head[:4] == b"fLaC":
        suffix = ".flac"
    elif head[:3] == b"ID3" or (len(audio_bytes) > 1 and audio_bytes[0] == 0xFF and (audio_bytes[1] & 0xE0) == 0xE0):
        suffix = ".mp3"
    else:
        # Unknown magic — let soundfile try.
        suffix = ".audio"

    try:
        import soundfile as sf  # local import: only needed for non-WAV path
        info = sf.info(io.BytesIO(audio_bytes))
        if info.samplerate < 8000 or info.samplerate > 48000:
            raise _BadRequest("ref_audio_b64: sample rate out of range (8-48 kHz)")
    except _BadRequest:
        raise
    except Exception as e:
        raise _BadRequest(f"ref_audio_b64: unsupported audio format ({type(e).__name__}: {e})") from None
    return suffix


def _parse_start_frame(
    raw: str,
    ref_cache: _RefAudioCache,
) -> tuple[str, str, str, str]:
    """Returns (text, ref_path, ref_text, language). Raises _BadRequest or
    _RefNotCached on protocol errors."""
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        raise _BadRequest(f"start frame not valid JSON: {e}")

    if not isinstance(obj, dict):
        raise _BadRequest("start frame must be a JSON object")
    if obj.get("type") != "start":
        raise _BadRequest("first frame must be {'type':'start',...}")

    text = obj.get("text")
    if not isinstance(text, str) or not text.strip():
        raise _BadRequest("start.text is required and must be a non-empty string")
    text = text.strip()
    if len(text) > MAX_TEXT_CHARS:
        raise _BadRequest(
            f"start.text too long ({len(text)} chars; max {MAX_TEXT_CHARS})"
        )

    ref_text = obj.get("ref_text", "")
    if not isinstance(ref_text, str):
        raise _BadRequest("start.ref_text must be a string when present")

    language = obj.get("language", "Auto")
    if not isinstance(language, str) or not language.strip():
        language = "Auto"
    if language not in _KNOWN_LANGUAGES:
        _log.debug("non-standard language %r — passing through", language)

    # Two ref-audio paths: full bytes (always works) or hash-only (chunks 2..N).
    ref_b64 = obj.get("ref_audio_b64")
    ref_hash = obj.get("ref_audio_sha256")

    if isinstance(ref_b64, str) and ref_b64:
        try:
            audio_bytes = base64.b64decode(ref_b64, validate=False)
        except Exception as e:
            raise _BadRequest(f"ref_audio_b64: base64 decode failed ({e})")
        if len(audio_bytes) > MAX_REF_AUDIO_BYTES:
            raise _BadRequest(
                f"ref_audio_b64: too large ({len(audio_bytes)} bytes; "
                f"max {MAX_REF_AUDIO_BYTES})"
            )
        suffix = _validate_audio(audio_bytes)
        computed_hash = hashlib.sha256(audio_bytes).hexdigest()
        if isinstance(ref_hash, str) and ref_hash and ref_hash.lower() != computed_hash:
            raise _BadRequest("ref_audio_sha256 does not match ref_audio_b64")
        path = ref_cache.put(computed_hash, audio_bytes, ref_text, suffix=suffix)
        return text, path, ref_text, language

    if isinstance(ref_hash, str) and ref_hash:
        key = ref_hash.strip().lower()
        if len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
            raise _BadRequest("ref_audio_sha256 must be a 64-char hex sha256")
        cached = ref_cache.get(key)
        if cached is None:
            raise _RefNotCached()
        path, cached_ref_text = cached
        return text, path, (ref_text or cached_ref_text), language

    raise _BadRequest("start frame requires ref_audio_b64 OR ref_audio_sha256")


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

class StreamingServer:
    def __init__(
        self,
        model: FasterQwen3TTS,
        *,
        api_key: str | None,
        cap: int,
        chunk_size: int,
        model_id: str,
    ) -> None:
        self._model = model
        self._model_id = model_id
        self._api_key = (api_key or "").strip()
        self._chunk_size = chunk_size
        self._inflight = _InflightTracker(cap=cap)
        self._gpu_lock = threading.Lock()
        self._ref_cache = _RefAudioCache(REF_CACHE_SIZE)
        self._metrics = _Metrics()

    async def healthz(self, _request: web.Request) -> web.Response:
        return web.json_response({
            "status": "ok",
            "service": "tts_streaming",
            "model_id": self._model_id,
            "sample_rate": SAMPLE_RATE,
            "frame_ms": FRAME_MS,
            "inflight": self._inflight.inflight,
            "cap": self._inflight.cap,
            "dev_stub": False,
        })

    async def metrics(self, request: web.Request) -> web.Response:
        """Scrape endpoint for the Vocence ops dashboard. Bearer-auth'd the
        same as /v1/voice-clone/stream so a public-IP rented box doesn't leak
        traffic counters to anyone who curls it."""
        if self._api_key:
            header = request.headers.get("Authorization", "")
            if header != f"Bearer {self._api_key}":
                return web.json_response(
                    {"type": "error", "code": "auth", "message": "missing or invalid bearer token"},
                    status=401,
                )
        snap = self._metrics.snapshot()
        snap["service"] = "tts_streaming"
        snap["inflight"] = self._inflight.inflight
        snap["cap"] = self._inflight.cap
        return web.json_response(snap)

    async def voice_clone_stream(self, request: web.Request) -> web.WebSocketResponse:
        # Auth before WS upgrade so we can return a clean 401.
        if self._api_key:
            header = request.headers.get("Authorization", "")
            if header != f"Bearer {self._api_key}":
                ws = web.WebSocketResponse()
                await ws.prepare(request)
                await _send_error(ws, "auth", "missing or invalid bearer token")
                await ws.close(code=WSCloseCode.POLICY_VIOLATION)
                return ws

        ws = web.WebSocketResponse(heartbeat=20.0, max_msg_size=MAX_REF_AUDIO_BYTES * 2)
        await ws.prepare(request)

        try:
            slot_cm = self._inflight.slot()
            await slot_cm.__aenter__()
        except _ServerBusy:
            await _send_error(ws, "server_busy", "inflight cap exhausted")
            await ws.close(code=WSCloseCode.TRY_AGAIN_LATER)
            return ws

        try:
            await self._handle_session(ws)
        finally:
            await slot_cm.__aexit__(None, None, None)

        return ws

    async def _handle_session(self, ws: web.WebSocketResponse) -> None:
        t_session = time.perf_counter()  # for metrics; covers handshake -> end

        def session_ms() -> float:
            return (time.perf_counter() - t_session) * 1000.0

        # 1. Read the `start` frame with an idle timeout.
        try:
            first = await asyncio.wait_for(ws.receive(), timeout=IDLE_START_TIMEOUT_S)
        except asyncio.TimeoutError:
            await _send_error(ws, "timeout", f"no start frame within {IDLE_START_TIMEOUT_S}s")
            await ws.close(code=WSCloseCode.GOING_AWAY)
            self._metrics.record_error("timeout", session_ms())
            return

        if first.type in (WSMsgType.CLOSE, WSMsgType.CLOSED, WSMsgType.ERROR):
            # Client gave up before sending start; not an error from our side.
            return
        if first.type != WSMsgType.TEXT:
            await _send_error(ws, "bad_request", "start frame must be a text JSON message")
            await ws.close(code=WSCloseCode.POLICY_VIOLATION)
            self._metrics.record_error("bad_request", session_ms())
            return

        try:
            text, ref_path, ref_text, language = _parse_start_frame(first.data, self._ref_cache)
        except _RefNotCached:
            await _send_error(
                ws, "ref_not_cached",
                "ref_audio_sha256 not in server cache — resend with ref_audio_b64",
            )
            await ws.close(code=WSCloseCode.POLICY_VIOLATION)
            self._metrics.record_error("ref_not_cached", session_ms())
            return
        except _BadRequest as e:
            await _send_error(ws, "bad_request", e.message)
            await ws.close(code=WSCloseCode.POLICY_VIOLATION)
            self._metrics.record_error("bad_request", session_ms())
            return

        # 2. Meta. Fixed by spec — the backend ignores values but other clients may use them.
        try:
            await ws.send_json({
                "type": "meta",
                "sample_rate": SAMPLE_RATE,
                "frame_ms": FRAME_MS,
                "encoding": "pcm16le",
                "channels": 1,
            })
        except Exception:
            # Client disconnected before we could send meta — count as bad_request
            # (we accepted the slot and parsed; not a clean cancellation).
            self._metrics.record_error("client_disconnect", session_ms())
            return

        # 3. Spin up the synthesis thread.
        loop = asyncio.get_running_loop()
        runner = _SynthesisRunner(
            self._model,
            self._gpu_lock,
            text=text,
            ref_path=ref_path,
            ref_text=ref_text,
            language=language,
            chunk_size=self._chunk_size,
            loop=loop,
        )

        # 4. Background watcher: client close -> cancel event -> synth aborts at next yield.
        close_watcher = asyncio.create_task(_watch_for_close(ws, runner.cancel_event))

        t_start = time.perf_counter()
        runner.start()
        ended_cleanly = False
        synth_error: str | None = None
        bytes_sent = 0
        cancelled_by_client = False

        try:
            while True:
                try:
                    item = await asyncio.wait_for(runner.queue.get(), timeout=SYNTH_HARD_TIMEOUT_S)
                except asyncio.TimeoutError:
                    runner.cancel_event.set()
                    synth_error = f"synth exceeded {SYNTH_HARD_TIMEOUT_S}s hard cap"
                    break

                if item is runner._CANCELLED:
                    cancelled_by_client = True
                    return
                if item is runner._DONE:
                    if runner.error is not None:
                        synth_error = f"engine: {type(runner.error).__name__}: {runner.error}"
                    else:
                        ended_cleanly = True
                    break

                try:
                    await ws.send_bytes(item)
                    bytes_sent += len(item)
                except (ConnectionResetError, asyncio.CancelledError, RuntimeError):
                    runner.cancel_event.set()
                    cancelled_by_client = True
                    return
                except Exception as e:
                    runner.cancel_event.set()
                    _log.warning("WS send_bytes failed: %s", e)
                    cancelled_by_client = True
                    return

            if ended_cleanly:
                duration_ms = int(runner.duration_samples * 1000 / SAMPLE_RATE)
                try:
                    await ws.send_json({"type": "end", "duration_ms": duration_ms})
                except Exception:
                    pass
                _log.info(
                    "voice-clone-stream: ok text=%d chars audio=%dms wall=%.2fs",
                    len(text), duration_ms, time.perf_counter() - t_start,
                )
                self._metrics.record_success(
                    session_ms(), bytes_sent=bytes_sent, audio_ms=duration_ms,
                )
            else:
                await _send_error(ws, "engine_failed", (synth_error or "unknown engine error"))
                code = "timeout" if synth_error and "hard cap" in synth_error else "engine_failed"
                self._metrics.record_error(code, session_ms())
        finally:
            # Client barge-in is NOT an error — it's normal voice-agent flow.
            # We don't record it as success either (no `end` sent); intentionally
            # untracked so error rates stay clean.
            _ = cancelled_by_client and not ended_cleanly  # noqa: keeps intent visible
            runner.cancel_event.set()
            close_watcher.cancel()
            try:
                await asyncio.wait_for(close_watcher, timeout=0.1)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
            if not ws.closed:
                try:
                    await asyncio.wait_for(ws.close(code=WSCloseCode.OK), timeout=0.3)
                except (asyncio.TimeoutError, Exception):
                    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _send_error(ws: web.WebSocketResponse, code: str, message: str) -> None:
    try:
        await ws.send_json({"type": "error", "code": code, "message": message[:200]})
    except Exception:
        pass


async def _watch_for_close(ws: web.WebSocketResponse, cancel_event: threading.Event) -> None:
    """Read from the WS until it closes; signal cancel on any close, RST, or
    unexpected frame. Spec §3.1: client MUST NOT send anything after `start`,
    so any extra frame is treated as an implicit cancel."""
    try:
        async for msg in ws:
            if msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSED, WSMsgType.ERROR):
                cancel_event.set()
                return
            cancel_event.set()
            return
    except asyncio.CancelledError:
        return
    except Exception:
        cancel_event.set()


def _warmup(model: FasterQwen3TTS, ref_path: str, ref_text: str, chunk_size: int) -> None:
    """Run one tiny synthesis at boot so CUDA graphs are captured before the
    first real request. Without this, the first WS connection eats ~2-5 s of
    graph capture latency and blows TTFA."""
    _log.info("warming up CUDA graphs…")
    t0 = time.perf_counter()
    try:
        gen = model.generate_voice_clone_streaming(
            text="Hello.",
            language="English",
            ref_audio=ref_path,
            ref_text=ref_text,
            chunk_size=chunk_size,
            max_new_tokens=64,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            top_p=TOP_P,
            do_sample=True,
            repetition_penalty=REPETITION_PENALTY,
            append_silence=True,
            non_streaming_mode=False,
        )
        for _ in gen:
            pass
        _log.info("warmup complete in %.2fs", time.perf_counter() - t0)
    except Exception as e:
        _log.warning("warmup failed (non-fatal): %s", e)


# ---------------------------------------------------------------------------
# App factory + entrypoint
# ---------------------------------------------------------------------------

def build_app(
    *,
    model: FasterQwen3TTS,
    api_key: str | None,
    cap: int,
    chunk_size: int,
    model_id: str,
) -> web.Application:
    server = StreamingServer(
        model,
        api_key=api_key,
        cap=cap,
        chunk_size=chunk_size,
        model_id=model_id,
    )
    app = web.Application()
    app.router.add_get("/healthz", server.healthz)
    app.router.add_get("/metrics", server.metrics)
    app.router.add_get("/v1/voice-clone/stream", server.voice_clone_stream)
    app["server"] = server
    return app


def run(
    *,
    model_id: str,
    host: str,
    port: int,
    api_key: str | None,
    cap: int,
    chunk_size: int,
    device: str,
    dtype: str,
    warmup_ref_audio: str | None,
    warmup_ref_text: str,
) -> None:
    if dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif dtype == "fp16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    _log.info("loading model %s on %s (%s)…", model_id, device, dtype)
    t0 = time.perf_counter()
    model = FasterQwen3TTS.from_pretrained(
        model_id,
        device=device,
        dtype=torch_dtype,
        attn_implementation="sdpa",
        max_seq_len=2048,
    )
    _log.info("model ready in %.1fs (sample rate %d Hz)", time.perf_counter() - t0, model.sample_rate)

    if model.sample_rate != SAMPLE_RATE:
        _log.error(
            "model sample rate %d != server SAMPLE_RATE %d — output will sound wrong",
            model.sample_rate, SAMPLE_RATE,
        )

    if warmup_ref_audio:
        _warmup(model, warmup_ref_audio, warmup_ref_text, chunk_size)
    else:
        _log.warning(
            "no warmup ref audio configured (--warmup-ref-audio); the first WS "
            "request will pay ~2-5s of CUDA graph capture latency"
        )

    app = build_app(
        model=model,
        api_key=api_key,
        cap=cap,
        chunk_size=chunk_size,
        model_id=model_id,
    )

    _log.info(
        "serving on %s:%d  (cap=%d chunk_size=%d max_chars=%d auth=%s)",
        host, port, cap, chunk_size, MAX_TEXT_CHARS,
        "on" if api_key else "OFF",
    )
    web.run_app(app, host=host, port=port, print=None)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="qwen3-tts-streaming",
        description="Voice-clone streaming WebSocket server for faster-qwen3-tts",
    )
    p.add_argument(
        "--model",
        default=os.environ.get("QWEN3_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"),
        help="HF model id or local path (default: Qwen3-TTS-12Hz-1.7B-Base)",
    )
    p.add_argument("--host", default=os.environ.get("QWEN3_TTS_HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=_env_int("QWEN3_TTS_PORT", 8111))
    p.add_argument(
        "--api-key",
        default=os.environ.get("QWEN3_TTS_API_KEY"),
        help="Required bearer token. Empty disables auth (dev only).",
    )
    p.add_argument("--cap", type=int, default=DEFAULT_CAP, help="Max concurrent WS connections")
    p.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
                   help="Codec tokens per streaming yield (8 = ~667ms audio)")
    p.add_argument("--device", default=os.environ.get("QWEN3_TTS_DEVICE", "cuda"))
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument(
        "--warmup-ref-audio",
        default=os.environ.get("QWEN3_TTS_WARMUP_REF_AUDIO") or DEFAULT_WARMUP_REF_AUDIO,
        help=(
            "WAV file used for the CUDA-graph warmup pass at boot. Defaults "
            "to the bundled samples/joker.wav so the server starts hot out of "
            "the box."
        ),
    )
    p.add_argument(
        "--warmup-ref-text",
        default=os.environ.get("QWEN3_TTS_WARMUP_REF_TEXT") or _load_default_warmup_ref_text(),
        help="Transcript matching --warmup-ref-audio (defaults to samples/joker.txt)",
    )
    p.add_argument(
        "--log-level",
        default=os.environ.get("QWEN3_TTS_LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run(
        model_id=args.model,
        host=args.host,
        port=args.port,
        api_key=args.api_key,
        cap=args.cap,
        chunk_size=args.chunk_size,
        device=args.device,
        dtype=args.dtype,
        warmup_ref_audio=args.warmup_ref_audio,
        warmup_ref_text=args.warmup_ref_text,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
