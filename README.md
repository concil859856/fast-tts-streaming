# qwen3-tts-streaming

WebSocket streaming server for [faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts).
Implements the wire protocol at [`vocence_website/dashboard-backend/docs/tts_server_spec.md`](../vocence_website/dashboard-backend/docs/tts_server_spec.md) as a drop-in replacement for the previous `qwen3-clone-streaming` service.

Designed for **one RTX 4090** running **Qwen3-TTS-12Hz-1.7B-Base**:

| Metric (warm) | Target | Measured (demo) |
|---|---|---|
| TTFA | < 200 ms | ~200 ms |
| RTF | > 4× real-time | ~5× |
| Cancel propagation | < 100 ms ceiling | ~120 ms (chunk_size=8) |
| Output | 24 kHz mono PCM16LE, 40 ms frames | matches spec |

## Install

```bash
cd /workspace/development/qwen3-tts-streaming

# Same Python/CUDA env that runs faster-qwen3-tts. Editable install picks
# up the sibling repo automatically.
pip install -e ../faster-qwen3-tts
pip install -e .
```

## Run

```bash
cp .env.example .env
# edit .env: set QWEN3_TTS_API_KEY, QWEN3_TTS_WARMUP_REF_AUDIO, QWEN3_TTS_WARMUP_REF_TEXT

# Option 1: via the installed console script
qwen3-tts-streaming

# Option 2: via the module
python -m qwen3_tts_streaming

# Option 3: with CLI flags
qwen3-tts-streaming \
  --host 127.0.0.1 --port 8111 \
  --api-key "$QWEN3_TTS_API_KEY" \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --warmup-ref-audio /abs/path/to/ref.wav \
  --warmup-ref-text "Transcript of the warmup reference audio."
```

On boot the server:

1. Loads the model in bf16 on `cuda`.
2. Runs **one warmup synthesis** so CUDA graphs are captured (skip this and the first real WS request will hang for 2-5 s).
3. Listens on `127.0.0.1:8111` by default.

## Wire protocol (summary)

See [`docs/tts_server_spec.md`](../vocence_website/dashboard-backend/docs/tts_server_spec.md) for the full normative spec.

### `GET /healthz`
```json
{
  "status": "ok",
  "model_id": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
  "sample_rate": 24000,
  "frame_ms": 40,
  "inflight": 0,
  "cap": 4,
  "dev_stub": false
}
```

### `WS /v1/voice-clone/stream`

Auth: `Authorization: Bearer <QWEN3_TTS_API_KEY>`.

**Client → server** (one text frame):
```json
{
  "type": "start",
  "text": "Sure — so Bittensor's basically a decentralized network…",
  "ref_audio_b64": "<base64 WAV>",
  "ref_text": "Transcript of the reference audio.",
  "language": "English"
}
```

For chunks 2..N of the same voice the client MAY send the SHA-256 hash instead, saving the ~540 KB upload:
```json
{
  "type": "start",
  "text": "Instead of one company controlling everything…",
  "ref_audio_sha256": "5c8e…",
  "language": "English"
}
```

If the hash is not in the server's LRU cache (e.g. cache evicted, server restart), the response is `{"type":"error","code":"ref_not_cached"}` and the client should retry with the full `ref_audio_b64`.

**Server → client** (in order):

1. `{"type":"meta", "sample_rate":24000, "frame_ms":40, "encoding":"pcm16le", "channels":1}` — once
2. Binary PCM frames — 1920 bytes each (24 kHz × 40 ms × int16 mono), many, until done
3. `{"type":"end", "duration_ms":<N>}` on success, OR
4. `{"type":"error", "code":"...", "message":"..."}` on failure

Error codes: `auth`, `bad_request`, `server_busy`, `engine_failed`, `timeout`, `ref_not_cached`.

## Barge-in / cancellation

When the client closes the WebSocket (graceful 1000 or RST), the server:

1. Sets a thread-safe cancel event within ~10 ms of the close handler firing
2. Closes the running generator at the next codec-chunk boundary (~120 ms wall on `chunk_size=8`)
3. Releases the GPU lock so the next WS connection can start its synthesis
4. Decrements the `inflight` counter via `@asynccontextmanager` — guaranteed on every code path

No `end` or `error` frame is sent after a client-initiated close — that's the spec contract.

## Concurrency model

- **Single GPU = single synthesis at a time** (no batching in faster-qwen3-tts).
- `QWEN3_TTS_CAP` controls **WebSocket connection slots**, not GPU concurrency. With `cap=4`, four clients can hold connections and queue at the GPU lock; a fifth gets `server_busy`.
- For a single agent talking to itself (sequential sentence chunks), `cap=1` is sufficient. `cap=4` covers 2-3 simultaneous voice agents queueing without rejection.

## Layout

```
qwen3_tts_streaming/
├── __init__.py
├── __main__.py        # python -m qwen3_tts_streaming
└── server.py          # aiohttp app + synthesis driver
tests/
└── test_protocol.py   # smoke tests for start-frame validation
```

## License

MIT.
