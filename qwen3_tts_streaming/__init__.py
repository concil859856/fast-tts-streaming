"""qwen3-tts-streaming — WebSocket streaming server for faster-qwen3-tts.

Implements the wire protocol at
vocence_website/dashboard-backend/docs/tts_server_spec.md, intended as a drop-in
replacement for the previous `qwen3-clone-streaming` service.
"""
from .server import StreamingServer, build_app, run

__version__ = "0.1.0"
__all__ = ["StreamingServer", "build_app", "run"]
