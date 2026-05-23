"""Allow `python -m qwen3_tts_streaming` as an entrypoint."""
from .server import main

if __name__ == "__main__":
    main()
