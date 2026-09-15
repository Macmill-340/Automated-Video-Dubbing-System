"""Allow ``python -m video_dubber`` as an alias for the ``dub-video`` CLI."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
