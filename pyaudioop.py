"""Compatibility shim for pyaudioop import.

Some environments expect a top-level `pyaudioop` module. Newer versions of
`pydub` include `pydub.pyaudioop`. This shim attempts to import
`pydub.pyaudioop` and expose it as `pyaudioop` so existing imports work.
"""
try:
    # pydub includes pyaudioop implementation
    from pydub import pyaudioop as _audioop
except Exception as _e:
    # If pydub isn't installed, raise ImportError to keep behavior consistent
    raise ImportError("pydub.pyaudioop not available; please install pydub") from _e

# Re-export names
for _name in dir(_audioop):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_audioop, _name)

# Also keep a module object reference
__all__ = [n for n in dir(_audioop) if not n.startswith("__")]
