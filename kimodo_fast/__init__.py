"""Kimodo, made small and fast enough to keep resident.

    from kimodo_fast import KimodoFast
    k = KimodoFast("./enc_nf4")        # ~12 s, once, 5.4 GB
    clips = k.generate(["a person walks forward."])
"""
from .serve import KimodoFast, _Table  # noqa: F401

__all__ = ["KimodoFast"]
__version__ = "0.1.0"
