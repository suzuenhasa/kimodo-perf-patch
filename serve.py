#!/usr/bin/env python
"""Compatibility shim: the implementation moved into the kimodo_fast package.

Kept because the measurement scripts in scripts/ import `serve` by path, and those
scripts are the record of how every number in FINDINGS.md was produced -- rewriting
their imports would silently change what was run.
"""
from kimodo_fast.serve import *          # noqa: F401,F403
from kimodo_fast.serve import KimodoFast, main   # noqa: F401

if __name__ == "__main__":
    main()
