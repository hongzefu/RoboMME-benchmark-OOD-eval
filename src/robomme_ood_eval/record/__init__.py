"""Per-episode recording: ``recorder`` (raw AV1 4:4:4 frames + arrays), ``trace_writer`` (trace.jsonl),
``official_render`` (official-layout rendering).

The modules are deliberately kept separate (a dozen callers load them by module name or file path); this file
imports nothing so that loading by path behaves exactly as before.
"""
