"""The corpus, calibration-record selection, persisted AR states, and the
prompt-context cache. Physically relocated here -- see
``scripts/prune/core/__init__.py`` for the relocation note and why this
package does not eagerly import its submodules (``source_target`` needs
``core.session``, which needs ``data.prompt_cache``).

Modules: chunk_states, corpus, prompt_cache, records, source_target.
"""
