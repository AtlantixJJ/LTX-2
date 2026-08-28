"""The T0-T3 gate metrics, decoding, the Phase 0 latency/FLOP baseline, the
per-sigma K/V cache, and the pruned-candidate gates themselves. Physically
relocated here -- see ``scripts/prune/core/__init__.py`` for the relocation
note.

(``evaluate/``, not ``eval/`` -- ``eval`` shadows the builtin and trips
ruff's ``A`` rules.)

Modules: bench_refiner, cross_kv_cache, decode, gates, head_ablation_eval,
metrics, phase1_gates, sampler_ab, timing.
"""
