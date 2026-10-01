# `core/ltx_adapter.py`

## Objective and data flow

Quarantine upstream private transformer context and VAE encoder/decoder access.
Thin context managers own resident model lifetimes. Sampling state builders and
steppers are outside this interface.

## Invariants and verification

Review private access when the upstream API changes. No other prune module
imports underscore-prefixed upstream symbols; `tests/test_ltx_adapter.py`
checks that boundary and the documented source revision.
