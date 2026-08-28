"""Phase 2/3 pruning: activation hooks and masks, the loss/least-squares
estimators, head and FFN scoring, the iterative pruning schedule, and
exporting a narrower checkpoint. Physically relocated here -- see
``scripts/prune/core/__init__.py`` for the relocation note.

Modules: export_pruned, ffn_scores, head_scores, hooks, losses, lstsq,
prune_schedule.
"""
