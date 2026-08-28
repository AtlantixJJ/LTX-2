"""Organizational grouping only -- see ``scripts/prune/core/__init__.py``.

The three bit-exactness gates: the pre-``refine_core`` registry-refactor
check, the deployed-method rollout parity check, and the audio-branch-drop
check.
"""

from scripts.prune import method_parity, parity_check, video_only_check

__all__ = ["method_parity", "parity_check", "video_only_check"]
