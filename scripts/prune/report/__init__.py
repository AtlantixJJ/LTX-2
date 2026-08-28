"""Organizational grouping only -- see ``scripts/prune/core/__init__.py``.

Collecting every gate/number into ``analysis_summary.json``, and plotting
head scores.
"""

from scripts.prune import plot_head_scores, summarize_phase0

__all__ = ["plot_head_scores", "summarize_phase0"]
