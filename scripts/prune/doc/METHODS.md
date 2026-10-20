# Pruning methods and research scope

## Current task

The supported task is a full-clip bidirectional D0 forward with clean capture frame zero.
Each schedule is exactly `[sigma, 0]`. The [core algorithm](core_algorithm.md) defines the numerical contract.
The [run guide](usage.md) gives current commands. The [verification guide](VALIDATION.md) defines independent acceptance gates.

## Training-free selection

The scorer measures native projected contributions on saved calibration inputs.
Width masks remove complete attention heads or FFN channels at their projection boundaries.
Depth export removes a declared set of complete blocks in original order.
A calibration score ranks interventions. It does not establish held-out quality or measured acceleration.
The package does not supply a native fitted-weight collector or exporter.

## Learned heads and channels

The head learner implements FP32 scalar gates and bounded V/output adapter recovery.
Its current DDP preset uses rank-128 attention adapters and continuous F18 optimizer crops.
The [protocol source](../learn/protocol.py) derives phase budgets from frozen configuration.
The [FFN family](learn/ffn.md) adds individual binary channel masks and FFN adapters.
Its explicit preset retains 14,848 channels per block and uses rank-16 FFN adapters.

The [V3 family](learn/joint.md) supports joint clipped gates, heads-only clipped gates and a matched-budget arm.
The [readmission policy](learn/readmission.md) owns return windows, permanent removals and terminal alignment.
The [shared DDP owner](learn/ddp_owner.md) runs native acceptance, optimizer updates, checkpoints and full-source monitoring.
The frozen config selects each arm and its exact ranks, schedules, adapters and update limits.

## Export arithmetic

A merged export adds FP32 adapter deltas before coupled structural slicing.
This changes BF16 rounding points and requires separate native parity evidence.
A [split bundle](score/export_joint_split.md) preserves the two trained adapter GEMMs.
Its full-width FFN down reduction must be included in timing.
`compact_faithful` reduces stored width while restoring original execution width.
A smaller file alone cannot establish speed or lower resident memory.

## Research and historical evidence

Current proposed comparisons live in the [workspace backlog](../../../../plans/RESEARCH_BACKLOG.md).
The [online-compaction plan](../../../../plans/2026-10-19-joint-head-ffn-online-compaction.md) remains a separate implementation and acceptance task.
The earlier literature review and completed-study recipes remain in the
[exact-byte method archive](../../../../expr/prune_package_cleanup_20261020/originals/doc/METHODS.md).
That archive includes historical status statements; use current designs for implemented behavior.
Saved studies and reports retain their original identities. This package makes no new quality or speed claim from CPU tests.
