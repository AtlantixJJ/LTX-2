# Plans and retained research

The workspace [avatar handoff](../../plans/2026-10-07-onestep-avatar-development-experiment-handoff.md)
owns avatar progress, stop instructions and next actions. Pruning study plans live
in the workspace [plans directory](../../plans/). This folder has no active execution plan.

- [Dormant research backlog](RESEARCH_BACKLOG.md): useful proposals extracted from retired plans, plus deferred test-tool options.
- [Historical evidence](HISTORY.md): dated observations, their limits, retained assets and source attribution.
- [Cleanup inventory](cleanup_2026-10-17.tsv): original paths, content hashes, Git blobs and disposition for all 23 former files.

Use the package [avatar run guide](../scripts/onestep_avatar/configs/README.md)
and [pruning run guide](../scripts/prune/doc/usage.md) for commands.
The [test-selector design](../scripts/onestep_avatar/doc/tests/select.md) owns T0–T3 selection.
The old test-suite proposal is implemented; its deferred options remain in the backlog.

The 2026-10-17 cleanup removed redundant plans and seven historical scripts.
Four measurement and presentation assets survive byte for byte under ignored
`expr/onestep_avatar/plan_cleanup_20261017/evidence/2026-09-26-d1-boundaries/`
in the workspace. The historical record links them. Removed tracked source remains
recoverable from the pinned Git revision. Rebuilds must use current package owners;
do not restore removed plans or executors into this folder.

Check preserved hashes, Git sources, links and absence of retired paths with the
[read-only validator](validate_cleanup.py), from LTX-2 in the `ltx` environment:

```bash
conda run -n ltx python plans/validate_cleanup.py
```
