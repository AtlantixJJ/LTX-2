# `scripts/prune/` working rules

## Active task and separate deployed regression

Pruning selection and acceptance use native whole-video D0: an original capture, one clean conditioning latent frame, exact saved noise at each sigma, one full bidirectional forward, and comparison of generated-frame noise direction plus VAE-decoded output. The k2 sliding-window refiner is a separate deployed method; its old pruning experiments are historical for this decision. Keep `checks.method_parity` for any tensor-affecting change to `core.refine_core` or the deployed `vae_refine_sliding_window.py` path.

Run from the LTX-2 root in the `ltx` conda environment, using `python -m scripts.prune.<subpackage>.<module>`. Check `nvidia-smi` before loading the 22B model. `scripts/` and `scripts/prune/` are PEP 420 namespace packages: do not add `__init__.py` there.

## Invariants

- `data.whole_clip` owns saved native D0 inputs. Calibration uses only the baseline manifest and checkpoint; pairwise evaluation separately verifies candidate setup and actual epsilon equality. Do not rebuild a different noise draw, fps, clean prefix, attention mask or geometry.
- `core.session` owns model lifetime and the one `DTYPE = torch.bfloat16` declaration. Model forwards use `torch.no_grad()` except explicit VJP estimators. `core.artifacts` owns standard `expr/refiner_prune/<model>` names and attributable run directories; explicit output arguments are allowed for matched D0 experiment directories.
- `score.hooks` validates model identity, mask widths and native D0 task provenance. Active export rejects historical/missing task stamps. `score.export_pruned` records mask hash and source fingerprint. Parity compares the functional mask with the export **before** output quality is interpreted.
- `core.ltx_adapter` contains private upstream API access. `core.refine_core` is the single k2 sliding-window rollout imported by deployment and its regression gate; do not duplicate it.
- The avatar package and deployed refiner use some shared prune utilities. Audit external imports before moving or deleting modules. Retain one `doc/<module>.md` design page per production module, updating it with interface, data-flow or invariant changes.

## Validation

Run focused CPU tests, Ruff on changed D0 modules, and saved-baseline forward parity for changes to D0 input construction or scoring. Use the D0 export parity command with a held-out capture and the documented 0.02 maximum absolute default. For deployed k2 tensor changes, run `python -m scripts.prune.checks.method_parity --model 2.5 --gpu-id N --windows 3`. Inspect synchronized VAE-decoded comparisons and benchmark drift before claiming quality or speed.

The [README](README.md) gives active commands. [METHODS](doc/METHODS.md), [ARCHITECTURE](doc/ARCHITECTURE.md), [VALIDATION](doc/VALIDATION.md), and [HISTORY](doc/HISTORY.md) explain the design; the [module index](doc/README.md) maps each implementation file.
