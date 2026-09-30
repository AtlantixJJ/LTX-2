# Pruning architecture

The active experiment is native bidirectional, whole-video D0. `data/whole_clip.py` reads the saved baseline capture, epsilon and manifest and constructs the one-step model input. The scorer consumes that input without a candidate. `score/hooks.py` owns functional masks and validates the native mask format. `score/export_pruned.py` writes a checkpoint with mask and source provenance. `checks/export_parity.py` compares the functional mask with that export on saved D0 inputs. `evaluate/whole_clip_d0.py` compares baseline and candidate directions and makes synchronized VAE-decoded media. `evaluate/bench_whole_clip_d0.py` times the matched forward on one device.

`core/session.py` owns model lifetime, dtype and prompt context; `core/artifacts.py` owns experiment output paths. The deployed sliding-window refiner still uses `core/refine_core.py`, and `checks/method_parity.py` guards it. The avatar scripts import `data/prompt_cache.py`, `evaluate/decode.py` and `evaluate/metrics.py`; the deployed refiner imports `evaluate/timing.py`. Keep those imports stable.

Historical k2 modules remain in the tree while their call sites and behavior are audited. They cannot supply native D0 masks without new D0 calibration. Each production module retains its design page in this folder.
