from __future__ import annotations

import json

import pytest

from scripts.prune.core import preflight, provenance, refine_core, refine_task


def test_sweep_inputs_bind_checkpoint_geometry_and_source(tmp_path, model):
    fingerprint = provenance.checkpoint_fingerprint(model.paths.transformer())
    geometry = refine_core.WindowGeometry(refine_task.WINDOW_FRAMES, refine_task.OVERLAP_FRAMES,
                                          model.scale_factors).as_dict()
    parity = {"pass": True, "provenance": {"model_key": model.key, "transformer_fingerprint": fingerprint},
              "geometry": geometry, "method_sources": provenance.method_source_hashes()}
    index = {"format": 2, "provenance": {"transformer_fingerprint": fingerprint},
             "records": [{"split": "calibration"}, {"split": "held_out"}]}
    baseline = {"provenance": {"transformer_fingerprint": fingerprint}, "geometry": geometry,
                "method_sources": provenance.method_source_hashes(), "T0": {"records": [1]}, "T2": {"windows": 2}}
    paths = [tmp_path / name for name in ("parity.json", "index.json", "baseline.json")]
    for path, payload in zip(paths, (parity, index, baseline), strict=True):
        path.write_text(json.dumps(payload))
    assert preflight.check_sweep_inputs(model, parity_path=paths[0], index_path=paths[1],
                                        baseline_path=paths[2])["calibration_records"] == 1
    parity["method_sources"]["scripts/prune/core/refine_core.py"] = "stale"
    paths[0].write_text(json.dumps(parity))
    with pytest.raises(ValueError, match="stale"):
        preflight.check_sweep_inputs(model, parity_path=paths[0], index_path=paths[1],
                                     baseline_path=paths[2])
