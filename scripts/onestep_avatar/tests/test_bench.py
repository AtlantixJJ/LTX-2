"""Ordinary mode measurement retains input/output identity and real call counts."""

import pytest
import torch

from scripts.onestep_avatar import bench
from scripts.onestep_avatar.model import common
from scripts.onestep_avatar.tests.test_causal_core import _geometry, _grid, _model
from scripts.onestep_avatar.training.config import BidirectionalSettings, CausalSettings


@pytest.mark.parametrize("mode", ["bidirectional", "causal"])
def test_measure_generation_fixed_inputs_and_complete_counts(mode):
    model = _model()
    grid = _grid(_geometry())
    capture = torch.zeros(1, 28, 8)
    context = torch.zeros(1, 3, 16)
    settings = BidirectionalSettings() if mode == "bidirectional" else CausalSettings()
    record = bench.measure_generation(
        model, context, grid, capture, capture, torch.ones_like(capture),
        device=torch.device("cpu"), repetitions=2, warmup=1,
        mode=mode, mode_settings=settings, guide_mode="d1", schedule=[0.725, 0], seed=4,
        predict_x0=common.denoised_from_velocity_model(model),
    )
    rows = record["repetitions"]
    assert len(rows) == 2
    assert rows[0]["output_sha256"] == rows[1]["output_sha256"]
    assert all(row["call_counts"]["model_calls"] == (1 if mode == "bidirectional" else 6) for row in rows)
    assert all(row["memory_peak_bytes"] is None for row in rows)
    assert all(row["encoded_frames"] == 7 for row in rows)
    assert all(row["covered_rgb_frames"] == 49 and row["generated_rgb_frames"] == 48 for row in rows)
    assert all(row["covered_rgb_frames_per_s"] == pytest.approx(49 / row["elapsed_s"]) for row in rows)
    assert all(row["generated_rgb_frames_per_s"] == pytest.approx(48 / row["elapsed_s"]) for row in rows)
    assert record["elapsed_s"]["minimum"] <= record["elapsed_s"]["median"] <= record["elapsed_s"]["maximum"]


@pytest.mark.parametrize("repetitions,warmup", [(0, 1), (1, -1)])
def test_invalid_measurement_counts_fail_before_model(repetitions, warmup):
    with pytest.raises(ValueError, match="positive repetitions"):
        bench.measure_generation(None, device=torch.device("cpu"), repetitions=repetitions, warmup=warmup)


@pytest.mark.parametrize('mode', ['bidirectional', 'causal'])
def test_explicit_mode_cli_measures_and_saves_ordinary_result(mode, tmp_path, monkeypatch):
    from scripts.onestep_avatar import evaluate
    from pathlib import Path
    import json

    model = _model()
    grid = _grid(_geometry())
    capture = torch.zeros(1, 28, 8)
    context = torch.zeros(1, 3, 16)

    def execute(args, sample_runner):
        output, record = sample_runner(
            model, context, grid, capture, None, torch.ones_like(capture),
            mode=args.mode, mode_settings=args.mode_settings, guide_mode=args.guide_mode,
            schedule=args.schedule, seed=args.seed,
            predict_x0=common.denoised_from_velocity_model(model),
        )
        evaluate.save_case(output, record, args.output)
        return 0

    monkeypatch.setattr(evaluate, 'execute_evaluation', execute)
    assert bench.main([
        '--mode', mode, '--subset', 'unused', '--output', str(tmp_path / 'result'),
        '--guide-mode', 'd0', '--schedule', '0.725', '0', '--repetitions', '2', '--warmup', '1',
    ]) == 0
    record = json.loads((tmp_path / 'result/result.json').read_text())
    assert Path(record['output']['path']).is_file()
    assert record['mode'] == mode
    timings = record['benchmark']
    assert timings['untimed_artifact_calls'] == 1 and timings['warmup'] == 1
    assert len(timings['repetitions']) == 2
    assert all(row['call_counts']['model_calls'] == (1 if mode == 'bidirectional' else 6) for row in timings['repetitions'])


@pytest.mark.parametrize('options', [['--repetitions', '0'], ['--warmup', '-1']])
def test_benchmark_cli_rejects_invalid_counts_before_input_access(options):
    with pytest.raises(SystemExit):
        bench.main(options)
