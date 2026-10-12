"""Membership pins and exact official-grid controls without model loading."""
import pytest

from scripts.prune.data.head_score_dataset import SIGMAS, audit, choose_cases, official_schedule


def test_heldout_cases_ignore_input_order():
    sources = [{'actor': actor, 'split': split, 'relative_dir': f'{actor}/view{view}'}
               for split, actors in [('train', ['a']), ('held_out', ['b', 'c', 'd', 'e'])]
               for actor in actors for view in range(3)]
    cases = choose_cases(sources)
    assert cases == choose_cases(sources[::-1])
    assert len({s['actor'] for s in cases}) == 3
    assert all(s['split'] == 'held_out' for s in cases)


def test_changed_membership_rejected_before_outputs(tmp_path):
    path = tmp_path / 'membership.json'
    path.write_text('{}')
    with pytest.raises(ValueError, match='pins changed'):
        audit(path, tmp_path / 'output')
    assert not (tmp_path / 'output').exists()


def test_official_schedule_preserves_tail(monkeypatch):
    from types import SimpleNamespace

    from scripts.prune.data import head_score_dataset
    monkeypatch.setattr(head_score_dataset, 'detect_params', lambda _: SimpleNamespace(num_inference_steps=30))
    schedule = official_schedule('fixture')
    assert schedule['sigmas'] == SIGMAS
    assert schedule['eligible_tail'] == schedule['original_grid'][8:]
    assert schedule['stock_step_indices'] == [8, 11, 14, 17, 20, 23, 26, 29]
    assert schedule['selected_tail_indices'] == [0, 3, 6, 9, 12, 15, 18, 21]
    assert len(schedule['eligible_tail']) == 23
    assert schedule['evaluation_schedules'] == [[s, 0.] for s in SIGMAS]
