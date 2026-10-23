"""Monitor comparison coverage and pairing must be exact across recipes."""

import copy

import pytest

from scripts.prune.evaluate.joint_monitor_media import paired_rows, presentation


def fixture():
    rows = [
        dict(
            actor=str(actor),
            view=f"view_{actor}",
            sigma=sigma,
            input_sha256=f"input_{actor}",
            noise_sha256=f"noise_{actor}",
        )
        for actor in range(4)
        for sigma in (0.4, 0.7, 0.9)
    ]
    return {arm: dict(cases=copy.deepcopy(rows)) for arm in ("native", "A", "B", "C")}


def test_complete_paired_monitor_cases():
    rows = paired_rows(fixture())
    assert len(rows["C"]) == 12


def test_exact_monitor_labels_fit_before_decoding():
    titles, question = presentation(1000)
    assert [role for role, _ in titles] == ["native", "A", "B", "C"]
    assert "1000" in question


@pytest.mark.parametrize("invalid", ["missing", "duplicate", "noise", "input", "view"])
def test_changed_monitor_pairing_is_rejected(invalid):
    monitors = fixture()
    if invalid == "missing":
        monitors["C"]["cases"].pop()
    elif invalid == "duplicate":
        monitors["C"]["cases"][-1] = monitors["C"]["cases"][0]
    elif invalid == "view":
        monitors["C"]["cases"][0]["view"] = "different"
    else:
        monitors["C"]["cases"][0][invalid + "_sha256"] = "different"
    with pytest.raises(ValueError):
        paired_rows(monitors)
