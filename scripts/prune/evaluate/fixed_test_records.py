"""Strict saved prediction scores and same-step sparsity/error comparisons.

Model execution belongs to fixed_test. This module opens only saved tensors
and receipts. It never fills missing cases, changes a mask, or averages a
partial checkpoint. All means use the frozen actor/view/sigma hierarchy.
"""

import json
import math
from pathlib import Path

import torch

from scripts.onestep_avatar.corpus.dataset import load_training_master
from scripts.prune.core.provenance import file_sha256
from scripts.prune.data.fixed_test_inputs import AGGREGATION, METRIC, SIGMAS
from scripts.prune.learn import checkpoints


def capture_mse(prediction, capture):
    """FP32 generated-frame reduction from the complete saved native latent."""
    if (
        capture.ndim != 4
        or prediction.shape != (1, *capture.shape)
        or capture.shape[1] < 2
        or not torch.isfinite(prediction).all()
        or not torch.isfinite(capture).all()
        or not torch.equal(prediction[0, :, :1], capture[:, :1])
    ):
        raise ValueError("fixed-test MSE requires finite full latents and the exact clean first frame")
    mse = float((prediction[0, :, 1:].float() - capture[:, 1:].float()).square().mean())
    if not math.isfinite(mse):
        raise ValueError("nonfinite fixed-test FP32 reduction")
    return mse


def prediction_identity(checkpoint, case, frozen):
    return dict(
        checkpoint_sha256=checkpoint["sha256"],
        checkpoint_identity=checkpoint["identity"],
        progressive_sha256=checkpoint["progressive_sha256"],
        alpha_sha256=checkpoint["alpha_sha256"],
        adapter_hashes_sha256=checkpoints.digest(checkpoint["adapter_hashes"]),
        update=checkpoint["update"],
        phase=checkpoint["phase"],
        draw_cursor=checkpoint["draw_cursor"],
        fixed_test_sha256=frozen["sha256"],
        fixed_test_content_sha256=frozen["content_sha256"],
        case=case,
        metric=METRIC,
        aggregation=AGGREGATION,
    )


def read_case(path, expected_identity, source):
    record = json.loads(Path(path).read_text())
    if record.get("format") != "clipped_reset_test_prediction_v1" or record.get("identity") != expected_identity:
        raise ValueError("cached fixed-test checkpoint, gate state, or case identity differs")
    pin = record["prediction"]
    if file_sha256(pin["path"]) != pin["sha256"]:
        raise ValueError("cached fixed-test prediction bytes changed")
    prediction = torch.load(pin["path"], map_location="cpu", weights_only=True)
    if checkpoints.tensor_hash(prediction) != pin["tensor_sha256"]:
        raise ValueError("cached fixed-test prediction tensor changed")
    if file_sha256(source["capture"]["path"]) != source["capture"]["sha256"]:
        raise ValueError("fixed-test capture GT bytes changed")
    capture, _ = load_training_master(Path(source["capture"]["path"]))
    if list(capture.shape) != source["shape"]:
        raise ValueError("fixed-test prediction uses a crop or a different source shape")
    mse = capture_mse(prediction, capture)
    if record["capture_mse"] != mse:
        raise ValueError("cached fixed-test score differs from the saved prediction and GT")
    return dict(
        case=expected_identity["case"],
        capture_mse=mse,
        receipt_path=str(Path(path).resolve()),
        receipt_sha256=file_sha256(path),
        prediction=pin,
    )


def aggregate(manifest, rows):
    expected = {c["id"]: c for c in manifest["cases"]}
    if len(expected) != 36 or len(rows) != 36 or {r["case"]["id"] for r in rows} != set(expected):
        raise ValueError("fixed-test score requires all 36 unique cases")
    by_id = {r["case"]["id"]: r for r in rows}
    if any(
        r["case"] != expected[r["case"]["id"]] or not math.isfinite(r["capture_mse"]) or r["capture_mse"] < 0
        for r in rows
    ):
        raise ValueError("fixed-test case identity or finite MSE differs")
    actors = manifest["actors"]
    if len(actors) != 6 or len(set(actors)) != 6:
        raise ValueError("fixed-test score requires six distinct actors")
    actor_means, view_means, sigma_means = {}, {}, {}
    for actor in actors:
        views = sorted({c["view"] for c in expected.values() if c["actor"] == actor})
        if len(views) != 2:
            raise ValueError("fixed-test score requires two views for every actor")
        for view in views:
            cases = [c for c in expected.values() if c["actor"] == actor and c["view"] == view]
            if sorted(c["sigma"] for c in cases) != sorted(SIGMAS):
                raise ValueError("fixed-test score requires three exact sigmas per actor/view")
            view_means[view] = math.fsum(by_id[c["id"]]["capture_mse"] for c in cases) / 3
        actor_means[actor] = math.fsum(view_means[v] for v in views) / 2
    for sigma in SIGMAS:
        # Equal actors, then equal views, for each sigma.
        sigma_means[str(sigma)] = (
            math.fsum(
                math.fsum(
                    by_id[c["id"]]["capture_mse"]
                    for c in expected.values()
                    if c["actor"] == actor and c["sigma"] == sigma
                )
                / 2
                for actor in actors
            )
            / 6
        )
    return dict(
        metric=METRIC,
        aggregation=AGGREGATION,
        case_count=36,
        mean_capture_mse=math.fsum(actor_means.values()) / 6,
        actor_means=actor_means,
        view_means=view_means,
        sigma_means=sigma_means,
        cases=[by_id[c["id"]] for c in manifest["cases"]],
    )


def paired_step(previous, new, previous_score, new_score, frozen, manifest):
    if (
        previous["update"] != new["update"]
        or previous["draw_cursor"] != new["draw_cursor"]
        or previous["consumed_draws_sha256"] != new["consumed_draws_sha256"]
    ):
        raise ValueError("fixed-test pair has unequal global steps or consumed draw prefixes")
    for score in (previous_score, new_score):
        recalculated = aggregate(manifest, score.get("cases", []))
        if any(score.get(k) != v for k, v in recalculated.items()):
            raise ValueError("fixed-test pair score differs from its complete case aggregation")
        if (
            score.get("fixed_test_sha256") != frozen["sha256"]
            or score.get("case_count") != 36
            or score.get("metric") != METRIC
            or score.get("aggregation") != AGGREGATION
            or not math.isfinite(score["mean_capture_mse"])
        ):
            raise ValueError("fixed-test pair has incomplete or changed test scores")
    for checkpoint, score in ((previous, previous_score), (new, new_score)):
        if score["checkpoint_sha256"] != checkpoint["sha256"]:
            raise ValueError("fixed-test score belongs to a different checkpoint")
    before, after = previous_score["mean_capture_mse"], new_score["mean_capture_mse"]

    def values(checkpoint, score):
        return dict(
            checkpoint=checkpoint,
            removed=checkpoint["removed"],
            retained=checkpoint["retained"],
            removed_fraction=checkpoint["removed"] / 1536,
            mean_capture_mse=score["mean_capture_mse"],
            actor_means=score["actor_means"],
            sigma_means=score["sigma_means"],
        )

    return dict(
        update=new["update"],
        draw_cursor=new["draw_cursor"],
        fixed_test_sha256=frozen["sha256"],
        previous=values(previous, previous_score),
        new=values(new, new_score),
        removed_difference=new["removed"] - previous["removed"],
        mse_difference=after - before,
        mse_relative_change=(after - before) / before if before else None,
        relative_change_defined=before != 0,
    )
