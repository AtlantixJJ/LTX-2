"""Serial native evaluation of exact-step checkpoints on one frozen test split.

This owner reproduces saved historical monitor outputs before test inference.
Every model keeps its saved gates and adapters. Saved case receipts are reused
only after tensor, GT, checkpoint and common-input verification.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from scripts.onestep_avatar.model import common
from scripts.prune.core import ltx_adapter
from scripts.prune.core.provenance import file_sha256
from scripts.prune.data import whole_clip
from scripts.prune.data.checkpoint_pairs import (
    CONTINUATION_ID,
    REMOVAL_ID,
    checkpoint_record,
    historical_study,
    ordinary_draws,
)
from scripts.prune.data.fixed_test_inputs import load_fixed_manifest, pin
from scripts.prune.data.learned_head_study import load_study
from scripts.prune.evaluate import fixed_test_records as records
from scripts.prune.learn import adapters, checkpoints
from scripts.prune.learn.gates import HeadGates
from scripts.prune.learn.heads import numerical_policy, open_current
from scripts.prune.learn.progressive import ProgressivePolicy
from scripts.prune.score import hooks


class FunctionalCheckpoint:
    """One provider follows the loaded policy without editing native connectivity."""

    def __init__(self, model, rank, device):
        self.attached = adapters.attach(model, 42, rank)
        self.device, self.gates = device, None
        model.eval().requires_grad_(False)

    def load(self, state, owner, *, method):
        policy = state.get("progressive")
        clipped = policy is not None and policy.get("format") == "progressive_clip_reset_v2"
        if (
            policy is None
            or state["identity"]["study"] != owner["identity"]
            or state["mask"] is not None
            or (method == "new") != clipped
            or (not clipped and policy.get("format") not in ("progressive_removal_v2", "progressive_zero_one_v1"))
        ):
            raise ValueError("functional test loader received the wrong saved policy or a hardened mask")
        gates = HeadGates(48, 32, activation="clip01_v1" if clipped else "sigmoid_v1").to(self.device)
        gates.alpha.data.copy_(state["alpha"].to(self.device))
        gates.alpha.requires_grad_(False)
        gates.progressive = ProgressivePolicy.from_state(policy, owner["config"]["progressive_policy"])
        gates.phase = "recovery" if clipped and gates.progressive.recovering else "soft"
        adapters.load_state(self.attached, state["adapters"])
        gates.microbatch()
        self.gates = gates

    def provider(self, name):
        return self.gates.provider(name)


def load_pairs(study):
    frozen = study["clipped_reset_protocol"]["checkpoint_pairs"]
    if file_sha256(frozen["path"]) != frozen["sha256"]:
        raise ValueError("frozen checkpoint pairing bytes changed")
    pairs = json.loads(Path(frozen["path"]).read_text())
    owners = []
    for row, identity in zip(pairs["previous_studies"], (REMOVAL_ID, CONTINUATION_ID), strict=True):
        if file_sha256(row["path"]) != row["sha256"]:
            raise ValueError("historical checkpoint owner changed")
        owners.append(historical_study(row["path"], identity))
    draws = ordinary_draws(study)
    if checkpoints.digest(draws) != pairs["draws_sha256"] or pairs["pair_grid"] != list(range(0, 801, 50)):
        raise ValueError("frozen checkpoint pair grid or common draw prefix changed")
    return {p["update"]: p["previous"] for p in pairs["pairs"]}, owners, draws


def checked_reference(owner, reference, draws):
    actual = checkpoint_record(owner, reference["path"], reference["update"], draws)
    if actual != reference:
        raise ValueError("historical checkpoint differs from its frozen pairing record")
    return checkpoints.load(reference["path"])


@torch.no_grad()
def historical_control(study, owner, reference, current, model, loader):
    """Reproduce all saved same-checkpoint monitor cases for one loader format."""
    update = reference["update"]
    if (checkpoints.digest(loader.gates.progressive.state_dict()) != reference["progressive_sha256"]
            or checkpoints.tensor_hash(loader.gates.alpha) != reference["alpha_sha256"]):
        raise ValueError("historical control must execute the actual saved gate policy")
    monitor_path = Path(owner["root"]) / f"checks/learned_progressive/replicate_42/monitor_{update:04d}.json"
    monitor = json.loads(monitor_path.read_text())
    if (
        monitor["study"] != owner["identity"]
        or monitor["update"] != update
        or checkpoints.digest(monitor["progressive"]) != reference["progressive_sha256"]
        or len(monitor["rows"]) != 12
    ):
        raise ValueError("historical loader control requires its actual complete same-checkpoint monitor")
    identity = dict(
        checkpoint=reference,
        monitor=pin(monitor_path),
        code=study["code"]["sha256"],
        environment=pin(Path(study["root"]) / "provenance/environment.json"),
    )
    directory = Path(study["root"]) / f"comparisons/loader_controls/update_{update:04d}"
    output = directory / "acceptance.json"
    previous = json.loads(output.read_text()) if output.exists() else None
    if previous is not None and (previous["identity"] != identity or not previous["passed"]):
        raise ValueError("completed historical loader acceptance changed")
    rows = []
    for row in monitor["rows"]:
        key = hashlib.sha256(row["view"].encode()).hexdigest()
        inputs = Path(owner["root"]) / f"inputs/{key}/seed_2042"
        if file_sha256(inputs / "manifest.json") != row["input_manifest_sha256"]:
            raise ValueError("historical loader input manifest changed")
        manifest = whole_clip.load_manifest(inputs)
        grid, video, c0, native = whole_clip.build_input(
            inputs,
            manifest,
            view=row["source"]["resolved_dir"],
            sigma=row["sigma"],
            current=current,
        )
        if native["artifacts"]["epsilon_sha256"] != row["epsilon_sha256"] or row[
            "context_sha256"
        ] != checkpoints.tensor_hash(current.context):
            raise ValueError("historical loader noise or context differs")
        expected_pin = row["states"]["trained"]
        if file_sha256(expected_pin["path"]) != expected_pin["sha256"]:
            raise ValueError("historical saved monitor prediction changed")
        expected = torch.load(expected_pin["path"], map_location="cpu", weights_only=True)
        case_path = directory / f"{key}_{row['sigma']}.pt"
        if previous is None:
            _, clean = ltx_adapter.velocity_forward(model, video)
            clean = common.with_clean_prefix(clean, c0)
            actual = grid.unpatchify_block(clean, grid.latent_frames).cpu()
            checkpoints.publish_tensor(case_path, actual)
        else:
            actual = torch.load(case_path, map_location="cpu", weights_only=True)
        if actual.shape != expected.shape or not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
            raise ValueError("historical loader control has nonfinite or different-shaped outputs")
        delta = actual.float() - expected.float()
        maximum = float(delta.abs().max())
        if maximum > 0.02:
            raise ValueError(f"historical loader reproduction exceeds frozen tolerance: {maximum}")
        rows.append(
            dict(
                view=row["view"],
                sigma=row["sigma"],
                reference=expected_pin,
                input_manifest_sha256=row["input_manifest_sha256"],
                output=pin(case_path),
                max_abs=maximum,
                relative_l2=float(delta.norm() / expected.float().norm().clamp_min(1e-12)),
            )
        )
    result = dict(identity=identity, rows=rows, max_abs_tolerance=0.02, passed=True)
    checkpoints.publish_json(output, result)
    return pin(output)


@torch.no_grad()
def score_checkpoint(study, record, current, model, manifest, frozen, directory):
    sources = {v["source"]["relative_dir"]: v for v in manifest["inputs"]}
    rows = []
    for case in manifest["cases"]:
        source = sources[case["view"]]
        identity = records.prediction_identity(record, case, frozen)
        output = directory / f"{case['id']}.json"
        if not output.exists():
            inputs = Path(source["input_manifest"]["path"]).parent
            native = whole_clip.load_manifest(inputs)
            grid, video, c0, _ = whole_clip.build_input(
                inputs,
                native,
                view=source["source"]["resolved_dir"],
                sigma=case["sigma"],
                current=current,
            )
            epsilon = whole_clip.load_epsilon(inputs, native["videos"][0])
            if checkpoints.tensor_hash(epsilon) != source["epsilon_tensor_sha256"]:
                raise ValueError("fixed-test actual epsilon differs from its frozen tensor hash")
            from scripts.onestep_avatar.corpus.dataset import load_training_master

            capture, _ = load_training_master(Path(source["capture"]["path"]))
            tokens = grid.patchify(capture.unsqueeze(0).to(current.device))
            if (
                checkpoints.tensor_hash(tokens) != source["capture_tokens_sha256"]
                or checkpoints.tensor_hash(c0) != source["clean_first_frame_sha256"]
                or grid.tokens_per_latent_frame != source["prefix_tokens"]
            ):
                raise ValueError("fixed-test actual GT tokens or clean first frame differ")
            _, clean = ltx_adapter.velocity_forward(model, video)
            clean = common.with_clean_prefix(clean, c0)
            prediction = grid.unpatchify_block(clean, grid.latent_frames).cpu()
            tensor_path = output.with_suffix(".pt")
            checkpoints.publish_tensor(tensor_path, prediction)
            checkpoints.publish_json(
                output,
                dict(
                    format="clipped_reset_test_prediction_v1",
                    identity=identity,
                    prediction=dict(**pin(tensor_path), tensor_sha256=checkpoints.tensor_hash(prediction)),
                    capture_mse=records.capture_mse(prediction, capture),
                ),
            )
        rows.append(records.read_case(output, identity, source))
    score = dict(
        **records.aggregate(manifest, rows), checkpoint_sha256=record["sha256"], fixed_test_sha256=frozen["sha256"]
    )
    checkpoints.publish_json(directory / "score.json", score)
    return score


def evaluate(study, steps, gpu_id):
    references, owners, draws = load_pairs(study)
    manifest, frozen = load_fixed_manifest(study)
    if not steps or steps != sorted(set(steps)) or any(t not in references for t in steps):
        raise ValueError("fixed-test steps must be unique ordered members of the frozen 50-update grid")
    root = Path(study["root"])
    current = open_current(study, gpu_id, serial=True)
    results = []
    with current.transformer(Path(study["components"]["transformer"]["path"])) as model:
        loader = FunctionalCheckpoint(model, study["config"]["rank"], current.device)
        live = hooks.attach_head_provider(model, loader.provider)
        try:
            controls = []
            # Step zero has no historical monitor. Validate the exact loader on
            # saved step-50 and step-550 checkpoints before scoring any test case.
            for update, owner in ((50, owners[0]), (550, owners[1])):
                reference = references[update]
                state = checked_reference(owner, reference, draws)
                loader.load(state, owner, method="previous")
                controls.append(historical_control(study, owner, reference, current, model, loader))
            for update in steps:
                scores, pins = {}, {}
                for method in ("previous", "new"):
                    if method == "previous":
                        owner = owners[0] if update <= 500 else owners[1]
                        record = references[update]
                        state = checked_reference(owner, record, draws)
                    else:
                        owner = study
                        path = root / f"training/learned_progressive/replicate_42/checkpoints/update_{update:04d}.pt"
                        record = checkpoint_record(study, path, update, draws)
                        state = checkpoints.load(path)
                    loader.load(state, owner, method=method)
                    pins[method] = record
                    directory = root / f"comparisons/predictions/{method}/update_{update:04d}"
                    scores[method] = score_checkpoint(study, record, current, model, manifest, frozen, directory)
                pair = records.paired_step(
                    pins["previous"], pins["new"], scores["previous"], scores["new"], frozen, manifest
                )
                pair["new_terminal_update"] = state["progressive"].get("finalization_start")
                pair["score_receipts"] = {
                    method: pin(root / f"comparisons/predictions/{method}/update_{update:04d}/score.json")
                    for method in ("previous", "new")
                }
                pair["historical_loader_controls"] = controls
                checkpoints.publish_json(root / f"comparisons/steps/update_{update:04d}.json", pair)
                results.append(pair)
        finally:
            live.detach_all()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True, type=Path)
    parser.add_argument("--steps", required=True, type=int, nargs="+")
    parser.add_argument("--gpu-id", type=int, default=6, choices=(6,))
    args = parser.parse_args()
    numerical_policy()
    torch.set_num_threads(4)
    study = load_study(args.study)
    results = evaluate(study, args.steps, args.gpu_id)
    print(json.dumps(dict(completed_steps=[r["update"] for r in results]), indent=2))


if __name__ == "__main__":
    main()
