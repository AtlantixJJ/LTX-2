"""Freeze the twelve full-source test inputs once, without model inference.

The GPU only constructs saved epsilon and native input tokens. Both methods
must reconstruct these same bundles; neither can generate new noise during
checkpoint evaluation. This repeatedly scored split never controls training.
"""

import argparse
import json
from pathlib import Path

from scripts.prune.core.provenance import file_sha256
from scripts.prune.data.checkpoint_pairs import test_sources
from scripts.prune.data.learned_head_study import load_study, study_sigmas
from scripts.prune.learn import checkpoints
from scripts.prune.learn.clipped_reset import is_clipped
from scripts.prune.learn.heads import input_case, numerical_policy, open_current

SIGMAS = [0.9489720463752747, 0.7663628458976746, 0.48016369342803955]
METRIC = "capture_latent_fp32_mse_exclude_first_latent_frame_v1"
AGGREGATION = "equal_six_actors_equal_two_views_equal_three_sigmas_v1"


def pin(path):
    return dict(path=str(Path(path).resolve()), sha256=file_sha256(path))


def freeze(study, current):
    if not is_clipped(study["config"].get("progressive_policy")):
        raise ValueError("fixed-test preparation requires the clipped-reset experiment")
    sources = test_sources(study)
    if [study_sigmas(study)[i] for i in (0, 7, 9)] != SIGMAS:
        raise ValueError("fixed-test exact direct schedules differ")
    if checkpoints.tensor_hash(current.context) != study["text_context"]["tensor_sha256"]:
        raise ValueError("fixed-test actual empty context bytes differ")
    root = Path(study["root"])
    context_path = root / "comparisons/inputs/empty_context.pt"
    checkpoints.publish_tensor(context_path, current.context)
    cases = []
    inputs = []
    for source in sources:
        # The same epsilon is used for all three direct schedules for this view.
        grid, video, c0, capture, epsilon, directory, manifest = input_case(study, current, source, 2042, SIGMAS[0])
        view = Path(source["resolved_dir"])
        row = manifest["videos"][0]
        crop = source["capture_encode_record"]
        source_pin = dict(
            source=source,
            capture=pin(source["capture"]),
            rgb=pin(view / "rgb.mp4"),
            clip_metadata=pin(view.parent.parent / "meta.json"),
            matte=pin(view / "mask.mp4"),
            crop=crop,
            crop_sha256=checkpoints.digest(crop),
            input_manifest=pin(directory / "manifest.json"),
            epsilon=pin(directory / row["artifacts"]["epsilon"]),
            epsilon_tensor_sha256=checkpoints.tensor_hash(epsilon),
            capture_tokens_sha256=checkpoints.tensor_hash(capture),
            clean_first_frame_sha256=checkpoints.tensor_hash(c0),
            fps=source["fps"],
            shape=source["shape"],
            geometry=manifest["geometry"],
            prefix_tokens=grid.tokens_per_latent_frame,
        )
        inputs.append(source_pin)
        for sigma in SIGMAS:
            case = dict(
                actor=source["actor"],
                view=source["relative_dir"],
                sigma=sigma,
                schedule=[sigma, 0.0],
                noise_seed=2042,
                source_pin_sha256=checkpoints.digest(source_pin),
            )
            case["id"] = checkpoints.digest(case)
            cases.append(case)
        del grid, video, c0, capture, epsilon
    result = dict(
        format="clipped_reset_fixed_test_v1",
        study=study["identity"],
        actors=study["splits"]["test_actors"],
        source_ids=[s["relative_dir"] for s in sources],
        inputs=inputs,
        cases=cases,
        context=pin(context_path),
        context_tensor_sha256=checkpoints.tensor_hash(current.context),
        components=study["components"],
        software=study["software"],
        numerics=study["config"]["numerics"],
        code_sha256=study["code"]["sha256"],
        clean_first_latent=True,
        guidance=False,
        prompt="",
        schedules=[[s, 0.0] for s in SIGMAS],
        metric=METRIC,
        aggregation=AGGREGATION,
        training_use=False,
        selection_use=False,
        stopping_use=False,
        repeatedly_scored=True,
        case_count=36,
    )
    path = root / "comparisons/fixed_test_manifest.json"
    validate_manifest(result, study, expected_digest=checkpoints.digest(result))
    checkpoints.publish_json(path, result)
    frozen = dict(**pin(path), content_sha256=checkpoints.digest(result), study=study["identity"])
    checkpoints.publish_json(root / "comparisons/fixed_test_manifest_pin.json", frozen)
    return frozen


def validate_manifest(manifest, study, *, expected_digest, verify_files=True):
    """Reject development-panel caches and any changed source, noise, or GT."""
    sources = test_sources(study)
    if (
        checkpoints.digest(manifest) != expected_digest
        or manifest.get("format") != "clipped_reset_fixed_test_v1"
        or manifest.get("study") != study["identity"]
        or manifest.get("metric") != METRIC
        or manifest.get("aggregation") != AGGREGATION
        or manifest.get("actors") != study["splits"]["test_actors"]
        or manifest.get("source_ids") != [s["relative_dir"] for s in sources]
        or manifest.get("schedules") != [[s, 0.0] for s in SIGMAS]
        or manifest.get("components") != study["components"]
        or manifest.get("software") != study["software"]
        or manifest.get("numerics") != study["config"]["numerics"]
        or manifest.get("context_tensor_sha256") != study["text_context"]["tensor_sha256"]
        or manifest.get("case_count") != 36
        or len(manifest.get("cases", [])) != 36
        or len(manifest.get("inputs", [])) != 12
    ):
        raise ValueError("fixed-test identity, metric, or complete case membership differs")
    expected = []
    for source, item in zip(sources, manifest["inputs"], strict=True):
        if item["source"] != source or item["capture"]["sha256"] != source["capture_latent_sha256"]:
            raise ValueError("fixed-test GT source differs")
        for sigma in SIGMAS:
            case = dict(
                actor=source["actor"],
                view=source["relative_dir"],
                sigma=sigma,
                schedule=[sigma, 0.0],
                noise_seed=2042,
                source_pin_sha256=checkpoints.digest(item),
            )
            case["id"] = checkpoints.digest(case)
            expected.append(case)
        if verify_files:
            for key in ("capture", "rgb", "clip_metadata", "matte", "input_manifest", "epsilon"):
                if file_sha256(item[key]["path"]) != item[key]["sha256"]:
                    raise ValueError(f"fixed-test {key} bytes changed")
    if manifest["cases"] != expected:
        raise ValueError("fixed-test saved case, noise, schedule, or GT pin changed")
    if verify_files and file_sha256(manifest["context"]["path"]) != manifest["context"]["sha256"]:
        raise ValueError("fixed-test context bytes changed")
    return manifest


def load_fixed_manifest(study, *, verify_files=True):
    """Use the immutable publication receipt, never a caller's replacement digest."""
    root = Path(study["root"]) / "comparisons"
    frozen = json.loads((root / "fixed_test_manifest_pin.json").read_text())
    path = root / "fixed_test_manifest.json"
    if (
        frozen["study"] != study["identity"]
        or Path(frozen["path"]).resolve() != path.resolve()
        or file_sha256(path) != frozen["sha256"]
    ):
        raise ValueError("fixed-test manifest publication pin differs")
    manifest = json.loads(path.read_text())
    validate_manifest(manifest, study, expected_digest=frozen["content_sha256"], verify_files=verify_files)
    return manifest, frozen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True, type=Path)
    parser.add_argument("--gpu-id", type=int, default=6, choices=(6,))
    args = parser.parse_args()
    numerical_policy()
    study = load_study(args.study)
    current = open_current(study, args.gpu_id, serial=True)
    result = freeze(study, current)
    validate_manifest(json.loads(Path(result["path"]).read_text()), study, expected_digest=result["content_sha256"])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
