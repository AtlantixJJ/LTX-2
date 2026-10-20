"""CPU preparation of complete, actor-disjoint learned-head experiments."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from scripts.onestep_avatar import media
from scripts.onestep_avatar.corpus.dataset import CaptureManifest, ClipStore
from scripts.onestep_avatar.corpus.subset import load_membership
from scripts.prune.core import model_registry
from scripts.prune.core.provenance import file_sha256
from scripts.prune.core.records import digest, publish, publish_json, tensor_hash
from scripts.prune.data import foreground, prompt_cache, training_windows, whole_clip
from scripts.prune.data.corpus_pins import CSV_SHA, JSON_SHA, SEMANTIC_SHA, official_schedule
from scripts.prune.data.study_io import code_identity, read_frozen, verify_files
from scripts.prune.learn.protocol import draws_per_update
from scripts.prune.learn.protocol import validate as validate_protocol

SALT = "learned-heads-dev8-v1"


def report_resources(output, plan, *, rank=16):
    """Install saved-only report tools and proposed questions before execution."""
    templates = Path(__file__).resolve().parents[1] / "configs/learned_heads_report"
    for template in templates.glob("*.py.txt"):
        target = output / "code" / template.name.removesuffix(".txt")
        publish(target, lambda p, template=template: p.write_bytes(template.read_bytes()))
        doc = output / "code/doc" / template.name.removesuffix(".py.txt")
        publish(
            doc.with_suffix(".md"),
            lambda p, target=target: p.write_text(
                f"# {target.name}\n\n## Objective\n\nRead saved experiment evidence only.\n\n## Usage\n\n"
                f"Run `conda run -n ltx python {target}` on CPU.\n\n## Data flow\n\n"
                "Declared narrative and saved artifacts enter the report reader. Missing evidence fails.\n\n"
                "## Organization logic\n\nValidate declared files before rendering or plotting. Never start models or repair missing runs.\n\n"
                "## Invariants\n\nKeep question, figure, source hash and media roles consistent.\n\n"
                "## Gotchas\n\nSaved schema validation cannot prove scientific acceptance or visual readability.\n\n"
                "## Tests\n\nCheck missing evidence with transformer and decoder imports blocked.\n"
            ),
        )
    text = plan.read_text()
    questions = []
    for match in re.finditer(r"^### (R\d+)\. ([^\n]+)\n(.*?)(?=^### |^## |\Z)", text, re.M | re.S):
        if int(match[1][1:]) <= 17:
            questions.append({"id": match[1], "title": match[2], "design": match[3].strip(), "status": "planned"})
    publish_json(
        output / "configs/narrative.json",
        {
            "problem": "Choose fewer executable heads while preserving people and motion in whole-clip D0 predictions.",
            "terms": [
                "Teacher: unchanged LTX-2.5 dev without recovery adapters.",
                "D0: one direct prediction from noised recorded capture.",
                "c0: supplied clean first latent frame.",
                "Soft gate: sigmoid of a learned scalar logit.",
                "Binary refinement: legal hard forwards with sigmoid straight-through gradients.",
                "Energy arm: fixed projected-energy ranking with matched adapter training.",
                f"Joint arm: learned gates and rank-{rank} V/output adapters.",
                "VAE GT: recorded capture latent decoded by the pinned VAE.",
                "RGB GT: recorded pixels after their original crop and white matte.",
                "Sigma: the saved FP32 input noise level.",
                "Block: one transformer layer.",
                "Fresh recovery: new zero-delta adapters and optimizer moments.",
            ],
            "setup": "White capture, empty text, full bidirectional attention, complete clips, clean c0 and exact independent [sigma,0] schedules.",
            "questions": questions,
            "required_files": [],
            "figures": [],
            "videos": [],
        },
    )


def order(role, value):
    return hashlib.sha256(f"{SALT}:{role}:{value}".encode()).hexdigest(), value


def normalized_actor(actor):
    value = str(actor).removeprefix("dna_actor:").strip()
    return str(int(value)) if value.isdecimal() else value


def view_pair(views):
    chosen = []
    for length in sorted({s["n_latent_frames"] for s in views}):
        chosen.append(
            min(
                (s for s in views if s["n_latent_frames"] == length), key=lambda s: order("learning", s["relative_dir"])
            )
        )
    remaining = sorted((s for s in views if s not in chosen), key=lambda s: order("learning", s["relative_dir"]))
    return (chosen + remaining)[:2]


def select_splits(sources, seen, *, validation_actors=20, learning_views=256, test_actors=6):
    grouped = defaultdict(list)
    for source in sources:
        actor = normalized_actor(source["actor"])
        grouped[actor].append({**source, "actor": actor})
    train = {s["actor"] for group in grouped.values() for s in group if s["split"] == "train"}
    held = {s["actor"] for group in grouped.values() for s in group if s["split"] == "held_out"}
    if train & held or len(train) <= validation_actors:
        raise ValueError("invalid actor split authority")
    validation = sorted(train, key=lambda a: order("validation", a))[:validation_actors]
    learning = sorted(
        train - set(validation), key=lambda a: (min(order("learning", s["relative_dir"]) for s in grouped[a]), a)
    )
    pools = {a: sorted(grouped[a], key=lambda s: order("learning", s["relative_dir"])) for a in learning}
    selected = []
    used_lengths = defaultdict(set)
    while len(selected) < learning_views:
        progressed = False
        for actor in learning:
            choices = pools[actor]
            if not choices:
                continue
            candidate = next((s for s in choices if s["n_latent_frames"] not in used_lengths[actor]), choices[0])
            choices.remove(candidate)
            selected.append(candidate)
            used_lengths[actor].add(candidate["n_latent_frames"])
            progressed = True
            if len(selected) == learning_views:
                break
        if not progressed:
            raise ValueError("insufficient complete learning views")
    blind = sorted(held - {normalized_actor(a) for a in seen}, key=lambda a: order("test", a))[:test_actors]
    return {
        "learning": selected,
        "validation": [s for a in validation for s in view_pair(grouped[a])],
        "test": [s for a in blind for s in view_pair(grouped[a])],
        "learning_actors": sorted(train - set(validation)),
        "validation_actors": validation,
        "test_actors": blind,
        "test_label": "blind" if len(blind) == test_actors else "held_out_generalization",
    }


def assign_buckets(sources, stats):
    motion = np.array([stats[s["relative_dir"]]["motion_amount"] for s in sources])
    area = np.array([stats[s["relative_dir"]]["person_area"] for s in sources])
    cuts = np.quantile(motion, [1 / 3, 2 / 3]).tolist()
    median = float(np.median(area))
    buckets = {
        s["relative_dir"]: int(np.searchsorted(cuts, motion[i], side="right")) * 2 + int(area[i] >= median)
        for i, s in enumerate(sources)
    }
    return {
        "motion_tertiles": cuts,
        "person_area_median": median,
        "buckets": buckets,
        "counts": {str(i): sum(b == i for b in buckets.values()) for i in range(6)},
    }


def study_sigmas(study):
    """Read the exact frozen learned-study grid; historical scoring keeps its own grid."""
    return study["config"]["sigmas"]


def monitor_sigma_indices(study):
    """Keep quick validation levels explicit when the optimizer grid expands."""
    count = len(study_sigmas(study))
    indices = study["config"].get("monitor_sigma_indices", [0, 4, count - 1])
    if len(indices) != 3 or len(set(indices)) != 3 or any(type(i) is not int or not 0 <= i < count for i in indices):
        raise ValueError("monitor sigma indices must name three distinct training levels")
    return indices


def validate_sigma_grid(config, schedule):
    """Accept exact positive dev scheduler levels, with complete draw coverage."""
    grid = config["sigmas"]
    if (
        not grid
        or any(type(s) not in (int, float) or not 0 < s <= 0.95 or s not in schedule["original_grid"] for s in grid)
        or grid != sorted(set(grid), reverse=True)
    ):
        raise ValueError("training sigmas must be distinct descending exact dev levels in (0, 0.95]")
    cycle = config["sigma_indices_cycle"]
    if len(cycle) != len(grid) or any(type(i) is not int for i in cycle) or sorted(cycle) != list(range(len(grid))):
        raise ValueError("frozen sigma cycle does not cover the declared grid once")
    monitor_sigma_indices({"config": config})


def draw_list(sources, seed, *, buckets=None, updates=1500, sigma_indices=None, draws_per_update=2):
    cycle = list(sigma_indices) if sigma_indices is not None else [0, 7, 1, 6, 2, 5, 3, 4]
    if not cycle or set(cycle) != set(range(len(cycle))) or any(type(i) is not int for i in cycle):
        raise ValueError("sigma cycle must contain each grid index exactly once")
    rng = random.Random(seed)
    groups = defaultdict(lambda: defaultdict(list))
    if buckets is not None:
        for source in sources:
            groups[buckets[source["relative_dir"]]][source["actor"]].append(source)
        if set(groups) != set(range(6)):
            raise ValueError("balanced arm requires all six motion/size buckets")
    draws = []
    if type(draws_per_update) is not int or draws_per_update < 1:
        raise ValueError("draws per update must be a positive integer")
    for j in range(updates * draws_per_update):
        if buckets is None:
            source = sources[rng.randrange(len(sources))]
        else:
            bucket = groups[(j // len(cycle)) % 6]
            actor = rng.choice(sorted(bucket))
            source = rng.choice(bucket[actor])
        draws.append(
            {
                "microbatch": j,
                "view": source["relative_dir"],
                "sigma_index": cycle[j % len(cycle)],
                "noise_seed": 1_000_000 + 10_000 * seed + j,
            }
        )
    return draws


def declared_evidence(path):
    """Read only explicitly declared, content-pinned prior evidence for actor exclusions."""
    path = Path(path).resolve()
    declaration_bytes = path.read_bytes()
    declaration = json.loads(declaration_bytes)
    if declaration.get("format") != "prune_seen_evidence_v1" or not isinstance(declaration.get("files"), list):
        raise ValueError("expected a declared prune_seen_evidence_v1 file list")
    actors, pins, seen = [], [{"path": str(path), "sha256": hashlib.sha256(declaration_bytes).hexdigest()}], set()
    for item in declaration["files"]:
        source = (path.parent / item["path"]).resolve()
        if source in seen or source.suffix not in (".md", ".json"):
            raise ValueError("duplicate or unsupported declared evidence")
        raw = source.read_bytes()
        if hashlib.sha256(raw).hexdigest() != item["sha256"]:
            raise ValueError(f"frozen input changed: {source}")
        seen.add(source)
        if source.suffix == ".json":
            value = json.loads(raw)
            rows = [*value.get("rows", []), *value.get("cases", [])]
            actors += [normalized_actor(row["actor"]) for row in rows if "actor" in row]
        else:
            text = raw.decode("utf-8")
            found = re.findall(r"Part_\d+/0*(\d+)_\d+/views/", text)
            found += re.findall(r'(?i)\bactor\s*[:=]\s*["\x27]?(\d+)', text)
            actors += [normalized_actor(actor) for actor in found]
        pins.append({"path": str(source), "sha256": item["sha256"]})
    return sorted(set(actors)), pins


def load_study(path, *, verify_components=True):
    study = read_frozen(path, formats=("learned_head_study_v1",))
    verify_files([study["plan"], *study["files"]])
    if verify_components:
        verify_files(study["components"].values())
    return study


def prepare_view(args):
    """Independent CPU audit; cache only fully checked source and GT statistics."""
    membership_path, source, is_learning, output, region_tracks = args[:5]
    audit_foreground = args[5] if len(args) > 5 else True
    import cv2

    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    membership = load_membership(Path(membership_path))
    store, crop = ClipStore(membership), CaptureManifest.load(Path(membership["corpus_root"]))
    output = Path(output)
    key = hashlib.sha256(source["relative_dir"].encode()).hexdigest()
    receipt_path = output / "provenance/source_audits" / f"{key}.json"
    reader_identity = digest(
        {str(p): file_sha256(p) for p in (Path(__file__), Path(foreground.__file__), Path(media.__file__))}
    )
    if receipt_path.exists():
        cached = json.loads(receipt_path.read_text())
        if (
            cached.get("reader_identity") != reader_identity
            or cached["source"]["capture_encode_record"] != source["capture_encode_record"]
        ):
            raise ValueError("cached source audit code or source changed; use a new preparation directory")
        for item in cached["files"]:
            if file_sha256(item["path"]) != item["sha256"]:
                raise ValueError("cached source audit input changed")
        return cached
    source_files, statistics = [], None
    view = (Path(membership["corpus_root"]) / source["relative_dir"]).resolve()
    identity = whole_clip.actor_identity(str(view))
    if identity != "dna_actor:" + source["actor"]:
        raise ValueError("actor metadata or alias differs")
    capture = view / "ltx_vae_latent_white.pt"
    for p, expected in (
        (capture, source["capture_latent_sha256"]),
        (view / "rgb.mp4", source["rgb_sha256"]),
        (view / "../../meta.json", source["clip_metadata_sha256"]),
    ):
        if file_sha256(p) != expected:
            raise ValueError(f"changed source: {p}")
        source_files.append({"path": str(p.resolve()), "sha256": expected})
    clip = store.load(source["relative_dir"], require_guide=False)
    if list(clip.z_y.shape) != source["shape"] or clip.fps != 30 or clip.z_y.dtype != torch.bfloat16:
        raise ValueError("full capture geometry changed")
    if list(crop.box_for(view)) != source["capture_encode_record"]["box_xyxy"]:
        raise ValueError("recorded crop differs from canonical crop authority")
    verified_source = {**source, "resolved_dir": str(view), "capture": str(capture)}
    if audit_foreground:
        matte, _, mask_identity = foreground.person_coverage(view, source["capture_encode_record"])
        for p in (view / "mask.mp4", view / "capture_mask_crop.mp4"):
            source_files.append({"path": str(p), "sha256": file_sha256(p)})
        key = hashlib.sha256(source["relative_dir"].encode()).hexdigest()
        publish_json(output / "regions" / key / "canonical_mask.json", mask_identity)
        if region_tracks["status"] == "verified":
            tracks = region_tracks["views"][source["relative_dir"]]
            for name in ("face", "left_hand", "right_hand"):
                _, coverage_record = foreground.region_coverage(
                    tracks[name], matte, source["capture_encode_record"], schema=tracks["schema"]
                )
                publish_json(output / "regions" / key / f"{name}.json", coverage_record)
        if is_learning:
            rgb = media.recorded_capture_rgb(
                {**source, "box_xyxy": source["capture_encode_record"]["box_xyxy"]},
                Path(membership["corpus_root"]),
                "white",
                source["n_latent_frames"],
            )
            statistics = foreground.gt_statistics(rgb, matte)
    print(f"verified {source['relative_dir']}", flush=True)
    result = {
        "source": verified_source,
        "files": source_files,
        "statistics": statistics,
        "reader_identity": reader_identity,
        "foreground_audit": "verified" if audit_foreground else "deferred",
    }
    publish_json(output / "provenance/source_audits" / f"{key}.json", result)
    return result


def prepare(config_path, plan, output, seen_path=None, region_path=None, *, uniform_f18=False, evidence_list=None):
    if evidence_list is None:
        raise ValueError("fresh preparation requires an explicit --evidence-list; no experiment-tree scan")
    prior_actors, evidence_pins = declared_evidence(evidence_list)
    config = json.loads(config_path.read_text())
    if uniform_f18:
        config["uniform_f18"] = True
    crop_training = training_windows.enabled(config)
    validate_protocol(config)
    if (
        config["target"] != "epsilon_minus_capture"
        or config["alpha_init"] != 3
        or config["temperature"] != 1
        or [config[k] for k in ("soft_updates", "binary_updates", "compact_updates")] != [700, 300, 500]
        or config["variant"] != "dev"
    ):
        raise ValueError("preset changes require an explicit protocol amendment")
    cycle = config["sigma_indices_cycle"]
    workspace = Path(__file__).resolve().parents[4]
    schedule = official_schedule(workspace / config["components"]["transformer"])
    validate_sigma_grid(config, schedule)
    alias = Path("/home/jianjinx/data2/VideoDiffusionModels")
    if alias.exists() and alias.resolve() != workspace.resolve():
        raise ValueError("workspace alias differs")
    membership_path = workspace / config["membership"]
    if file_sha256(membership_path) != JSON_SHA or file_sha256(membership_path.with_suffix(".csv")) != CSV_SHA:
        raise ValueError("membership source pins changed")
    membership = load_membership(membership_path)
    if membership["sha256"] != SEMANTIC_SHA:
        raise ValueError("membership semantic identity changed")
    seen = {"actors": config["known_seen_actors"], "evidence": ["known inspected IDs in the frozen protocol"]}
    if seen_path:
        extra = json.loads(seen_path.read_text())
        seen["actors"] = sorted(set(seen["actors"]) | set(extra["actors"]))
        seen["evidence"] += extra["evidence"]
    audit_foreground = config.get("foreground_preparation", "full") != "deferred"
    region_tracks = {"status": "not_verified", "views": {}}
    if not audit_foreground and region_path:
        raise ValueError("reviewed regional preparation requires foreground replay")
    if region_path:
        region_tracks = json.loads(region_path.read_text())
        if region_tracks.get("status") != "verified" or not region_tracks.get("views"):
            raise ValueError("region input requires reviewed GT tracks for the selected views")
    seen["actors"] += prior_actors
    seen["actors"] = sorted(set(seen["actors"]))
    seen["evidence"] += evidence_pins
    splits = select_splits(
        membership["sources"],
        seen["actors"],
        validation_actors=config["validation_actors"],
        learning_views=config["learning_views"],
        test_actors=config["test_actors"],
    )
    output = output.resolve()
    if crop_training and any(s["n_latent_frames"] < 18 for s in splits["learning"]):
        raise ValueError("uniform F18 learning sources must have at least 18 latent frames")
    effective_config = output / "provenance/effective_config.json" if uniform_f18 else config_path
    if uniform_f18:
        publish_json(effective_config, config)
    report_resources(output, plan, rank=config["rank"])
    frozen_plan = output / "provenance/plan.md"
    publish(frozen_plan, lambda p: p.write_bytes(plan.read_bytes()))
    publish_json(output / "provenance/seen_actors.json", seen)
    publish_json(output / "provenance/splits.json", splits)
    from concurrent.futures import ProcessPoolExecutor

    stats, verified, files = {}, {}, list(evidence_pins)
    if uniform_f18:
        files.append({"path": str(effective_config), "sha256": file_sha256(effective_config)})
    selected = splits["learning"] + splits["validation"] + splits["test"]
    learning_ids = {s["relative_dir"] for s in splits["learning"]}
    tasks = [
        (str(membership_path), s, s["relative_dir"] in learning_ids, str(output), region_tracks, audit_foreground)
        for s in selected
    ]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for result in pool.map(prepare_view, tasks):
            source = result["source"]
            verified[source["relative_dir"]] = source
            files += result["files"]
            if result["statistics"] is not None:
                stats[source["relative_dir"]] = result["statistics"]
    for role in ("learning", "validation", "test"):
        splits[role] = [verified[s["relative_dir"]] for s in splits[role]]
    coverage = (
        assign_buckets(splits["learning"], stats)
        if audit_foreground
        else {
            "status": "deferred",
            "counts": {},
            "buckets": {},
            "reason": "unweighted GT pilot consumes capture latents; foreground replay and buckets are deferred",
        }
    )
    publish_json(output / "coverage/learning_buckets.json", {**coverage, "statistics": stats})
    draws = {}
    for seed in config["replicate_seeds"]:
        for policy, buckets in (("ordinary", None), ("balanced", coverage["buckets"])):
            if buckets is not None and (not audit_foreground or any(v == 0 for v in coverage["counts"].values())):
                continue
            path = output / f"draws/replicate_{seed}_{policy}.json"
            publish_json(
                path,
                draw_list(
                    splits["learning"],
                    seed,
                    buckets=buckets,
                    sigma_indices=cycle,
                    draws_per_update=draws_per_update(config),
                ),
            )
            draws[f"{seed}:{policy}"] = {"path": str(path), "sha256": file_sha256(path)}
            files.append(draws[f"{seed}:{policy}"])
    components = {
        role: {"path": str((workspace / p).resolve()), "file_sha256": file_sha256(workspace / p)}
        for role, p in config["components"].items()
    }
    model = model_registry.resolve("2.5", **{f"{role}_path": value["path"] for role, value in components.items()})
    cache = prompt_cache.cache_path("2.5", "")
    if not cache.is_file():
        raise FileNotFoundError(f"empty-context cache required before freeze: {cache}")
    context = torch.load(cache, map_location="cpu", weights_only=True)
    files.append({"path": str(cache.resolve()), "sha256": file_sha256(cache)})
    if region_path:
        files.append({"path": str(region_path.resolve()), "sha256": file_sha256(region_path)})
    files += [
        {"path": str(membership_path), "sha256": JSON_SHA},
        {"path": str(membership_path.with_suffix(".csv")), "sha256": CSV_SHA},
        {
            "path": str(Path(membership["corpus_root"]) / "capture_latent_manifest.json"),
            "sha256": file_sha256(Path(membership["corpus_root"]) / "capture_latent_manifest.json"),
        },
    ]
    study = {
        "format": "learned_head_study_v1",
        "config": config,
        "config_sha256": file_sha256(effective_config),
        "plan": {"path": str(frozen_plan), "sha256": file_sha256(frozen_plan), "original_path": str(plan.resolve())},
        "components": components,
        "splits": splits,
        "seen": seen,
        "coverage": coverage,
        "draws": draws,
        "files": files,
        "text_context": {
            "path": str(cache.resolve()),
            "tensor_sha256": tensor_hash(context),
            "bytes_sha256": hashlib.sha256(context.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),
        },
        "schedule": schedule,
        "scale_factors": list(model.scale_factors),
        "code": code_identity(),
        "root": str(output),
        "region_tracks": region_tracks,
        "length_counts": dict(Counter(s["n_latent_frames"] for s in splits["learning"])),
    }
    study["software"] = {
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "safetensors", "transformers")},
    }
    metric_spec = importlib.util.find_spec("lpips")
    alexnet = Path(torch.hub.get_dir()) / "checkpoints/alexnet-owt-7be5be79.pth"
    if metric_spec is not None and alexnet.is_file():
        metric_root = Path(metric_spec.origin).parent
        weights = metric_root / "weights/v0.1/alex.pth"
        study["perceptual_metric"] = {
            "status": "pinned",
            "backbone": "alex",
            "version": importlib.metadata.version("lpips"),
            "preprocess": "GT-fixed boxes; 256px bilinear resize; RGB [0,1] -> [-1,1]",
            "weights": [{"path": str(p), "sha256": file_sha256(p)} for p in (weights, alexnet)],
        }
        study["files"] += study["perceptual_metric"]["weights"]
    else:
        study["perceptual_metric"] = {"status": "missing", "backbone": "alex", "quality_gate": "cannot_pass"}
    study["identity"] = digest(study)
    publish_json(output / "configs/frozen_study.json", study)
    return study


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/learned_heads_dev10_ddp_gt_uniform_f18_rank128.json",
    )
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seen-actors", type=Path)
    p.add_argument(
        "--evidence-list", type=Path, required=True, help="Pinned JSON list of previously inspected evidence."
    )
    p.add_argument("--region-tracks", type=Path)
    p.add_argument(
        "--uniform-f18", action="store_true", help="Train on continuous 18-latent windows from full sources."
    )
    args = p.parse_args()
    torch.set_num_threads(4)
    try:
        prepare(
            args.config,
            args.plan,
            args.output,
            args.seen_actors,
            args.region_tracks,
            uniform_f18=args.uniform_f18,
            evidence_list=args.evidence_list,
        )
    except Exception as exc:
        publish_json(
            args.output / "logs/preparation.failure.json",
            {"phase": "preparation", "error": type(exc).__name__, "message": str(exc)},
        )
        raise


if __name__ == "__main__":
    main()
