"""Check navigable docs and selected source facts without loading a model.

Module docstrings replace mirrored pages. The file/entry maps must cover current
sources; contracts have a fixed reading order. Static references, named labels,
placeholder recipes, config values and exporter keys are checked against owners.
Negative fixtures ensure a plausible but nonexistent function is rejected.
"""

from __future__ import annotations

import ast
import json
import re
from functools import cache
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
REQUIRED = ("Objective", "Data flow", "Organization logic", "Invariants", "Gotchas", "Tests")
CONTRACTS = (
    "data/whole_clip.md", "score/export_pruned.md", "score/export_depth.md",
    "learn/heads.md", "learn/readmission.md", "learn/ddp_owner.md",
)


def sources():
    return sorted(
        path for path in ROOT.rglob("*.py")
        if path.relative_to(ROOT).parts[0] not in ("doc", "configs", "tests") and "__pycache__" not in path.parts
    )


def pages():
    return [ROOT / "README.md", ROOT / "CLAUDE.md", *sorted((ROOT / "doc").rglob("*.md"))]


def entry_modules():
    return {".".join(p.relative_to(ROOT).with_suffix("").parts) for p in sources() if "__main__" in p.read_text()}


def test_file_map_and_source_docstrings():
    section = (ROOT / "doc/README.md").read_text().split("## Current source owners", 1)[1].split("\n## ", 1)[0]
    listed = re.findall(r"\[`([^`]+\.py)`\]", section)
    actual = [str(p.relative_to(ROOT)) for p in sources()]
    assert sorted(listed) == actual
    for path in sources():
        if path.name != "__init__.py":
            assert ast.get_docstring(ast.parse(path.read_text())), path
        for ref in re.findall(r"doc/[a-zA-Z0-9_/]+\.md", path.read_text()):
            assert (ROOT / ref).is_file(), (path, ref)
    assert (ROOT / "AGENTS.md").is_symlink()
    assert (ROOT / "AGENTS.md").resolve() == (ROOT / "CLAUDE.md").resolve()
    assert len((ROOT / "CLAUDE.md").read_text().splitlines()) <= 50


def contract_errors(text):
    headings = re.findall(r"^## (.+)$", text, re.M)
    expected = list(REQUIRED)
    if "Usage" in headings:
        expected.insert(1, "Usage")
    valid = headings == expected and not re.search(r"^#{1,6} ", text.split("## Tests", 1)[-1], re.M)
    return [] if valid else [f"expected ordered sections {expected}, found {headings}"]


def test_contract_order_and_tests_last():
    for name in CONTRACTS:
        assert contract_errors((ROOT / "doc" / name).read_text()) == [], name
    for page in pages():
        if "## Tests" in page.read_text():
            assert not re.search(r"^#{1,6} ", page.read_text().split("## Tests", 1)[1], re.M), page


def slug(heading):
    text = re.sub(r"[`*~]", "", heading.strip().lower())
    return re.sub(r"[^\w\- ]", "", text).replace(" ", "-")


def anchors(path):
    found, seen, fenced = set(), {}, False
    for line in path.read_text().splitlines():
        if line.startswith("```"):
            fenced = not fenced
        match = None if fenced else re.match(r"^#+\s+(.*)$", line)
        if match:
            value = slug(match.group(1))
            found.add(value if value not in seen else f"{value}-{seen[value]}")
            seen[value] = seen.get(value, 0) + 1
    return found


def test_local_links_and_anchors_resolve():
    broken = []
    for page in pages():
        fenced = False
        for number, line in enumerate(page.read_text().splitlines(), 1):
            if line.startswith("```"):
                fenced = not fenced
                continue
            if fenced:
                continue
            for target in re.findall(r"!?\[[^\]]*\]\(([^)\s]+)\)", line):
                if re.match(r"^[a-z]+://", target):
                    continue
                relative, _, fragment = target.partition("#")
                destination = (page.parent / relative).resolve() if relative else page
                if not destination.exists() or (
                    fragment and destination.suffix == ".md" and fragment not in anchors(destination)
                ):
                    broken.append(f"{page.relative_to(ROOT)}:{number} {target}")
    assert broken == []


@cache
def module_symbols(path):
    tree = ast.parse(path.read_text())
    names = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names |= {n.id for target in targets for n in ast.walk(target) if isinstance(n, ast.Name)}
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names |= {n.asname or n.name.split(".")[0] for n in node.names}
    return names, tree


def symbol_exists(value):
    value = value.removeprefix("scripts.prune.")
    parts = value.split(".")
    for size in range(len(parts), 0, -1):
        path = ROOT.joinpath(*parts[:size]).with_suffix(".py")
        if not path.is_file():
            continue
        remaining = parts[size:]
        if not remaining:
            return True
        names, tree = module_symbols(path)
        if remaining[0] not in names:
            return False
        # Class method references are statically resolvable too; runtime
        # instance attributes are described as data rather than module symbols.
        if len(remaining) > 1:
            cls = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == remaining[0]), None)
            return bool(cls and len(remaining) == 2 and any(getattr(n, "name", None) == remaining[1] for n in cls.body))
        return True
    return False


def reference_errors(text, page):
    errors = []
    # Explicit placeholders name generated artifacts rather than source files.
    for code in re.findall(r"(?<!`)`([^`\n]+)`(?!`)", text):
        if "<" in code:
            continue
        if re.fullmatch(r"[a-zA-Z0-9_./-]+\.py", code):
            candidates = [page.parent / code, ROOT / code, REPO / code, ROOT / "tests" / code]
            if not any(p.is_file() for p in candidates):
                errors.append(code)
        for value in re.findall(r"(?<![\w.])(?:scripts\.prune\.)?(?:checks|core|data|evaluate|learn|score)\.[\w.]+", code):
            if not symbol_exists(value.rstrip(".")):
                errors.append(value)
    return errors


def test_named_source_references_resolve():
    problems = {str(page.relative_to(ROOT)): errors for page in pages() if (errors := reference_errors(page.read_text(), page))}
    assert problems == {}


def labels(text):
    result = set(re.findall(r"\b(?:V\d+|P\d+|G\d+)\b", text))
    result |= set(re.findall(r"\barms? +([A-Z])\b", text))
    for group in re.findall(r"\barms? +([A-Z](?:/[A-Z])+)\b", text):
        result.update(group.split("/"))
    return result


def test_short_labels_have_glossary_definitions():
    glossary = (ROOT / "doc/glossary.md").read_text()
    defined = set()
    for term in re.findall(r"^- \*\*([^*]+)\*\*:", glossary, re.M):
        defined |= labels(term)
        defined |= set(re.findall(r"\b[A-Z]\b", term))
    assert {str(page): labels(page.read_text()) - defined for page in pages() if labels(page.read_text()) - defined} == {}
    for page in [ROOT / "README.md", ROOT / "CLAUDE.md", *(ROOT / "doc").glob("*.md")]:
        if page.name != "glossary.md":
            assert "glossary.md" in page.read_text(), page


def dated_usage(text, *, run_guide=False):
    usage = text if run_guide else "\n".join(re.findall(r"^## Usage\n(.*?)(?=^## |\Z)", text, re.M | re.S))
    commands = re.findall(r"```(?:bash|sh)?\n(.*?)```", usage, re.S)
    return [block for block in commands if re.search(r"\b(?:20\d{6}|20\d{2}-\d{2}-\d{2})\b|(?<=_)20\d{6}(?=\D|$)", block)]


def test_usage_uses_placeholders_instead_of_dated_studies():
    for page in pages():
        assert not dated_usage(page.read_text(), run_guide=page.name == "usage.md"), page


def parser_flags(path):
    text = path.read_text()
    found = {"--help"}
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument":
            found |= {a.value for a in node.args if isinstance(a, ast.Constant) and str(a.value).startswith("--")}
    for match in re.finditer(r'for (\w+) in \(([^)]*)\):\s*\n\s*\w+\.add_argument\("--" \+ \1', text):
        found |= {"--" + name for name in re.findall(r'"([^"]+)"', match.group(2))}
    if "add_model_args" in text:
        found |= {"--model", "--gpu-id", "--seed"}
    if "add_prompt_args" in text:
        found |= {"--prompt", "--prompt-file"}
    return found


def test_documented_commands_use_real_flags():
    command = re.compile(r"(?:python (?:-u )?|torchrun [^\n`]*?)-m scripts\.prune\.([a-z0-9_]+\.[a-z0-9_]+)([^\n`]*)")
    problems = []
    for page in pages():
        for module, rest in command.findall(page.read_text().replace("\\\n", " ")):
            path = ROOT / (module.replace(".", "/") + ".py")
            if not path.exists():
                problems.append((str(page), module))
                continue
            used = set(re.findall(r"(?<![\w-])(--[a-z][a-z0-9-]*)", rest))
            unknown = sorted(used - parser_flags(path))
            if unknown:
                problems.append((str(page.relative_to(ROOT)), module, unknown))
    assert problems == []


def test_entry_table_lists_every_entry_module():
    section = (ROOT / "doc/README.md").read_text().split("## Entry scripts", 1)[1].split("\n## ", 1)[0]
    listed = re.findall(r"^\| `([a-z0-9_.]+)` \|", section, re.M)
    assert sorted(listed) == sorted(entry_modules())


def test_documented_preset_values_match_config():
    usage = (ROOT / "doc/usage.md").read_text()
    config = json.loads((ROOT / "configs/learned_heads_dev10_ddp_gt_uniform_f18_rank128.json").read_text())
    for key in (
        "soft_updates", "binary_updates", "compact_updates", "gate_lr", "adapter_lr", "rank", "lora_alpha",
        "allowed_counts", "sparsity_penalty", "sparsity_logit_offset", "lambda_max", "lambda_warmup",
        "lambda_start_multiplier", "monitor_reference", "monitor_sigma_indices", "sigma_indices_cycle",
    ):
        values = re.findall(rf"\b{key}=(\[[^\]]+\]|[^,\s`|]+)", usage)
        parsed = []
        for value in values:
            try:
                parsed.append(json.loads(value))
            except json.JSONDecodeError:
                parsed.append(value)
        assert config[key] in parsed, key
    words = {"eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}
    count = re.search(r"preset's (\w+) exact `sigmas`", usage).group(1)
    assert words[count] == len(config["sigmas"])
    ffn = json.loads((ROOT / "configs/learned_ffn_dev10_ddp_joint.json").read_text())
    for key in ("selection_updates", "recovery_updates", "head_soft_updates", "head_rank", "ffn_rank", "ffn_kept", "ffn_width", "ffn_alignment", "ffn_warmup", "ffn_ramp_end"):
        assert f"{key}={ffn[key]}" in usage, key


def test_event_defaults_and_runtime_keys_match_source():
    algorithm = (ROOT / "doc/core_algorithm.md").read_text()
    tree = ast.parse((ROOT / "learn/readmission.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "JointPolicy")
    init = next(n for n in cls.body if getattr(n, "name", "") == "__init__")
    defaults = {a.arg: ast.literal_eval(v) for a, v in zip(init.args.kwonlyargs, init.args.kw_defaults, strict=True)}
    rows = re.findall(r"\|[^\n|]+\| `policy\.(\w+)` \| ([^|]+)\|", algorithm)
    assert {key for key, _ in rows} == {"interval", "late", "earliest", "limit", "recovery", "alignment"}
    for key, value in rows:
        assert int(re.search(r"[\d,]+", value).group().replace(",", "")) == defaults[key]
    exporter = ast.parse((ROOT / "score/export_pruned.py").read_text())
    keys = {n.value for n in ast.walk(exporter) if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value.startswith(("per_layer_", "video_pruning_"))}
    contract = (ROOT / "doc/score/export_pruned.md").read_text()
    assert all(f"`{key}`" in contract for key in keys)
    inputs = (ROOT / "doc/data/whole_clip.md").read_text()
    owner = ast.parse((ROOT / "data/whole_clip.py").read_text())
    families = next(n.value for n in owner.body if isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "ENVELOPE_FAMILIES" for t in n.targets))
    assert all(f"`{key}`" in inputs for key in ast.literal_eval(families))


@pytest.mark.parametrize("reference", ["score.estimators.exact_local_head_energy", "learn.readmission.JointPolicy.missing", "tests/test_missing.py"])
def test_reference_checker_rejects_plausible_stale_names(reference):
    assert reference_errors(f"Use `{reference}`.", ROOT / "doc/core_algorithm.md")


def test_structure_checks_reject_append_only_and_dated_usage():
    complete = "\n".join("## " + h for h in REQUIRED)
    assert contract_errors(complete) == []
    assert contract_errors("No sections")
    assert contract_errors(complete + "\n## Added later")
    assert dated_usage("## Usage\n```bash\npython --study learned_heads_20261012_r2/study.json\n```\n## Data flow")
    assert "V9" not in labels((ROOT / "doc/glossary.md").read_text())
    assert labels("arm Z and V9") == {"Z", "V9"}
