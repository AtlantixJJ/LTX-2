"""The package documentation contract, checked mechanically on every CPU run.

The design index must list every source with its exact physical line count;
modules over 100 lines need a design page with the required headings; entry
pages need Usage; local links and anchors must resolve; documented launch
commands may use only real parser flags; every entry module is in the entry table.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ("Objective", "Data flow", "Organization logic", "Invariants", "Gotchas", "Tests")


def sources():
    for path in sorted(ROOT.rglob("*")):
        rel = path.relative_to(ROOT)
        if not path.is_file() or "__pycache__" in rel.parts or rel.parts[0] in ("doc", "configs", "tests"):
            continue
        if path.suffix == ".py" or path.name.endswith(".py.txt"):
            yield path


def lines(path):
    with path.open("rb") as stream:
        return sum(1 for _ in stream)


def index_rows():
    rows = {}
    for line in (ROOT / "doc/README.md").read_text().splitlines():
        match = re.match(r"\|\s*`([^`]+)`\s*\|.*\|\s*(\d+)\s*\|\s*$", line)
        if match:
            rows[match.group(1)] = int(match.group(2))
    return rows


def design(path):
    return ROOT / "doc" / re.sub(r"\.py(\.txt)?$", ".md", str(path.relative_to(ROOT)))


def entry_modules():
    return sorted(
        ".".join(p.relative_to(ROOT).with_suffix("").parts)
        for p in sources()
        if p.suffix == ".py" and "__main__" in p.read_text()
    )


def test_index_lists_every_source_with_its_line_count():
    rows = index_rows()
    actual = {str(p.relative_to(ROOT)): lines(p) for p in sources()}
    assert set(rows) == set(actual), {
        "missing": sorted(set(actual) - set(rows)),
        "stale": sorted(set(rows) - set(actual)),
    }
    assert {k: rows[k] for k in rows if rows[k] != actual[k]} == {}


def test_large_modules_have_complete_design_pages():
    problems = []
    for path in sources():
        page = design(path)
        if lines(path) > 100 and path.name != "__init__.py" and not page.exists():
            problems.append(f"missing design: {page.relative_to(ROOT)}")
        if page.exists():
            headings = re.findall(r"^## (.+)$", page.read_text(), re.M)
            missing = [h for h in REQUIRED if h not in headings]
            if "__main__" in path.read_text() and "Usage" not in headings:
                missing.append("Usage")
            if missing:
                problems.append(f"{page.relative_to(ROOT)} lacks {missing}")
        for ref in re.findall(r"doc/[a-z0-9_/]+\.md", path.read_text()):
            if not (ROOT / ref).exists():
                problems.append(f"{path.relative_to(ROOT)} cites missing {ref}")
    for page in (ROOT / "doc").rglob("*.md"):
        rel = page.relative_to(ROOT / "doc")
        stem = str(rel)[:-3]
        if len(rel.parts) > 1 and not any((ROOT / (stem + s)).exists() for s in (".py", ".py.txt")):
            problems.append(f"orphan design page: {rel}")
    assert problems == []


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
    for page in sorted(ROOT.rglob("*.md")):
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
                if not destination.exists():
                    broken.append(f"{page.relative_to(ROOT)}:{number} {target}")
                elif fragment and destination.suffix == ".md" and fragment not in anchors(destination):
                    broken.append(f"{page.relative_to(ROOT)}:{number} {target}")
    assert broken == []


def parser_flags(path):
    text = path.read_text()
    found = set()
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
    for page in sorted([*(ROOT / "doc").rglob("*.md"), ROOT / "README.md", ROOT / "CLAUDE.md"]):
        for module, rest in command.findall(page.read_text().replace("\\\n", " ")):
            path = ROOT / (module.replace(".", "/") + ".py")
            if not path.exists():
                problems.append(f"{page.relative_to(ROOT)}: unknown module {module}")
                continue
            used = set(re.findall(r"(?<![\w-])(--[a-z][a-z0-9-]*)", rest.split(">")[0]))
            unknown = sorted(used - parser_flags(path))
            if unknown:
                problems.append(f"{page.relative_to(ROOT)}: {module} {unknown}")
    assert problems == []


def test_entry_table_lists_every_entry_module():
    section = (ROOT / "doc/README.md").read_text().split("## Entry scripts", 1)[1].split("\n## ", 1)[0]
    listed = {m.replace("/", ".") for m in re.findall(r"\[`([a-z0-9_./]+)`\]", section)}
    assert [m for m in entry_modules() if m not in listed] == []
