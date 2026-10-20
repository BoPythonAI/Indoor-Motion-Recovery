"""Verify the published source, result counts, media identities and local links.

This check uses only the Python standard library; it does not run simulation.
"""
import ast
import hashlib
import json
import re
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    python_files = list(ROOT.rglob("*.py"))
    for path in python_files:
        if ".git" not in path.parts:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    expected = {
        "baseline_development.json": [39, 33, 21],
        "teacher_development.json": [38, 32, 30],
        "bc_development.json": [38, 33, 27],
        "diffusion_development.json": [34, 33, 29],
        "nominal_only.json": [37, 35, 30],
    }
    for name, values in expected.items():
        result = load("evidence/" + name)["summary"]
        assert [result[k]["success"] for k in ["0.0", "0.45", "0.75"]] == values
        assert all(result[k]["total"] == 40 for k in ["0.0", "0.45", "0.75"])
    selection = load("evidence/selection.json")
    assert selection["status"] == "FROZEN_QUALIFIED"
    assert selection["horizon"] == selection["execute_steps"] == 1
    events = load("evidence/demo_events.json")
    assert len(events["scenes"]) == 3
    videos = 0
    for scene in events["scenes"]:
        directory = ROOT / "docs/media" / scene["id"]
        audit = json.loads((directory / "audit.json").read_text(encoding="utf-8"))
        assert audit["all_source_frames_covered"]
        for mode, n in audit["source_frames"].items():
            indices = audit["frame_mapping"][mode]
            assert len(indices) == audit["output_frames"]
            assert set(indices) == set(range(n))
            if mode != "mesh":
                rows = scene["annotations"][mode]
                assert len(rows) == n
                assert sum(x["push_active"] for x in rows) == 6
        for filename, digest in audit["output_sha256"].items():
            path = directory / filename
            assert sha(path) == digest, path
            assert b"ftyp" in path.read_bytes()[:64], path
            videos += 1
        for name in ["poster.jpg", "push.jpg", "preview.gif"]:
            assert (directory / name).is_file()
    assert videos == 12
    for item in load("evidence/source_provenance.json")["files"]:
        assert sha(ROOT / item["file"]) == item["published_sha256"], item["file"]
    for path in (ROOT / "docs/assets").glob("*.svg"):
        ET.parse(path)
    text_suffixes = {".py", ".md", ".json", ".html", ".js", ".css", ".sh", ".svg", ".yml", ".txt"}
    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts or path.suffix not in text_suffixes:
            continue
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"[\u4e00-\u9fff]", text), f"Non-English text: {path}"
        secret = re.search(
            r"gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{30,}|"
            r"-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----", text)
        assert secret is None, f"Credential-shaped content: {path}"
        if path.suffix == ".md":
            for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", text):
                if target.startswith(("https://", "http://", "#", "mailto:")):
                    continue
                assert (path.parent / target.split("#", 1)[0]).exists(), (path, target)
    print(f"PASS: {len(python_files)} Python files, frozen results, 3 scenes, 12 MP4s, "
          "full frame coverage, English text and local links.")
    print("Scope: release integrity. GPU physics was not rerun.")


if __name__ == "__main__":
    main()
