"""The 3D viewer (viewer/index.html) re-implements the mapping rules in JavaScript.
This test makes sure its copy stays identical to cadt/mapping.py."""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cadt.mapping import RULES  # noqa: E402


def test_viewer_rules_match_python():
    html = (ROOT / "viewer" / "index.html").read_text(encoding="utf-8")
    block = re.search(r'<script type="application/json" id="rules">(.*?)</script>', html, re.S)
    assert block, "rules block not found in viewer/index.html"
    js_rules = json.loads(block.group(1))
    py_rules = {name: {"baseline": r.baseline, "weights": r.weights, "bounds": list(r.bounds)}
                for name, r in RULES.items()}
    assert js_rules == py_rules
