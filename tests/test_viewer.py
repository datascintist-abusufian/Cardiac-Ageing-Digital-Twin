"""The 3D viewer (viewer/index.html) re-implements the mapping rules in JavaScript.
This test makes sure its copies stay identical to cadt.calibration.RULESETS."""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cadt.calibration import RULESETS  # noqa: E402


def test_viewer_rules_match_python():
    html = (ROOT / "viewer" / "index.html").read_text(encoding="utf-8")
    block = re.search(r'<script type="application/json" id="rules">(.*?)</script>', html, re.S)
    assert block, "rules block not found in viewer/index.html"
    js_sets = json.loads(block.group(1))
    py_sets = {name: {k: {"baseline": r.baseline, "weights": r.weights, "bounds": list(r.bounds)}
                      for k, r in rules.items()}
               for name, rules in RULESETS.items()}
    assert js_sets == py_sets
