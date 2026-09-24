"""Shared pytest hooks.

Prints the per-feature summary table for tests/test_phase_3_complete.py.
Other test modules are unaffected.
"""
import sys
from collections import OrderedDict

import pytest

PHASE3_MODULE = "test_phase_3_complete.py"
_phase3_results = OrderedDict()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if PHASE3_MODULE not in item.nodeid or item.cls is None:
        return
    feature = getattr(item.cls, "FEATURE", item.cls.__name__)
    coverage = getattr(item.cls, "COVERAGE", "")
    entry = _phase3_results.setdefault(feature, {"passed": 0, "failed": 0, "xfailed": 0,
                                                 "skipped": 0, "coverage": coverage})
    if report.when == "call":
        if hasattr(report, "wasxfail"):
            entry["xfailed"] += 1
        elif report.passed:
            entry["passed"] += 1
        elif report.failed:
            entry["failed"] += 1
    elif report.when == "setup" and (report.failed or report.skipped):
        entry["failed" if report.failed else "skipped"] += 1


def pytest_terminal_summary(terminalreporter):
    if not _phase3_results:
        return
    ok, bad, warn = "✅", "❌", "⚠️"
    encoding = getattr(sys.stdout, "encoding", None) or "ascii"
    try:
        (ok + bad + warn).encode(encoding)
    except (UnicodeEncodeError, LookupError):
        ok, bad, warn = "[PASS]", "[FAIL]", "[WARN]"

    tr = terminalreporter
    tr.section("Phase 3 feature summary")
    tr.write_line(f"   {'Feature':<32} | {'Tests Passed':<14} | Coverage")
    tr.write_line("   " + "-" * 32 + "-+-" + "-" * 14 + "-+-" + "-" * 60)
    for feature, r in _phase3_results.items():
        run = r["passed"] + r["failed"]
        mark = bad if r["failed"] else (warn if r["xfailed"] or r["skipped"] else ok)
        cell = f"{r['passed']}/{run}" + (f" +{r['xfailed']} known" if r["xfailed"] else "")
        tr.write_line(f"{mark} {feature:<32} | {cell:<14} | {r['coverage']}")
