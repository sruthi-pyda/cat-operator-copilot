"""Replay one recorded session as a live shift.

    python replay.py --session S000001
    python replay.py --session S000001 --limit 40 --delay 0.2

Prints a timeline of what the system would have decided at each telemetry row.
The Buddy verdict is evaluated live; safety events are the ones the dataset
recorded during that interval.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from features.dashboard.data import DashboardData, MissingDatasetError  # noqa: E402
from features.dashboard.replay import TelemetryReplay  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay a recorded session as a live shift.")
    parser.add_argument("--session", required=True, help="Session id, e.g. S000001")
    parser.add_argument("--limit", type=int, default=None, help="Stop after this many rows.")
    parser.add_argument(
        "--delay", type=float, default=0.0, help="Seconds between steps, for a live-looking demo."
    )
    parser.add_argument("--list-sessions", action="store_true", help="Print some session ids and exit.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data = DashboardData.from_settings()

    try:
        if args.list_sessions:
            sessions = data.sessions().session_id.head(20).tolist()
            print("\n".join(sessions))
            return 0

        replay = TelemetryReplay(data, args.session)
        rows = replay.telemetry()
    except MissingDatasetError as exc:
        print(exc)
        return 1

    if rows.empty:
        print(f"No telemetry for session {args.session!r}. Try --list-sessions.")
        return 1

    print(f"Replaying {args.session}: {len(rows)} telemetry rows")
    print("-" * 78)

    buddy_windows = 0
    events_seen = 0
    for step in replay.steps(limit=args.limit):
        print(step.summary())
        buddy_windows += step.buddy_available
        events_seen += len(step.safety_events)
        if args.delay:
            import time

            time.sleep(args.delay)

    print("-" * 78)
    print(f"Buddy available on {buddy_windows} steps; {events_seen} safety events recorded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
