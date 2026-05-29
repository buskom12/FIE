from __future__ import annotations

import argparse

from learning.collective_learning_loop import mark_latest_outcome, run_learning_update


def main() -> None:
    parser = argparse.ArgumentParser(description="Collective intelligence learning loop runner")
    parser.add_argument("--event", type=str, help="Event title to mark latest outcome for")
    parser.add_argument("--outcome", type=float, help="Real outcome (0/1)")
    parser.add_argument("--reward", type=float, default=1.08, help="Reward factor for correct role")
    parser.add_argument("--penalty", type=float, default=0.92, help="Penalty factor for wrong role")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for class")
    args = parser.parse_args()

    if args.event is not None and args.outcome is not None:
        ok = mark_latest_outcome(args.event, args.outcome)
        print({"marked_outcome": ok, "event": args.event, "outcome": args.outcome})

    stats = run_learning_update(
        reward_factor=args.reward,
        penalty_factor=args.penalty,
        threshold=args.threshold,
    )
    print(stats)


if __name__ == "__main__":
    main()
