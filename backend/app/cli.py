"""CLI for testing question generation without a webserver.

Examples:
    python -m app.cli example --level 1
    python -m app.cli example --level 7 -n 5 --mode normal
    python -m app.cli example --level 5 --mode hard --seed 42
    python -m app.cli solve   --group oneworld --a LHR --b MIA
    python -m app.cli validate --group oneworld --a LHR --b MIA \\
        --leg LHR:MAD:IB --leg MAD:MIA:IB
    python -m app.cli sample-histogram --n 1000  # quick distribution sample
"""
from __future__ import annotations

import argparse
import random
import sys
import time

from .data import load_dataset
from .questions import VALID_MODES, Leg, Question, QuestionGenerator


def _format_question(q: Question, gen: QuestionGenerator) -> str:
    direct_note = " (direct exists; indirect-only)" if q.direct_available else ""
    return (
        f"  L{q.level:2d}  {q.display_a()} -> {q.display_b()}\n"
        f"        group: {q.group_name}  [obscurity={q.obscurity}, conn_tier={q.conn_tier}]\n"
        f"        min stops: {q.min_stops}{direct_note}  mode: {q.mode}"
    )


def _format_legs(legs: list[Leg], gen: QuestionGenerator) -> str:
    parts = []
    for i, leg in enumerate(legs, 1):
        parts.append(
            f"    leg {i}: {leg.src} -> {leg.dst} on {leg.airline} ({gen.airline_name(leg.airline)})"
        )
    return "\n".join(parts)


def cmd_example(gen: QuestionGenerator, level: int, n: int, mode: str, seed: int | None) -> int:
    rng = random.Random(seed) if seed is not None else random
    qs = []
    for _ in range(n):
        try:
            qs.append(gen.sample(level, mode=mode, rng=rng))  # type: ignore[arg-type]
        except (ValueError, RuntimeError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
    print(f"Level {level} ({mode} mode): {n} example question(s)")
    for q in qs:
        print(_format_question(q, gen))
    return 0


def cmd_solve(gen: QuestionGenerator, group: str, a: str, b: str, mode: str, seed: int | None) -> int:
    rng = random.Random(seed) if seed is not None else random
    try:
        q = gen.question_for(group, a, b, mode=mode)  # type: ignore[arg-type]
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(_format_question(q, gen))
    legs = gen.example_answer(q, rng=rng)
    print("  example answer:")
    print(_format_legs(legs, gen))
    return 0


def cmd_validate(
    gen: QuestionGenerator,
    group: str,
    a: str,
    b: str,
    leg_specs: list[str],
    mode: str,
) -> int:
    try:
        q = gen.question_for(group, a, b, mode=mode)  # type: ignore[arg-type]
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(_format_question(q, gen))
    try:
        legs = [Leg.parse(s) for s in leg_specs]
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print("  candidate answer:")
    print(_format_legs(legs, gen))
    result = gen.validate_answer(q, legs)
    if result.valid:
        print(f"  -> VALID ({result.stops} stop(s))")
    else:
        print(f"  -> INVALID: {result.reason}")
    return 0 if result.valid else 1


def cmd_sample_histogram(gen: QuestionGenerator, n_per_level: int, seed: int | None) -> int:
    """Time `sample()` at each level n_per_level times and report success rates."""
    rng = random.Random(seed) if seed is not None else random
    print(f"Sampling {n_per_level} questions per level (seed={seed})")
    print(f"  {'level':>5}  {'success':>8}  {'fail':>5}  {'avg ms':>8}")
    print(f"  {'-'*5}  {'-'*8}  {'-'*5}  {'-'*8}")
    for lvl in range(1, 11):
        ok = 0
        fails = 0
        total_ms = 0.0
        for _ in range(n_per_level):
            t0 = time.perf_counter()
            try:
                gen.sample(lvl, rng=rng)
                ok += 1
            except (ValueError, RuntimeError):
                fails += 1
            total_ms += (time.perf_counter() - t0) * 1000
        avg = total_ms / n_per_level if n_per_level else 0.0
        print(f"  {lvl:>5}  {ok:>8}  {fails:>5}  {avg:>8.2f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="app.cli")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ex = sub.add_parser("example", help="Sample example question(s) at a level.")
    p_ex.add_argument("--level", type=int, required=True, help="Difficulty level 1-10.")
    p_ex.add_argument("-n", type=int, default=1, help="How many examples to sample.")
    p_ex.add_argument("--mode", choices=VALID_MODES, default="normal")
    p_ex.add_argument("--seed", type=int, default=None)

    p_solve = sub.add_parser("solve", help="Print a worked example answer for a (group, A, B) triple.")
    p_solve.add_argument("--group", required=True)
    p_solve.add_argument("--a", required=True)
    p_solve.add_argument("--b", required=True)
    p_solve.add_argument("--mode", choices=VALID_MODES, default="normal")
    p_solve.add_argument("--seed", type=int, default=None)

    p_val = sub.add_parser("validate", help="Validate a candidate answer for a (group, A, B) triple.")
    p_val.add_argument("--group", required=True)
    p_val.add_argument("--a", required=True)
    p_val.add_argument("--b", required=True)
    p_val.add_argument(
        "--leg",
        action="append",
        required=True,
        dest="legs",
        help="Repeatable. Format SRC:DST:AIRLINE (e.g. LHR:MAD:IB).",
    )
    p_val.add_argument("--mode", choices=VALID_MODES, default="normal")

    p_hist = sub.add_parser(
        "sample-histogram",
        help="Sample N questions per level, report success rate and timing.",
    )
    p_hist.add_argument("--n", type=int, default=200)
    p_hist.add_argument("--seed", type=int, default=None)

    args = parser.parse_args(argv)

    dataset = load_dataset()
    gen = QuestionGenerator(dataset)

    if args.cmd == "example":
        return cmd_example(gen, args.level, args.n, args.mode, args.seed)
    if args.cmd == "solve":
        return cmd_solve(gen, args.group, args.a, args.b, args.mode, args.seed)
    if args.cmd == "validate":
        return cmd_validate(gen, args.group, args.a, args.b, args.legs, args.mode)
    if args.cmd == "sample-histogram":
        return cmd_sample_histogram(gen, args.n, args.seed)
    parser.error(f"unknown command: {args.cmd}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
