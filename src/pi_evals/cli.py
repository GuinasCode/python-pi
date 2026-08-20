"""CLI entry point for running Pi evals — the Python counterpart to
``scripts/run-evals.mjs``.

Runs pytest-evals' two phases (``--run-eval`` then ``--run-eval-analysis``)
in one invocation, forwarding any extra arguments (test paths, ``-k``) to
both. ``--provider``/``--model`` set ``PI_PROVIDER``/``PI_MODEL`` for
harnesses that don't pin their own model via
``PiCodingAgentHarnessOptions.model``.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    try:
        import pytest
    except ImportError:
        print(
            "pi-evals requires the eval extra: install with `pip install python-pi[eval]` "
            "(or `uv sync --extra eval`).",
            file=sys.stderr,
        )
        return 1

    parser = argparse.ArgumentParser(
        prog="pi-evals",
        description="Run Pi behavioral evals (pytest-evals eval + analysis phases, in one command).",
    )
    parser.add_argument("--provider", help="Default model provider (sets PI_PROVIDER)")
    parser.add_argument("--model", help="Default model id (sets PI_MODEL)")
    args, pytest_args = parser.parse_known_args(argv)

    if bool(args.provider) != bool(args.model):
        parser.error("--provider and --model must be supplied together")
    if args.provider and args.model:
        os.environ["PI_PROVIDER"] = args.provider
        os.environ["PI_MODEL"] = args.model

    eval_exit_code = int(pytest.main(["--run-eval", "--supress-failed-exit-code", *pytest_args]))
    if eval_exit_code != 0:
        # A nonzero exit here means pytest itself errored (collection, a bad
        # -k, etc.) — --supress-failed-exit-code already keeps individual
        # failed eval CASES from surfacing here. Bail before running
        # analysis against a run that never completed.
        return eval_exit_code

    return int(pytest.main(["--run-eval-analysis", *pytest_args]))


if __name__ == "__main__":
    sys.exit(main())
