"""CLI entry point for pi."""

from __future__ import annotations

import sys

from pi_coding_agent import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
