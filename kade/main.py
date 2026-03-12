from __future__ import annotations

import argparse

from kade.ui.app import main as run_ui


def main() -> None:
    parser = argparse.ArgumentParser(description="Kade local launcher")
    parser.add_argument("--ui", action="store_true", help="Start local browser UI")
    args = parser.parse_args()
    if args.ui:
        run_ui()


if __name__ == "__main__":
    main()
