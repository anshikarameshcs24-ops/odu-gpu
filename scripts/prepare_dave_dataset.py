"""Download or enumerate DAVE benchmark samples into a local manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["epic", "ego4d"], required=True)
    parser.add_argument("--output_manifest", required=True)
    args = parser.parse_args()

    from datasets import load_dataset

    dataset = load_dataset(
        "gorjanradevski/dave",
        split=args.split,
        keep_in_memory=True,
        trust_remote_code=True,
    )

    output_path = Path(args.output_manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for sample in dataset:
            handle.write(json.dumps(sample) + "\n")

    print(f"Saved {len(dataset)} DAVE records to {output_path}")


if __name__ == "__main__":
    main()

