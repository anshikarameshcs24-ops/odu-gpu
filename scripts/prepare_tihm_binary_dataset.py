"""Prepare binary low-vs-elevated TIHM early-warning sequences."""

from __future__ import annotations

import argparse
import json

from src.data.tihm_binary_dataset import build_tihm_binary_sequence_records, save_binary_sequence_splits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="data/external/tihm/Dataset")
    parser.add_argument("--output_dir", default="data/processed/tihm_binary")
    parser.add_argument("--sequence_length", type=int, default=30)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--lead_minutes", type=int, default=8)
    args = parser.parse_args()

    records = build_tihm_binary_sequence_records(
        root_dir=args.input_dir,
        output_dir=args.output_dir,
        sequence_length=args.sequence_length,
        stride=args.stride,
        lead_minutes=args.lead_minutes,
    )
    save_binary_sequence_splits(records, args.output_dir)
    print(json.dumps(records["counts"], indent=2))
    print(f"Saved TIHM binary sequence splits to {args.output_dir}")


if __name__ == "__main__":
    main()
