"""Convert TIHM CSV tables into temporal training sequences."""

from __future__ import annotations

import argparse

from src.data.tihm_dataset import build_tihm_sequence_records, save_sequence_splits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Directory containing TIHM CSV tables.")
    parser.add_argument("--output_dir", required=True, help="Directory to write processed TIHM sequences.")
    parser.add_argument("--sequence_length", type=int, default=30)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--pre_agitation_minutes", type=int, default=8)
    parser.add_argument("--imminent_minutes", type=int, default=3)
    args = parser.parse_args()

    records = build_tihm_sequence_records(
        root_dir=args.input_dir,
        output_dir=args.output_dir,
        sequence_length=args.sequence_length,
        stride=args.stride,
    )
    save_sequence_splits(records, args.output_dir)
    print(f"Saved TIHM sequence splits to {args.output_dir}")


if __name__ == "__main__":
    main()

