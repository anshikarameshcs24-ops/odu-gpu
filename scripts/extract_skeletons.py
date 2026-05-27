"""Optional skeleton extraction stub for future pose modeling."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.parse_args()
    raise RuntimeError(
        "Skeleton extraction is not implemented in this scaffold. "
        "Integrate ViTPose or MMPose here if pose features are required."
    )


if __name__ == "__main__":
    main()

