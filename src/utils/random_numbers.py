"""Utility script to generate random numbers.

Run this script from the command line to print a list of random integers.

Usage:
    python -m src.utils.random_numbers [--count N] [--min MIN] [--max MAX]

Options:
    --count N   Number of random numbers to generate (default: 10)
    --min MIN   Minimum value (inclusive) (default: 0)
    --max MAX   Maximum value (inclusive) (default: 100)

The script uses the standard library only.
"""

import argparse
import random
from typing import List


def generate_random_numbers(count: int = 10, min_val: int = 0, max_val: int = 100) -> List[int]:
    """Return a list of *count* random integers between *min_val* and *max_val*.

    Args:
        count: Number of random numbers to generate.
        min_val: Lower bound (inclusive).
        max_val: Upper bound (inclusive).
    """
    if min_val > max_val:
        raise ValueError("min_val must be less than or equal to max_val")
    return [random.randint(min_val, max_val) for _ in range(count)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate random integers.")
    parser.add_argument("--count", type=int, default=10, help="Number of random numbers to generate")
    parser.add_argument("--min", type=int, default=0, help="Minimum value (inclusive)")
    parser.add_argument("--max", type=int, default=100, help="Maximum value (inclusive)")
    args = parser.parse_args()

    numbers = generate_random_numbers(count=args.count, min_val=args.min, max_val=args.max)
    print("Generated numbers:", numbers)


if __name__ == "__main__":
    main()
