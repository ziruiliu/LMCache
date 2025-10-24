#!/usr/bin/env python3
"""Utility script to verify skipping tokens keeps suffix hashes unchanged."""

from __future__ import annotations

# Standard
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    # First Party
    from lmcache.v1.token_database import ChunkedTokenDatabase
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency guard
    if exc.name == "torch":
        print(
            "PyTorch is required to run this script because ChunkedTokenDatabase "
            "depends on torch. Please install torch before running the check."
        )
        sys.exit(1)
    raise


def compute_skip_index(ends: list[int], skip_n_tokens: int) -> int:
    """Return the number of full chunks contained in ``skip_n_tokens``."""

    skip_index = 0
    for end in ends:
        if end <= skip_n_tokens:
            skip_index += 1
        else:
            break
    return skip_index


def main() -> None:
    """Compare chunk hashes before and after skipping prefix tokens."""

    db = ChunkedTokenDatabase()

    # Generate a deterministic token sequence with several chunks.
    random.seed(0)
    total_chunks = 4
    tokens_per_chunk = db.chunk_size
    tokens = [random.randint(0, 32000) for _ in range(total_chunks * tokens_per_chunk)]

    chunk_infos = list(db.process_tokens(tokens, make_key=False))
    ends = [end for _, end, _ in chunk_infos]

    # Skip exactly half of the chunks.
    skip_n_tokens = (total_chunks // 2) * tokens_per_chunk
    skip_index = compute_skip_index(ends, skip_n_tokens)

    suffix_hashes_full = [info[2] for info in chunk_infos[skip_index:]]

    truncated_tokens = tokens[skip_n_tokens:]
    truncated_hashes = [
        info[2] for info in db.process_tokens(truncated_tokens, make_key=False)
    ]

    print(f"Chunk size: {db.chunk_size}")
    print(f"Skip tokens: {skip_n_tokens}")
    print("Hashes after skip (using full tokens):", suffix_hashes_full)
    print("Hashes when recomputing on truncated tokens:", truncated_hashes)
    print(
        "Hashes preserved after skipping full chunks:",
        suffix_hashes_full == [info[2] for info in chunk_infos][skip_index:],
    )


if __name__ == "__main__":
    main()
