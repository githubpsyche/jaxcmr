"""Generate semantic embeddings for VRT film clips.

This script creates a feature matrix for use with semantic CRP analysis.
Each clip's description is embedded using sentence-transformers, producing
a matrix aligned with pres_itemids indexing:
  - Rows 0-10: emotional clips (pres_itemids 1-11)
  - Rows 11-21: neutral clips (pres_itemids 12-22)

Usage:
    python generate_clip_embeddings.py [--output PATH] [--model NAME]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# Reuse parsing and encoding from VRT_data_preparation
from VRT_data_preparation import (
    parse_vrt_clips,
    load_sentence_transformer,
    encode_texts,
    VrtClip,
)


def build_clip_texts(clips: list[VrtClip]) -> tuple[list[str], list[str]]:
    """Return description texts for emotional and neutral clips, ordered by clip number.

    Args:
        clips: Parsed clip descriptions from the scoring document.

    Returns:
        (emotional_texts, neutral_texts): Lists of 11 description strings each,
        ordered by clip_number (1-11).
    """
    emotional = sorted(
        [c for c in clips if c.film == "emotional"],
        key=lambda c: c.clip_number,
    )
    neutral = sorted(
        [c for c in clips if c.film == "neutral"],
        key=lambda c: c.clip_number,
    )

    if len(emotional) != 11 or len(neutral) != 11:
        raise ValueError(
            f"Expected 11 clips per film, got {len(emotional)} emotional, {len(neutral)} neutral"
        )

    emotional_texts = [c.description for c in emotional]
    neutral_texts = [c.description for c in neutral]
    return emotional_texts, neutral_texts


def main(argv: list[str] | None = None) -> int:
    """Generate and save clip embeddings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scoring-md",
        type=Path,
        default=Path("data/raw/VRT_Data_Scoring_Info.md"),
        help="Path to the VRT scoring info markdown (clip descriptions).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/VRT_clip_embeddings.npy"),
        help="Output path for the embeddings .npy file.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="all-mpnet-base-v2",
        help="SentenceTransformer model name for embedding.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional torch device override (e.g., 'cpu', 'cuda').",
    )
    args = parser.parse_args(argv)

    # Parse clip descriptions
    print(f"Parsing clips from: {args.scoring_md}")
    clips = parse_vrt_clips(args.scoring_md)
    emotional_texts, neutral_texts = build_clip_texts(clips)

    # Combine: emotional first (rows 0-10), then neutral (rows 11-21)
    all_texts = emotional_texts + neutral_texts
    print(f"Embedding {len(all_texts)} clip descriptions with {args.model_name}")

    # Load model and encode
    model = load_sentence_transformer(args.model_name, args.device)
    embeddings = encode_texts(model, all_texts, batch_size=32)

    # Verify shape
    print(f"Embeddings shape: {embeddings.shape}")
    if embeddings.shape[0] != 22:
        raise ValueError(f"Expected 22 rows, got {embeddings.shape[0]}")

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, embeddings)
    print(f"Saved embeddings to: {args.output}")

    # Print alignment info
    print("\nAlignment with pres_itemids:")
    print("  Emotional clips (pres_itemids 1-11) -> rows 0-10")
    print("  Neutral clips (pres_itemids 12-22) -> rows 11-21")
    print("  Access: features[pres_itemids - 1]")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
