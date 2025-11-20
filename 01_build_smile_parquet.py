#!/usr/bin/env python

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import pandas as pd


# Classes are fixed: 0 = no_smiling, 1 = smiling.
CLASS_MAP = {
    "no_smiling": 0,
    "smiling": 1,
}


@dataclass(frozen=True)
class SampleRow:
    split: str
    image_path: str
    image_bytes: bytes
    class_id: int
    label: str
    source: str
    filename: str
    video_id: str
    timestamp: int
    person_id: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build dataset.parquet for the HSC pipeline from FERPlus_smile_color "
            "folder structure with embedded image bytes."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/FERPlus_smile_color"),
        help="Root directory containing train/validation subfolders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/dataset.parquet"),
        help="Destination parquet file path.",
    )
    parser.add_argument(
        "--extensions",
        nargs="*",
        default=[".png", ".jpg", ".jpeg"],
        help="Image file extensions to include (default: .png .jpg .jpeg).",
    )
    return parser.parse_args()


def iter_images(base: Path, split: str, extensions: Iterable[str]) -> Iterable[tuple[Path, str]]:
    allowed = {ext.lower() for ext in extensions}
    for label_name, class_id in CLASS_MAP.items():
        folder = base / split / label_name
        if not folder.is_dir():
            continue
        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.suffix.lower() in allowed:
                rel_path = path.relative_to(base).as_posix()
                yield path, rel_path


def build_dataframe(root: Path, extensions: Iterable[str]) -> pd.DataFrame:
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    rows: List[SampleRow] = []
    person_counter = 1

    for split in ("train", "validation"):
        for path, rel_path in iter_images(root, split, extensions):
            label_folder = Path(rel_path).parts[1]  # split / label / file
            class_id = CLASS_MAP[label_folder]
            label = label_folder
            rows.append(
                SampleRow(
                    split=split,
                    image_path=rel_path,
                    image_bytes=path.read_bytes(),
                    class_id=class_id,
                    label=label,
                    source=root.name,
                    filename=path.name,
                    video_id=Path(rel_path).parent.as_posix(),
                    timestamp=0,
                    person_id=person_counter,
                )
            )
            person_counter += 1

    if not rows:
        raise RuntimeError(f"No images found under {root}.")

    df = pd.DataFrame(rows)
    df = df.sort_values(["split", "class_id", "image_path"]).reset_index(drop=True)
    return df


def main() -> None:
    args = parse_args()
    df = build_dataframe(args.root, args.extensions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"Wrote {len(df)} rows to {args.output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
