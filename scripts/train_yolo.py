#!/usr/bin/env python3
"""Prepare the local resource dataset and train an Ultralytics YOLO model."""

from __future__ import annotations

import argparse
import random
import shutil
from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class Sample:
    image: Path
    label: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a custom Ultralytics YOLO model from resource/figure images "
            "and resource/labels annotations."
        )
    )
    parser.add_argument("--images", type=Path, default=Path("resource/figure"))
    parser.add_argument("--labels", type=Path, default=Path("resource/labels"))
    parser.add_argument("--out", type=Path, default=Path("resource/yolo_dataset"))
    parser.add_argument(
        "--model",
        default="yolo11m.pt",
        help=(
            "Ultralytics model checkpoint/name. Use yolo11m.pt for box labels, "
            "or yolo11m-seg.pt only if labels are segmentation polygons."
        ),
    )
    parser.add_argument(
        "--task",
        choices=("auto", "detect", "segment"),
        default="auto",
        help="Training task. auto infers segment when the model name contains '-seg'.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=None, help="Example: 0, cpu, mps. Default lets YOLO choose.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project", default="runs/yolo")
    parser.add_argument("--name", default="roboclaw")
    parser.add_argument(
        "--names",
        default=None,
        help="Comma-separated class names. Default: class_0,class_1,...",
    )
    parser.add_argument(
        "--link",
        action="store_true",
        help="Symlink images/labels into the prepared dataset instead of copying.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only prepare and validate the YOLO dataset; do not start training.",
    )
    return parser.parse_args()


def resolve_task(model: str, task: str) -> str:
    if task != "auto":
        return task
    return "segment" if "-seg" in Path(model).stem else "detect"


def collect_samples(images_dir: Path, labels_dir: Path) -> list[Sample]:
    if not images_dir.is_dir():
        raise SystemExit(f"Image directory not found: {images_dir}")
    if not labels_dir.is_dir():
        raise SystemExit(f"Label directory not found: {labels_dir}")

    images = sorted(path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_EXTS)
    samples = [Sample(image=image, label=labels_dir / f"{image.stem}.txt") for image in images]
    samples = [sample for sample in samples if sample.label.is_file()]

    if not samples:
        raise SystemExit(f"No image/label pairs found in {images_dir} and {labels_dir}")
    return samples


def validate_labels(samples: list[Sample], task: str) -> int:
    class_ids: set[int] = set()

    for sample in samples:
        for line_no, raw_line in enumerate(sample.label.read_text(encoding="utf-8").splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split()
            try:
                class_id = int(parts[0])
                values = [float(part) for part in parts[1:]]
            except (IndexError, ValueError) as exc:
                raise SystemExit(f"Invalid label at {sample.label}:{line_no}: {raw_line}") from exc

            if task == "detect" and len(parts) != 5:
                raise SystemExit(
                    f"Detection labels must have 5 columns, got {len(parts)} at "
                    f"{sample.label}:{line_no}"
                )
            if task == "segment" and (len(parts) < 7 or len(values) % 2 != 0):
                raise SystemExit(
                    "Segmentation training needs polygon labels: "
                    f"class x1 y1 x2 y2 ...; got box-style label at {sample.label}:{line_no}. "
                    "Use --model yolo11m.pt for the current labels."
                )
            if any(value < 0.0 or value > 1.0 for value in values):
                raise SystemExit(f"Label values must be normalized to 0..1 at {sample.label}:{line_no}")
            if class_id < 0:
                raise SystemExit(f"Class id must be non-negative at {sample.label}:{line_no}")
            class_ids.add(class_id)

    if not class_ids:
        raise SystemExit("Labels are empty; add at least one object annotation before training.")
    return max(class_ids) + 1


def split_samples(samples: list[Sample], val_fraction: float, seed: int) -> tuple[list[Sample], list[Sample]]:
    if not 0.0 <= val_fraction < 1.0:
        raise SystemExit("--val-fraction must be >= 0.0 and < 1.0")

    shuffled = samples[:]
    random.Random(seed).shuffle(shuffled)

    if len(shuffled) == 1 or val_fraction == 0.0:
        return shuffled, shuffled

    val_count = max(1, round(len(shuffled) * val_fraction))
    val_count = min(val_count, len(shuffled) - 1)
    return shuffled[val_count:], shuffled[:val_count]


def class_names(names_arg: str | None, nc: int) -> list[str]:
    if names_arg is None:
        return [f"class_{idx}" for idx in range(nc)]

    names = [name.strip() for name in names_arg.split(",") if name.strip()]
    if len(names) != nc:
        raise SystemExit(f"--names must provide exactly {nc} names, got {len(names)}")
    return names


def reset_dataset_dir(out_dir: Path) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def place_file(src: Path, dst: Path, link: bool) -> None:
    if link:
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


def prepare_dataset(
    train_samples: list[Sample],
    val_samples: list[Sample],
    out_dir: Path,
    names: list[str],
    link: bool,
) -> Path:
    reset_dataset_dir(out_dir)

    for split, split_samples in (("train", train_samples), ("val", val_samples)):
        for sample in split_samples:
            place_file(sample.image, out_dir / "images" / split / sample.image.name, link)
            place_file(sample.label, out_dir / "labels" / split / sample.label.name, link)

    names_yaml = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(names))
    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {out_dir.resolve().as_posix()}",
                "train: images/train",
                "val: images/val",
                f"nc: {len(names)}",
                "names:",
                names_yaml,
                "",
            ]
        ),
        encoding="utf-8",
    )
    return data_yaml


def train(args: argparse.Namespace, data_yaml: Path, task: str) -> None:
    from ultralytics import YOLO

    model = YOLO(args.model, task=task)
    train_kwargs = {
        "data": str(data_yaml),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "project": args.project,
        "name": args.name,
        "seed": args.seed,
    }
    if args.device is not None:
        train_kwargs["device"] = args.device

    model.train(**train_kwargs)


def main() -> None:
    args = parse_args()
    task = resolve_task(args.model, args.task)
    samples = collect_samples(args.images, args.labels)
    nc = validate_labels(samples, task)
    train_samples, val_samples = split_samples(samples, args.val_fraction, args.seed)
    names = class_names(args.names, nc)
    data_yaml = prepare_dataset(train_samples, val_samples, args.out, names, args.link)

    print(f"Prepared YOLO dataset: {data_yaml}")
    print(f"Task: {task}")
    print(f"Classes: {', '.join(names)}")
    print(f"Train images: {len(train_samples)}")
    print(f"Val images: {len(val_samples)}")

    if args.dry_run:
        return

    train(args, data_yaml, task)


if __name__ == "__main__":
    main()
