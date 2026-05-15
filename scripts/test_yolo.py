#!/usr/bin/env python3
"""Test a trained Ultralytics YOLO model on a new image folder or YOLO dataset."""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
DEFAULT_MODEL = Path("runs/detect/runs/yolo/roboclaw/weights/best.pt")


@dataclass(frozen=True)
class Sample:
    image: Path
    label: Path | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run prediction on a new folder, and run validation metrics when matching "
            "YOLO label files are present."
        )
    )
    parser.add_argument("source", type=Path, help="Image file, image folder, or YOLO dataset folder.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--mode", choices=("predict", "val", "both"), default="both")
    parser.add_argument(
        "--task",
        choices=("auto", "detect", "segment"),
        default="auto",
        help="auto infers segment when the model filename contains '-seg'.",
    )
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None, help="Example: 0, cpu, mps. Default lets YOLO choose.")
    parser.add_argument("--project", default="runs/predict")
    parser.add_argument("--name", default="roboclaw_test")
    parser.add_argument("--val-out", type=Path, default=Path("resource/yolo_test_dataset"))
    parser.add_argument("--save-txt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-conf", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--link",
        action="store_true",
        help="Symlink validation images/labels instead of copying them.",
    )
    return parser.parse_args()


def resolve_task(model: Path, task: str) -> str:
    if task != "auto":
        return task
    return "segment" if "-seg" in model.stem else "detect"


def collect_images(source: Path) -> list[Path]:
    if source.is_file():
        if source.suffix.lower() not in IMAGE_EXTS:
            raise SystemExit(f"Not an image file: {source}")
        return [source]

    if not source.is_dir():
        raise SystemExit(f"Source not found: {source}")

    preferred_dirs = [source / "images", source / "figure", source / "figures"]
    for directory in preferred_dirs:
        if directory.is_dir():
            images = sorted(path for path in directory.rglob("*") if path.suffix.lower() in IMAGE_EXTS)
            if images:
                return images

    ignored_parts = {"labels", "label", "yolo_dataset", "yolo_test_dataset"}
    return sorted(
        path
        for path in source.rglob("*")
        if path.suffix.lower() in IMAGE_EXTS and ignored_parts.isdisjoint(path.relative_to(source).parts)
    )


def build_label_index(source: Path) -> dict[str, Path]:
    if source.is_file():
        label = source.with_suffix(".txt")
        return {source.stem: label} if label.is_file() else {}

    labels: dict[str, Path] = {}
    for path in source.rglob("*.txt"):
        if "labels" not in path.parts and "label" not in path.parts:
            continue
        labels.setdefault(path.stem, path)
    return labels


def collect_samples(source: Path) -> list[Sample]:
    images = collect_images(source)
    if not images:
        raise SystemExit(f"No images found in: {source}")

    label_index = build_label_index(source)
    return [Sample(image=image, label=label_index.get(image.stem)) for image in images]


def validate_labels(samples: list[Sample], task: str) -> int:
    class_ids: set[int] = set()
    labelled = [sample for sample in samples if sample.label is not None]

    for sample in labelled:
        assert sample.label is not None
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
                    "Segmentation validation needs polygon labels: "
                    f"class x1 y1 x2 y2 ...; got box-style label at {sample.label}:{line_no}."
                )
            if any(value < 0.0 or value > 1.0 for value in values):
                raise SystemExit(f"Label values must be normalized to 0..1 at {sample.label}:{line_no}")
            class_ids.add(class_id)

    if not class_ids:
        raise SystemExit("No usable labels found for validation.")
    return max(class_ids) + 1


def reset_val_dir(out_dir: Path) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)


def place_file(src: Path, dst: Path, link: bool) -> None:
    if link:
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


def model_names(model: object, nc: int) -> list[str]:
    names = getattr(model, "names", None)
    if isinstance(names, dict):
        return [str(names.get(idx, f"class_{idx}")) for idx in range(nc)]
    if isinstance(names, list) and len(names) >= nc:
        return [str(names[idx]) for idx in range(nc)]
    return [f"class_{idx}" for idx in range(nc)]


def prepare_val_dataset(samples: list[Sample], out_dir: Path, names: list[str], link: bool) -> Path:
    labelled = [sample for sample in samples if sample.label is not None]
    reset_val_dir(out_dir)

    for sample in labelled:
        assert sample.label is not None
        place_file(sample.image, out_dir / "images" / "val" / sample.image.name, link)
        place_file(sample.label, out_dir / "labels" / "val" / sample.label.name, link)

    names_yaml = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(names))
    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {out_dir.resolve().as_posix()}",
                "train: images/val",
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


def run_predict(model: object, args: argparse.Namespace, samples: list[Sample]) -> None:
    predict_kwargs = {
        "source": [str(sample.image) for sample in samples],
        "conf": args.conf,
        "iou": args.iou,
        "imgsz": args.imgsz,
        "save": True,
        "save_txt": args.save_txt,
        "save_conf": args.save_conf,
        "project": args.project,
        "name": args.name,
    }
    if args.device is not None:
        predict_kwargs["device"] = args.device
    model.predict(**predict_kwargs)


def run_val(model: object, samples: list[Sample], args: argparse.Namespace, task: str) -> None:
    labelled_count = sum(1 for sample in samples if sample.label is not None)
    if labelled_count != len(samples):
        print(
            f"Validation skipped: found labels for {labelled_count}/{len(samples)} images. "
            "Prediction images were still saved."
        )
        return

    nc = validate_labels(samples, task)
    data_yaml = prepare_val_dataset(samples, args.val_out, model_names(model, nc), args.link)
    val_kwargs = {
        "data": str(data_yaml),
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
    }
    if args.device is not None:
        val_kwargs["device"] = args.device
    print(f"Prepared validation dataset: {data_yaml}")
    model.val(**val_kwargs)


def main() -> None:
    args = parse_args()
    task = resolve_task(args.model, args.task)
    samples = collect_samples(args.source)

    from ultralytics import YOLO

    model = YOLO(str(args.model), task=task)
    print(f"Loaded model: {args.model}")
    print(f"Task: {task}")
    print(f"Images: {len(samples)}")
    print(f"Images with labels: {sum(1 for sample in samples if sample.label is not None)}")

    if args.mode in ("predict", "both"):
        run_predict(model, args, samples)
    if args.mode in ("val", "both"):
        run_val(model, samples, args, task)


if __name__ == "__main__":
    main()
