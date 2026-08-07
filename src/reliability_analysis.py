"""Repeatedly analyse a fixed random sample of unmasked sequence images."""

from __future__ import annotations

import argparse
import csv
import random
import re
import shutil
import tempfile
from pathlib import Path
from typing import Iterable

import json_repair
from tqdm import tqdm


DEFAULT_MODEL = "OpenGVLab/InternVL3-38B"
DEFAULT_RUNS = 100
EXPECTED_SEQUENCE_COUNT = 5
IMAGES_PER_SEQUENCE = 2
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg"}
DIMENSIONS = ("safe", "lively", "beauty", "wealthy", "depressing", "boring")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def image_paths(sequence_path: Path) -> list[Path]:
    """Return the unmasked image files directly inside a sequence directory."""
    return sorted(
        path
        for path in sequence_path.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )


def discover_sequences(image_root: Path) -> list[Path]:
    """Find immediate child directories containing unmasked images."""
    if not image_root.is_dir():
        raise FileNotFoundError(f"Image root does not exist: {image_root}")
    return sorted(
        (path for path in image_root.iterdir() if path.is_dir() and image_paths(path)),
        key=lambda path: path.name,
    )


def resolve_sequences(image_root: Path, sequence_ids: Iterable[str] | None) -> list[Path]:
    """Resolve either explicitly requested sequences or all discoverable sequences."""
    if sequence_ids:
        sequences = [image_root / sequence_id for sequence_id in sequence_ids]
        missing = [path.name for path in sequences if not path.is_dir()]
        if missing:
            raise FileNotFoundError(f"Sequence directories not found: {', '.join(missing)}")
    else:
        sequences = discover_sequences(image_root)

    if len(sequences) != EXPECTED_SEQUENCE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_SEQUENCE_COUNT} sequences to select 10 images, "
            f"but found {len(sequences)}. Pass exactly five IDs with --sequence-ids."
        )
    return sequences


def select_images(sequence_paths: Iterable[Path], seed: int) -> list[tuple[str, Path]]:
    """Randomly select two images from every sequence using a reproducible seed."""
    rng = random.Random(seed)
    selected: list[tuple[str, Path]] = []
    for sequence_path in sequence_paths:
        candidates = image_paths(sequence_path)
        if len(candidates) < IMAGES_PER_SEQUENCE:
            raise ValueError(
                f"Sequence {sequence_path.name} contains {len(candidates)} images; "
                f"at least {IMAGES_PER_SEQUENCE} are required."
            )
        selected.extend(
            (sequence_path.name, path)
            for path in rng.sample(candidates, IMAGES_PER_SEQUENCE)
        )
    return selected


def parse_response(response: str) -> tuple[dict[str, object], str]:
    """Flatten the six JSON dimension objects returned by InternVL."""
    values = {
        f"{dimension}_{field}": ""
        for dimension in DIMENSIONS
        for field in ("score", "confidence", "rationale")
    }

    fenced_json = re.search(r"```(?:json)?\s*(.*?)```", response, flags=re.DOTALL | re.IGNORECASE)
    json_text = fenced_json.group(1) if fenced_json else response
    try:
        parsed = json_repair.loads(json_text)
    except Exception as exc:
        return values, f"{type(exc).__name__}: {exc}"

    if not isinstance(parsed, list):
        return values, f"Expected a JSON array, received {type(parsed).__name__}"

    errors = []
    by_dimension = {}
    for item in parsed:
        if not isinstance(item, dict):
            errors.append(f"Non-object array item: {item!r}")
            continue
        characteristic = str(item.get("characteristic", "")).strip().lower()
        if characteristic == "beautiful":
            characteristic = "beauty"
        if characteristic not in DIMENSIONS:
            errors.append(f"Unknown characteristic: {characteristic or '<missing>'}")
            continue
        by_dimension[characteristic] = item

    for dimension in DIMENSIONS:
        item = by_dimension.get(dimension)
        if item is None:
            errors.append(f"Missing characteristic: {dimension}")
            continue
        for field in ("score", "confidence", "rationale"):
            value = item.get(field, "")
            if value == "":
                errors.append(f"Missing {field}: {dimension}")
            values[f"{dimension}_{field}"] = value

    return values, "; ".join(errors)


def output_fieldnames() -> tuple[list[str], list[str]]:
    """Return the full CSV schema and the flattened model-result fields."""
    metadata_fields = [
        "sequence_id",
        "image_id",
        "image_path",
        "run",
        "selection_seed",
        "model_name",
    ]
    result_fields = [
        f"{dimension}_{field}"
        for dimension in DIMENSIONS
        for field in ("score", "confidence", "rationale")
    ]
    fieldnames = metadata_fields + result_fields + [
        "generation_error",
        "parse_error",
        "unparsed_response",
    ]
    return fieldnames, result_fields


def load_completed_rows(
    output_path: Path,
    fieldnames: list[str],
    result_fields: list[str],
    selected: list[tuple[str, Path]],
    runs: int,
    seed: int,
    model_name: str,
) -> tuple[dict[tuple[int, str, str], dict[str, str]], list[tuple[int, str, str]]]:
    """Validate a prior CSV and retain only completely successful task rows."""
    expected_keys = [
        (run_number, sequence_id, path.stem)
        for run_number in range(1, runs + 1)
        for sequence_id, path in selected
    ]
    expected_set = set(expected_keys)
    completed = {}

    with output_path.open(newline="", encoding="utf-8") as output_file:
        reader = csv.DictReader(output_file)
        if reader.fieldnames != fieldnames:
            raise ValueError(
                "Cannot resume because the existing CSV schema differs from the "
                "current structured reliability schema."
            )
        for line_number, row in enumerate(reader, start=2):
            try:
                key = (int(row["run"]), row["sequence_id"], row["image_id"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid task metadata on CSV line {line_number}") from exc
            if key not in expected_set:
                raise ValueError(
                    f"CSV line {line_number} does not match the requested image/run set: {key}"
                )
            if row["selection_seed"] != str(seed):
                raise ValueError(
                    f"CSV line {line_number} used selection seed {row['selection_seed']}, "
                    f"not {seed}."
                )
            if row["model_name"] != model_name:
                raise ValueError(
                    f"CSV line {line_number} used model {row['model_name']}, not {model_name}."
                )

            complete = (
                not row["generation_error"].strip()
                and not row["parse_error"].strip()
                and all(row[field].strip() for field in result_fields)
            )
            if complete:
                if key in completed:
                    raise ValueError(f"Duplicate successful task in existing CSV: {key}")
                completed[key] = row

    return completed, expected_keys


def compact_resume_file(
    output_path: Path,
    fieldnames: list[str],
    completed: dict[tuple[int, str, str], dict[str, str]],
    expected_keys: list[tuple[int, str, str]],
) -> Path:
    """Atomically replace the active CSV with successful rows, retaining a backup."""
    backup_path = output_path.with_name(
        f"{output_path.stem}.pre_resume{output_path.suffix}"
    )
    counter = 2
    while backup_path.exists():
        backup_path = output_path.with_name(
            f"{output_path.stem}.pre_resume_{counter}{output_path.suffix}"
        )
        counter += 1
    shutil.copy2(output_path, backup_path)

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            prefix=f".{output_path.stem}_resume_",
            suffix=".tmp",
            dir=output_path.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            writer = csv.DictWriter(temporary_file, fieldnames=fieldnames)
            writer.writeheader()
            for key in expected_keys:
                if key in completed:
                    writer.writerow(completed[key])
        temporary_path.replace(output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return backup_path


def run_reliability_analysis(
    image_root: Path,
    prompt_path: Path,
    output_path: Path,
    sequence_ids: Iterable[str] | None = None,
    runs: int = DEFAULT_RUNS,
    seed: int = 42,
    model_name: str = DEFAULT_MODEL,
    quantize: bool = True,
    overwrite: bool = False,
    resume: bool = False,
) -> None:
    """Run each selected image through InternVL repeatedly and save every response."""
    if runs < 1:
        raise ValueError("runs must be at least 1")
    if not prompt_path.is_file():
        raise FileNotFoundError(f"Prompt file does not exist: {prompt_path}")
    if overwrite and resume:
        raise ValueError("--overwrite and --resume cannot be used together")
    if resume and not output_path.exists():
        raise FileNotFoundError(f"Cannot resume because the output does not exist: {output_path}")
    if output_path.exists() and not overwrite and not resume:
        raise FileExistsError(
            f"Output already exists: {output_path}. Use --resume or --overwrite."
        )

    sequences = resolve_sequences(image_root, sequence_ids)
    selected = select_images(sequences, seed)
    if len(selected) != 10:
        raise RuntimeError(f"Expected 10 selected images, got {len(selected)}")

    fieldnames, result_fields = output_fieldnames()
    total_calls = len(selected) * runs
    completed: dict[tuple[int, str, str], dict[str, str]] = {}
    if resume:
        completed, expected_keys = load_completed_rows(
            output_path=output_path,
            fieldnames=fieldnames,
            result_fields=result_fields,
            selected=selected,
            runs=runs,
            seed=seed,
            model_name=model_name,
        )
        backup_path = compact_resume_file(
            output_path, fieldnames, completed, expected_keys
        )
        print(
            f"Resume found {len(completed)}/{total_calls} successful calls; "
            f"the original CSV was backed up to {backup_path}."
        )
        if len(completed) == total_calls:
            print("All requested calls are already complete.")
            return

    # Import heavyweight ML modules only when an analysis is actually run. This keeps
    # image selection and command-line validation testable without a GPU environment.
    from vlm_analysis import ImageAnalyzer
    from vlm_utils import build_generation_config, load_model

    prompt = prompt_path.read_text(encoding="utf-8")
    model, tokenizer = load_model(model_name, quant=quantize)
    model_config = build_generation_config(tokenizer)
    analyzer = ImageAnalyzer(model, tokenizer, model_config)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if resume else ("w" if overwrite else "x")
    failures = 0

    with output_path.open(mode, newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        if not resume:
            writer.writeheader()

        progress = tqdm(
            total=total_calls,
            initial=len(completed),
            desc="Reliability analysis",
            unit="call",
        )
        for run_number in range(1, runs + 1):
            for sequence_id, path in selected:
                task_key = (run_number, sequence_id, path.stem)
                if task_key in completed:
                    continue
                response = ""
                generation_error = ""
                parse_error = ""
                fatal_generation_error = False
                parsed_values = {field: "" for field in result_fields}
                try:
                    response = analyzer.analyze_image(path, prompt)
                    parsed_values, parse_error = parse_response(response)
                except Exception as exc:  # Preserve failures in the output before continuing.
                    failures += 1
                    generation_error = f"{type(exc).__name__}: {exc}"
                    error_text = str(exc).lower()
                    fatal_generation_error = isinstance(exc, RuntimeError) and any(
                        marker in error_text
                        for marker in (
                            "cuda error",
                            "cuda out of memory",
                            "device-side assert",
                        )
                    )

                if parse_error:
                    failures += 1

                writer.writerow(
                    {
                        "sequence_id": sequence_id,
                        "image_id": path.stem,
                        "image_path": path.as_posix(),
                        "run": run_number,
                        "selection_seed": seed,
                        "model_name": model_name,
                        **parsed_values,
                        "generation_error": generation_error,
                        "parse_error": parse_error,
                        # Retain raw text only when it could not be parsed completely.
                        "unparsed_response": response if parse_error else "",
                    }
                )
                output_file.flush()
                progress.update(1)
                if fatal_generation_error:
                    progress.close()
                    raise RuntimeError(
                        "Stopping after a fatal CUDA error; the failed row was saved. "
                        "Continuing would only produce more invalid rows."
                    ) from None
        progress.close()

    if failures:
        raise RuntimeError(
            f"Reliability analysis completed with {failures} failed calls. "
            f"Details were saved to {output_path}."
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select two unmasked images from each of five sequences and analyse "
            "each image repeatedly with InternVL."
        )
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=PROJECT_ROOT / "images",
        help="Directory whose immediate child folders are image sequences (default: images).",
    )
    parser.add_argument(
        "--sequence-ids",
        nargs="+",
        help="Exactly five sequence directory names; otherwise they are auto-discovered.",
    )
    parser.add_argument(
        "--prompt",
        type=Path,
        default=PROJECT_ROOT / "prompts" / "prompt.txt",
        help="VLM prompt file (default: prompts/prompt.txt).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "reliability_analysis.csv",
        help="Output CSV path (default: results/reliability_analysis.csv).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_RUNS,
        help=f"Calls per selected image (default: {DEFAULT_RUNS}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random image-selection seed (default: 42).",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model name.")
    parser.add_argument(
        "--no-quantization",
        dest="quantize",
        action="store_false",
        help="Load the model without the default 8-bit quantization.",
    )
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output CSV instead of refusing to overwrite it.",
    )
    output_mode.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Keep successful rows in an existing CSV and retry failed or missing "
            "image/run combinations. The original CSV is backed up first."
        ),
    )
    parser.set_defaults(quantize=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_reliability_analysis(
        image_root=args.image_root,
        prompt_path=args.prompt,
        output_path=args.output,
        sequence_ids=args.sequence_ids,
        runs=args.runs,
        seed=args.seed,
        model_name=args.model,
        quantize=args.quantize,
        overwrite=args.overwrite,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
