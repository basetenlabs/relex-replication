"""Paired base -> target comparison: which base failures does the target repair?

Each question is binned by the *base* output: heavy repetition (more than half
of its whitespace 10-word windows are repeats), output-cap hit without heavy
repetition, or neither. The bins are descriptive; they do not show that
looping caused a failure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evaluate import read_jsonl

COHORTS = ("heavy_repetition", "cap_without_repetition", "neither")


def repeated_fraction(text: str, window: int = 10) -> float:
    """Fraction of whitespace word windows that repeat an earlier window."""
    words = text.split()
    windows = [tuple(words[i:i + window]) for i in range(len(words) - window + 1)]
    if not windows:
        return 0.0
    return (len(windows) - len(set(windows))) / len(windows)


def is_heavy_repetition(text: str) -> bool:
    return repeated_fraction(text) > 0.5


def is_cap_hit(generation: dict) -> bool:
    return generation["finish_reason"] == "length" or generation["output_tokens"] >= generation["output_cap"]


def cohort(generation: dict) -> str:
    if is_heavy_repetition(generation["response"]):
        return "heavy_repetition"
    return "cap_without_repetition" if is_cap_hit(generation) else "neither"


def load(run: Path) -> tuple[list[dict], list[dict]]:
    return read_jsonl(run / "generations.jsonl"), read_jsonl(run / "scores.jsonl")


def compare(base: tuple[list[dict], list[dict]], target: tuple[list[dict], list[dict]]) -> dict:
    (base_gens, base_scores), (target_gens, target_scores) = base, target
    ids = [row["question_id"] for row in base_scores]
    if ids != [row["question_id"] for row in target_scores]:
        raise ValueError("base and target cover different questions")
    table = {name: dict.fromkeys(
        ("questions", "base_failures", "repairs", "regressions", "net",
         "repairs_target_heavy_repetition", "repairs_target_cap_hit"), 0) for name in COHORTS}
    for base_gen, base_score, target_gen, target_score in zip(base_gens, base_scores, target_gens, target_scores,
                                                              strict=True):
        row = table[cohort(base_gen)]
        row["questions"] += 1
        row["base_failures"] += not base_score["correct"]
        if target_score["correct"] and not base_score["correct"]:
            row["repairs"] += 1
            row["repairs_target_heavy_repetition"] += is_heavy_repetition(target_gen["response"])
            row["repairs_target_cap_hit"] += is_cap_hit(target_gen)
        elif base_score["correct"] and not target_score["correct"]:
            row["regressions"] += 1
    for row in table.values():
        row["net"] = row["repairs"] - row["regressions"]
    table["total"] = {key: sum(table[name][key] for name in COHORTS) for key in table["neither"]}
    return table


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", type=Path, required=True, help="evaluate.py output for the base model")
    parser.add_argument("--target", type=Path, required=True, help="evaluate.py output for the target model")
    parser.add_argument("--json", type=Path, help="also write the table here")
    args = parser.parse_args(argv)
    table = compare(load(args.base), load(args.target))
    columns = list(table["total"])
    print("| base cohort | " + " | ".join(columns) + " |")
    print("|---" * (len(columns) + 1) + "|")
    for name, row in table.items():
        print(f"| {name} | " + " | ".join(str(row[c]) for c in columns) + " |")
    if args.json:
        args.json.write_text(json.dumps(table, indent=2) + "\n")


if __name__ == "__main__":
    main()
