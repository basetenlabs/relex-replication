"""Greedy MATH-test evaluation with the official RELEX extractor and grader.

Generation runs in a spawned child so that scoring, whose symbolic timeout
forks worker processes, happens in a parent that never initialized CUDA.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
from pathlib import Path

from .data import DEFAULT_CONFIG, completion_cap, load_config, official_extractor, test_questions


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    partial = path.with_suffix(".partial")
    partial.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    partial.replace(path)


def generate(config: dict, model: str, revision: str | None, output: Path) -> None:
    """Generate one greedy response per question in fixed batches, resuming per batch."""
    from vllm import LLM, SamplingParams

    evaluation, data = config["evaluation"], config["data"]
    questions = test_questions(config)
    batch_size = evaluation["batch_size"]
    pending = [start for start in range(0, len(questions), batch_size)
               if not (output / f"batch_{start // batch_size}.jsonl").exists()]
    if not pending:
        return
    # Every arm uses the pinned base tokenizer so prompt token IDs are identical.
    llm = LLM(
        model=model,
        revision=revision,
        tokenizer=config["model"]["id"],
        tokenizer_revision=config["model"]["revision"],
        tensor_parallel_size=1,
        dtype=config["model"]["dtype"],
        gpu_memory_utilization=evaluation["gpu_memory_utilization"],
        trust_remote_code=False,
        seed=evaluation["seed"],
        max_model_len=data["context_tokens"],
    )
    tokenizer = llm.get_tokenizer()
    for start in pending:
        batch = questions[start:start + batch_size]
        prompt_tokens = [len(tokenizer.encode(q["prompt"], add_special_tokens=False)) for q in batch]
        caps = [completion_cap(n, evaluation["max_output_tokens"], data["context_tokens"], data["max_prompt_tokens"])
                for n in prompt_tokens]
        params = [SamplingParams(n=1, temperature=0.0, top_p=1.0, top_k=-1, max_tokens=cap, seed=evaluation["seed"])
                  for cap in caps]
        outputs = llm.generate([q["prompt"] for q in batch], sampling_params=params, use_tqdm=True,
                               tokenization_kwargs={"add_special_tokens": False})
        records = []
        for question, n_prompt, cap, result in zip(batch, prompt_tokens, caps, outputs, strict=True):
            if len(result.prompt_token_ids) != n_prompt:
                raise RuntimeError(f"vLLM tokenized question {question['question_id']} differently")
            completion = result.outputs[0]
            records.append({
                "question_id": question["question_id"],
                "response": completion.text,
                "finish_reason": completion.finish_reason,
                "prompt_tokens": n_prompt,
                "output_tokens": len(completion.token_ids),
                "output_cap": cap,
            })
        write_jsonl(output / f"batch_{start // batch_size}.jsonl", records)
        print(f"{model}: {start + len(batch)}/{len(questions)}", flush=True)


def score(config: dict, output: Path) -> dict:
    """Grade serially, as the official evaluator does, with its 1s symbolic timeout."""
    from ._vendor.relex_eval.grader import math_equal

    extract = official_extractor()
    questions = test_questions(config)
    batches = sorted(output.glob("batch_*.jsonl"), key=lambda p: int(p.stem.split("_")[1]))
    generations = [row for path in batches for row in read_jsonl(path)]
    if [row["question_id"] for row in generations] != [q["question_id"] for q in questions]:
        raise RuntimeError("generations are incomplete or out of order")
    scores = []
    for question, row in zip(questions, generations, strict=True):
        prediction = extract(row["response"])
        scores.append({
            "question_id": question["question_id"],
            "gold_answer": question["gold_answer"],
            "prediction": prediction,
            "correct": bool(math_equal(prediction, question["gold_answer"], timeout=True)),
        })
    write_jsonl(output / "generations.jsonl", generations)
    write_jsonl(output / "scores.jsonl", scores)
    count = len(scores)
    correct = sum(s["correct"] for s in scores)
    cap_hits = sum(g["finish_reason"] == "length" and g["output_tokens"] >= g["output_cap"] for g in generations)
    summary = {
        "questions": count,
        "correct": correct,
        "accuracy": correct / count,
        "cap_hit_rate": cap_hits / count,
        "mean_output_tokens": sum(g["output_tokens"] for g in generations) / count,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="local model directory or Hugging Face repo id")
    parser.add_argument("--revision", help="Hugging Face revision for a hub model")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    args.output.mkdir(parents=True, exist_ok=True)
    child = multiprocessing.get_context("spawn").Process(
        target=generate, args=(config, args.model, args.revision, args.output))
    child.start()
    child.join()
    if child.exitcode != 0:
        raise SystemExit(f"generation failed with exit code {child.exitcode}")
    summary = score(config, args.output)
    print(f"{args.model}: {summary['correct']}/{summary['questions']} = {summary['accuracy']:.2%} "
          f"(cap hits {summary['cap_hit_rate']:.2%})")


if __name__ == "__main__":
    main()
