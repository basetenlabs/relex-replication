"""Greedy test-set evaluation with each environment's official scorer.

math: 5,000 MATH test questions, official RELEX extractor and grader.
kk: 498 screened Knights & Knaves test puzzles, Logic-RL reward (relaxed closing think tag).
ifeval: Google's 541 IFEval prompts; loose prompt accuracy is primary, strict is also reported.
fc: the 3,641 official BFCL single-turn items, also reported on the 2,517 calling items.

Generation runs in a spawned child so that scoring, whose symbolic timeout
forks worker processes, happens in a parent that never initialized CUDA.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import re
from pathlib import Path

from .data import DEFAULT_CONFIG, ENV_NAMES, completion_cap, load_config, read_jsonl, sha256_file
from .envs import get_env


def prepare_output(config: dict, env_name: str, model: str, revision: str | None, output: Path) -> None:
    """Bind resumable batches to their weights and resolved recipe."""
    local = Path(model)
    if local.is_dir():
        files = sorted(p for p in local.iterdir() if p.suffix in {".safetensors", ".json"})
        if not any(p.suffix == ".safetensors" for p in files):
            raise ValueError(f"no safetensors model in {local}")
        identity = {"path": str(local.resolve()), "files": {p.name: sha256_file(p) for p in files}}
    else:
        if not revision or not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
            raise ValueError("Hub evaluation needs --revision with an immutable commit")
        identity = {"repo_id": model, "revision": revision}
    manifest = {"config": config, "env": env_name, "model": identity}
    path = output / "evaluation.json"
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError("evaluation output belongs to different weights or settings; use a new --output")
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("evaluation output has no identity; use a new --output")
        output.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=2) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    partial = path.with_suffix(".partial")
    partial.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    partial.replace(path)


def batches(env, questions: list[dict], batch_size: int) -> list[list[dict]]:
    return [group[start:start + batch_size] for group in env.generation_groups(questions)
            for start in range(0, len(group), batch_size)]


def generate(config: dict, env_name: str, model: str, revision: str | None, output: Path) -> None:
    """Generate one greedy response per question in fixed batches, resuming per batch."""
    from vllm import LLM, SamplingParams

    evaluation = config["evaluation"]
    env = get_env(env_name, config)
    planned = enumerate(batches(env, env.test_rows(config), evaluation["batch_size"]))
    pending = []
    for index, batch in planned:
        path = output / f"batch_{index}.jsonl"
        if path.exists():
            if [r["question_id"] for r in read_jsonl(path)] != [q["question_id"] for q in batch]:
                raise ValueError(f"saved batch {index} does not match the planned questions")
        else:
            pending.append((index, batch))
    if not pending:
        return
    context = evaluation["context_tokens"]
    prompt_limit = evaluation.get("max_prompt_tokens", config["data"]["max_prompt_tokens"])
    # Every arm uses the pinned base tokenizer so prompt token IDs are identical.
    llm = LLM(
        model=model,
        revision=revision,
        tokenizer=config["model"]["id"],
        tokenizer_revision=config["model"]["revision"],
        dtype=config["model"]["dtype"],
        trust_remote_code=False,
        seed=evaluation["seed"],
        max_model_len=context,
        **evaluation["vllm"],
    )
    tokenizer = llm.get_tokenizer()
    for index, batch in pending:
        token_ids = [tokenizer.encode(q["prompt"], add_special_tokens=False) for q in batch]
        prompt_tokens = [len(ids) for ids in token_ids]
        caps = [completion_cap(n, evaluation["max_output_tokens"], context, prompt_limit) for n in prompt_tokens]
        params = [SamplingParams(n=1, temperature=0.0, top_p=1.0, top_k=-1, max_tokens=cap, seed=evaluation["seed"])
                  for cap in caps]
        outputs = llm.generate([{"prompt_token_ids": ids} for ids in token_ids], sampling_params=params, use_tqdm=True)
        records = []
        for question, ids, cap, result in zip(batch, token_ids, caps, outputs, strict=True):
            if result.prompt_token_ids != ids or len(result.outputs) != 1:
                raise RuntimeError(f"vLLM tokenized question {question['question_id']} differently")
            completion = result.outputs[0]
            records.append({
                "question_id": question["question_id"],
                "response": completion.text,
                "finish_reason": completion.finish_reason,
                "prompt_tokens": len(ids),
                "output_tokens": len(completion.token_ids),
                "output_cap": cap,
            })
        write_jsonl(output / f"batch_{index}.jsonl", records)
        print(f"{model}: batch {index} done", flush=True)


def score(config: dict, env_name: str, output: Path) -> dict:
    """Grade serially in question order, as the official evaluators do."""
    env = get_env(env_name, config)
    questions = env.test_rows(config)
    files = sorted(output.glob("batch_*.jsonl"), key=lambda p: int(p.stem.split("_")[1]))
    records = [row for path in files for row in read_jsonl(path)]
    by_id = {row["question_id"]: row for row in records}
    if len(by_id) != len(records):
        raise RuntimeError("duplicate question IDs in generation batches")
    if set(by_id) != {q["question_id"] for q in questions}:
        raise RuntimeError("generations are incomplete or belong to another question set")
    generations = [by_id[q["question_id"]] for q in questions]
    scores = [{"question_id": q["question_id"], **env.score(q, g["response"])}
              for q, g in zip(questions, generations, strict=True)]
    write_jsonl(output / "generations.jsonl", generations)
    write_jsonl(output / "scores.jsonl", scores)
    count = len(scores)
    correct = sum(s["correct"] for s in scores)
    cap_hits = sum(g["finish_reason"] == "length" and g["output_tokens"] >= g["output_cap"] for g in generations)
    summary = {
        "env": env_name,
        "questions": count,
        "correct": correct,
        "accuracy": correct / count,
        "cap_hit_rate": cap_hits / count,
        "mean_output_tokens": sum(g["output_tokens"] for g in generations) / count,
        **env.summarize(scores),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="local model directory or Hugging Face repo id")
    parser.add_argument("--revision", help="Hugging Face revision for a hub model")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--env", choices=ENV_NAMES, default="math")
    args = parser.parse_args(argv)
    config = load_config(args.config, args.env)
    prepare_output(config, args.env, args.model, args.revision, args.output)
    child = multiprocessing.get_context("spawn").Process(
        target=generate, args=(config, args.env, args.model, args.revision, args.output))
    child.start()
    child.join()
    if child.exitcode != 0:
        raise SystemExit(f"generation failed with exit code {child.exitcode}")
    summary = score(config, args.env, args.output)
    print(f"{args.model}: {summary['correct']}/{summary['questions']} = {summary['accuracy']:.2%} "
          f"(cap hits {summary['cap_hit_rate']:.2%})")
    if "callable" in summary:
        calls = summary["callable"]
        print(f"calling items: {calls['correct']}/{calls['questions']} = {calls['accuracy']:.2%}")


if __name__ == "__main__":
    main()
