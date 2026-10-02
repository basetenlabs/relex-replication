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
from pathlib import Path

from .data import DEFAULT_CONFIG, ENV_NAMES, completion_cap, load_config
from .envs import get_env


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    partial = path.with_suffix(".partial")
    partial.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
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
    pending = [(index, batch) for index, batch in planned if not (output / f"batch_{index}.jsonl").exists()]
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
        prompt_tokens = [len(tokenizer.encode(q["prompt"], add_special_tokens=False)) for q in batch]
        caps = [completion_cap(n, evaluation["max_output_tokens"], context, prompt_limit) for n in prompt_tokens]
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
        write_jsonl(output / f"batch_{index}.jsonl", records)
        print(f"{model}: batch {index} done", flush=True)


def score(config: dict, env_name: str, output: Path) -> dict:
    """Grade serially in question order, as the official evaluators do."""
    env = get_env(env_name, config)
    questions = env.test_rows(config)
    files = sorted(output.glob("batch_*.jsonl"), key=lambda p: int(p.stem.split("_")[1]))
    by_id = {row["question_id"]: row for path in files for row in read_jsonl(path)}
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
    args.output.mkdir(parents=True, exist_ok=True)
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
