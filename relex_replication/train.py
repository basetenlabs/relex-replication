"""500-step GRPO (TRL + colocated vLLM) on one environment, saving the checkpoints RELEX needs.

Launch on the config's world size (2 nodes x 8 H200 for 1.5B/4B, 1 node x 8 B200 for 8B), e.g.
    accelerate launch --multi_gpu --num_machines 2 --num_processes 16 --machine_rank R \
        --main_process_ip HOST --mixed_precision bf16 --dynamo_backend no \
        -m relex_replication.train --config configs/qwen3_4b.json --env kk --output runs/4b-kk

Writes base_model/ (the pinned base), trajectory/global_step_N/ (model-only
snapshots) and resume/checkpoint-N/ (full Trainer state every 10 steps).
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path

import torch
from torch.utils.data import Sampler

from .data import DEFAULT_CONFIG, ENV_NAMES, completion_cap, load_config
from .envs import get_env


def parse_steps(spec: str) -> set[int]:
    """'1-3,10' -> {1, 2, 3, 10}."""
    steps = set()
    for part in spec.split(","):
        first, _, last = part.partition("-")
        steps.update(range(int(first), int(last or first) + 1))
    return steps


class RestartStableRepeatSampler(Sampler[int]):
    """TRL 1.11 ``RepeatSampler`` ordering that also honours ``set_epoch``.

    Upstream keeps one generator that advances every epoch but has no
    ``set_epoch``, so a resumed Trainer would replay epoch 0's permutation.
    Replaying the preceding ``randperm`` calls from the seed restores the
    exact upstream order for any epoch; a fresh run is identical to upstream.
    """

    def __init__(self, data_source, mini_repeat_count: int, batch_size: int = 1, repeat_count: int = 1,
                 shuffle: bool = True, seed: int | None = None) -> None:
        self.num_samples = len(data_source)
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _permutation(self, epoch: int) -> list[int]:
        if not self.shuffle:
            return list(range(self.num_samples))
        generator = torch.Generator()
        if self.seed is not None:
            generator.manual_seed(self.seed)
        for _ in range(epoch + 1):
            indexes = torch.randperm(self.num_samples, generator=generator).tolist()
        return indexes

    def __iter__(self) -> Iterator[int]:
        indexes = self._permutation(self.epoch)
        self.epoch += 1
        for start in range(0, len(indexes) - self.batch_size + 1, self.batch_size):
            chunk = indexes[start:start + self.batch_size]
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield index

    def __len__(self) -> int:
        return (self.num_samples // self.batch_size) * self.batch_size * self.mini_repeat_count * self.repeat_count


@contextmanager
def without_fused_allreduce_rms():
    """vLLM 0.19 auto-enables a FlashInfer all-reduce/RMSNorm fusion that, under
    TRL's offset tensor-parallel groups, uses the TP-relative rank as the CUDA
    device. TRL exposes no vLLM kwargs, so inject the opt-out at construction."""
    from trl.generation import vllm_generation

    original = vllm_generation.LLM

    def llm(*args, **kwargs):
        return original(*args, compilation_config={"pass_config": {"fuse_allreduce_rms": False}}, **kwargs)

    vllm_generation.LLM = llm
    try:
        yield
    finally:
        vllm_generation.LLM = original


def install_completion_caps(trainer, requested: int, context: int, prompt_limit: int) -> None:
    """Give each rollout min(requested, context - prompt_tokens) tokens instead of
    TRL's single max_completion_length, so long prompts never overflow context."""
    llm = trainer.vllm_generation.llm
    original = llm.generate

    def generate(prompts, *args, sampling_params, **kwargs):
        if sampling_params.max_tokens != requested:
            raise RuntimeError("TRL changed the requested completion length")
        capped = []
        for prompt in prompts:
            params = sampling_params.clone()
            params.max_tokens = completion_cap(len(prompt["prompt_token_ids"]), requested, context, prompt_limit)
            capped.append(params)
        return original(list(prompts), *args, sampling_params=capped, **kwargs)

    llm.generate = generate


def build_trainer_class(snapshot_steps: set[int], trajectory: Path, data: Mapping):
    from transformers import TrainerCallback
    from trl import GRPOTrainer

    class SnapshotTrainer(GRPOTrainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            install_completion_caps(self, self.args.max_completion_length, data["context_tokens"],
                                    data["max_prompt_tokens"])

        def _tokenize_prompts(self, prompts: list[str]):
            # Prompts are already rendered with the chat markup; tokenize them literally.
            encoded = self.processing_class(text=prompts, add_special_tokens=False, truncation=False, padding=False)
            return encoded["input_ids"], None, {}

        def _get_train_sampler(self, dataset=None):
            return RestartStableRepeatSampler(
                data_source=self.train_dataset if dataset is None else dataset,
                mini_repeat_count=self.num_generations,
                batch_size=self.args.generation_batch_size // self.num_generations,
                repeat_count=self.num_iterations * self.args.steps_per_generation,
                shuffle=self.shuffle_dataset,
                seed=self.args.seed,
            )

        def _save_checkpoint(self, model, trial):
            step = self.state.global_step
            if step in snapshot_steps:
                self.save_model(str(trajectory / f"global_step_{step}"))
            if step % self.args.save_steps == 0 or step == self.args.max_steps:
                super()._save_checkpoint(model, trial)

    class SaveAtSnapshots(TrainerCallback):
        def on_step_end(self, args, state, control, **_):
            if state.global_step in snapshot_steps:
                control.should_save = True
            return control

    return SnapshotTrainer, SaveAtSnapshots


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--env", choices=ENV_NAMES, default="math")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path, help="resume/checkpoint-N directory to continue from")
    args = parser.parse_args(argv)

    multiprocessing.set_start_method("spawn", force=True)
    from accelerate import PartialState
    from datasets import Dataset
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    from trl import GRPOConfig

    config = load_config(args.config, args.env)
    env = get_env(args.env, config)
    model, data, training = config["model"], config["data"], config["training"]
    base_path = args.output / "base_model"
    with PartialState().local_main_process_first():
        snapshot_download(repo_id=model["id"], revision=model["revision"], local_dir=base_path)
        tokenizer = AutoTokenizer.from_pretrained(base_path, local_files_only=True)
        rows = env.train_rows(config, tokenizer)
    if args.env != "math":
        # As in the original non-MATH runs.
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
    dataset = Dataset.from_list(rows)
    longest = max(len(ids) for ids in tokenizer(dataset["prompt"], add_special_tokens=False)["input_ids"])
    if longest > data["max_prompt_tokens"]:
        raise ValueError(f"a training prompt has {longest} tokens; limit is {data['max_prompt_tokens']}")

    # Keep 32 prompts x 8 completions per optimizer update on any world size.
    # Only the config's world size reproduces the original run's data sharding.
    grpo = dict(training["grpo"])
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    per_update = grpo["per_device_train_batch_size"] * world_size
    if training["completions_per_update"] % per_update:
        raise ValueError(f"{training['completions_per_update']} completions do not split over {world_size} ranks")
    grpo["gradient_accumulation_steps"] = training["completions_per_update"] // per_update
    if world_size != training["world_size"]:
        print(f"warning: world size {world_size} != {training['world_size']} used for the reference run")

    grpo_config = GRPOConfig(
        output_dir=str(args.output / "resume"),
        model_init_kwargs={"dtype": model["dtype"], "attn_implementation": model["attn_implementation"],
                           "local_files_only": True, "trust_remote_code": False},
        **grpo,
    )
    trainer_class, callback_class = build_trainer_class(
        parse_steps(training["snapshot_steps"]), args.output / "trajectory", data)
    with without_fused_allreduce_rms():
        trainer = trainer_class(
            model=str(base_path),
            reward_funcs=env.reward,
            args=grpo_config,
            train_dataset=dataset,
            processing_class=tokenizer,
            callbacks=[callback_class()],
        )
    trainer.train(resume_from_checkpoint=str(args.resume) if args.resume else None)


if __name__ == "__main__":
    main()
