import copy
import hashlib
import io
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from relex_replication import data, evaluate, train
from relex_replication.analyze import compare
from relex_replication.envs import get_env


def test_jsonl_preserves_unicode_line_separators(tmp_path):
    rows = [{"question_id": 1, "response": "first\u2028second\u2029third\x85fourth"}]
    path = tmp_path / "batch_0.jsonl"
    evaluate.write_jsonl(path, rows)
    assert data.read_jsonl(path) == rows


def test_concurrent_cache_misses_publish_verified_files(tmp_path, monkeypatch):
    payload = b"pinned dataset bytes"
    barrier = threading.Barrier(2)

    def open_url(*args, **kwargs):
        barrier.wait(timeout=5)
        return io.BytesIO(payload)

    monkeypatch.setattr(data, "urlopen", open_url)
    digest = hashlib.sha256(payload).hexdigest()
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(
            lambda _: data.download_verified("https://example.org/data", digest, tmp_path), range(2)))
    assert results[0] == results[1]
    assert results[0].read_bytes() == payload
    assert list(results[0].parent.iterdir()) == [results[0]]


def test_download_failure_cleans_partial_file(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "urlopen", lambda *args, **kwargs: io.BytesIO(b"wrong bytes"))
    with pytest.raises(ValueError, match="sha256"):
        data.download_verified("https://example.org/data", "0" * 64, tmp_path)
    assert not list(tmp_path.rglob("*.*"))


def test_evaluation_resume_rejects_changed_model_or_recipe(run, tmp_path):
    model, output = run / "base_model", tmp_path / "eval"
    config = data.load_config()
    evaluate.prepare_output(config, "math", str(model), None, output)
    evaluate.prepare_output(config, "math", str(model), None, output)
    altered = copy.deepcopy(config)
    altered["evaluation"]["max_output_tokens"] -= 1
    with pytest.raises(ValueError, match="different weights or settings"):
        evaluate.prepare_output(altered, "math", str(model), None, output)
    (model / "config.json").write_text('{"model_type":"changed"}')
    with pytest.raises(ValueError, match="different weights or settings"):
        evaluate.prepare_output(config, "math", str(model), None, output)


def test_evaluation_refuses_unidentified_existing_batches(tmp_path):
    (tmp_path / "batch_0.jsonl").write_text('{}\n')
    with pytest.raises(ValueError, match="no identity"):
        evaluate.prepare_output({}, "math", "hub/model", "a" * 40, tmp_path)


def test_scoring_rejects_duplicate_generations(tmp_path, monkeypatch):
    class Environment:
        def test_rows(self, config):
            return [{"question_id": 0}]

    monkeypatch.setattr(evaluate, "get_env", lambda *args: Environment())
    for batch in range(2):
        evaluate.write_jsonl(tmp_path / f"batch_{batch}.jsonl", [{"question_id": 0, "response": ""}])
    with pytest.raises(RuntimeError, match="duplicate"):
        evaluate.score({}, "math", tmp_path)


def test_paired_analysis_rejects_misaligned_generations():
    generations = [{"question_id": i} for i in (1, 0)]
    scores = [{"question_id": i, "correct": False} for i in (0, 1)]
    with pytest.raises(ValueError, match="same order"):
        compare((generations, scores), (generations, scores))


@pytest.mark.parametrize(("requested", "context"), [(0, 4096), (10, 100), (10, 99)])
def test_completion_cap_requires_room_for_output(requested, context):
    with pytest.raises(ValueError, match="completion budget"):
        data.completion_cap(100, requested, context, 2048)


def test_training_output_cannot_mix_recipes_or_restart_over_snapshots(tmp_path):
    config = data.load_config()
    train.prepare_output(config, "math", 16, tmp_path, None)
    checkpoint = tmp_path / "resume/checkpoint-10"
    checkpoint.mkdir(parents=True)
    (checkpoint / "trainer_state.json").write_text('{"global_step": 10}')
    train.prepare_output(config, "math", 16, tmp_path, checkpoint)
    with pytest.raises(ValueError, match="already started"):
        train.prepare_output(config, "math", 16, tmp_path, None)
    with pytest.raises(ValueError, match="different recipe"):
        train.prepare_output(config, "math", 8, tmp_path, checkpoint)
    with pytest.raises(ValueError, match="this run"):
        train.prepare_output(config, "math", 16, tmp_path, Path("elsewhere/checkpoint-10"))


@pytest.mark.parametrize(("model", "world", "accumulation"), [
    ("qwen25_math_1.5b", 16, 8), ("qwen3_4b", 16, 16), ("qwen3_8b", 8, 32),
])
def test_original_training_batch_arithmetic(model, world, accumulation):
    config = data.load_config(data.CONFIGS / f"{model}.json", "kk" if model.startswith("qwen3") else "math")
    args = train.training_arguments(config, world)
    assert args["gradient_accumulation_steps"] == accumulation
    assert args["generation_batch_size"] == world * args["per_device_train_batch_size"] * accumulation
    with pytest.raises(ValueError):
        train.training_arguments(config, 1)


def test_sampler_resume_matches_continuous_epochs():
    kwargs = dict(data_source=range(65), mini_repeat_count=2, batch_size=32, repeat_count=3, seed=1)
    continuous = train.RestartStableRepeatSampler(**kwargs)
    epochs = [list(continuous) for _ in range(3)]
    restored = train.RestartStableRepeatSampler(**kwargs)
    restored.set_epoch(2)
    assert list(restored) == epochs[2]
    assert len(epochs[0]) == len(continuous) == 64 * 2 * 3
    assert epochs[0] != epochs[1]


def test_function_calling_environments_do_not_share_mutable_config():
    config = data.load_config(data.CONFIGS / "qwen3_4b.json", "fc")
    first = get_env("fc", config)
    changed = copy.deepcopy(config)
    changed["data"]["bfcl"]["revision"] = "different"
    second = get_env("fc", changed)
    assert first.revision == config["data"]["bfcl"]["revision"]
    assert second.revision == "different"


def test_rollout_caps_preserve_sampling_and_cover_long_prompts():
    from types import SimpleNamespace

    calls = []

    def original(prompts, *, sampling_params, **kwargs):
        calls.append((prompts, sampling_params, kwargs))
        return "generated"

    class Params:
        max_tokens = 2048
        temperature = 1.0

        def clone(self):
            return copy.copy(self)

    llm = SimpleNamespace(generate=original)
    trainer = SimpleNamespace(vllm_generation=SimpleNamespace(llm=llm))
    train.install_completion_caps(trainer, requested=2048, context=4096, prompt_limit=2070)
    prompts = [{"prompt_token_ids": [1] * length} for length in (100, 2070)]
    params = Params()
    assert llm.generate(prompts, sampling_params=params, use_tqdm=False) == "generated"
    assert [p.max_tokens for p in calls[0][1]] == [2048, 2026]
    assert all(p.temperature == 1.0 for p in calls[0][1])
    assert params.max_tokens == 2048
    assert calls[0][0] == prompts


def test_generation_resumes_only_matching_batches(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    questions = [{"question_id": i, "prompt": "x" * (i + 1)} for i in range(3)]
    calls = []

    class Environment:
        def generation_groups(self, rows):
            return [rows]

        def test_rows(self, config):
            return questions

    class LLM:
        def __init__(self, **kwargs):
            pass

        def get_tokenizer(self):
            return SimpleNamespace(encode=lambda text, **kwargs: [7] * len(text))

        def generate(self, prompts, *, sampling_params, **kwargs):
            calls.append(prompts)
            return [SimpleNamespace(prompt_token_ids=p["prompt_token_ids"], outputs=[SimpleNamespace(
                text="answer\u2028with separator", token_ids=[8], finish_reason="stop")]) for p in prompts]

    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=LLM, SamplingParams=SimpleNamespace))
    monkeypatch.setattr(evaluate, "get_env", lambda *args: Environment())
    config = data.load_config()
    config["evaluation"]["batch_size"] = 2
    evaluate.generate(config, "math", "unused", None, tmp_path)
    assert calls == [[{"prompt_token_ids": [7]}, {"prompt_token_ids": [7, 7]}], [{"prompt_token_ids": [7, 7, 7]}]]
    assert data.read_jsonl(tmp_path / "batch_0.jsonl")[0]["response"] == "answer\u2028with separator"
    evaluate.generate(config, "math", "unused", None, tmp_path)
    assert len(calls) == 2
    evaluate.write_jsonl(tmp_path / "batch_0.jsonl", [{"question_id": 999}])
    with pytest.raises(ValueError, match="planned questions"):
        evaluate.generate(config, "math", "unused", None, tmp_path)
