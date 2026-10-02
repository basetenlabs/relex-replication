import json

import pytest
import torch
from safetensors.torch import load_file

from relex_replication import extrapolate as ex
from relex_replication.data import CONFIGS, load_config
from relex_replication.envs import FunctionCalling, IFEval, KnightsKnaves, bfcl_inputs, normalize_missing_think_end
from relex_replication.train import parse_steps

# Golden cases below are taken from the original runs' test suite.
KK_TRUTH = {"solution_text_format": "Alice is a knight\nBob is a knave"}
KK_ANSWER = "<answer>Alice is a knight\nBob is a knave</answer>"


@pytest.mark.parametrize(("text", "expected"), [
    ("Reasoning." + KK_ANSWER, 3),
    (KK_ANSWER, 3),
    ("Reasoning.</think>" + KK_ANSWER, 3),
    ("Reasoning." + KK_ANSWER.replace("Bob is a knave", "Bob is a knight"), -0.5),
    ("Reasoning." + KK_ANSWER.replace("Bob is a knave", ""), -1),
    ("Reasoning." + KK_ANSWER.replace("</answer>", ""), -3),
    ("Reasoning." + KK_ANSWER + KK_ANSWER, -3),
    ("<think>Reasoning." + KK_ANSWER, -3),
    ("Reasoning." + KK_ANSWER + "</think>", -3),
    ("Reasoning without an answer", -3),
    ("Assistant: Reasoning." + KK_ANSWER, -3),
])
def test_kk_relaxed_logic_rl_reward(text, expected):
    env = KnightsKnaves()
    assert env.reward([text], [json.dumps(KK_TRUTH)]) == [expected]
    assert env.score({"ground_truth": KK_TRUTH}, text)["correct"] is (expected == 3)
    normalized = normalize_missing_think_end(text)
    assert normalized == text or normalized.replace("</think>", "", 1) == text


@pytest.mark.parametrize(("name", "args", "good", "bad"), [
    ("verify_keywords", {"keyword_list": ["red", "blue"]}, "red blue", "red"),
    ("validate_forbidden_words", {"forbidden_words": ["cat"]}, "dog", "cat"),
    ("validate_word_constraint", {"N": 2, "quantifier": "at most"}, "one two", "one two three"),
    ("verify_bullet_points", {"N": 2}, "* one\n* two", "* one"),
    ("validate_title", {}, "<<title>>\nbody", "title body"),
    ("validate_json_format", {}, '{"x":1}', "not json"),
    ("validate_uppercase", {}, "HELLO", "Hello"),
    ("validate_end", {"end_phrase": "END"}, "hello END", "END hello"),
    ("validate_no_commas", {}, "hello world", "hello, world"),
])
def test_ifeval_training_reward(name, args, good, bad):
    truth = json.dumps({"func_name": name, **args})
    assert IFEval().reward([good, bad, "<think>x</think>" + good], [truth] * 3) == [1.0, 0.0, 1.0]


def test_ifeval_official_scorer_strict_and_loose():
    row = {"ground_truth": {"key": 1, "prompt": "Write without commas.",
                            "instruction_id_list": ["punctuation:no_comma"], "kwargs": [{}]}}
    assert IFEval().score(row, "No commas here.")["correct"]
    result = IFEval().score(row, "Commas, here.")
    assert not result["correct"] and not result["strict_correct"]
    assert IFEval().summarize([{**result, "question_id": 1}])["strict_prompt_accuracy"] == 0


FC_GOLD = {"tools": [{"name": "math.add", "parameters": {"a": {"type": "int", "required": True},
                                                         "b": {"type": "int", "required": True}}}],
           "answers": [{"name": "math.add", "arguments": {"a": 1, "b": 2}}]}


@pytest.mark.parametrize(("completion", "expected"), [
    (json.dumps([{"name": "math.add", "arguments": {"b": 2, "a": 1}}]), 1.0),
    (json.dumps([{"name": "math.add", "arguments": {"a": True, "b": 2}}]), 0.0),
    (json.dumps([{"name": "math.add", "arguments": {"a": 1.0, "b": 2}}]), 0.0),
    (json.dumps([{"name": "math.add", "arguments": {"a": 1}}]), 0.0),
    (json.dumps([{"name": "math.add", "arguments": {"a": 1, "b": 2, "c": 3}}]), 0.0),
    ("", 0.0), ("math.add(a=1,b=2)", 0.0), ('[{"name":"math.add","arguments":null}]', 0.0),
])
def test_function_calling_reward_delegates_to_bfcl(completion, expected):
    assert FunctionCalling().reward([completion], [json.dumps(FC_GOLD)]) == [expected]


def test_bfcl_string_normalization_and_schema_conversion():
    truth = {"tools": [{"name": "f", "parameters": {"x": {"type": "str", "required": True}}},
                       {"name": "g", "parameters": {"y": {"type": "List[int], optional"}}}],
             "answers": [{"name": "f", "arguments": {"x": "New York"}}, {"name": "g", "arguments": {}}]}
    env = FunctionCalling()
    calls = '[{"name":"g","arguments":{}},{"name":"f","arguments":{"x":"new-york"}}]'
    assert env.reward([calls, calls.replace("new-york", "Boston")], [json.dumps(truth)] * 2) == [1.0, 0.0]
    assert bfcl_inputs(truth)[0][1]["parameters"]["properties"]["y"]["items"] == {"type": "integer"}
    with pytest.raises(ValueError, match="unsupported"):
        bfcl_inputs({"tools": [{"name": "f", "parameters": {"x": {"type": "mystery"}}}],
                     "answers": [{"name": "f", "arguments": {"x": 1}}]})


def test_configs_resolve_to_the_original_recipes():
    four_kk, four_if = load_config(CONFIGS / "qwen3_4b.json", "kk"), load_config(CONFIGS / "qwen3_4b.json", "ifeval")
    assert (four_kk["training"]["grpo"]["beta"], four_kk["training"]["grpo"]["max_completion_length"]) == (0.0, 2048)
    assert four_if["training"]["grpo"]["vllm_max_model_length"] == 8192
    assert four_if["evaluation"]["max_output_tokens"] == 4096
    four_math = load_config(CONFIGS / "qwen3_4b.json", "math")
    assert four_math["training"]["grpo"]["beta"] == 0.001 and four_math["evaluation"]["context_tokens"] == 16384
    assert "enable_prefix_caching" not in four_math["evaluation"]["vllm"]
    eight_if, eight_fc = load_config(CONFIGS / "qwen3_8b.json", "ifeval"), load_config(CONFIGS / "qwen3_8b.json", "fc")
    assert eight_if["training"]["grpo"]["learning_rate"] == 1e-5 and eight_if["evaluation"]["context_tokens"] == 4096
    assert eight_fc["training"]["grpo"]["learning_rate"] == 3e-6 and eight_fc["evaluation"]["context_tokens"] == 8192
    assert parse_steps(eight_fc["training"]["snapshot_steps"]) >= set(range(1, 56)) | {100, 500}
    assert len(json.loads((CONFIGS / "xlam_train_indices.json").read_text())) == 8457
    with pytest.raises(ValueError):
        load_config(CONFIGS / "qwen3_8b.json", "math")


def test_lam_rescales_the_relex_coefficient(run, tmp_path):
    ex.build(run, tmp_path / "lam1", prefix=3, lam=1.0)
    ex.build(run, tmp_path / "plain", prefix=3)
    ex.build(run, tmp_path / "half", prefix=3, lam=0.5)
    plain, half = load_file(tmp_path / "plain/model.safetensors"), load_file(tmp_path / "half/model.safetensors")
    assert all(torch.equal(t, load_file(tmp_path / "lam1/model.safetensors")[n]) for n, t in plain.items())
    base = ex.TensorStore(run / "base_model")
    snapshots = [ex.TensorStore(run / "trajectory" / f"global_step_{s}") for s in (1, 2, 3)]
    for name in base.names:
        direction, coefficients = ex.stream_rank1(base, snapshots, name)
        coefficient = 0.5 * ex.predict_coefficient(coefficients, 500)
        w0 = base.read_block(name, 0, base.shape(name)[0]).reshape(-1)
        expected = ex.materialize(w0, torch.from_numpy(direction) * coefficient)
        assert torch.equal(half[name].reshape(-1), expected)
    with pytest.raises(ValueError):
        ex.build(run, tmp_path / "bad", alpha=2.0, lam=0.5)


def test_fp16_first_update_scaling_is_exact(run, tmp_path):
    ex.build(run, tmp_path / "out", alpha=500.0, fp16_delta=True)
    base = load_file(run / "base_model/model.safetensors")
    step1 = load_file(run / "trajectory/global_step_1/model.safetensors")
    out = load_file(tmp_path / "out/model.safetensors")
    for name, w0 in base.items():
        expected = (w0.float() + 500.0 * (step1[name].half() - w0.half()).float()).to(torch.bfloat16)
        assert torch.equal(out[name], expected)
