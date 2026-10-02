"""Config, pinned data downloads, MATH prompts and completion caps."""

from __future__ import annotations

import hashlib
import json
import warnings
from functools import cache
from pathlib import Path
from urllib.request import urlopen

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
DEFAULT_CONFIG = CONFIGS / "qwen25_math_1.5b.json"
DATA_CACHE = Path.home() / ".cache/relex_replication"
ENV_NAMES = ("math", "kk", "ifeval", "fc")


def merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        nested = isinstance(value, dict) and isinstance(merged.get(key), dict)
        merged[key] = merge(merged[key], value) if nested else value
    return merged


def load_config(path: Path = DEFAULT_CONFIG, env: str = "math") -> dict:
    """Model config, then configs/envs.json[env], then the model's own envs[env] overrides.

    A config without an `envs` table (the original 1.5B MATH run) is used as is.
    """
    config = json.loads(Path(path).read_text())
    if "envs" not in config:
        if env != "math":
            raise ValueError(f"{path} only defines the math environment")
        return config
    overrides = config.pop("envs")
    if env not in overrides:
        raise ValueError(f"{path} has no {env} run")
    shared = json.loads((CONFIGS / "envs.json").read_text()).get(env, {})
    return merge(merge(config, shared), overrides[env])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def download_verified(url: str, sha256: str, cache_dir: Path = DATA_CACHE) -> Path:
    """Download once, refusing bytes that differ from the pinned digest."""
    destination = cache_dir / sha256 / Path(url).name
    if destination.exists() and sha256_file(destination) == sha256:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(".partial")
    with urlopen(url, timeout=120) as response, open(partial, "wb") as handle:
        while chunk := response.read(1 << 23):
            handle.write(chunk)
    if (observed := sha256_file(partial)) != sha256:
        partial.unlink()
        raise ValueError(f"{url}: sha256 {observed} != {sha256}")
    partial.replace(destination)
    return destination


def hf_file(spec: dict) -> Path:
    """A pinned Hugging Face dataset file (gated repos need HF_TOKEN), checked by digest."""
    from huggingface_hub import hf_hub_download

    path = Path(hf_hub_download(spec["repo_id"], spec["file"], repo_type="dataset", revision=spec["revision"]))
    if (observed := sha256_file(path)) != spec["sha256"]:
        raise ValueError(f"{spec['repo_id']}/{spec['file']}: sha256 {observed} != {spec['sha256']}")
    return path


def math_parquet(config: dict, split: str) -> Path:
    spec = config["data"][split]
    return download_verified(spec["url"], spec["sha256"])


def render_prompt(template: str, question: str) -> str:
    # str.format would choke on the literal braces in "\boxed{}".
    if template.count("{input}") != 1 or not question:
        raise ValueError("template needs one {input} marker and a nonempty question")
    return template.replace("{input}", question)


def completion_cap(prompt_tokens: int, requested: int, context: int, prompt_limit: int) -> int:
    """Native-context cap: min(requested, context - prompt_tokens), never truncating prompts."""
    if not 0 < prompt_tokens <= prompt_limit:
        raise ValueError(f"prompt has {prompt_tokens} tokens; limit is {prompt_limit}")
    return min(requested, context - prompt_tokens)


@cache
def official_extractor():
    # Upstream utils.py has two invalid escape sequences that warn on Python 3.12.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        from ._vendor.relex_eval.utils import extract_answer_math
    return extract_answer_math


def test_questions(config: dict) -> list[dict]:
    """The 5,000 MATH test questions with official-extractor gold answers."""
    import pyarrow.parquet as pq

    rows = pq.read_table(math_parquet(config, "test")).to_pylist()
    if len(rows) != config["data"]["test"]["rows"]:
        raise ValueError(f"expected {config['data']['test']['rows']} test rows, got {len(rows)}")
    extract = official_extractor()
    questions = []
    for position, row in enumerate(rows):
        truth = row["reward_model"]["ground_truth"]
        if row["extra_info"]["index"] != position or row["prompt"][0]["content"] != truth["question"]:
            raise ValueError(f"test row {position} does not match the pinned schema")
        questions.append({
            "question_id": position,
            "question": truth["question"],
            "gold_answer": extract(truth["solution"]),
            "prompt": render_prompt(config["data"]["prompt_template"], truth["question"]),
        })
    return questions
