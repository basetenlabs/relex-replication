"""Per-environment prompts, training rewards and evaluation scorers.

Every reward and score delegates to the vendored upstream grader; the code
here only adapts data schemas, exactly as the original runs did.
"""

from __future__ import annotations

import ast
import contextlib
import copy
import json
import os
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from functools import cache
from pathlib import Path

from . import _vendor  # noqa: F401 - puts upstream top-level packages on sys.path
from .data import DATA_CACHE, download_verified, hf_file, math_parquet, read_jsonl, render_prompt, test_questions

REPO = Path(__file__).resolve().parents[1]


def chatml(user: str, system: str | None = None) -> str:
    messages = ([("system", system)] if system is not None else []) + [("user", user)]
    return "".join(f"<|im_start|>{role}\n{text}<|im_end|>\n" for role, text in messages) + "<|im_start|>assistant\n"


def expect(rows: list, count: int, what: str) -> list:
    if len(rows) != count:
        raise ValueError(f"expected {count} {what}, got {len(rows)}")
    return rows


def quietly(function, *args):
    # Upstream graders print per answer; silence them without touching the scores.
    with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
        return function(*args)


class Env:
    def generation_groups(self, rows: list[dict]) -> list[list[dict]]:
        return [rows]

    def summarize(self, scores: list[dict]) -> dict:
        return {}


class Math(Env):
    def train_rows(self, config: dict, tokenizer) -> list[dict]:
        import pyarrow.parquet as pq

        data = config["data"]
        rows = pq.read_table(math_parquet(config, "train")).to_pylist()
        truths = [row["reward_model"]["ground_truth"] for row in rows]
        return expect([{"prompt": render_prompt(data["prompt_template"], truth["question"]), "ground_truth": truth,
                        "source_index": row["extra_info"]["index"]} for row, truth in zip(rows, truths, strict=True)],
                      data["train"]["rows"], "rows")

    def reward(self, completions: Sequence[str], ground_truth: Sequence[Mapping], **_) -> list[float]:
        """Binary RLVR-Decomposed reward: official boxed-answer extraction vs the target."""
        from ._vendor.rlvr_math.math import compute_score

        return [float(compute_score(c, t)) for c, t in zip(completions, ground_truth, strict=True)]

    def test_rows(self, config: dict) -> list[dict]:
        return test_questions(config)

    def score(self, row: dict, response: str) -> dict:
        from ._vendor.relex_eval.grader import math_equal
        from .data import official_extractor

        prediction = official_extractor()(response)
        return {"gold_answer": row["gold_answer"], "prediction": prediction,
                "correct": bool(math_equal(prediction, row["gold_answer"], timeout=True))}


class JsonTruthEnv(Env):
    """Training ground truth travels through Arrow as a JSON string."""

    def reward(self, completions: Sequence[str], ground_truth: Sequence[str], **_) -> list[float]:
        truths = [json.loads(t) if isinstance(t, str) else t for t in ground_truth]
        return [quietly(self.training_reward, c, t) for c, t in zip(completions, truths, strict=True)]


# ---------------------------------------------------------------- Knights & Knaves

KK_PREFIX = "<|im_start|>assistant\n<think>"


def normalize_missing_think_end(text: str) -> str:
    """Approved relaxation: insert one missing </think> before <answer> if that alone
    makes the upstream structure check pass. Nothing else is repaired."""
    from ._vendor.logic_rl import kk

    response = KK_PREFIX + text
    if "</think>" in response:
        return text
    _, processed = kk.extract_solution(response)
    if "<answer>" not in processed:
        return text
    boundary = processed.index("<answer>")
    if not kk.validate_response_structure(processed[:boundary] + "</think>" + processed[boundary:]):
        return text
    offset = len(response) - len(processed) + boundary - len(KK_PREFIX)
    if not 0 <= offset <= len(text):
        raise ValueError("upstream assistant extraction produced an invalid boundary")
    return text[:offset] + "</think>" + text[offset:]


def kk_score(completion: str, truth: Mapping) -> tuple[float, bool]:
    """Logic-RL compute_score on the relaxed text; returns (reward, format_valid)."""
    from ._vendor.logic_rl import kk

    response = KK_PREFIX + normalize_missing_think_end(completion)
    value = kk.compute_score(response, truth)
    _, processed = kk.extract_solution(response)
    return float(value), bool(kk.validate_response_structure(processed))


class KnightsKnaves(JsonTruthEnv):
    def source(self, data: dict, people: int, split: str) -> list[dict]:
        import pyarrow.parquet as pq

        key = f"{people}-{split}"
        path = download_verified(data["url"].format(people=people, split=split), data["sha256"][key])
        return expect(pq.read_table(path).to_pylist(), 900 if split == "train" else 100, f"{key} rows")

    def rows(self, config: dict, split: str) -> list[dict]:
        data = config["data"]
        rows = []
        for people in data["people"]:
            for position, row in enumerate(self.source(data, people, split)):
                source_id = f"{people}ppl/{split}/{position}"
                if source_id not in data["test_exclude"]:
                    rows.append({"question_id": source_id, "prompt": row["prompt"][0]["content"],
                                 "ground_truth": dict(row["reward_model"]["ground_truth"])})
        return rows

    def train_rows(self, config: dict, tokenizer) -> list[dict]:
        rows = expect(self.rows(config, "train"), config["data"]["train_rows"], "training rows")
        return [{"prompt": r["prompt"], "ground_truth": json.dumps(r["ground_truth"], allow_nan=False)} for r in rows]

    def training_reward(self, completion: str, truth: Mapping) -> float:
        return kk_score(completion, truth)[0]

    def test_rows(self, config: dict) -> list[dict]:
        return expect(self.rows(config, "test"), config["data"]["test_rows"], "test rows")

    def score(self, row: dict, response: str) -> dict:
        value, format_valid = quietly(kk_score, response, row["ground_truth"])
        return {"reward": value, "format_valid": format_valid, "correct": value == 3}

    def summarize(self, scores: list[dict]) -> dict:
        return {"format_failures": sum(not s["format_valid"] for s in scores)}


# ---------------------------------------------------------------- IFEval

def install_punkt_tab(spec: dict, directory: Path = DATA_CACHE / "nltk") -> None:
    """Extract only the pinned English punkt_tab tables; never unpickle downloads."""
    import zipfile

    import nltk

    with zipfile.ZipFile(download_verified(spec["url"], spec["sha256"])) as archive:
        for member in archive.infolist():
            if member.filename.startswith("punkt_tab/english/") and not member.is_dir():
                target = directory / "tokenizers" / member.filename
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(member))
    if str(directory) not in nltk.data.path:
        nltk.data.path.insert(0, str(directory))


class IFEval(JsonTruthEnv):
    def train_rows(self, config: dict, tokenizer) -> list[dict]:
        import pyarrow.parquet as pq

        data = config["data"]
        source = expect(pq.read_table(hf_file(data["train"])).to_pylist(), data["train"]["source_rows"], "source rows")
        rows = []
        for row in source:
            (message,) = row["messages"]
            prompt = chatml(message["content"])
            # Approved length filter: drop over-long prompts, never truncate.
            if len(tokenizer(prompt, add_special_tokens=False)["input_ids"]) <= data["max_prompt_tokens"]:
                rows.append({"prompt": prompt, "ground_truth": json.dumps(json.loads(row["ground_truth"]))})
        return expect(rows, data["train"]["rows"], "length-filtered training rows")

    def training_reward(self, completion: str, truth: dict) -> float:
        from ._vendor.tulu3.verify import IFEvalVerifierOld

        return IFEvalVerifierOld()([], completion, copy.deepcopy(truth)).score

    def test_rows(self, config: dict) -> list[dict]:
        data = config["data"]
        install_punkt_tab(data["nltk_punkt_tab"])
        rows = read_jsonl(hf_file(data["test"]))
        return expect([{"question_id": f"google_ifeval/{r['key']}", "prompt": chatml(r["prompt"]),
                        "ground_truth": {k: r[k] for k in ("key", "prompt", "instruction_id_list", "kwargs")}}
                       for r in rows], data["test"]["rows"], "test rows")

    def score(self, row: dict, response: str) -> dict:
        """Google's official strict and loose checks, seeded per prompt as in the original runs."""
        import langdetect
        from instruction_following_eval import evaluation_lib

        langdetect.DetectorFactory.seed = 1
        truth = row["ground_truth"]
        example, responses = evaluation_lib.InputExample(**truth), {truth["prompt"]: response}
        state = random.getstate()
        try:
            random.seed(truth["key"])
            strict = evaluation_lib.test_instruction_following_strict(example, responses)
            random.seed(truth["key"])
            loose = evaluation_lib.test_instruction_following_loose(example, responses)
        finally:
            random.setstate(state)
        return {"correct": bool(loose.follow_all_instructions), "strict_correct": bool(strict.follow_all_instructions),
                "strict_instructions": list(strict.follow_instruction_list),
                "loose_instructions": list(loose.follow_instruction_list)}

    def summarize(self, scores: list[dict]) -> dict:
        total = sum(len(s["strict_instructions"]) for s in scores)
        return {"primary_metric": "loose_prompt_accuracy",
                "strict_prompt_accuracy": sum(s["strict_correct"] for s in scores) / len(scores),
                "strict_instruction_accuracy": sum(sum(s["strict_instructions"]) for s in scores) / total,
                "loose_instruction_accuracy": sum(sum(s["loose_instructions"]) for s in scores) / total}


# ---------------------------------------------------------------- function calling (xLAM -> BFCL)

FC_SYSTEM = ("Select calls to the provided tools. Return only a JSON array of objects with "
             "name and arguments. Do not execute calls.\nTools:\n")
BASE_TYPES = {"int": "integer", "integer": "integer", "long": "integer", "bigint": "integer",
              "float": "float", "number": "float", "double": "float",
              "str": "string", "string": "string", "char": "string",
              "bool": "boolean", "boolean": "boolean",
              "list": "array", "array": "array", "tuple": "tuple",
              "dict": "dict", "object": "dict", "any": "any"}
BFCL_CATEGORIES = ("simple_python", "simple_java", "simple_javascript", "multiple", "parallel", "parallel_multiple",
                   "irrelevance", "live_simple", "live_multiple", "live_parallel", "live_parallel_multiple",
                   "live_irrelevance", "live_relevance")
IRRELEVANCE = ("irrelevance", "live_irrelevance")


def split_top_level(text: str) -> list[str]:
    parts, depth, current = [], 0, ""
    for character in text:
        if character == "," and depth == 0:
            parts.append(current)
            current = ""
            continue
        depth += (character == "[") - (character == "]")
        if depth < 0:
            raise ValueError(f"unsupported xLAM type: {text!r}")
        current += character
    if depth:
        raise ValueError(f"unsupported xLAM type: {text!r}")
    return [*parts, current]


def annotation_schema(text) -> dict:
    """xLAM's Python type spelling ('List[int]', 'str, optional') -> BFCL schema, or fail.

    Optionality stays with BFCL's required list; nothing is widened to 'any'."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"unsupported xLAM type: {text!r}")
    segments = split_top_level(text)
    if len(segments) == 2 and segments[1].strip().lower() == "optional":
        segments = segments[:1]
    if len(segments) != 1:
        raise ValueError(f"unsupported xLAM type: {text!r}")
    head, _, rest = segments[0].strip().partition("[")
    key = head.strip().lower()
    arguments = None
    if rest:
        if not rest.endswith("]"):
            raise ValueError(f"unsupported xLAM type: {text!r}")
        arguments = [a.strip() for a in split_top_level(rest[:-1])]
        if not all(arguments):
            raise ValueError(f"unsupported xLAM type: {text!r}")
    if key == "optional":
        if arguments is None or len(arguments) != 1:
            raise ValueError(f"unsupported xLAM type: {text!r}")
        return annotation_schema(arguments[0])
    if key not in BASE_TYPES:
        raise ValueError(f"unsupported xLAM type: {text!r}")
    kind = BASE_TYPES[key]
    if arguments is None or kind == "dict":  # BFCL's dict checker ignores declared key/value types
        return {"type": kind}
    if kind not in {"array", "tuple"}:
        raise ValueError(f"unsupported xLAM type: {text!r}")
    items = [annotation_schema(a) for a in arguments]
    if any(item != items[0] for item in items[1:]):
        raise ValueError(f"unsupported xLAM type: {text!r}")
    return {"type": kind, "items": items[0]}


def parameter_schema(spec: dict) -> dict:
    result = copy.deepcopy(spec)
    schema = annotation_schema(result.get("type"))
    result["type"] = schema["type"]
    result.pop("required", None)
    if result["type"] in {"array", "tuple"}:
        result["items"] = schema.get("items") or parameter_schema(result.get("items", {"type": "any"}))
    return result


def reference_value(value, schema: dict) -> list:
    """One reference value in BFCL's list-of-acceptable-values encoding."""
    if schema["type"] == "dict" and isinstance(value, dict):
        return [{key: [item] for key, item in value.items()}]
    if schema["type"] in {"array", "tuple"} and schema["items"]["type"] == "dict" and isinstance(value, list):
        return [[{key: [item] for key, item in obj.items()} for obj in value]]
    return [value]


def bfcl_inputs(truth: dict) -> tuple[list, list]:
    tools, answers = truth["tools"], truth["answers"]
    if not tools or not answers:
        raise ValueError("xLAM requires tool schemas and reference calls")
    descriptions = [{"name": tool["name"], "description": tool.get("description", ""), "parameters": {
        "type": "dict", "properties": {k: parameter_schema(v) for k, v in tool["parameters"].items()},
        "required": [k for k, v in tool["parameters"].items() if v.get("required", False)]}} for tool in tools]
    by_name = {d["name"]: d for d in descriptions}
    if len(by_name) != len(descriptions):
        raise ValueError("duplicate tool names")
    possible = []
    for call in answers:
        properties = by_name[call["name"]]["parameters"]["properties"]
        if set(call["arguments"]) - set(properties):
            raise ValueError("reference argument absent from tool schema")
        possible.append({call["name"]: {k: reference_value(v, properties[k]) for k, v in call["arguments"].items()}})
    return descriptions, possible


def check_calls(decoded: list, truth: dict) -> dict:
    from bfcl_eval.constants.enums import Language
    from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker

    descriptions, possible = bfcl_inputs(truth)
    category = "parallel_multiple" if len(possible) > 1 else "multiple"
    return ast_checker(descriptions, decoded, possible, Language.PYTHON, category, "xlam-json")


@cache
def bfcl_functions(root: str) -> dict:
    """Named official BFCL runner functions, executed verbatim from hash-checked upstream files."""
    from bfcl_eval.constants.enums import Language, ReturnFormat
    from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker

    symbols = {"utils.py": ("is_function_calling_format_output", "is_empty_output", "is_js", "is_java",
                            "_get_language_specific_hint", "_func_doc_language_specific_pre_processing"),
               "eval_checker/eval_runner.py": ("_evaluate_single_relevance_entry", "_evaluate_single_ast_entry")}
    namespace = {"BaseHandler": object, "Language": Language, "ReturnFormat": ReturnFormat, "json": json,
                 "ast_checker": ast_checker}
    for relative, names in symbols.items():
        text = (Path(root) / relative).read_text()
        nodes = {n.name: n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef)}
        for name in names:
            exec(compile(ast.get_source_segment(text, nodes[name]), f"bfcl/{relative}", "exec"), namespace)
    return namespace


class FunctionCalling(JsonTruthEnv):
    def train_rows(self, config: dict, tokenizer) -> list[dict]:
        data = config["data"]["train"]
        source = expect(json.loads(hf_file(data).read_text()), data["source_rows"], "source rows")
        indices = json.loads((REPO / data["indices"]).read_text())
        rows = []
        for index in indices:
            tools, answers = json.loads(source[index]["tools"]), json.loads(source[index]["answers"])
            system = FC_SYSTEM + json.dumps(tools, ensure_ascii=False, sort_keys=True)
            rows.append({"prompt": chatml(source[index]["query"], system),
                         "ground_truth": json.dumps({"tools": tools, "answers": answers}, allow_nan=False)})
        return expect(rows, data["rows"], "training rows")

    def training_reward(self, completion: str, truth: dict) -> float:
        from bfcl_eval.salesforce_decoder import SalesforceDecoder

        bfcl_inputs(truth)  # bad gold must raise, not score zero
        try:
            decoded = SalesforceDecoder().decode_ast(completion, "Python", False)
        except (ValueError, TypeError, KeyError, AttributeError, RecursionError):
            return 0.0
        if not all(isinstance(args, dict) for call in decoded for args in call.values()):
            return 0.0
        return float(check_calls(decoded, truth)["valid"])

    def bfcl_root(self, config: dict) -> str:
        spec = config["data"]["bfcl"]
        root = DATA_CACHE / "bfcl" / spec["revision"]
        for relative, sha256 in spec["files"].items():
            download_verified(spec["url"].format(path=relative), sha256, destination=root / relative)
        return str(root)

    def test_rows(self, config: dict) -> list[dict]:
        root = self.bfcl_root(config)
        functions = bfcl_functions(root)
        rows = []
        for category in BFCL_CATEGORIES:
            entries = read_jsonl(Path(root) / f"data/BFCL_v4_{category}.json")
            # Upstream aligns gold answers by position; some gold IDs use older names.
            gold = [None] * len(entries) if "relevance" in category else [
                g["ground_truth"] for g in read_jsonl(Path(root) / f"data/possible_answer/BFCL_v4_{category}.json")]
            for entry, answer in zip(entries, gold, strict=True):
                (messages,) = entry["question"]
                tools = functions["_func_doc_language_specific_pre_processing"](copy.deepcopy(entry["function"]),
                                                                              entry["id"].rsplit("_", 1)[0])
                system = FC_SYSTEM + json.dumps(tools, ensure_ascii=False, sort_keys=True)
                prompt = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
                                 for m in [{"role": "system", "content": system}, *messages])
                rows.append({"question_id": entry["id"], "category": category, "entry": entry, "gold": answer,
                             "prompt": prompt + "<|im_start|>assistant\n"})
        return expect(rows, config["data"]["bfcl"]["rows"], "BFCL rows")

    def generation_groups(self, rows: list[dict]) -> list[list[dict]]:
        return [rows[0::2], rows[1::2]]  # the original two alternating-index inference shards

    def score(self, row: dict, response: str) -> dict:
        from bfcl_eval.constants.enums import Language, ReturnFormat
        from bfcl_eval.salesforce_decoder import SalesforceDecoder

        functions = bfcl_functions(str(DATA_CACHE / "bfcl" / self.revision))
        args = (SalesforceDecoder(), row["question_id"], response)
        category = row["category"]
        if "relevance" in category:
            result = functions["_evaluate_single_relevance_entry"](*args, copy.deepcopy(row["entry"]), "xlam-json",
                                                                   category)
        else:
            language, return_format = {"simple_java": (Language.JAVA, ReturnFormat.JAVA),
                                       "simple_javascript": (Language.JAVASCRIPT, ReturnFormat.JAVASCRIPT)}.get(
                category, (Language.PYTHON, ReturnFormat.PYTHON))
            result = functions["_evaluate_single_ast_entry"](
                *args, copy.deepcopy(row["gold"]), copy.deepcopy(row["entry"]), "xlam-json", category,
                language=language, return_format=return_format, has_tool_call_tag=False)
        return {"category": category, "correct": bool(result["valid"])}

    def summarize(self, scores: list[dict]) -> dict:
        def counts(subset):
            correct = sum(s["correct"] for s in subset)
            return {"questions": len(subset), "correct": correct, "accuracy": correct / len(subset)}

        by_category = Counter(s["category"] for s in scores)
        return {"callable": counts([s for s in scores if s["category"] not in IRRELEVANCE]),
                "exact_overlap_excluded": counts([s for s in scores if s["question_id"] not in self.overlap]),
                "categories": {c: counts([s for s in scores if s["category"] == c]) for c in by_category}}

    def configure(self, config: dict) -> None:
        self.revision = config["data"]["bfcl"]["revision"]
        self.overlap = set(config["data"]["bfcl"]["exact_query_overlap"])


ENVS = {"math": Math, "kk": KnightsKnaves, "ifeval": IFEval, "fc": FunctionCalling}


def get_env(name: str, config: dict):
    env = ENVS[name]()
    if hasattr(env, "configure"):
        env.configure(config)
    return env
