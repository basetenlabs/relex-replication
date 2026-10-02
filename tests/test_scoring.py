import hashlib

import pytest

from relex_replication.analyze import cohort, compare, is_heavy_repetition, repeated_fraction
from relex_replication.data import completion_cap, load_config, official_extractor, render_prompt


def test_prompt_template_is_pinned():
    template = load_config()["data"]["prompt_template"]
    assert hashlib.sha256(template.encode()).hexdigest() == (
        "a16c3b73f538a640cd637ce7a6baadcbcf7f80152d319e7c212e73cc03212a2e")
    assert render_prompt(template, "1+1?").count("1+1?\nPlease reason step by step") == 1


def test_completion_cap():
    assert completion_cap(100, 4096, 4096, 2048) == 3996
    assert completion_cap(100, 3072, 4096, 2048) == 3072
    with pytest.raises(ValueError):
        completion_cap(2049, 4096, 4096, 2048)


@pytest.mark.parametrize(("response", "gold", "correct"), [
    (r"So the answer is $\boxed{\frac{1}{2}}$.", r"\frac{1}{2}", True),
    (r"\boxed{0.50}", "0.5", True),
    (r"Thus $\boxed{3x^2-3x-6}$", "3x^2-3x-6", True),
    (r"\boxed{7}", "8", False),
    # Known false negative of the official checker, kept as-is.
    (r"\boxed{10000}", "10{,}000", False),
])
def test_official_grader(response, gold, correct):
    from relex_replication._vendor.relex_eval.grader import math_equal

    assert math_equal(official_extractor()(response), gold, timeout=True) is correct


def test_training_reward_golden_cases():
    from relex_replication.train import math_reward

    cases = [(r"Reasoning. \boxed{2}", "2", 1.0), (r"Reasoning. \boxed{3}", "2", 0.0),
             (r"Reasoning. \boxed{\frac{2}{4}}", r"\frac{2}{4}", 1.0)]
    truths = [{"question": "q", "solution": target, "target": target} for _, target, _ in cases]
    assert math_reward([c for c, _, _ in cases], truths) == [r for _, _, r in cases]


def test_repetition_classifier():
    loop = "the answer is the answer is " * 50
    assert repeated_fraction("one two three") == 0.0
    assert is_heavy_repetition(loop)
    assert not is_heavy_repetition(" ".join(f"w{i}" for i in range(200)))
    capped = {"response": "fresh words " + " ".join(map(str, range(50))), "finish_reason": "length",
              "output_tokens": 10, "output_cap": 10}
    assert cohort({**capped, "response": loop}) == "heavy_repetition"
    assert cohort(capped) == "cap_without_repetition"
    assert cohort({**capped, "finish_reason": "stop", "output_tokens": 5}) == "neither"


def test_compare_counts_repairs_by_base_cohort():
    def gen(i, response, finish="stop"):
        return {"question_id": i, "response": response, "finish_reason": finish, "output_tokens": 1, "output_cap": 9}

    def score(i, correct):
        return {"question_id": i, "correct": correct}

    loop = "x y " * 100
    base = ([gen(0, loop), gen(1, "ok", "length"), gen(2, "ok"), gen(3, "ok")],
            [score(0, False), score(1, False), score(2, True), score(3, False)])
    target = ([gen(i, "fine") for i in range(4)], [score(0, True), score(1, True), score(2, False), score(3, False)])
    table = compare(base, target)
    assert table["heavy_repetition"]["repairs"] == 1
    assert table["cap_without_repetition"]["repairs"] == 1
    assert table["neither"] == {**table["neither"], "questions": 2, "base_failures": 1, "repairs": 0,
                                "regressions": 1, "net": -1}
    assert table["total"]["net"] == 1
