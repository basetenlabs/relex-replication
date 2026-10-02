"""Exact upstream legacy IFEval method and dependencies; inference framework omitted."""
import dataclasses
import json
import logging
from .if_functions import IF_FUNCTIONS_MAP

logger = logging.getLogger(__name__)

@dataclasses.dataclass
class VerificationResult:
    score: float
    cost: float = 0.0
    reasoning: str | None = None

def remove_thinking_section(prediction: str) -> str:
    prediction = prediction.replace("<|assistant|>", "").strip()
    # remove thinking section from the prediction
    prediction = prediction.split("</think>")[-1]
    # remove answer tags from the prediction
    prediction = prediction.replace("<answer>", "").replace("</answer>", "")
    return prediction.strip()

class IFEvalVerifierOld:
    def __call__(
        self,
        tokenized_prediction: list[int],
        prediction: str,
        label: str | dict,
        query: str | None = None,
        rollout_state: dict | None = None,
    ) -> VerificationResult:
        constraint = label
        answer = remove_thinking_section(prediction)
        if isinstance(constraint, str):
            constraint = json.loads(constraint)
        if "func_name" not in constraint:
            logger.warning("Constraint missing 'func_name': %s", constraint)
            return VerificationResult(score=0.0)
        func_name = constraint.pop("func_name")
        func = IF_FUNCTIONS_MAP[func_name]
        non_none_args = {k: v for k, v in constraint.items() if v is not None}
        if not constraint:
            return VerificationResult(score=float(func(answer)))
        return VerificationResult(score=float(func(answer, **non_none_args)))
