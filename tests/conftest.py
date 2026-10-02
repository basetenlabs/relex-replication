import json

import pytest
import torch
from safetensors.torch import save_file


def write_model(path, tensors):
    path.mkdir(parents=True)
    save_file(tensors, str(path / "model.safetensors"), metadata={"format": "pt"})
    (path / "config.json").write_text(json.dumps({"model_type": "toy"}))


@pytest.fixture
def run(tmp_path):
    torch.manual_seed(0)
    base = {"a.weight": torch.randn(6, 4).bfloat16(), "b.bias": torch.randn(5).bfloat16()}
    direction = {name: torch.randn(t.shape) * 1e-2 for name, t in base.items()}
    write_model(tmp_path / "base_model", base)
    for step in (1, 2, 3):
        snapshot = {name: (t.float() + step * direction[name]).bfloat16() for name, t in base.items()}
        write_model(tmp_path / "trajectory" / f"global_step_{step}", snapshot)
    return tmp_path
