import json

import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file

from relex_replication import extrapolate as ex


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


def test_first_update_scaling_is_exact(run, tmp_path):
    ex.build(run, tmp_path / "out", alpha=1000.0)
    base = load_file(run / "base_model/model.safetensors")
    step1 = load_file(run / "trajectory/global_step_1/model.safetensors")
    out = load_file(tmp_path / "out/model.safetensors")
    for name, w0 in base.items():
        expected = (w0.float() + 1000.0 * (step1[name].float() - w0.float())).to(torch.bfloat16)
        assert out[name].dtype == torch.bfloat16
        assert torch.equal(out[name], expected)
    assert (tmp_path / "out/config.json").exists()


def test_relex_matches_reference_formula(run, tmp_path):
    ex.build(run, tmp_path / "out", prefix=3, target_step=500)
    base = ex.TensorStore(run / "base_model")
    snapshots = [ex.TensorStore(run / "trajectory" / f"global_step_{s}") for s in (1, 2, 3)]
    out = load_file(tmp_path / "out/model.safetensors")
    for name in base.names:
        w0 = base.read_block(name, 0, base.shape(name)[0])
        history = np.stack([
            (s.read_block(name, 0, base.shape(name)[0]).half() - w0.half()).reshape(-1).float().numpy()
            for s in snapshots
        ])
        u, sigma, vt = np.linalg.svd(history.astype(np.float64), full_matrices=False)
        coefficients = u[:, 0] * sigma[0]
        slope, intercept = np.polyfit(np.arange(3), coefficients, 1)
        reference = w0.float().reshape(-1) + torch.from_numpy(vt[0] * (slope * 499 + intercept)).float()
        torch.testing.assert_close(out[name].float().reshape(-1), reference, atol=1e-2, rtol=1e-2)


def test_exact_rank1_line_extrapolates_linearly():
    history = np.outer(np.arange(1, 6, dtype=np.float32), np.array([0.5, -2.0, 1.0], dtype=np.float32))
    left, top = ex.rank1_from_gram(history @ history.T)
    direction = history.T @ left / np.float32(top)
    direction, coefficients = ex.canonicalize_sign(direction, history @ direction)
    assert direction[np.argmax(np.abs(direction))] > 0
    np.testing.assert_allclose(np.outer(coefficients, direction), history, atol=1e-5)
    predicted = ex.predict_coefficient(coefficients, 500) * direction
    np.testing.assert_allclose(predicted, 500 * history[0], rtol=1e-4)


def test_blocking_matches_single_pass(tmp_path):
    torch.manual_seed(1)
    shape = (5, 300_000)  # three rows per 1M-element block -> two blocks
    base = torch.randn(shape).bfloat16()
    write_model(tmp_path / "base_model", {"w": base})
    for step in (1, 2):
        noise = torch.randn(shape) * 1e-3
        weights = {"w": (base.float() + step * 1e-2 + noise).bfloat16()}
        write_model(tmp_path / "trajectory" / f"global_step_{step}", weights)
    assert len(list(ex.blocks(shape))) == 2
    stores = [ex.TensorStore(tmp_path / "trajectory" / f"global_step_{s}") for s in (1, 2)]
    direction, coefficients = ex.stream_rank1(ex.TensorStore(tmp_path / "base_model"), stores, "w")
    history = ex.history_block(ex.TensorStore(tmp_path / "base_model"), stores, "w", 0, shape[0])
    left, top = ex.rank1_from_gram(history @ history.T)
    expected = history.T @ left / np.float32(top)
    expected, expected_coefficients = ex.canonicalize_sign(expected, history @ expected)
    np.testing.assert_allclose(direction, expected, rtol=1e-4, atol=1e-7)
    np.testing.assert_allclose(coefficients, expected_coefficients, rtol=1e-4)
