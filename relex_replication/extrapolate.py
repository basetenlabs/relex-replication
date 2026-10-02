"""Build RELEX rank-1 and first-update-scaling extrapolations from saved checkpoints.

RELEX prefix N -> target T: per tensor, stack FP16 displacements W_t - W_0 for
t = 1..N, take the uncentered temporal rank-1 factor, fit the coefficients by OLS
against rows 0..N-1 and evaluate at row T-1.

First-update scaling: BF16(FP32(W0) + alpha * (FP32(W1) - FP32(W0))).
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

# Gram matrices and coefficients are accumulated block by block in FP32, so
# the block size is part of the numerics and must stay fixed.
CHUNK_ELEMENTS = 1_000_000
READ_WORKERS = 16


class TensorStore:
    """Lazy axis-0 block reads from a safetensors checkpoint directory."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        index = self.root / "model.safetensors.index.json"
        if index.exists():
            weight_map = json.loads(index.read_text())["weight_map"]
            self.files = {name: self.root / file for name, file in weight_map.items()}
        else:
            self.files = {}
            for shard in sorted(self.root.glob("*.safetensors")):
                with safe_open(shard, framework="pt") as handle:
                    self.files.update(dict.fromkeys(handle.keys(), shard))
        if not self.files:
            raise ValueError(f"no safetensors weights in {self.root}")
        self.meta = {}
        for shard in set(self.files.values()):
            with safe_open(shard, framework="pt") as handle:
                for name in handle.keys():  # noqa: SIM118 - safe_open is not iterable
                    sliced = handle.get_slice(name)
                    self.meta[name] = (tuple(sliced.get_shape()), sliced.get_dtype())

    @property
    def names(self) -> list[str]:
        return sorted(self.files)

    def shape(self, name: str) -> tuple[int, ...]:
        return self.meta[name][0]

    def read_block(self, name: str, start: int, end: int) -> torch.Tensor:
        with safe_open(self.files[name], framework="pt") as handle:
            if not self.shape(name):
                return handle.get_tensor(name)
            return handle.get_slice(name)[start:end]


def blocks(shape: tuple[int, ...], chunk_elements: int = CHUNK_ELEMENTS):
    """Yield (row_start, row_end, flat_start, flat_end) blocks along axis 0."""
    if not shape:
        yield 0, 1, 0, 1
        return
    row_elements = math.prod(shape[1:])
    rows = max(1, chunk_elements // row_elements)
    for start in range(0, shape[0], rows):
        end = min(start + rows, shape[0])
        yield start, end, start * row_elements, end * row_elements


def fp16_delta(snapshot: torch.Tensor, base: torch.Tensor) -> np.ndarray:
    """Cast both weights to FP16, then subtract, as the authors' precompute does."""
    return (snapshot.to(torch.float16) - base.to(torch.float16)).reshape(-1).to(torch.float32).numpy()


def history_block(base: TensorStore, snapshots: list[TensorStore], name: str, start: int, end: int) -> np.ndarray:
    base_block = base.read_block(name, start, end)
    with ThreadPoolExecutor(max_workers=min(READ_WORKERS, len(snapshots))) as pool:
        rows = list(pool.map(lambda store: fp16_delta(store.read_block(name, start, end), base_block), snapshots))
    return np.stack(rows).astype(np.float32, copy=False)


def rank1_from_gram(gram: np.ndarray) -> tuple[np.ndarray, float]:
    """Leading left singular vector and value of the history from its temporal Gram."""
    eigenvalues, eigenvectors = np.linalg.eigh(np.asarray(gram, dtype=np.float32))
    order = np.argsort(eigenvalues)[::-1]
    singular_values = np.sqrt(np.maximum(eigenvalues[order], 0.0).astype(np.float32)).astype(np.float32)
    return np.asarray(eigenvectors[:, order[0]], dtype=np.float32), float(singular_values[0])


def canonicalize_sign(direction: np.ndarray, coefficients: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if direction.size and direction[np.argmax(np.abs(direction))] < 0:
        return -direction, -coefficients
    return direction, coefficients


def stream_rank1(base: TensorStore, snapshots: list[TensorStore], name: str) -> tuple[np.ndarray, np.ndarray]:
    """Two-pass uncentered rank-1 of one tensor's [steps, elements] history."""
    shape = base.shape(name)
    gram = np.zeros((len(snapshots), len(snapshots)), dtype=np.float32)
    for start, end, _, _ in blocks(shape):
        matrix = history_block(base, snapshots, name, start, end)
        gram += np.asarray(matrix @ matrix.T, dtype=np.float32)
    left, top = rank1_from_gram(gram)
    direction = np.zeros(math.prod(shape), dtype=np.float32)
    coefficients = np.zeros(len(snapshots), dtype=np.float32)
    if top > 0.0:
        for start, end, flat_start, flat_end in blocks(shape):
            matrix = history_block(base, snapshots, name, start, end)
            block_direction = np.asarray(matrix.T @ left / np.float32(top), dtype=np.float32)
            direction[flat_start:flat_end] = block_direction
            coefficients += np.asarray(matrix @ block_direction, dtype=np.float32)
        direction, coefficients = canonicalize_sign(direction, coefficients)
    if not (np.isfinite(direction).all() and np.isfinite(coefficients).all()):
        raise ValueError(f"nonfinite rank-1 factor for {name}")
    return direction, coefficients


def predict_coefficient(coefficients: np.ndarray, target_step: int) -> float:
    """OLS of coefficients against rows 0..N-1, evaluated at row target_step - 1."""
    values = np.asarray(coefficients, dtype=np.float32).astype(np.float64).reshape(-1)
    rows = np.arange(values.size, dtype=np.float64)
    slope, intercept = np.linalg.lstsq(np.column_stack((rows, np.ones_like(rows))), values, rcond=None)[0]
    return float(slope) * (float(target_step) - 1.0) + float(intercept)


def materialize(base: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    return (base.to(torch.float32) + delta.to(torch.float32)).to(torch.bfloat16)


def scaled_first_update(base: torch.Tensor, step1: torch.Tensor, alpha: float) -> torch.Tensor:
    return materialize(base.float(), alpha * (step1.float() - base.float()))


def copy_ancillary(base_root: Path, destination: Path) -> None:
    """Copy config/tokenizer files (everything except weights) from the base model."""
    for source in base_root.rglob("*"):
        relative = source.relative_to(base_root)
        if source.is_dir() or ".cache" in relative.parts or source.suffix == ".safetensors":
            continue
        (destination / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination / relative)


def build(run: Path, output: Path, *, prefix: int | None = None, alpha: float | None = None,
          target_step: int = 500) -> None:
    if (prefix is None) == (alpha is None):
        raise ValueError("choose exactly one of prefix or alpha")
    if prefix is not None and prefix < 2:
        raise ValueError("RELEX needs at least two checkpoints")
    if alpha is not None and not math.isfinite(alpha):
        raise ValueError("alpha must be finite")
    base = TensorStore(run / "base_model")
    steps = range(1, (prefix or 1) + 1)
    snapshots = [TensorStore(run / "trajectory" / f"global_step_{step}") for step in steps]
    for store in snapshots:
        if store.names != base.names or any(store.meta[n] != base.meta[n] for n in base.names):
            raise ValueError(f"{store.root} differs from the base tensor universe")
    if any(dtype != "BF16" for _, dtype in base.meta.values()):
        raise ValueError("base and trajectory weights must be BF16")

    output.mkdir(parents=True, exist_ok=False)
    copy_ancillary(base.root, output)
    for shard in sorted(set(base.files.values())):
        tensors = {}
        for name in (n for n in base.names if base.files[n] == shard):
            shape = base.shape(name)
            tensor = torch.empty(shape, dtype=torch.bfloat16)
            if prefix is not None:
                direction, coefficients = stream_rank1(base, snapshots, name)
                coefficient = predict_coefficient(coefficients, target_step)
            for start, end, flat_start, flat_end in blocks(shape):
                base_block = base.read_block(name, start, end).reshape(-1)
                if prefix is not None:
                    delta = torch.from_numpy(direction[flat_start:flat_end]) * coefficient
                    weight = materialize(base_block, delta)
                else:
                    step1 = snapshots[0].read_block(name, start, end).reshape(-1)
                    weight = scaled_first_update(base_block, step1, alpha)
                if not torch.isfinite(weight).all():
                    raise ValueError(f"nonfinite weights for {name}")
                if shape:
                    tensor[start:end].copy_(weight.reshape((end - start, *shape[1:])))
                else:
                    tensor.copy_(weight.reshape(()))
            tensors[name] = tensor
        save_file(tensors, str(output / shard.name), metadata={"format": "pt"})
    print(f"wrote {output}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True, help="training output with base_model/ and trajectory/")
    parser.add_argument("--output", type=Path, required=True, help="new model directory")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prefix", type=int, help="RELEX from checkpoints 1..PREFIX")
    mode.add_argument("--alpha", type=float, help="scale the first saved update by ALPHA")
    parser.add_argument("--target-step", type=int, default=500, help="RELEX extrapolation step")
    args = parser.parse_args(argv)
    build(args.run, args.output, prefix=args.prefix, alpha=args.alpha, target_step=args.target_step)


if __name__ == "__main__":
    main()
