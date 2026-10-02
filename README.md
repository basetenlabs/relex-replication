# RELEX replication

Replicates and stress-tests RELEX from [You Only Need Minimal RLVR Training: Extrapolating LLMs via Rank-1 Trajectories](https://arxiv.org/abs/2605.21468) ([authors' code](https://github.com/weizhepei/RELEX)) on MATH with Qwen2.5-Math-1.5B, and on MATH, Knights & Knaves (`kk`), IFEval (`ifeval`) and function calling
(`fc`: xLAM training, BFCL evaluation) with Qwen3-4B-Base and Qwen3-8B-Base.

## Setup and training

Run commands from this repository's root. The reference environment is Python **3.12**, Torch 2.10 with
CUDA 12.9, and `vllm/vllm-openai:v0.19.0` (image digest
`sha256:7a0f0fdd2771464b6976625c2b2d5dd46f566aa00fbc53eceab86ef50883da90`).
Use that image on Linux GPU hosts, then install the requirements. The repository is run directly from
source; `pip install .` is not the supported setup.

```bash
python -m pip install -r requirements.txt
export PYTHONHASHSEED=1 VLLM_WORKER_MULTIPROC_METHOD=spawn TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
```

For CPU tests and extrapolation (including on macOS), install `requirements-cpu.txt` instead and run
`python -m pytest -q`. Training and generation require CUDA GPUs; CPU tests do not validate those paths.
Data and tokenizers download automatically at their pinned revisions. xLAM is gated: accept its terms
on Hugging Face and set `HF_TOKEN` before function-calling training.

| Recipe | Config | Environment | Training hardware | Snapshot weights alone |
|---|---|---|---|---|
| Qwen2.5-Math-1.5B | `configs/qwen25_math_1.5b.json` | `math` | 2 nodes × 8 H200 | ~230 GiB |
| Qwen3-4B-Base | `configs/qwen3_4b.json` | `math`, `kk`, `ifeval`, `fc` | 2 nodes × 8 H200 | ~600 GiB |
| Qwen3-8B-Base | `configs/qwen3_8b.json` | `kk`, `ifeval`, `fc` | 1 node × 8 B200 | ~920 GiB |

Allow additional disk for the base model, two full optimizer checkpoints, download caches and derived
models. For the 8B runs, plan at least 1.5 TiB per experiment. Each environment is a separate training run.
The two-node runs require a **shared output filesystem at the same path on both nodes**, plus network
connectivity to node 0. Keep the reference GPU count to preserve data sharding.

For 1.5B, set these variables on **both** nodes, using machine rank 0 on the first and 1 on the second.
For 4B, change `C`, `E`, and `RUN` to the chosen row above and a distinct output path.

```bash
C=configs/qwen25_math_1.5b.json
E=math
RUN=/shared/runs/1.5b-math
# Set MAIN_PROCESS_IP to node 0's reachable address and MACHINE_RANK to 0 or 1.
: "${MAIN_PROCESS_IP:?set the address of node 0}"
: "${MACHINE_RANK:?set 0 on node 0 or 1 on node 1}"
accelerate launch --multi_gpu --num_machines 2 --num_processes 16 \
  --machine_rank "$MACHINE_RANK" --main_process_ip "$MAIN_PROCESS_IP" --main_process_port 29500 \
  --mixed_precision bf16 --dynamo_backend no \
  -m relex_replication.train --config "$C" --env "$E" --output "$RUN"
```

For 8B, run once on its eight-B200 node:

```bash
C=configs/qwen3_8b.json
E=kk                         # kk, ifeval, or fc
RUN=/mnt/checkpoints/runs/8b-$E
accelerate launch --multi_gpu --num_machines 1 --num_processes 8 --machine_rank 0 \
  --mixed_precision bf16 --dynamo_backend no \
  -m relex_replication.train --config "$C" --env "$E" --output "$RUN"
```

Training writes `base_model/`, `trajectory/global_step_N/`, `resume/checkpoint-N/` and the resolved
recipe in `run.json`. To resume, repeat the same launch command with
`--resume "$RUN/resume/checkpoint-N"`, using a completed checkpoint and the original `--output`.
Optimizer, scheduler and Trainer RNG state resume; vLLM sampling is not guaranteed bitwise identical
after a restart. Fresh training cannot overwrite an existing trajectory.

## Qwen2.5-Math-1.5B experiments

A fresh trajectory reruns the method; reproducing the recorded numbers requires the **original saved
weights**. The original 1.5B producer is Baseten project `qklp483`, job `w5j4g73`, checkpoint prefix
`relex_qwen25_math15b_method/` (access required). Retain its `base_model/` and
`trajectory/global_step_N/` layout and point `RUN` there. Original Qwen3 producers are 4B:
K&K `w7r5463`, IFEval `wx2zn73`, FC `32o07vw`; 8B: K&K `wp1rv9w`, IFEval `q498j1w`, FC `q8zem8q`.
Those archives may require exporting their checkpoints into this layout before using `--run`.
The reference repository's source manifests describe their archive paths.

After training or restoring the 1.5B trajectory:

```bash
RUN=/shared/runs/1.5b-math
python -m relex_replication.extrapolate --run "$RUN" --prefix 2 --output models/1.5b-relex2
python -m relex_replication.extrapolate --run "$RUN" --alpha 1000 --output models/1.5b-alpha1000

# MATH evaluation uses one GPU. Choose the GPU with CUDA_VISIBLE_DEVICES if needed.
python -m relex_replication.evaluate --model "$RUN/base_model" --output evals/1.5b-base
python -m relex_replication.evaluate --model "$RUN/trajectory/global_step_500" --output evals/1.5b-step500
python -m relex_replication.evaluate --model models/1.5b-alpha1000 --output evals/1.5b-alpha1000
python -m relex_replication.evaluate --model models/1.5b-relex2 --output evals/1.5b-relex2
python -m relex_replication.evaluate --model relex-rlvr/RELEX-Qwen2.5-Math-1.5B \
  --revision 458878c19dd5abbdbf89de91f7267ad249fca00a --output evals/1.5b-released-relex
python -m relex_replication.analyze --base evals/1.5b-base --target evals/1.5b-relex2
```

For the original full sweeps, repeat the build/evaluate pair at prefixes `2 5 10 25 50 75` and alphas
`0 1 10 30 100 300 500 1000 3000`, with a distinct model and evaluation directory per setting.
Extrapolation runs on CPU and needs the base plus every checkpoint through the selected prefix.
A build requires a new output directory and currently restarts from scratch if interrupted.
Evaluation resumes completed batches only when the model hashes and resolved settings match.
It writes `evaluation.json`, raw `generations.jsonl`, `scores.jsonl` and `summary.json`.

| Model | MATH test accuracy |
|---|---:|
| Base | 46.86% |
| Raw step 500 | 63.80% |
| First update, alpha = 1000 | 66.66% |
| RELEX prefix 2 -> 500 | 66.02% |

Numbers are from our original training run; a retrained run will differ slightly. Alpha was chosen on the test set.

## Qwen3 multi-environment runs

Keep `--config` and `--env` on **every** Qwen3 evaluation; otherwise evaluation defaults to
1.5B MATH. K&K, IFEval and FC evaluation use **two GPUs**, while MATH uses one.

To rerun the 4B table for one environment, set the same `C`, `E`, and `RUN` used in training:

```bash
C=configs/qwen3_4b.json
E=kk                         # math, kk, ifeval, or fc
RUN=/shared/runs/4b-$E
python -m relex_replication.evaluate --config "$C" --env "$E" \
  --model "$RUN/base_model" --output "evals/4b-$E-base"
python -m relex_replication.evaluate --config "$C" --env "$E" \
  --model "$RUN/trajectory/global_step_500" --output "evals/4b-$E-step500"
python -m relex_replication.extrapolate --run "$RUN" --alpha 300 --fp16-delta \
  --output "models/4b-$E-alpha300"
python -m relex_replication.evaluate --config "$C" --env "$E" \
  --model "models/4b-$E-alpha300" --output "evals/4b-$E-alpha300"
for N in 2 20 50 75; do
  python -m relex_replication.extrapolate --run "$RUN" --prefix "$N" --output "models/4b-$E-relex$N"
  python -m relex_replication.evaluate --config "$C" --env "$E" \
    --model "models/4b-$E-relex$N" --output "evals/4b-$E-relex$N"
done
for L in 0.3 0.5; do
  python -m relex_replication.extrapolate --run "$RUN" --prefix 75 --lam "$L" --output "models/4b-$E-lam$L"
  python -m relex_replication.evaluate --config "$C" --env "$E" \
    --model "models/4b-$E-lam$L" --output "evals/4b-$E-lam$L"
done
```

For 8B, evaluate the base and raw checkpoints, then run the horizon and scale grids:

```bash
C=configs/qwen3_8b.json
E=kk                         # kk, ifeval, or fc
RUN=/mnt/checkpoints/runs/8b-$E
python -m relex_replication.evaluate --config "$C" --env "$E" \
  --model "$RUN/base_model" --output "evals/8b-$E-base"
for N in 55 500; do
  python -m relex_replication.evaluate --config "$C" --env "$E" \
    --model "$RUN/trajectory/global_step_$N" --output "evals/8b-$E-step$N"
done
for K in 100 200 300 400 500; do
  python -m relex_replication.extrapolate --run "$RUN" --prefix 55 --target-step "$K" \
    --output "models/8b-$E-k$K"
  python -m relex_replication.evaluate --config "$C" --env "$E" \
    --model "models/8b-$E-k$K" --output "evals/8b-$E-k$K"
done
for L in 0.05 0.1 0.3 0.5; do
  python -m relex_replication.extrapolate --run "$RUN" --prefix 55 --lam "$L" --output "models/8b-$E-lam$L"
  python -m relex_replication.evaluate --config "$C" --env "$E" \
    --model "models/8b-$E-lam$L" --output "evals/8b-$E-lam$L"
done
```

Each build refits the SVD, including when only `--target-step` or `--lam` changes. This keeps the
implementation small but rereads the prefix; budget time for those disk reads. `--lam` scales the
whole predicted displacement. `--fp16-delta` is needed for Qwen3 first-update scaling; prefix RELEX
always uses FP16-subtracted displacements.

Expected test accuracy (%) from our original runs. `kk`: relaxed Logic-RL reward, n=498. `ifeval`: official loose
prompt accuracy, n=541. `fc`: official BFCL single-turn on the 2,517 calling items (all 3,641 items in brackets).
MATH: n=5,000, greedy, 16K budget. Grids were evaluated on the test set, so their maxima are oracle choices.

| Qwen3-4B-Base | MATH | kk | ifeval | fc |
|---|---:|---:|---:|---:|
| Base | 71.46 | 5.42 | 44.18 | 23.44 (40.65) |
| Raw step 500 | 76.66 | 19.88 | 48.43 | 78.07 (55.37) |
| First update, `--alpha 300 --fp16-delta` | 73.60 | 0.20 | 42.70 | (40.57) |
| RELEX `--prefix 2` -> 500 | 67.30 | 0.40 | 44.92 | 23.44 (40.57) |
| RELEX `--prefix 20` -> 500 | 74.70 | 0.00 | 48.43 | 77.23 (54.68) |
| RELEX `--prefix 50` -> 500 | 75.88 | 0.00 | 48.43 | 77.27 (54.33) |
| RELEX `--prefix 75` -> 500 | | 3.41 | 47.87 | 75.96 (53.03) |
| `--prefix 75 --lam 0.3` | | 13.25 | 45.84 | 77.12 (55.75) |
| `--prefix 75 --lam 0.5` | | 14.26 | 47.13 | 77.00 (54.49) |

| Qwen3-8B-Base (`--prefix 55`) | kk | ifeval | fc |
|---|---:|---:|---:|
| Base | 20.48 | 52.50 | 31.31 (43.70) |
| Raw step 55 / 500 | 54.62 / 95.18 | 70.06 / 65.25 | 82.64 / 82.64 |
| RELEX -> K = 100 / 200 / 300 / 400 / 500 | 46.18 / 49.80 / 40.16 / 23.29 / 7.23 | 58.41 / 42.51 / 34.01 / 26.43 / 22.37 | 80.97 / 79.22 / 75.29 / 63.09 / 45.05 |
| K = 500, `--lam` 0.05 / 0.1 / 0.3 / 0.5 | 33.13 / 37.95 / 45.58 / 47.19 | 61.37 / 64.70 / 52.50 / 36.78 | 81.57 / 81.92 / 80.25 / 78.35 |

8B used learning rates 1e-5 (kk, ifeval) and 3e-6 (fc) and a 4K training context. Evaluation uses 4K for kk/ifeval
and 8K for official BFCL, whose longest prompts need it. These runs are not comparable row-for-row
with 4B. Data licenses: K&K puzzles are CC-BY-NC-SA-4.0
(non-commercial); xLAM is CC-BY-4.0 but gated and described by its authors as research-only; RLVR-IFeval is
ODC-BY; IFEval, BFCL and MATH are Apache-2.0/MIT.
