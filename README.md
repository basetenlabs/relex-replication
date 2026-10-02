# RELEX replication

Replicates and stress-tests RELEX from [You Only Need Minimal RLVR Training: Extrapolating LLMs via Rank-1 Trajectories](https://arxiv.org/abs/2605.21468) ([authors' code](https://github.com/weizhepei/RELEX)) on MATH with Qwen2.5-Math-1.5B, and on MATH, Knights & Knaves (`kk`), IFEval (`ifeval`) and function calling
(`fc`: xLAM training, BFCL evaluation) with Qwen3-4B-Base and Qwen3-8B-Base.

## Reproduce

```bash
pip install -r requirements.txt          # Python 3.12, CUDA GPUs
export PYTHONHASHSEED=1 VLLM_WORKER_MULTIPROC_METHOD=spawn TOKENIZERS_PARALLELISM=false

# Train: 500 GRPO steps on 16 GPUs (2 nodes x 8; run once per node with its --machine_rank)
accelerate launch --multi_gpu --num_machines 2 --num_processes 16 --machine_rank 0 \
  --main_process_ip $HOST --mixed_precision bf16 --dynamo_backend no \
  -m relex_replication.train --output runs/qwen

# Extrapolate
python -m relex_replication.extrapolate --run runs/qwen --prefix 2 --output models/relex_prefix2
python -m relex_replication.extrapolate --run runs/qwen --alpha 1000 --output models/alpha1000

# Evaluate (5,000 MATH test questions, greedy, one GPU)
python -m relex_replication.evaluate --model runs/qwen/base_model --output evals/base
python -m relex_replication.evaluate --model runs/qwen/trajectory/global_step_500 --output evals/step500
python -m relex_replication.evaluate --model models/alpha1000 --output evals/alpha1000
python -m relex_replication.evaluate --model models/relex_prefix2 --output evals/relex_prefix2
python -m relex_replication.evaluate --model relex-rlvr/RELEX-Qwen2.5-Math-1.5B \
  --revision 458878c19dd5abbdbf89de91f7267ad249fca00a --output evals/released_relex

# Analyze: repaired base failures by base failure mode
python -m relex_replication.analyze --base evals/base --target evals/relex_prefix2

pytest   # CPU tests
```

| Model | MATH test accuracy |
|---|---:|
| Base | 46.86% |
| Raw step 500 | 63.80% |
| First update, alpha = 1000 | 66.66% |
| RELEX prefix 2 -> 500 | 66.02% |

Numbers are from our original training run; a retrained run will differ slightly. Alpha was chosen on the test set.

## Qwen3 multi-environment runs

`configs/qwen3_4b.json` and `configs/qwen3_8b.json` hold the recipes; `configs/envs.json` holds the pinned
data and evaluation budget for `kk`, `ifeval` and `fc`. Data is downloaded at run time (xLAM is gated: accept its
terms on Hugging Face and set `HF_TOKEN`). 4B trains on 2 x 8 H200 like the 1.5B run; 8B on one node of 8 B200.

```bash
C=configs/qwen3_4b.json E=kk   # E in math, kk, ifeval, fc (8B: kk, ifeval, fc)
accelerate launch ... -m relex_replication.train --config $C --env $E --output runs/4b-$E
python -m relex_replication.extrapolate --run runs/4b-$E --alpha 300 --fp16-delta --output models/4b-$E-s300
python -m relex_replication.extrapolate --run runs/4b-$E --prefix 20 --output models/4b-$E-relex20
python -m relex_replication.extrapolate --run runs/4b-$E --prefix 75 --lam 0.5 --output models/4b-$E-lam0.5
python -m relex_replication.evaluate --config $C --env $E --model models/4b-$E-lam0.5 --output evals/4b-$E-lam0.5

# 8B: one 55-update fit read at K = 100..500, and rescaled at K = 500
python -m relex_replication.extrapolate --run runs/8b-kk --prefix 55 --target-step 200 --output models/8b-kk-k200
python -m relex_replication.extrapolate --run runs/8b-kk --prefix 55 --lam 0.3 --output models/8b-kk-lam0.3
```

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

8B used learning rates 1e-5 (kk, ifeval) and 3e-6 (fc) and a 4K context everywhere, so it is not comparable
row-for-row with 4B. Each `extrapolate` call refits the SVD. Data licenses: K&K puzzles are CC-BY-NC-SA-4.0
(non-commercial); xLAM is CC-BY-4.0 but gated and described by its authors as research-only; RLVR-IFeval is
ODC-BY; IFEval, BFCL and MATH are Apache-2.0/MIT.
