# RELEX replication

Replicates and stress-tests RELEX from [You Only Need Minimal RLVR Training: Extrapolating LLMs via Rank-1 Trajectories](https://arxiv.org/abs/2605.21468) ([authors' code](https://github.com/weizhepei/RELEX)) on MATH with Qwen2.5-Math-1.5B.

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
