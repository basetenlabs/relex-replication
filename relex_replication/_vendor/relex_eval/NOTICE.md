# Vendored RELEX evaluator

`utils.py` and `grader.py` are byte-identical copies of `scripts/utils.py` and
`scripts/grader.py` from [weizhepei/RELEX](https://github.com/weizhepei/RELEX)
at commit `e4548babfd23c9c0657b87c36164c1425533ec6c`, distributed under the MIT
License reproduced in `LICENSE` (Copyright (c) 2026 Zhepei Wei).

SHA-256:

- `utils.py`: `b6632d5129ba71ea685c28d36d58a7524f6774081bf1559a94ffa1bc80b5eedf`
- `grader.py`: `cf981c6ebf6ae56c8c55d5838c710e317d514f8aa132b65a804c1e21fd5c689c`

`grader.py` attributes parts of its logic to the Hendrycks MATH release,
ProphetNet/CRITIC, PRM800K, ToRA and DeepSeek-Math; those notices are retained
verbatim in the file. Do not edit these files: they are the evaluation
instrument.
