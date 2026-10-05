# Vendored BFCL AST checker

From [ShishirPatil/gorilla](https://github.com/ShishirPatil/gorilla)
`berkeley-function-call-leaderboard/bfcl_eval/` at commit
`6ea57973c7a6097fd7c5915698c54c17c5b1b6c8`, Apache License 2.0 (`LICENSE`).

- `constants/enums.py`, `constants/type_mappings.py`,
  `eval_checker/ast_eval/ast_checker.py` and
  `eval_checker/ast_eval/type_convertor/{java,js}_type_converter.py` are
  byte-identical; their `bfcl_eval.` imports resolve through the `sys.path`
  entry added in `relex_replication/_vendor/__init__.py`.
- `salesforce_decoder.py`: `SalesforceLlamaHandler.decode_ast` from
  `model_handler/local_inference/salesforce_llama.py`, method body unchanged,
  inference base class and decorator omitted.
- `constants/model_config.py` is a local shim (not upstream) supplying the one
  `MODEL_CONFIG_MAPPING` entry the checker reads.
- Empty `__init__.py` files are local.

The official BFCL data and the two evaluation-runner modules
(`utils.py`, `eval_checker/eval_runner.py`) are downloaded at evaluation time,
hash-checked, and only the named functions are executed (see `envs.py`).
