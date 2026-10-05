# Vendored Tulu 3 / open-instruct IFEval training verifier

From [allenai/open-instruct](https://github.com/allenai/open-instruct) at commit
`172e379eada34137885bd1c36543c4f59cb113e4`, Apache License 2.0 (`LICENSE`).

- `if_functions.py`: byte-identical copy of `open_instruct/if_functions.py`
  (SHA-256 `afea345e3bc100e6d5588ca14e7269141e8810e1489e60ad64a59cd57360201c`).
- `verify.py`: `VerificationResult`, `remove_thinking_section` and
  `IFEvalVerifierOld.__call__` extracted unchanged from
  `open_instruct/ground_truth_utils.py` (upstream SHA-256
  `9c5a5fe88e44bc8f6bc11da622e28947da8ef75d91d16511ddc30d45b4943dd4` is not
  vendored because it imports the whole training stack); the framework base
  class and constructor are omitted and the import of `IF_FUNCTIONS_MAP` is
  relative.
