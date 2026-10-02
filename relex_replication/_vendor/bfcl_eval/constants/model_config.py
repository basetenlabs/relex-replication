"""Local shim for BFCL's inference-provider registry (not upstream code).

The checker only reads `underscore_to_dot` for the xLAM JSON protocol, which
permits dotted function names.
"""
from types import SimpleNamespace

MODEL_CONFIG_MAPPING = {"xlam-json": SimpleNamespace(underscore_to_dot=False)}
