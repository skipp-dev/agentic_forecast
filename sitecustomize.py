"""Compatibility shim for legacy tests.

This file previously monkey patched the hyperparameter agent to inject
risk-mode behavior. The logic now lives inside the agent itself, so the module
simply logs when it is imported (to preserve optional import semantics).
"""

from __future__ import annotations

import logging

_logger = logging.getLogger(__name__)
_logger.debug("sitecustomize imported - no runtime patches applied (inlined risk mode)")
