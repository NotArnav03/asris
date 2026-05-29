"""pytest fixtures for the FAIMR test suite.

Auto-resets the in-memory caches that the runtime classifier and
audit-time ECE lookup use, so each test starts from a clean state.
Without this, monkey-patched ECE dicts and singleton-loaded models
leak between tests.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _reset_global_caches():
    """Reset the model-card ECE cache before each test.

    The classifier singleton is NOT dropped by default — loading the
    model takes ~1s and the vast majority of tests share the same
    on-disk model.  Tests that need a fresh classifier should call
    ``reset_classifier_singleton()`` themselves.
    """
    try:
        from fairness.bias_detector import reset_model_card_ece_cache
        reset_model_card_ece_cache()
    except Exception:
        pass
    yield
    try:
        from fairness.bias_detector import reset_model_card_ece_cache
        reset_model_card_ece_cache()
    except Exception:
        pass
