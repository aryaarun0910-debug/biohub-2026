"""Byte-exact CC0 model definitions from the public reference baseline.

Two files, copied without a character changed, so that a checkpoint published
against them can be instantiated at all. They are vendored for the same reason
the official metric is: an architecture reimplemented from a description is a
different architecture, and ``strict=True`` loading would either fail or, worse,
silently succeed against the wrong shapes.

Upstream: ``repo/src/biohub_tracking/models/`` inside the Kaggle dataset
``pilkwang/biohub-tracking-support-pack-50ep-v1``, licensed CC0-1.0. The
identities of both files are bound in ``registry/reference.yaml`` under R-0004.

Nothing here is Biohub-X's work and nothing here confers belief. The weights
these definitions load are quarantined ``reference_only`` and may not support a
held-out finding; see ``registry/models.yaml``.

This package is excluded from ruff and mypy, exactly as
``_vendor/official_competition`` is, because formatting or linting vendored code
would be a silent fork of the thing whose identity is being asserted.
"""
