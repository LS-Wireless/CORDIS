"""
CORDIS experiments package — Stage 8a.

Helpers for managing experiment outputs with consistent path
conventions that respect environment variables for cluster portability.

(More to come in Stage 8b: experiment-spec library, sweep helpers,
registry.)
"""
from cordis.experiments.io import (
    experiment_dir,
    figure_dir,
    latest_result,
)

__all__ = ["experiment_dir", "figure_dir", "latest_result"]

