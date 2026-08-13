"""Data quality checks.

Read-only integrity checks over ingested data. Run before trusting a season for
modelling: an unnoticed coverage gap or arithmetic inconsistency becomes a
silent bias in every model trained on it.
"""

from euroleague_ai.quality.checks import (
    ALL_CHECKS,
    Finding,
    QualityReport,
    Severity,
    run_quality_checks,
)

__all__ = ["ALL_CHECKS", "Finding", "QualityReport", "Severity", "run_quality_checks"]
