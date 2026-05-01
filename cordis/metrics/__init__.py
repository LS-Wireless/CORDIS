"""
cordis.metrics
==============
Performance metrics for the CORDIS simulation framework.

Public API
----------
SINR
    SINRMetrics, compute_sinr, sinr_to_rate, rate_to_sinr
SCNR
    SCNRMetrics, compute_scnr, compute_sensing_surrogate
Aggregation
    SINRStatistics, SCNRStatistics, aggregate_sinr, aggregate_scnr
Fronthaul
    FronthaulOverhead, compute_fronthaul_overhead, compare_overhead
"""

from cordis.metrics.sinr import (
    SINRMetrics, compute_sinr, sinr_to_rate, rate_to_sinr,
)
from cordis.metrics.scnr import (
    SCNRMetrics, compute_scnr, compute_sensing_surrogate,
)
from cordis.metrics.aggregation import (
    SINRStatistics, SCNRStatistics, aggregate_sinr, aggregate_scnr,
)
from cordis.metrics.fronthaul import (
    FronthaulOverhead, compute_fronthaul_overhead, compare_overhead,
)

__all__ = [
    "SINRMetrics", "compute_sinr", "sinr_to_rate", "rate_to_sinr",
    "SCNRMetrics", "compute_scnr", "compute_sensing_surrogate",
    "SINRStatistics", "SCNRStatistics", "aggregate_sinr", "aggregate_scnr",
    "FronthaulOverhead", "compute_fronthaul_overhead", "compare_overhead",
]
