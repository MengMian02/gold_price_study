"""Single source of truth for round-level arithmetic.

The nearest-level rounding rule and the distance-to-level calculation were
previously duplicated across the Stage 2-5 scripts. They live here now so every
stage rounds and measures distance identically. Both functions accept a pandas
Series or a numpy array (or a scalar) and return the same type.
"""

from __future__ import annotations

import numpy as np

LEVEL_STEP = 50.0


def round_half_up_to_step(values, step=LEVEL_STEP):
    # Explicit half-up rule: exact midpoints between two levels choose the higher level.
    return step * np.floor((values / step) + 0.5)


def distance_to_level(prices, step=LEVEL_STEP):
    return np.abs(prices - round_half_up_to_step(prices, step))
