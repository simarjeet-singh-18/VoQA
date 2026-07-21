"""
Trivial baseline that ignores the video/audio entirely and picks a
random option. Useful for (a) sanity-checking the eval pipeline end
to end without any GPU/model dependency, and (b) reporting a
random/chance-level row in the paper's results table.
"""

import random

from .base_model import BenchmarkModel, PredictionResult


class RandomBaselineModel(BenchmarkModel):
    name = "random_baseline"

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def predict(self, sample) -> PredictionResult:
        idx = self.rng.randrange(len(sample.choices))
        return PredictionResult(predicted_index=idx, raw_response=f"[random choice] -> {idx}")
