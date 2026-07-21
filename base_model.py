"""
Common interface every benchmarked model must implement.

To add a new model for the paper:
  1. Create models/<your_model>.py
  2. Subclass BenchmarkModel and implement predict()
  3. Register it in evaluate.py's get_model() factory

Keeping this interface identical across models is what lets evaluate.py
and metrics.py stay unchanged as you add more models.
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class PredictionResult:
    predicted_index: Optional[int]   # 0-based index into sample.choices, None if unparseable
    raw_response: str                # full text the model produced, kept for auditing/debugging
    error: Optional[str] = None      # set if inference itself failed (OOM, bad file, etc.)


class BenchmarkModel(ABC):
    name: str = "base_model"

    @abstractmethod
    def predict(self, sample) -> PredictionResult:
        """sample is an AVQASample (see dataset.py)."""
        raise NotImplementedError

    def build_prompt(self, sample) -> str:
        letters = "ABCDEFGH"
        options_block = "\n".join(
            f"{letters[i]}. {c}" for i, c in enumerate(sample.choices)
        )
        return (
            "Watch and listen to the video, then answer the question using "
            "the visual and audio content.\n\n"
            f"Question: {sample.question_text}\n\n"
            f"Options:\n{options_block}\n\n"
            "Respond with only the single letter of the correct option."
        )

    @staticmethod
    def parse_letter_answer(text: str, num_choices: int) -> Optional[int]:
        """Best-effort parse of a free-text model response into a choice index."""
        if not text:
            return None
        letters = "ABCDEFGH"[:num_choices]
        m = re.search(rf"\b([{letters}])\b", text.strip().upper())
        if m:
            return letters.index(m.group(1))
        return None
