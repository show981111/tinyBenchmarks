from typing import Any

import numpy as np

from custom_tiny_bench.processor.benchmark_processor import BenchmarkProcessor


class POPEprocessor(BenchmarkProcessor):
    """Class for processing GQA result."""

    def get_correctness(self, predicted: str, answer: str) -> float:
        """Based on the prediction and answer, give the score."""
        return 1 if predicted.lower() == answer.lower() else 0

    def format_questions(self, questions: Any) -> list[dict]:
        data = []
        for q in questions.values():
            q["question_id"] = str(q["question_id"])
            q["answers"] = q.pop("answer")
            data.append(q)

        return data

    def format_predictions(self, predictions: dict) -> dict[str, str]:
        data = {}
        for k, p in predictions.items():
            data[k] = ["yes", "no"][np.argmax(p["yes_no_probabilities"])]
        return data
