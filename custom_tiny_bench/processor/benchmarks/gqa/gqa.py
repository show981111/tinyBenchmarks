from typing import Any

from custom_tiny_bench.processor.benchmark_processor import BenchmarkProcessor


class GQAprocessor(BenchmarkProcessor):
    """Class for processing GQA result."""

    def get_correctness(self, predicted: str, answer: str) -> float:
        """Based on the prediction and answer, give the score."""
        return 1 if predicted.lower() == answer.lower() else 0

    def format_questions(self, questions: Any) -> list[dict]:
        data = []
        for k, q in questions.items():
            questions[k]["question_id"] = k
            questions[k]["detailed_type"] = q["types"]["detailed"]
            questions[k]["semantic_type"] = q["types"]["semantic"]
            questions[k]["structural_type"] = q["types"]["structural"]
            questions[k]["answers"] = q["answer"]
            data.append(questions[k])

        return data

    def format_predictions(self, predictions: dict) -> dict[str, str]:
        if isinstance(predictions, list):
            return {str(p["questionId"]): p["prediction"] for p in predictions}
        return {str(p["question_id"]): p["model_output"] for p in predictions.values()}