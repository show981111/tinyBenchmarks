from typing import Any, Union

from custom_tiny_bench.processor.benchmark_processor import (
    BenchmarkConfig,
    BenchmarkProcessor,
)

from custom_tiny_bench.processor.benchmarks.text_vqa.m4c_evaluator import (
    TextVQAAccuracyEvaluator,
)


class TextVqaProcessor(BenchmarkProcessor):
    """Class for processing GQA result."""

    def __init__(self, bm_config: BenchmarkConfig) -> None:
        self.evaluator = TextVQAAccuracyEvaluator()
        super().__init__(bm_config)

    def get_correctness(self, predicted: str, answer: Union[str, list[str]]) -> float:
        """Based on the prediction and answer, give the score."""
        return self.evaluator.get_score(predicted, answer)

    def format_questions(self, questions: Any) -> dict:
        data = []
        for k, q in questions.items():
            questions[k]["question_id"] = k
            data.append(questions[k])

        return data

    def format_predictions(self, predictions: dict) -> dict:
        preds: dict[str, str] = {}
        for k, v in predictions.items():
            preds[str(k)] = v["model_output_ocr"]
        return preds
