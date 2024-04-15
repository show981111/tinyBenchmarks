"""Future work: Integrate open leaderboard data, so that we can train IRT as hugging face updates more models to open leaderboard."""

# from custom_tiny_bench.processor.benchmark_processor import BenchmarkProcessor


# class OpenLeaderBoardProcessor(BenchmarkProcessor):
#     """Processor for open leaderboard."""

#     def get_correctness(self, predicted: str, answer: str) -> float:
#         """Based on the prediction and answer, give the score."""
#         return 1 if predicted == answer else 0

#     def format_questions(self, questions: dict) -> dict:
#         return questions

#     def format_predictions(self, predictions: dict) -> dict:
#         return {str(p["questionId"]): p["prediction"] for p in predictions}
