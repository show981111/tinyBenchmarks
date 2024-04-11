from abc import ABC, abstractmethod

from collections import defaultdict
from tqdm import tqdm
import numpy as np


class BenchMarkResultProcessor(ABC):
    """Process benchmark result to create the formatted data to train IRT model."""

    @abstractmethod
    def get_metric(self, predicted: str, answer: str) -> float:
        """Based on the prediction and answer, give the score."""
        return 1 if predicted == answer else 0

    def create_correctness(
        self,
        models: list[str],
        subscenario_keyword: str,
        subscenarios: list[str],
        questions: dict,
        predictions: dict,
    ) -> dict:
        data = {}
        for model in models:
            data[model] = self._create_correctness_for_model(
                subscenario_keyword, subscenarios, questions, predictions
            )
        correctness_per_subscenario = self._collect_correctness_per_subscenario(
            data, models
        )

        return correctness_per_subscenario

    def _create_correctness_for_model(
        self,
        subscenario_keyword: str,
        subscenarios: list[str],
        questions: dict,
        predictions: dict,
    ) -> dict:
        """Based on questions and predictions, create correctness dictionary.

        Subsecnario here means the type of the question we want to categorize. The questions dictionary should include "subscenario_keyword"
        field for each question where the value is the corresponding subscenario.

        Args:
            subscenario_keyword: name of the type keyword ex) structural_type, detailed_type
            subscenarios: list of subscenario ex) choose, logical etc
            questions: format should be
                key = 'question_id'
                value = {'answer': answer_string, 'subscenario': subscenario_string} should at least be included.
            predictions: format should be
                key = 'question_id'
                value = answer_string
        """
        data = {type: defaultdict(list) for type in subscenarios}

        for qid, question in tqdm(questions.items()):
            answer = question["answer"]
            predicted = predictions[qid]

            data[question[subscenario_keyword]]["correctness"].append(
                self.get_metric(predicted, answer)
            )

        return data

    def _collect_correctness_per_subscenario(
        self, correctness_dict: dict[str, dict[str, list]], models: list[str]
    ) -> dict:
        """
        Returns:
            dictionary of 'data' and 'model'
            'data' contains correctness per each subscenario (type). Shape is (number of prompts, number of models)
            'model' contains the name of all models
        """
        data = {}
        data["data"] = {}
        data["models"] = models

        for ty in correctness_dict[list(correctness_dict.keys())[0]].keys():
            data["data"][ty] = {}
            data["data"][ty]["correctness"] = []

            for model in models:
                data["data"][ty]["correctness"].append(
                    correctness_dict[model][ty]["correctness"]
                )

            data["data"][ty]["correctness"] = np.array(
                data["data"][ty]["correctness"]
            ).T.astype(float)

        return data

    def prepare_responses(self, data: dict, types: list[str]) -> np.NDArray:
        """Stack all responses of different subscenarios"""
        responses = [np.vstack([data["data"][sub]["correctness"] for sub in types]).T]
        return np.hstack(responses)


# types = ["choose", "compare", "logical", "query", "verify"]
# models = [
#     "llava-v1.5-7b",
#     "instructblip-vicuna-7b",
#     "prism-clip+7b",
#     "prism-dinosiglip+7b",
#     "prism-siglip+7b",
# ]
# data = {}
# for model in models:
#     data[model] = create_correctness_for_model(
#         "structural_type", types, questions, predictions
#     )
