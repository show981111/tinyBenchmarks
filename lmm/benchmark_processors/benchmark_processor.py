from abc import ABC, abstractmethod

from collections import defaultdict
import json
from pathlib import Path
from typing import Any
from pydantic import BaseModel, FilePath, root_validator
from tqdm import tqdm
import numpy as np


class EvaluationResult(BaseModel):
    prediction_file: FilePath
    model: str


class BenchmarkConfig(BaseModel):
    name: str
    results: list[EvaluationResult]
    question_file: FilePath
    subscenario_keyword: str
    models: list[str] = []  # populated automatically

    @root_validator(skip_on_failure=True)
    def _check_default_batch_size(cls, values: dict[str, Any]) -> dict[str, Any]:
        results: list[EvaluationResult] = values["results"]
        models: list[str] = values["models"]

        for res in results:
            models.append(res.model)
        return values


# Should check if all benchmark has the same number of evaluation result from each model.
# Which means, BenchmarkConfig.models this list should be the same for all!


class BenchmarkProcessor(ABC):
    """Process benchmark result to create the formatted data to train IRT model."""

    def __init__(self, bm_config: BenchmarkConfig) -> None:
        self.subscenario_keyword = bm_config.subscenario_keyword
        self.models = bm_config.models
        self.questions: dict = {}
        self.predictions: dict[str, dict] = {}

        print(f"Config: {self.models}")

        with open(bm_config.question_file) as file:
            self.questions = self.format_questions(json.load(file))

        for result in bm_config.results:
            with open(result.prediction_file) as file:
                self.predictions[result.model] = self.format_predictions(
                    json.load(file)
                )

        if self.subscenario_keyword == "N/A":
            self.subscenarios = ["N/A"]
        else:
            temp: set[str] = set()
            for v in self.questions.values():
                temp.add(v[self.subscenario_keyword])
            self.subscenarios = list(temp)

    @abstractmethod
    def format_questions(self, questions: dict) -> dict:
        return questions

    @abstractmethod
    def format_predictions(self, predictions: dict) -> dict:
        return predictions

    @abstractmethod
    def get_metric(self, predicted: str, answer: str) -> float:
        """Based on the prediction and answer, give the score."""

    def create_correctness_for_all_models(self) -> dict:
        """Create the formatted correctness result per model."""
        data = {}
        for model in self.models:
            data[model] = self._create_correctness_for_model(
                self.questions, self.predictions[model]
            )

        return data

    def _create_correctness_for_model(
        self,
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
        data = {sub: defaultdict(list) for sub in self.subscenarios}

        for qid, question in tqdm(questions.items()):
            answer = question["answer"]
            predicted = predictions[qid]
            subscenario = (
                question[self.subscenario_keyword]
                if self.subscenario_keyword != "N/A"
                else "N/A"
            )

            data[subscenario]["correctness"].append(self.get_metric(predicted, answer))

        return data

    @staticmethod
    def collect_correctness_per_subscenario(
        correctness_dict: dict[str, dict[str, list]], models: list[str]
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
