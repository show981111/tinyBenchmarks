from abc import ABC, abstractmethod

from collections import OrderedDict, defaultdict
import json
from pathlib import Path
from typing import Any
from pydantic import BaseModel, FilePath, root_validator
from tqdm import tqdm
import numpy as np
import numpy.typing as npt


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


class Question(BaseModel):
    id: str
    answer: str
    subscenario: str


class Prediction(BaseModel):
    question_id: str
    prediction: str


class QuestionDict(BaseModel):
    # Question id -> Question
    data: OrderedDict[str, Question]


class PredictionDict(BaseModel):
    # model name -> dict[question_id -> prediction]
    predictions_per_model: OrderedDict[str, dict[str, Prediction]]


# Should check if all benchmark has the same number of evaluation result from each model.
# Which means, BenchmarkConfig.models this list should be the same for all!


class BenchmarkProcessor(ABC):
    """Process benchmark result to create the formatted data to train IRT model."""

    def __init__(self, bm_config: BenchmarkConfig) -> None:
        self.subscenario_keyword = bm_config.subscenario_keyword
        self.models = bm_config.models
        self.questions = QuestionDict()
        self.predictions = PredictionDict()

        print(f"Config: {self.models}")

        with open(bm_config.question_file) as file:
            formatted_question = self.format_questions(json.load(file))
            for qid, v in formatted_question.items():
                sub = (
                    v[self.subscenario_keyword]
                    if self.subscenario_keyword in v
                    else f"{bm_config}_sub"
                )
                self.questions[qid] = Question(
                    id=qid, answer=v["answer"], subscenario=sub
                )

        for result in bm_config.results:
            with open(result.prediction_file) as file:
                formatted_predictions = self.format_predictions(json.load(file))
                for q_id, pred in formatted_predictions:
                    self.predictions.predictions_per_model.get(result.model, {})[
                        q_id
                    ] = pred

        unique_sub: set[str] = set()
        for v in self.questions.data.values():
            unique_sub.add(v.subscenario)
        self.subscenarios = list(unique_sub)

    @abstractmethod
    def format_questions(self, questions: dict) -> dict:
        """Format the questions dictionary so that it is in the form of {question_id: {answer: <answer string>, <subscenario_keyword>: <subscenario>}}

        This function is applied to the json.load(file) where the file is the input question file.
        question_id and answer are required, subscenario can be omitted. If omitted, default to <benchmark_name>_sub : <benchmark_name>_sub.
        As long as those fields are present, it is okay to have different fields.
        """
        return questions

    @abstractmethod
    def format_predictions(self, predictions: dict) -> dict[str, str]:
        """Format the predictions so that it looks like {"question_id" : "prediction", "question_id" : "prediction", ... }

        This function is applied to the json.load(file) where the file is the input prediction file.
        """
        return predictions

    @abstractmethod
    def get_correctness(self, predicted: str, answer: str) -> float:
        """Based on the prediction and answer, give the score(correctness)."""

    """
    TODO: A function that creates (number_of_models, questions) dimension ND array where each cell is the correctness[model_idx][question_idx]
    Need a bookkeeping where (1) which answers are from which subscenario. For subcenario "A", which columns are in "A"? == subscenario position
    ++ (0) From the top level, we need which questions are from which scenario/benchmark (which columns are from benchmark "B")? == scenario position
    """

    def create_correctness_array(self, train: bool = True) -> npt.NDArray:
        """Create an IRT train data: Dimension of (model_size * questions) where each cell is correctness, and record the position of each question."""

        # order of keys are deterministic since we are using ordereddict.
        question_ids = list(self.questions.data.keys())
        models = list(self.predictions.predictions_per_model.keys())

        self.question_id_to_idx: dict[str, int] = {}
        self.idx_to_question_id: dict[int, str] = {}

        self.model_to_row_idx: dict[str, int] = {}
        self.row_idx_to_model: dict[int, str] = {}

        self.subcenarios_position: defaultdict[str, list[int]] = defaultdict(list)

        for idx, qid in enumerate(question_ids):
            self.question_id_to_idx[qid] = idx
            self.idx_to_question_id[idx] = qid
        for idx, model in enumerate(models):
            self.model_to_row_idx[model] = idx
            self.row_idx_to_model[idx] = model

        self.correctness_array: npt.NDArray = np.zeros(
            [
                len(self.predictions.predictions_per_model.keys()),
                len(self.questions.data.keys()),
            ]
        )

        for qid, data in self.questions.data.items():
            self.subcenarios_position[data.subscenario].append(
                self.question_id_to_idx[qid]
            )

            for model, pred in self.predictions.predictions_per_model.items():
                correctness = np.nan
                if not (qid in pred[qid].prediction):
                    if train:
                        raise Exception(
                            "Train data should have predictions for all questions."
                        )
                    # else: if it is not for train, we can ommit predictions for evaluation.
                else:
                    correctness = self.get_correctness(
                        predicted=pred[qid].prediction, answer=data.answer
                    )

                self.correctness_array[
                    self.model_to_row_idx[model], self.question_id_to_idx[qid]
                ] = correctness

        assert sum(
            [len(indices) for indices in self.subcenarios_position.values()]
        ) == len(self.questions.data.keys())

        return self.correctness_array

    # def create_correctness_for_all_models(self) -> dict:
    #     """Create the formatted correctness result per model."""
    #     data = {}
    #     for model in self.models:
    #         data[model] = self._create_correctness_for_model(
    #             self.questions, self.predictions[model]
    #         )

    #     return data

    # def _create_correctness_for_model(
    #     self,
    #     questions: dict,
    #     predictions: dict,
    # ) -> dict:
    #     """Based on questions and predictions, create correctness dictionary.

    #     Subsecnario here means the type of the question we want to categorize. The questions dictionary should include "subscenario_keyword"
    #     field for each question where the value is the corresponding subscenario.

    #     Args:
    #         subscenario_keyword: name of the type keyword ex) structural_type, detailed_type
    #         subscenarios: list of subscenario ex) choose, logical etc
    #         questions: format should be
    #             key = 'question_id'
    #             value = {'answer': answer_string, 'subscenario': subscenario_string} should at least be included.
    #         predictions: format should be
    #             key = 'question_id'
    #             value = answer_string
    #     """
    #     data = {sub: defaultdict(list) for sub in self.subscenarios}

    #     for qid, question in tqdm(questions.items()):
    #         answer = question["answer"]
    #         predicted = predictions[qid]
    #         subscenario = (
    #             question[self.subscenario_keyword]
    #             if self.subscenario_keyword != "N/A"
    #             else "N/A"
    #         )

    #         data[subscenario]["correctness"].append(
    #             self.get_correctness(predicted, answer)
    #         )

    #     return data

    # @staticmethod
    # def collect_correctness_per_subscenario(
    #     correctness_dict: dict[str, dict[str, list]], models: list[str]
    # ) -> dict:
    #     """
    #     Returns:
    #         dictionary of 'data' and 'model'
    #         'data' contains correctness per each subscenario (type). Shape is (number of prompts, number of models)
    #         'model' contains the name of all models
    #     """
    #     data = {}
    #     data["data"] = {}
    #     data["models"] = models

    #     for sub in correctness_dict[list(correctness_dict.keys())[0]].keys():
    #         data["data"][sub] = {}
    #         data["data"][sub]["correctness"] = []

    #         for model in models:
    #             data["data"][sub]["correctness"].append(
    #                 correctness_dict[model][sub]["correctness"]
    #             )

    #         data["data"][sub]["correctness"] = np.array(
    #             data["data"][sub]["correctness"]
    #         ).T.astype(float)

    #     return data
