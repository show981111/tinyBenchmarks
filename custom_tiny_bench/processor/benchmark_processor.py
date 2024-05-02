import json
import logging
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from typing import Any, Optional, Union

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, FilePath, root_validator
from tqdm import tqdm


class EvaluationResult(BaseModel):
    prediction_file: FilePath
    model: str


class BenchmarkConfig(BaseModel):
    name: str
    results: list[EvaluationResult]
    question_file: FilePath
    subscenario_keyword: Optional[str] = None
    _models: list[str] = []  # populated automatically

    @root_validator(skip_on_failure=True)
    def _populate_missing_fields(cls, values: dict[str, Any]) -> dict[str, Any]:
        results: list[EvaluationResult] = values["results"]
        results = sorted(results, key=lambda x: x.model)

        models: list[str] = []
        if len(results) == 0:
            raise ValueError("Empty results.")
        for res in results:
            models.append(res.model)
        values["_models"] = models
        sub_key = values["subscenario_keyword"]
        if sub_key is None:
            values["subscenario_keyword"] = values["name"] + "_sub_key"
        print(models)
        return values


class Question(BaseModel):
    id: str
    answers: Union[str, list[str]]
    subscenario: str


class Prediction(BaseModel):
    question_id: str
    correctness: float


class QuestionDict(BaseModel):
    # Question id -> Question
    data: OrderedDict[str, Question]


class PredictionDict(BaseModel):
    # model name -> dict[question_id -> prediction]
    predictions_per_model: OrderedDict[str, dict[str, Prediction]]


# Should check if all benchmark has the same number of evaluation result from each model.
# Which means, BenchmarkConfig.models this list should be the same for all!

logger = logging.getLogger(__name__)


class BenchmarkProcessor(ABC):
    """Process benchmark result to create the formatted data to train IRT model."""

    def __init__(self, bm_config: BenchmarkConfig) -> None:
        self.subscenario_keyword = bm_config.subscenario_keyword
        self.questions = QuestionDict(data={})
        self.predictions = PredictionDict(predictions_per_model={})

        with open(bm_config.question_file) as file:
            formatted_question = self.format_questions(json.load(file))
            for q in formatted_question:
                sub = (
                    q[self.subscenario_keyword]
                    if self.subscenario_keyword in q
                    else f"{bm_config.name}_sub_default"
                )  # If subscenario is not provided for each question, default it to {name of benchmark}_sub
                self.questions.data[q["question_id"]] = Question(
                    id=q["question_id"], answers=q["answers"], subscenario=sub
                )

        for result in bm_config.results:
            sum = 0
            logger.info("Opening %s", result.prediction_file)
            with open(result.prediction_file) as file:
                formatted_predictions = self.format_predictions(json.load(file))
                logger.info("Number of predictions: %d", len(formatted_predictions))

                self.predictions.predictions_per_model[result.model] = {}
                for q_id, pred in formatted_predictions.items():
                    c = self.get_correctness(
                        predicted=pred, answer=self.questions.data[q_id].answers
                    )
                    sum += c
                    self.predictions.predictions_per_model[result.model][q_id] = (
                        Prediction(question_id=q_id, correctness=c)
                    )

        self.models = list(self.predictions.predictions_per_model.keys())
        print(f"Config: {self.models}")

        unique_sub: set[str] = set()
        for v in self.questions.data.values():
            unique_sub.add(v.subscenario)
        self.subscenarios = list(unique_sub)

    @abstractmethod
    def format_questions(self, questions: Any) -> list[dict]:
        """Format the questions dictionary so that it is in the form of
            [{question_id: <question id>, answers: <answer string or list>, <subscenario_keyword>(optional): <subscenario>}]

        This function is applied to the json.load(file) where the file is the input question file.
        question_id and answers are required, subscenario can be omitted. If omitted, default to <benchmark_name>_sub_key : <benchmark_name>_sub_default.
        As long as those fields are present, it is okay to have different fields.

        Default implementation sets the question_id to index. Use super.format_questions(questions) to use default.
        """
        for idx, q in enumerate(questions):
            q["question_id"] = idx
        return questions

    @abstractmethod
    def format_predictions(self, predictions: dict) -> dict[str, str]:
        """Format the predictions so that it looks like {"question_id" : "prediction", "question_id" : "prediction", ... }

        This function is applied to the json.load(file) where the file is the input prediction file.
        """
        return predictions

    @abstractmethod
    def get_correctness(self, predicted: str, answer: Union[str, list[str]]) -> float:
        """Based on the prediction and answer, give the score(correctness)."""

    # @abstractmethod
    # def format_correctness(self, result_file): ...

    def _create_correctness_per_subscenario(
        self, train: bool = True
    ) -> None:  # sub -> correctness array.
        self.correctness_per_subscenario: defaultdict[str, list[list[Prediction]]] = (
            defaultdict(list)
        )

        for qid, question_detail in self.questions.data.items():
            answer_from_all_models = []
            for model in self.models:
                if not (qid in self.predictions.predictions_per_model[model].keys()):
                    if train:
                        raise Exception(
                            "Train data should have predictions for all questions."
                        )
                    else:
                        answer_from_all_models.append(
                            Prediction(question_id=qid, correctness=np.nan)
                        )
                        # if it is not for train, we can ommit predictions for evaluation.
                else:
                    answer_from_all_models.append(
                        self.predictions.predictions_per_model[model][qid]
                    )

            assert len(answer_from_all_models) == len(self.models)
            self.correctness_per_subscenario[question_detail.subscenario].append(
                answer_from_all_models
            )

    def create_correctness_array(self, train: bool = True) -> npt.NDArray:
        """Create an IRT train data: Dimension of (model_size * questions) where each cell is correctness, and record the position of each question."""

        self._create_correctness_per_subscenario(train)
        # order of keys are deterministic since we are using ordereddict.
        self.idx_to_question_id: dict[int, str] = {}

        model_to_row_idx: dict[str, int] = {}
        self.row_idx_to_model: dict[int, str] = {}

        self.subcenarios_position: defaultdict[str, list[int]] = defaultdict(list)

        for idx, model in enumerate(self.models):
            model_to_row_idx[model] = idx
            self.row_idx_to_model[idx] = model

        col_idx = 0
        correctness_np_list: list[npt.NDArray] = []
        for (
            sub,
            correctness_list_of_questions,
        ) in self.correctness_per_subscenario.items():
            for correctness_of_one_question in correctness_list_of_questions:
                correctness_np_list.append(
                    np.array([cor.correctness for cor in correctness_of_one_question])
                )  # Each element's shape is [number_of_models * 1]

                self.subcenarios_position[sub].append(col_idx)
                self.idx_to_question_id[col_idx] = correctness_of_one_question[
                    0
                ].question_id
                col_idx += 1

        assert len(correctness_np_list) == len(self.questions.data)
        self.correctness_array = np.vstack(correctness_np_list).T.astype(float)

        assert self.correctness_array.shape == (
            len(self.models),
            len(self.questions.data),
        )
        assert sum(
            [len(indices) for indices in self.subcenarios_position.values()]
        ) == len(self.questions.data.keys())

        logger.info(
            "[create_correctness_array] Shape of correctness array %s",
            str(self.correctness_array.shape),
        )

        need_binarize = np.any(
            (self.correctness_array != 0) & (self.correctness_array != 1)
        )

        if need_binarize:
            prev_mean = self.correctness_array.mean(axis=1)
            cs = np.linspace(0.01, 0.99, 100)  # Threshold values to consider

            # Find the best threshold value that minimizes the difference between averages
            c = cs[
                np.argmin(
                    [
                        np.mean(
                            (
                                np.abs(
                                    (self.correctness_array > c).mean(axis=1)
                                    - self.correctness_array.mean(axis=1)
                                )
                            )
                        )
                        for c in tqdm(cs)
                    ]
                )
            ]
            # Apply the threshold to train and test responses
            self.correctness_array = (self.correctness_array > c).astype(int)
            logger.info(
                "After binarize, error is %s",
                str(self.correctness_array.mean(axis=1) - prev_mean),
            )

        return self.correctness_array
