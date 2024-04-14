import logging
from pathlib import Path
import pickle
from typing import Literal, Union
import numpy as np
import numpy.typing as npt

from TinyBenchYourOwn.estimator import Estimator
from TinyBenchYourOwn.irt_trainer.trainer import Anchor, IrtTrainer
from TinyBenchYourOwn.processor.benchmark_processor import (
    BenchmarkConfig,
    BenchmarkProcessor,
)
from TinyBenchYourOwn.processor.registry import BENCHMARK2PROCESSOR

logger = logging.getLogger(__name__)


class TinyBenchmark:
    """Top most class that creates tiny benchmarks and estimate the performance.

    Receives (1) Question files, Predictions from the user
    (2) Process them and prepare the train data for IRT model
    (3) Train IRT model and save the model
    (4) Generates anchor points per bench mark and save everything we need to estimate the
        performance on predictions on this anchor points
    ------------
    (5) If the user gives, predictions on the anchor point, + data we need to estimate the performance, estimate the performance
    """

    def __init__(self, save_dir: Path = Path("tiny_benchmark_results")):
        self.trainer = IrtTrainer(save_dir)
        self.save_dir = save_dir

    def prepare_data(self, benchmark_configs: list[BenchmarkConfig]) -> None:
        """Prepare train data for IRT. Record the position of each senario in the train data.

        This function creates:
            1. train data
            2. Position of each scenario's questions
            3. balance weights for balancing different subscenarios

        Train data has the dimension of (number_of_models * number of questions from all benchmarks).
        Balance weights has the dimension of (number of questions * 1)
        """
        models = []
        self.bm_to_proc: dict[str, BenchmarkProcessor] = {}
        train_data_list: list[npt.NDArray] = []

        # benchmark name -> indices of questions in train_data
        self.scenarios_position: dict[str, list[int]] = {}

        col_idx = 0
        for bm_config in benchmark_configs:
            if models == []:
                models = bm_config.models
            else:
                if models != bm_config.models:
                    raise Exception(
                        "Should evaluate all benchmarks with the same models!"
                    )
            try:
                self.bm_to_proc[bm_config.name] = BENCHMARK2PROCESSOR[bm_config.name](
                    bm_config
                )
            except KeyError:
                print(
                    f"Unkown benchmark name: {bm_config.name}. Please create the processor for the benchmark and register in processor.registry!"
                )
                return
            d = self.bm_to_proc[bm_config.name].create_correctness_array()
            train_data_list.append(d)
            self.scenarios_position[bm_config.name] = [
                range(
                    col_idx, col_idx + d.shape[1]
                )  # From the offset, add the number of questions this benchmark have.
            ]
            col_idx += d.shape[1]

        self.train_data = np.hstack(train_data_list)
        assert self.train_data.shape == (len(models), col_idx)

        # get the balance weights for each scenario
        self.balance_weights = self._get_balance_weights(self.train_data.shape[1])

    def _get_balance_weights(self, number_of_questions: int) -> npt.NDArray:
        balance_weights = np.ones(number_of_questions)

        for bm, proc in self.bm_to_proc.items():
            n_sub = len(proc.subscenarios)  # number of subscenarios
            if n_sub > 1:
                N = len(self.scenarios_position[bm])  # number of questions in that bm
                for sub in proc.subscenarios:
                    n_i = len(proc.subcenarios_position[sub])
                    balance_weights[proc.subcenarios_position[sub]] = N / (n_sub * n_i)

        return balance_weights

    def train_irt(
        self,
        train_size: Union[int, float],
        device: Literal["cpu", "cuda"] = "cpu",
    ):
        self.trainer.train(
            self.scenarios_position,
            self.balance_weights,
            self.train_data,
            train_size,
            device,
            self.save_dir,
        )

    def get_anchors(
        self,
        model_dir: Path = Path("data/irt_model"),
        number_item: int = 100,
        random_state=42,
        clustering: Union[Literal["irt"], Literal["correct."]] = "irt",
    ) -> Anchor:
        anchor_data = self.trainer.extract_anchors(
            self.balance_weights,
            self.scenarios_position,
            model_dir=model_dir / "irt_model",
            save_dir=self.save_dir,
            number_item=number_item,
            random_state=random_state,
            clustering=clustering,
            train_data=self.train_data,
        )

        # save anchors as question ids
        for scenario in self.scenarios_position.keys():
            with open(
                self.save_dir / f"anchors/{scenario}_question_ids.pickle", "wb"
            ) as handle:
                anchor_data.weights[scenario]
                pickle.dump(
                    [
                        self.bm_to_proc[scenario].idx_to_question_id[idx]
                        for idx in anchor_data.weights[scenario]
                    ],
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )

        return anchor_data

    def estimate_performance(self):
        estimator = Estimator()
        # estimator.get_estimates()
