import json
import logging
from pathlib import Path
import pickle
from typing import Literal, Optional, Tuple, Union
import numpy as np
import numpy.typing as npt

from custom_tiny_bench.estimator import Estimator
from custom_tiny_bench.irt_trainer.trainer import Anchor, IrtTrainer
from custom_tiny_bench.processor.benchmark_processor import (
    BenchmarkConfig,
    BenchmarkProcessor,
)
from custom_tiny_bench.processor.registry import BENCHMARK2PROCESSOR

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
        """Initialize TinyBenchmark.

        Args:
            save_dir:
                If using this for training, this path is the path for saving models and anchors.
                If using this for estimating, this path is the path that has models and anchors.
        """
        self.trainer = IrtTrainer(save_dir)
        self.save_dir = save_dir
        self.train_size: Optional[Union[int, float]] = None
        self.train_data: Optional[npt.NDArray] = None
        self.test_data: Optional[npt.NDArray] = None

        self.correctness_array: Optional[npt.NDArray] = None
        self.scenarios_position: Optional[dict[str, list[int]]] = None
        self.balance_weights: Optional[npt.NDArray] = None

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
        correctness_list: list[npt.NDArray] = []

        # benchmark name -> indices of questions in train_data
        self.scenarios_position: dict[str, list[int]] = {}

        col_idx = 0
        for bm_config in benchmark_configs:
            if models == []:
                models = bm_config._models
            else:
                if models != bm_config._models:
                    raise Exception(
                        "Should evaluate all benchmarks with the same models!"
                    )
            try:
                self.bm_to_proc[bm_config.name] = BENCHMARK2PROCESSOR[bm_config.name](
                    bm_config
                )
            except KeyError:
                logger.error(
                    "Unkown benchmark name: %s. Please create the processor for the benchmark and register in processor.registry!",
                    bm_config.name,
                )
                return
            d = self.bm_to_proc[bm_config.name].create_correctness_array()
            correctness_list.append(d)
            self.scenarios_position[bm_config.name] = list(
                range(col_idx, col_idx + d.shape[1])
            )  # From the offset, add the number of questions this benchmark have.

            col_idx += d.shape[1]

        self.correctness_array = np.hstack(correctness_list)

        logger.info(
            "[prepare_data] correctness_array.shape == %s", str((len(models), col_idx))
        )
        assert self.correctness_array.shape == (len(models), col_idx)

        # get the balance weights for each scenario
        self.balance_weights = self._get_balance_weights(
            self.correctness_array.shape[1]
        )

    def _get_balance_weights(self, number_of_questions: int) -> npt.NDArray:
        balance_weights = np.ones(number_of_questions)

        for bm, proc in self.bm_to_proc.items():
            n_sub = len(proc.subscenarios)  # number of subscenarios
            if n_sub > 1:
                N = len(self.scenarios_position[bm])  # number of questions in that bm
                for sub in proc.subscenarios:
                    n_i = len(proc.subcenarios_position[sub])
                    logger.info("%s has %d number of questions", sub, n_i)
                    balance_weights[proc.subcenarios_position[sub]] = N / (n_sub * n_i)

        return balance_weights

    @staticmethod
    def split_data(
        data: npt.NDArray, train_size: Union[int, float]
    ) -> Tuple[npt.NDArray, npt.NDArray]:
        """Split the data based on train size. train_size can be either int or float, which is the ratio.

        Returns:
            Tuple of train_data, test_date
        """
        if isinstance(train_size, int):
            return (data[:train_size], data[train_size:])
        else:
            return (
                data[: round(train_size * data.shape[0])],
                data[round(train_size * data.shape[0]) :],
            )

    def train_irt(
        self,
        train_size: Union[int, float],
        device: Literal["cpu", "cuda"] = "cpu",
        epochs: int = 2000,
    ):
        self.train_size = train_size
        self.train_data, self.test_data = TinyBenchmark.split_data(
            self.correctness_array, train_size
        )

        self.trainer.train(
            self.scenarios_position,
            self.balance_weights,
            self.train_data,
            device=device,
            epochs=epochs,
        )

    def get_anchors(
        self,
        number_item: int = 100,
        random_state=42,
        clustering: Union[Literal["irt"], Literal["correct."]] = "irt",
    ) -> Anchor:
        if self.train_data is None or self.test_data is None:
            raise Exception(
                "Need to train the irt first before getting anchors. Call train_irt() first!"
            )

        anchor_data = self.trainer.extract_anchors(
            self.balance_weights,
            self.scenarios_position,
            number_item=number_item,
            random_state=random_state,
            clustering=clustering,
            train_data=self.train_data,
        )

        # save anchors as question ids
        for scenario in self.scenarios_position.keys():
            save_anchor_dir = self.save_dir / f"anchors/{scenario}"
            save_anchor_dir.mkdir(exist_ok=True, parents=True)
            with open(save_anchor_dir / f"{scenario}_question_ids.txt", "w") as handle:
                handle.write(
                    str(
                        [
                            self.bm_to_proc[scenario].idx_to_question_id[idx]
                            for idx in anchor_data.points[scenario]
                        ]
                    )
                )

        print("Points", anchor_data.points[scenario])
        for scenario in self.scenarios_position.keys():
            Y_anchor = self.test_data[:, self.scenarios_position[scenario]][
                :, anchor_data.points[scenario]
            ]
            Y_hat = (Y_anchor * anchor_data.weights[scenario]).sum(axis=1)
            Y_true = (self.balance_weights * self.test_data)[
                :, self.scenarios_position[scenario]
            ].mean(axis=1)

            logger.info(
                f"[Anchor points] scenario: {scenario}, avg. error: {np.abs(Y_hat-Y_true).mean():.10f}"
            )

        return anchor_data

    def estimate_performance(self, p_irt: bool = True, gp_irt=True):
        if (
            self.correctness_array is None
            or self.scenarios_position is None
            or self.balance_weights is None
        ):
            raise Exception(
                "Need to prepare the data first! call prepare_data() before estimating the performance."
            )

        estimator = Estimator()
        return estimator.get_estimates(
            self.correctness_array if self.test_data is None else self.test_data,
            self.save_dir,
            self.scenarios_position,
            self.balance_weights,
            p_irt=p_irt,
            gp_irt=gp_irt,
        )
