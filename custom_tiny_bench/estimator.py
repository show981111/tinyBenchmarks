from dataclasses import dataclass
import json
import logging
from pathlib import Path
import pickle
from typing import Callable, Tuple, Union

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from custom_tiny_bench.irt_trainer.irt import (
    estimate_ability_parameters,
    load_irt_parameters,
)
from custom_tiny_bench.irt_trainer.trainer import Anchor
from custom_tiny_bench.processor.benchmark_processor import Prediction
from custom_tiny_bench.irt_trainer.utils import item_curve

logger = logging.getLogger(__name__)


class Estimator:

    def get_estimates(
        self,
        model_correctness: npt.NDArray,
        saved_dir: Path,
        scenarios_position: dict[str, list[int]],
        balance_weights: npt.NDArray,
        p_irt: bool = True,
        gp_irt: bool = True,
    ) -> Tuple[
        dict[str, npt.NDArray],
        Union[None, dict[str, npt.NDArray]],
        Union[None, dict[str, npt.NDArray]],
    ]:
        anchor_data = Anchor({}, {})
        for scenario in scenarios_position.keys():
            with open(
                saved_dir / f"anchors/{scenario}/{scenario}_indices.pickle", "rb"
            ) as handle:
                anchor_data.points[scenario] = pickle.load(handle)
            with open(
                saved_dir / f"anchors/{scenario}/{scenario}_weights.pickle", "rb"
            ) as handle:
                anchor_data.weights[scenario] = pickle.load(handle)
        self.anc = anchor_data
        A, B, _ = load_irt_parameters(saved_dir / "irt_model")
        seen_items = np.hstack(
            [
                np.array(scenarios_position[scenario])[anchor_data.points[scenario]]
                for scenario in scenarios_position.keys()
            ]
        ).tolist()

        assert np.nan not in seen_items

        unseen_items = [
            i for i in range(model_correctness.shape[1]) if i not in seen_items
        ]

        thetas = [
            estimate_ability_parameters(
                model_correctness[j][seen_items],
                A[:, :, seen_items],
                B[:, :, seen_items],
            )
            for j in tqdm(range(model_correctness.shape[0]))
        ]

        pirt_preds = None
        gpirt_preds = None

        preds = {}
        for scenario in scenarios_position.keys():
            Y_anchor = model_correctness[:, scenarios_position[scenario]][
                :, anchor_data.points[scenario]
            ]
            preds[scenario] = (Y_anchor * anchor_data.weights[scenario]).sum(
                axis=1
            )  # Predictions

            for idx, score in enumerate(preds[scenario]):
                logger.info(
                    f"[IRT] predicted score for {idx}_th model in {scenario}: {score:.6f}"
                )

        if p_irt:
            pirt_preds = {}
            for scenario in scenarios_position.keys():

                ind_seen = [u for u in seen_items if u in scenarios_position[scenario]]
                ind_unseen = [
                    u for u in unseen_items if u in scenarios_position[scenario]
                ]
                pirt_lambd = anchor_data.points[scenario].shape[0] / len(
                    scenarios_position[scenario]
                )

                pirt_pred = []

                for j in range(model_correctness.shape[0]):
                    data_part = (balance_weights * model_correctness)[
                        j, ind_seen
                    ].mean()
                    irt_part = (balance_weights * item_curve(thetas[j], A, B))[
                        0, ind_unseen
                    ].mean()
                    pirt_pred.append(
                        pirt_lambd * data_part + (1 - pirt_lambd) * irt_part
                    )

                pirt_preds[scenario] = np.array(pirt_pred)  # Predictions

                for idx, score in enumerate(pirt_preds[scenario]):
                    logger.info(
                        f"[p-IRT] predicted score for {idx}_th model in {scenario}: {score:.6f}"
                    )

        if gp_irt:
            with open(saved_dir / "lambds.pickle", "rb") as handle:
                lambds = pickle.load(handle)

            gpirt_preds = {}
            for scenario in scenarios_position.keys():
                gpirt_preds[scenario] = (
                    lambds[scenario] * preds[scenario]
                    + (1 - lambds[scenario]) * pirt_preds[scenario]
                )

                for idx, score in enumerate(gpirt_preds[scenario]):
                    logger.info(
                        f"[gp-IRT] predicted score for {idx}_th model in {scenario}: {score:.6f}"
                    )

        return (preds, pirt_preds, gpirt_preds)
