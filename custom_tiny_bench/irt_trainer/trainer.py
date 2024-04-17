import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
import numpy.typing as npt
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances
from tqdm import tqdm

from custom_tiny_bench.irt_trainer.irt import (
    create_irt_dataset,
    estimate_ability_parameters,
    load_irt_parameters,
    train_irt_model,
)
from custom_tiny_bench.irt_trainer.utils import item_curve


@dataclass
class Anchor:
    points: dict[str, npt.NDArray]
    weights: dict[str, npt.NDArray]


class IrtTrainer:

    def __init__(self, save_dir: Path = Path("data")) -> None:
        self.save_dir = save_dir

    def train(
        self,
        scenarios_position: dict[str, list[int]],
        balance_weights: npt.NDArray,
        data: npt.NDArray,
        device: Union[Literal["cpu"], Literal["cuda"]] = "cpu",
        epochs: int = 2000,  # Number of epochs for IRT model training (py-irt default is 2000)
    ):
        """Train IRT model and save lamdas for estimating the accuracy."""

        Y_train = data

        Ds = [5, 10]  # Dimensions to try
        device = "cpu"  # Either 'cuda' or 'cpu'
        lr = 0.1  # Learning rate for IRT model training (py-irt default is .1)

        val_ind = list(range(0, Y_train.shape[0], 5))  # Validation indices
        train_ind = [i for i in range(Y_train.shape[0]) if i not in val_ind]

        # Saving the training dataset in the needed format
        create_irt_dataset(Y_train[train_ind], "data/irt_val_dataset.jsonlines")

        # Trying different Ds
        errors = []
        errors2: list[list] = []

        for D in tqdm(Ds):
            dataset_name = self.save_dir / "irt_val_dataset.jsonlines"
            model_name = self.save_dir / "irt_val_model/"

            # Load trained IRT model parameters
            train_irt_model(dataset_name, model_name, D, lr, epochs, device)
            A, B, Theta = load_irt_parameters(model_name)

            # Determine seen and unseen items for validation
            seen_items = list(range(0, Y_train.shape[1], 2))
            unseen_items = list(range(1, Y_train.shape[1], 2))

            # Estimate ability parameters for the validation set
            thetas = [
                estimate_ability_parameters(
                    Y_train[val_ind][j][seen_items],
                    A[:, :, seen_items],
                    B[:, :, seen_items],
                )
                for j in range(len(val_ind))
            ]

            # Compute validation errors for each scenario and update the errors list (in the end, we give the same weight for all scenarios)
            errors2.append([])
            for scenario, position in scenarios_position.items():
                ind = [u for u in unseen_items if u in position]
                errors2[-1].append(
                    np.mean(
                        [
                            abs(
                                (balance_weights * item_curve(thetas[j], A, B))[
                                    0, ind
                                ].mean()
                                - Y_train[val_ind][j, ind].mean()
                            )
                            for j in range(len(val_ind))
                        ]
                    )
                )
            errors.append(np.mean(errors2[-1]))

        ind_D = np.argmin(np.array(errors))
        D = Ds[ind_D]

        create_irt_dataset(Y_train, self.save_dir / "irt_dataset.jsonlines")

        # Train irt
        train_irt_model(
            dataset_name=self.save_dir / "irt_dataset.jsonlines",
            model_name=self.save_dir / "irt_model/",
            D=D,
            lr=lr,
            epochs=epochs,
            device=device,
        )

        def get_lambda(b, v):
            return (b**2) / (v + (b**2))

        number_item = 100

        lambds = {}

        for i, scenario in enumerate(list(scenarios_position.keys())):
            v = np.var(Y_train[:, scenarios_position[scenario]], axis=1).mean()
            b = np.mean(errors2[ind_D][i])
            lambds[scenario] = get_lambda(b, v / (4 * number_item))

        save_file_path = self.save_dir / "lambds.pickle"
        print(save_file_path)
        save_file_path.parent.mkdir(exist_ok=True, parents=True)

        with open(save_file_path, "wb") as handle:
            pickle.dump(lambds, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def extract_anchors(
        self,
        balance_weights: npt.NDArray,
        scenarios_position: dict[str, list[int]],
        number_item: int = 100,
        random_state: int = 42,
        clustering: Union[Literal["irt"], Literal["correct."]] = "irt",
        train_data: Optional[npt.NDArray] = None,
    ) -> Anchor:
        """Extract the anchor points using irt/correctness clustring.

        Returns:
            Tuple of anchor points and weights. Indices in anchor points are indices of train_data[scenario].
            Which means, Indices in the Nd array of training data filtered by each scenario.
        """
        anchor_points = {}
        anchor_weights = {}
        model_dir = self.save_dir / "irt_model"

        for scenario, pos in scenarios_position.items():

            if clustering == "correct.":
                if train_data is None:
                    raise Exception(
                        "Train data is required for correctness clustering!"
                    )
                X = train_data[:, pos].T
            elif clustering == "irt":
                A, B, _ = load_irt_parameters(model_dir)
                X = np.vstack((A.squeeze(), B.squeeze().reshape((1, -1)))).T
                X = X[pos]
            else:
                raise NotImplementedError

            # Normalizing balance_weights, so their sum is one within each scenario
            norm_balance_weights = balance_weights[pos]
            norm_balance_weights /= norm_balance_weights.sum()

            # Fitting the KMeans model
            kmeans = KMeans(
                n_clusters=number_item, n_init="auto", random_state=random_state
            )
            kmeans.fit(X, sample_weight=norm_balance_weights)

            # Calculating anchor points
            anchor_points[scenario] = pairwise_distances(
                kmeans.cluster_centers_, X, metric="euclidean"
            ).argmin(axis=1)

            # Calculating anchor weights
            anchor_weights[scenario] = np.array(
                [
                    np.sum(norm_balance_weights[kmeans.labels_ == c])
                    for c in range(number_item)
                ]
            )

            path_save_indices = (
                self.save_dir / f"anchors/{scenario}/{scenario}_indices.pickle"
            )
            path_save_weights = (
                self.save_dir / f"anchors/{scenario}/{scenario}_weights.pickle"
            )
            path_save_indices.parent.mkdir(exist_ok=True, parents=True)
            path_save_weights.parent.mkdir(exist_ok=True, parents=True)
            with open(path_save_indices, "wb") as handle:
                pickle.dump(
                    anchor_points[scenario],
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            with open(path_save_weights, "wb") as handle:
                pickle.dump(
                    anchor_weights[scenario],
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )

        return Anchor(points=anchor_points, weights=anchor_weights)
