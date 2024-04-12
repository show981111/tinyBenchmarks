import json
from typing import Tuple
from dataclasses import dataclass
import numpy.typing as npt
from pydantic import BaseModel
from lmm.benchmark_processors.benchmark_processor import (
    BenchmarkConfig,
    BenchmarkProcessor,
)
from lmm.utils import prepare_data, create_responses
from lmm.benchmark_processors import GQAprocessor

# Register your own processor here.
BENCHMARK2PROCESSOR = {"gqa": GQAprocessor}


@dataclass
class IrtTrainData:
    scenarios_position: dict
    subscenarios_position: dict
    scenarios: dict
    train_data: npt.NDArray


def generate_irt_train_data(
    benchmark_configurations: dict[str, BenchmarkConfig]
) -> IrtTrainData:
    """

    Returns:
        Tuple of scenarios_position, subscenarios_position, scenarios, Y
    """
    bm_to_proc: dict[str, BenchmarkProcessor] = {}
    scenarios = {}
    models = []

    correctness_per_model = {}
    for bm, bm_config in benchmark_configurations.items():
        if models == []:
            models = bm_config.models
        else:
            if models != bm_config.models:
                raise Exception("Should evaluate all benchmarks with the same models!")

        bm_to_proc[bm] = BENCHMARK2PROCESSOR[bm](bm_config)
        scenarios[bm] = bm_to_proc[bm].subscenarios
        correctness_per_model |= bm_to_proc[
            bm
        ].create_correctness_for_all_models()  # concat two dictionaries

    correctness_per_subscenario = (
        BenchmarkProcessor.collect_correctness_per_subscenario(
            correctness_per_model, models
        )
    )
    scenarios_position, subscenarios_position = prepare_data(
        scenarios, correctness_per_subscenario
    )
    Y = create_responses(scenarios, correctness_per_subscenario)

    return IrtTrainData(scenarios_position, subscenarios_position, scenarios, Y)
