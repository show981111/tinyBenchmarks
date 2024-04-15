import ast
import json
import logging
from pathlib import Path
import pickle
from typing import Literal
from custom_tiny_bench.processor.benchmark_processor import (
    BenchmarkConfig,
    EvaluationResult,
)
from custom_tiny_bench.tiny_benchmark import TinyBenchmark

logging.basicConfig(level=logging.DEBUG)


def main():
    save_dir = Path("data")
    bm_configs: list[BenchmarkConfig] = [
        BenchmarkConfig(
            name="gqa",
            results=[
                EvaluationResult(
                    prediction_file="data/gqa/test-model/gqa-formatted-predictions.json",
                    model="test-model",
                ),
            ],
            question_file="data/gqa/questions.json",
            subscenario_keyword="structural_type",
        )
    ]
    p_irt: bool = True
    gp_irt = True

    tinybm = TinyBenchmark(save_dir)
    tinybm.eval()
    tinybm.prepare_data(bm_configs)
    irt, p_irt, gp_irt = tinybm.estimate_performance(p_irt, gp_irt)

    print("irt:", irt, "p_irt", p_irt, "gp_irt", gp_irt)


if __name__ == "__main__":
    main()
