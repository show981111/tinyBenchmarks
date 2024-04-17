import sys

# In order to import custom tiny bench
sys.path.append("../")

import logging
from pathlib import Path
from typing import Literal
from custom_tiny_bench.processor.benchmark_processor import (
    BenchmarkConfig,
    EvaluationResult,
)
from custom_tiny_bench.tiny_benchmark import TinyBenchmark

logging.basicConfig(level=logging.DEBUG)


def main():
    ####### Configurations #########
    save_dir = Path("../data")
    bm_configs: list[BenchmarkConfig] = [
        BenchmarkConfig(
            name="gqa",
            results=[
                EvaluationResult(
                    prediction_file="../data/gqa/instructblip-vicuna-7b/gqa-formatted-predictions.json",
                    model="instructblip-vicuna-7b",
                ),
                EvaluationResult(
                    prediction_file="../ata/gqa/llava-v1.5-7b/gqa-formatted-predictions.json",
                    model="llava-v1.5-7b",
                ),
                EvaluationResult(
                    prediction_file="../data/gqa/prism-clip+7b/gqa-formatted-predictions.json",
                    model="prism-clip+7b",
                ),
                EvaluationResult(
                    prediction_file="../data/gqa/prism-dinosiglip+7b/gqa-formatted-predictions.json",
                    model="prism-dinosiglip+7b",
                ),
                EvaluationResult(
                    prediction_file="../data/gqa/prism-siglip+7b/gqa-formatted-predictions.json",
                    model="prism-siglip+7b",
                ),
            ],
            question_file="../data/gqa/questions.json",
            subscenario_keyword="structural_type",
        )
    ]
    train_size: int | float = 4
    device = "cpu"
    number_item: int = 100
    random_state: int = 42
    clustering: Literal["irt", "correct."] = "irt"
    p_irt: bool = True
    gp_irt = True
    epochs = 2000

    ####### Training Irt #########
    tinybm = TinyBenchmark(save_dir)
    tinybm.prepare_data(bm_configs)
    tinybm.train_irt(train_size, device, epochs)

    ####### Extract Anchors based on Irt Model and estimate perofrmance #########
    anchor = tinybm.get_anchors(number_item, random_state, clustering=clustering)
    irt, p_irt, gp_irt = tinybm.estimate_performance(p_irt, gp_irt)

    print("irt:", irt, "p_irt", p_irt, "gp_irt", gp_irt)


if __name__ == "__main__":
    main()
