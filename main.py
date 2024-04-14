from typing import Literal
from custom_tiny_bench.processor.benchmark_processor import (
    BenchmarkConfig,
    EvaluationResult,
)
from custom_tiny_bench.tiny_benchmark import TinyBenchmark


def main():
    save_dir = "data"
    bm_configs: list[BenchmarkConfig] = [
        BenchmarkConfig(
            name="gqa",
            results=[
                EvaluationResult(
                    prediction_file="data/gqa/instructblip-vicuna-7b/gqa-formatted-predictions.json",
                    model="instructblip-vicuna-7b",
                ),
                EvaluationResult(
                    prediction_file="data/gqa/prism-clip+7b/gqa-formatted-predictions.json",
                    model="prism-clip+7b",
                ),
                EvaluationResult(
                    prediction_file="data/gqa/prism-clip+7b/gqa-formatted-predictions.json",
                    model="prism-clip+7b",
                ),
                EvaluationResult(
                    prediction_file="data/gqa/prism-dinosiglip+7b/gqa-formatted-predictions.json",
                    model="prism-dinosiglip+7b",
                ),
                EvaluationResult(
                    prediction_file="data/gqa/prism-siglip+7b/gqa-formatted-predictions.json",
                    model="prism-siglip+7b",
                ),
            ],
            question_file="data/gqa/questions.json",
        )
    ]
    train_size: int | float = 0.8
    device = "cpu"
    number_item: int = 100
    random_state: int = 42
    clusterting: Literal["irt", "correct."]
    p_irt: bool = True
    gp_irt = True

    tinybm = TinyBenchmark(save_dir)
    tinybm.prepare_data(bm_configs)
    tinybm.train_irt(train_size, device)
    tinybm.get_anchors(number_item, random_state, clusterting)
    tinybm.estimate_performance(p_irt, gp_irt)


if __name__ == "__main__":
    main()
