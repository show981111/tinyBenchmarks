import sys

# In order to import custom tiny bench from example directory
sys.path.append("../")

import logging
from pathlib import Path
from custom_tiny_bench.processor.benchmark_processor import (
    BenchmarkConfig,
    EvaluationResult,
)
from custom_tiny_bench.tiny_benchmark import TinyBenchmark

logging.basicConfig(level=logging.DEBUG)


def main():
    save_dir = Path("../data")
    bm_configs: list[BenchmarkConfig] = [
        BenchmarkConfig(
            name="gqa",
            results=[
                EvaluationResult(
                    prediction_file="../data/gqa/prism-siglip+7b/gqa-formatted-predictions.json",
                    model="prism-siglip+7b",
                ),
                EvaluationResult(
                    prediction_file="../data/gqa/mordal-clip-b-vicuna-v15-7b-mlp/gqa-formatted-predictions.json",
                    model="mordal-clip-b-vicuna-v15-7b-mlp",
                ),
                EvaluationResult(
                    prediction_file="../data/gqa/mordal-clip-l-llama2-7b-chat-mlp/gqa-formatted-predictions.json",
                    model="mordal-clip-l-llama2-7b-chat-mlp",
                ),
                EvaluationResult(
                    prediction_file="../data/gqa/mordal-clip-l-vicuna-v15-7b-mlp/gqa-formatted-predictions.json",
                    model="mordal-clip-l-vicuna-v15-7b-mlp",
                ),
                EvaluationResult(
                    prediction_file="../data/gqa/instructblip-vicuna-7b/gqa-formatted-predictions.json",
                    model="instructblip-vicuna-7b",
                ),
            ],
            question_file="../data/gqa/questions.json",
            subscenario_keyword="structural_type",
        )
    ]
    p_irt: bool = True
    gp_irt = True

    tinybm = TinyBenchmark(save_dir, balance=True)
    tinybm.eval()
    tinybm.prepare_data(bm_configs)
    irt, p_irt, gp_irt = tinybm.estimate_performance(p_irt, gp_irt)

    print("irt:", irt, "p_irt", p_irt, "gp_irt", gp_irt)


if __name__ == "__main__":
    main()
