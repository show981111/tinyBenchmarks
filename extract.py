import ast
import json
from pathlib import Path
import pickle
from typing import Literal
from custom_tiny_bench.processor.benchmark_processor import (
    BenchmarkConfig,
    EvaluationResult,
)
from custom_tiny_bench.tiny_benchmark import TinyBenchmark


def main():
    save_dir = Path("data")
    scenario = "gqa"

    with open(
        save_dir / f"anchors/{scenario}/{scenario}_question_ids.json", "r"
    ) as handle:
        anchor = json.load(handle)

    l = []
    with open(
        save_dir / f"{scenario}/prism-siglip+7b/gqa-formatted-predictions.json", "r"
    ) as handle:
        pred = json.load(handle)
        pred = {str(p["questionId"]): p["prediction"] for p in pred}
        # print(pred[:100])
        for i in anchor:
            l.append({"questionId": i, "prediction": pred[i]})
    with open(
        save_dir / f"{scenario}/test-model/gqa-formatted-predictions.json", "w"
    ) as handle:
        json.dump(l, handle)
    print(len(l))


if __name__ == "__main__":
    main()
