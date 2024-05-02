# Build your own tiny benchmarks

Using `custom_tiny_bench`, you can easily add any benchmark you want to make the tiny version.
Only things you need to supply are (1) benchmark data files (2) model's predictions on the benchmark.

## Supported benchmarks

- GQA
- Text-VQA
- Pope

## Adding your own benchmark

You can add your own benchmark by inheriting the `custom_tiny_bench.processor.BenchmarkProcessor`.
You can refer to `custom_tiny_bench.processor.benchmarks.gqa.py` for an example.
Also, refer to `custom_tiny_bench.processor.__init__.py` for the detailed steps of adding your own benchmarks.

## Examples

Refer to `examples/train_example.py` for how to train your own IRT model and extract anchors and `examples/evaluation_example.py` for how to estimate the model performance on anchors.
