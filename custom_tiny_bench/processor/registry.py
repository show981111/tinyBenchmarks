from custom_tiny_bench.processor.benchmarks.gqa.gqa import GQAprocessor
from custom_tiny_bench.processor.benchmarks.pope.pope import POPEprocessor
from custom_tiny_bench.processor.benchmarks.text_vqa.text_vqa import TextVqaProcessor

# list of processors.
BENCHMARK2PROCESSOR = {
    "gqa": GQAprocessor,
    "text-vqa": TextVqaProcessor,
    "pope": POPEprocessor,
}
