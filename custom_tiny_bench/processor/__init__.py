"""Benchmark processor module. 

For each benchmark, we need to format them to be suitable for training IRT model. 
From predictions and questions, we (1) get correctness (2) gather based on subscenario (type) (3) prepare data of shape (number_of_model * number of questions)
where each cell (i,j) represents the correctness of model i for the question j, and (4) record the position of each question in the final train data. 

In order to add a new benchmark, you need to implement get_correctness, format_questions, and format_predictions. 
get_correctness: return the correctness based on the ground truth and the prediction.
format_questions: format the question file, so that it is in the certaion format. Refer to `format_question`[benchmark_processor.BenchmarkProcessor.format_questions]
format_predictions: format the prediction file (prediction of the model you want to evaluation)

Then, you should add your benchmark processor to `custom_tiny_bench.processor.registry.py`.
"""
