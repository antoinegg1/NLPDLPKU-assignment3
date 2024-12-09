# NLPDL - Assignment 3

## Environment 

Initialize the environment by:
```bash
pip install -r requirements.txt
```

## Code Structure

### `task1/`
- **简介**：与实验任务1相关的代码和数据。
- **文件说明**：
  - `compare_experiment.py`：用于进行比较不同缓存和量化策略实验。
  - `customized_gpt2.py`：实现了customized KV-Cache。
  - `data.txt`：任务1使用的实验数据。
  - `main.py`：任务1的主程序，customized KV-Cache推理速度比较的实验。

### `task2/`
- **简介**：与实验任务2相关的代码和结果。
- **文件说明**：
  - `result/`：存储任务2中不同推理方法的结果，按照实验方法分类存储（如Direct Answer, CoT, ICL等）。
  - `main.py`：任务2的主程序，实现了结果生成和推理方法的逻辑。

## task1

通过以下代码,设置设置不同的max_new_tokens和num_repeats参数运行Compare Experiment:
```bash
python ./task1/compare_experiment.py --max_new_tokens 100 --num_repeats 5
```

通过以下代码运行Customized KV-Cache与Golden Greedy Decoding方法Decode时间比较的实验:
```bash
python ./task1/main.py
```

## task2

通过以下代码运行不同推理技术比较的实验
```bash
python ./task2/main.py  --output_file ./task2/result/gsm8k_direct_answer_results.json --type direct_answer
python ./task2/main.py  --output_file ./task2/result/gsm8k_cot_answer_results.json --type cot
python ./task2/main.py  --output_file ./task2/result/gsm8k_cot_reflexion_answer_results.json --type cot_reflexion
python ./task2/main.py  --input_file ./task2/result/gsm8k_direct_answer_results.json --output_file ./task2/result/gsm8k_cot_icl_answer_results.json --type cot_icl
python ./task2/main.py  --input_file ./task2/result/gsm8k_direct_answer_results.json --output_file ./task2/result/gsm8k_cot_icl_reflexion_answer_results.json --type cot_icl_reflexion
```