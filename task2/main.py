import json
import time
import requests
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
from openai import OpenAI
import re
import json
import random
import argparse

def get_deepseek_response(API_KEY, API_URL, system_prompt, prompt):
    client = OpenAI(api_key=API_KEY, base_url=API_URL)
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        stream=False
    )
    return response.choices[0].message.content

def is_correct(generated_answer, direct_answer):
    # 简单的字符串比较，可以根据需要改进
    match = re.findall(r'\[(.*?)\]', generated_answer)
    if match is None or len(match) == 0:
        return False
    return match[-1] == direct_answer

def run_experiment(input_file, output_file, experiment_type):
    # experiment_type可取：
    # direct_answer, cot, cot_reflexion, cot_icl, cot_icl_reflexion

    # 请替换为实际的API端点与API密钥
    API_URL = "https://api.deepseek.com"
    API_KEY = "YOUR_API_KEY"  # 修改为你的API Key

    # 加载GSM8K数据集的测试集，用于计算总题目数
    dataset = load_dataset("openai/gsm8k", "main")["test"]

    # 仅在需要时加载 `reflexion_dataset`
    reflexion_dataset = None
    if experiment_type in ["cot_reflexion", "cot_icl", "cot_icl_reflexion"]:
        if not input_file:
            raise ValueError(f"Experiment type {experiment_type} requires an input dataset.")
        with open(input_file, "r", encoding="utf-8") as file:
            reflexion_dataset = json.load(file)

    # 用于提取标准答案的正则表达式
    direct_ans_pattern = r"####\s*(.*)"

    # 准备few-shot样例（当使用ICL时需要）
    if experiment_type in ["cot_icl", "cot_icl_reflexion"]:
        if not reflexion_dataset:
            raise ValueError("Few-shot examples are required for ICL experiments.")
        few_shot_items = random.sample(reflexion_dataset, 3)
        question_1 = few_shot_items[0]['question']
        answer_1 = few_shot_items[0]["correct_answer"]
        question_2 = few_shot_items[1]['question']
        answer_2 = few_shot_items[1]["correct_answer"]
        question_3 = few_shot_items[2]['question']
        answer_3 = few_shot_items[2]["correct_answer"]
        few_shot_prompt = '''You will be provided with a few examples of questions and answers as few-shot samples...
        '''
        few_shot_prompt = few_shot_prompt.format(
            question_1=question_1, answer_1=answer_1,
            question_2=question_2, answer_2=answer_2,
            question_3=question_3, answer_3=answer_3
        )
    else:
        few_shot_prompt = ""

    # 根据experiment_type设置不同的提示语策略
    base_system_prompt = "You are a helpful assistant."
    direct_answer_system_prompt = base_system_prompt + "\nSolve the following question and provide the answer directly in [ ]. The answer should be a number without any additional text.\n"
    cot_system_prompt = base_system_prompt + "\nSolve the following question by thinking step by step and provide the answer in the end in [ ]. The final answer should be a number without any additional text.\n"

    # 根据类型确定最终system_prompt和是否需要reflexion
    if experiment_type == "direct_answer":
        system_prompt = few_shot_prompt + direct_answer_system_prompt
        use_cot = False
        use_reflexion = False
    elif experiment_type == "cot":
        system_prompt = few_shot_prompt + cot_system_prompt
        use_cot = True
        use_reflexion = False
    elif experiment_type == "cot_reflexion":
        system_prompt = few_shot_prompt + cot_system_prompt
        use_cot = True
        use_reflexion = True
    elif experiment_type == "cot_icl":
        system_prompt = few_shot_prompt + cot_system_prompt
        use_cot = True
        use_reflexion = False
    elif experiment_type == "cot_icl_reflexion":
        system_prompt = few_shot_prompt + cot_system_prompt
        use_cot = True
        use_reflexion = True
    else:
        raise ValueError("Unsupported experiment_type")

    results = []
    total_count = len(dataset)
    correct_count = 0

    # 处理数据集（仅在需要时）
    examples = reflexion_dataset if reflexion_dataset else dataset
    for example in tqdm(examples, desc="Processing Dataset"):
        question = example["question"]
        correct_answer = example.get("correct_answer", "#### 0")
        match = re.search(direct_ans_pattern, correct_answer)
        if match:
            direct_answer = match.group(1).strip()
        else:
            # 若无匹配到direct answer，则跳过
            continue

        if use_reflexion and "icl_answer" in example:
            original_cot_icl_answer = example["icl_answer"]
            reflection_prompt = f"""
            The following is your initial answer:
            {original_cot_icl_answer}
            Please reflect on the answer and identify any potential errors or areas for improvement.
            """
            prompt = question + reflection_prompt
        else:
            prompt = question

        # 调用API
        generated_answer = get_deepseek_response(API_KEY, API_URL, system_prompt, prompt)

        # 判断正确性
        correct = is_correct(generated_answer, direct_answer)
        if correct:
            correct_count += 1

        # 保存结果
        results.append({
            "experiment_type": experiment_type,
            "question": question,
            "correct_answer": correct_answer,
            "final_answer": generated_answer,
            "is_correct": correct
        })

    # 保存结果
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    # 打印Accuracy
    print(f"{experiment_type} Accuracy: {correct_count}/{total_count} = {correct_count/total_count:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Run GSM8K experiment with different settings.")
    parser.add_argument("--input_file", type=str, default=None, help="Path to input JSON file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to output JSON file.")
    parser.add_argument("--type", type=str, required=True, choices=["direct_answer","cot","cot_reflexion","cot_icl","cot_icl_reflexion"],
                        help="Experiment type.")
    args = parser.parse_args()

    run_experiment(args.input_file, args.output_file, args.type)

if __name__ == "__main__":
    main()
