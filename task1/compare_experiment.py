import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import time
import os
import argparse

def measure_throughput(model, tokenizer, input_text, use_kv_cache, device, generate_kwargs, num_repeats=1):
    model.eval()
    if not use_kv_cache:
        model.config.use_cache = False

    # 准备输入数据
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
    attention_mask = torch.ones_like(input_ids)

    # 预热 GPU
    with torch.no_grad():
        _ = model.generate(input_ids, attention_mask=attention_mask, use_cache=use_kv_cache, **generate_kwargs)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    total_time = 0
    for _ in range(num_repeats):
        start_time = time.time()
        with torch.no_grad():
            output = model.generate(
                input_ids,
                attention_mask=attention_mask,
                use_cache=use_kv_cache,
                **generate_kwargs
            )
        end_time = time.time()
        total_time += (end_time - start_time)

    avg_time = total_time / num_repeats
    generated_tokens = output.shape[1] - input_ids.shape[1]
    throughput = generated_tokens / avg_time
    peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # 转换为 MB

    return throughput, peak_memory


def inference_pipeline(
    mode="naive",
    input_text="",
    max_new_tokens=50,
    quantization_config=None,
    num_repeats=3
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = "openai-community/gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if quantization_config == 'fp16':
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
        model.to(device)
    elif quantization_config == 'int8':
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map='auto')
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.to(device)

    use_kv_cache = mode == "kv_cache"
    generate_kwargs = {
        'max_new_tokens': max_new_tokens,
        'do_sample': False,
    }

    throughput, peak_memory = measure_throughput(
        model, tokenizer, input_text, use_kv_cache=use_kv_cache,
        device=device, generate_kwargs=generate_kwargs, num_repeats=num_repeats
    )

    mode_name = "KV-Cache" if use_kv_cache else "Naive"
    if quantization_config:
        mode_name = f"{quantization_config.upper()} Quantization"

    print(f"{mode_name} Throughput: {throughput:.2f} tokens/second")
    print(f"{mode_name} Peak Memory Usage: {peak_memory:.2f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference Benchmark")
    parser.add_argument("--max_new_tokens", type=int, default=50, help="Maximum number of new tokens to generate")
    parser.add_argument("--num_repeats", type=int, default=3, help="Number of repeat measurements for throughput")
    args = parser.parse_args()

    input_text = "In the year 3025, humanity has colonized the moons of Jupiter."

    print("Running Naive Inference:")
    inference_pipeline(mode="naive", input_text=input_text, max_new_tokens=args.max_new_tokens, num_repeats=args.num_repeats)

    print("\nRunning KV-Cache Inference:")
    inference_pipeline(mode="kv_cache", input_text=input_text, max_new_tokens=args.max_new_tokens, num_repeats=args.num_repeats)

    print("\nRunning Quantized Inference with FP16:")
    inference_pipeline(mode="kv_cache", input_text=input_text, max_new_tokens=args.max_new_tokens, quantization_config='fp16', num_repeats=args.num_repeats)

    print("\nRunning Quantized Inference with INT8:")
    inference_pipeline(mode="kv_cache", input_text=input_text, max_new_tokens=args.max_new_tokens, quantization_config='int8', num_repeats=args.num_repeats)
