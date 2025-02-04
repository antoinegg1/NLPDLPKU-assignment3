import time
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

from customized_gpt2 import CustomizedGPT2LMHeadModel


os.environ["CUDA_VISIBLE_DEVICES"] = "6"

@torch.no_grad()
def customized_greedy_decoding(batch):
    """
    Perform greedy decoding using KV cache for efficient generation.

    Args:
        batch (list of str): List of input texts to decode from.
        tokenizer: Pretrained tokenizer for tokenization and detokenization.
        custom_model: Custom model with KV cache support.
        max_new_length (int): Maximum number of tokens to generate.

    Returns:
        Tuple[torch.Tensor, float]: Generated token IDs and total decoding time.
    """
    device=custom_model.device
    tokenized_batch = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    # Extract initial input_ids and attention_mask
    input_ids = tokenized_batch['input_ids']
    attention_mask = tokenized_batch['attention_mask']
    res = input_ids.clone()
    past_key_values = None
    start_time = time.time()
    seq_len=input_ids.shape[1]
    # Initialize kv cache
    for t in range(seq_len-1):
        input_ids_t = input_ids[:, t:t+1]
        attention_mask_t = attention_mask[:, :t+1]
        outputs = custom_model(
            input_ids = input_ids_t,
            attention_mask=attention_mask_t,
            past_key_values=past_key_values,
            use_cache=True, 
        )
        past_key_values = outputs['past_key_values']
    
    
    for t in range(MAX_NEW_LENGTH):
        outputs = custom_model(
            input_ids=input_ids,
            attention_mask=attention_mask ,
            past_key_values=past_key_values,
            use_cache=True
        )

        # Update past_key_values with the latest outputs
        past_key_values = outputs["past_key_values"]
        
        output_tokens = torch.argmax(outputs['logits'][:, -1], dim=-1, keepdim=True)
        
        # Append the new tokens to the result
        res = torch.cat([res, output_tokens], dim=-1)
        
        # Prepare new input_ids and attention_mask for the next iteration
        input_ids = output_tokens
        # input_ids = output_tokens  # Only the newly generated token is needed
        attention_mask = torch.cat(
            [attention_mask, torch.ones_like(output_tokens, device=device)], dim=-1
        )

    return res, time.time() - start_time


@torch.no_grad()
def golden_greedy_decoding_wo_cache(batch):
    tokenized_batch = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to('cuda')
    res = tokenized_batch['input_ids']
    start_time = time.time()
    for timestep in range(MAX_NEW_LENGTH):
        
        tokenized_batch = original_model.prepare_inputs_for_generation(**tokenized_batch)
        outputs = original_model(**tokenized_batch)
        output_tokens = torch.argmax(outputs['logits'][:,-1], dim=-1, keepdim=True)
        tokenized_batch = {
            "input_ids": torch.cat([tokenized_batch['input_ids'], output_tokens], dim=-1),
            "attention_mask": torch.cat([tokenized_batch['attention_mask'], torch.ones_like(output_tokens)], dim=-1),
        }
        res = torch.cat([res, output_tokens], dim=-1)
    
    return res, time.time() - start_time


if __name__ == "__main__":
    MAX_NEW_LENGTH = 100
    bsz = 16
    times = [0, 0]

    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    original_model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2", attn_implementation="eager", device_map='cuda')
    custom_model = CustomizedGPT2LMHeadModel.from_pretrained("openai-community/gpt2", attn_implementation="eager", device_map="cuda")

    with open("data.txt") as f:
        prompt_dataset = [i.strip() for i in f.readlines()]
    
    for i in range(0, (len(prompt_dataset) + bsz - 1) // bsz):
        batch = prompt_dataset[i * bsz: (i + 1) * bsz]
        golden_wo_cache_res, golden_wo_cache_time = golden_greedy_decoding_wo_cache(batch)
        custom_res, custom_time = customized_greedy_decoding(batch)

        times[0] += golden_wo_cache_time
        times[1] += custom_time
        
        assert torch.equal(golden_wo_cache_res, custom_res.to(golden_wo_cache_res.device)), "Decoding results are not equal"

    print("Time taken for golden greedy decoding without KV cache: ", times[0])
    print("Time taken for customized greedy decoding: ", times[1])
