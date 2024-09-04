import json
import time
import os
import re
import torch
import argparse
from tqdm import tqdm

import sys
sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)
            )
        )
    )
)

from utils import add_common_args, load_tokenizer, read_model_name

import tensorrt_llm
from tensorrt_llm.runtime import PYTHON_BINDINGS, ModelRunner

if PYTHON_BINDINGS:
    from tensorrt_llm.runtime import ModelRunnerCpp

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DTYPE_STR_MAPPING = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

def load_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [json.loads(line) for line in file]


JUDGE_PROMPT = "[Instruction]\n：あなたは、プロの翻訳家です。以下の英語の文章を日本語に翻訳してください。直訳はせず、日本語として適切な表現を利用して翻訳してください。翻訳した訳文だけを表示するべきであり、余計な表現は出力しないでください。では、次の英語の文章を日本語に翻訳してください。\n[English text]:"


def write_results(data, output_path, mode='w'):
    with open(output_path, mode, encoding='utf-8') as file:
        for entry in data:
            json.dump(entry, file, ensure_ascii=False)
            file.write('\n')

def process_batch(model, tokenizer, batch, args):
    batch_input_ids = []
    input_lengths = []

    for item in batch:
        judged_text: str = JUDGE_PROMPT + item["question"]
        input_text: str = tokenizer.apply_chat_template(
            [{"role": "user", "content": judged_text}],
            tokenize=False,
            add_generation_prompt=True,
        ) + "[Japanese text]:"

        inputs = tokenizer.encode(
            input_text,
            return_tensors="pt",
            add_special_tokens=False
        ).squeeze(0)

        if inputs.shape[0] <= 3000:
            batch_input_ids.append(inputs)
            input_lengths.append(inputs.shape[0])

    if not batch_input_ids:
        return []
    runtime_rank = tensorrt_llm.mpi_rank()

    max_length = max(input_lengths)
    paddings = [torch.ones(max_length - l, dtype=torch.int32) * args.pad_id for l in input_lengths]
    batch_input_ids = [torch.cat([x, pad]) for x, pad in zip(batch_input_ids, paddings)]
    batch_input_ids = torch.stack(batch_input_ids).cuda()

    output_len = 4096 - max_length

    with torch.no_grad():
        outputs = model.generate(
            batch_input_ids,
            max_new_tokens=output_len,
            max_attention_window_size=args.max_attention_window_size,
            end_id=args.end_id,
            pad_id=args.pad_id,
            top_k=1,
            top_p=0.0,
        )
        torch.cuda.synchronize()
        if runtime_rank == 0:
            output_ids = outputs[0, 0, input_lengths[0]:]

    results = []
    if runtime_rank == 0:
            output_text = tokenizer.decode(output_ids, skip_special_tokens=True)

    if args.verbose and runtime_rank == 0:
            print(output_text, flush=True)
    results.append({
        "id": item["id"],
        "system_prompt": item["system_prompt"],
        "translated_text": output_text,
        "original_question": item["question"],
    })

    return results

def main(args):
    if args.tokenizer_dir is None:
        args.tokenizer_dir = args.hf_model_dir

    # Load the language model
    runtime_rank = tensorrt_llm.mpi_rank()
    model_name, model_version = read_model_name(args.engine_dir)
    tokenizer, pad_id, end_id = load_tokenizer(
        tokenizer_dir=args.tokenizer_dir,
        vocab_file=args.vocab_file,
        model_name=model_name,
        model_version=model_version,
    )
    end_id = 1
    print(f"pad_id: {pad_id}, end_id: {end_id}")
    assert pad_id == 0, "pad_id must be 0"
    assert end_id == 1, "end_id must be 1"
    args.pad_id = pad_id
    args.end_id = end_id

    runner_cls = ModelRunner if not PYTHON_BINDINGS else ModelRunnerCpp
    runner_kwargs = {}
    if PYTHON_BINDINGS:
        runner_kwargs.update(max_beam_width=1)
    runner_kwargs.update(
        max_tokens_in_paged_kv_cache=args.max_tokens_in_paged_kv_cache,
        kv_cache_enable_block_reuse=args.kv_cache_enable_block_reuse,
        kv_cache_free_gpu_memory_fraction=args.
        kv_cache_free_gpu_memory_fraction,
        enable_chunked_context=args.enable_chunked_context,
        multi_block_mode=args.multi_block_mode)
    model = runner_cls.from_dir(
        args.engine_dir,
        rank=runtime_rank,
        **runner_kwargs
    )

    data = load_jsonl(args.jsonl_path)
    start_index = 0
    if args.resume and os.path.exists(args.output_path):
        with open(args.output_path, 'r', encoding='utf-8') as file:
            start_index = sum(1 for _ in file)
        print(f"Resuming from index {start_index}")
    else:
        with open(args.output_path, 'w', encoding='utf-8') as file:
            file.write('')

    batch_size = args.batch_size
    processed_data = []

    for i in tqdm(range(start_index, len(data), batch_size)):
        batch = data[i:i+batch_size]
        results = process_batch(model, tokenizer, batch, args)

        for j, result in enumerate(results):
            item = batch[j]
            item.update(result)
            item["index"] = i + j
            processed_data.append(item)

        if len(processed_data) >= 10:
            write_results(processed_data, args.output_path, mode='a')
            processed_data = []

    if processed_data:
        write_results(processed_data, args.output_path, mode='a')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Batch scoring dataset by language model')
    parser.add_argument('--jsonl-path', help='Path to the input JSONL file')
    parser.add_argument('--output-path', help='Path to save the output JSONL file with Japanese entries')
    parser.add_argument('--verbose', action='store_true', help='Print verbose output')
    parser.add_argument('--resume', action='store_true', help='Resume from the last processed index')
    parser = add_common_args(parser)
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for inference')
    args = parser.parse_args()
    main(args)
