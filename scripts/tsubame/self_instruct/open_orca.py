import argparse
import os
import json
from tqdm import tqdm
from typing import Any

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

import torch
from utils import (
    DEFAULT_HF_MODEL_DIRS,
    add_common_args,
    load_tokenizer,
    read_model_name,
    supports_inflight_batching
)

import tensorrt_llm
import tensorrt_llm.profiler as profiler
from tensorrt_llm._utils import mpi_broadcast
from tensorrt_llm.logger import logger
from tensorrt_llm.runtime import PYTHON_BINDINGS, ModelRunner

if PYTHON_BINDINGS:
    from tensorrt_llm.runtime import ModelRunnerCpp

def load_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [json.loads(line) for line in file]


def write_results(data, output_path, mode='w'):
    with open(output_path, mode, encoding='utf-8') as file:
        for entry in data:
            json.dump(entry, file, ensure_ascii=False)
            file.write('\n')


SELF_INSTRUCT = f"""
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. Please answer in Japanese. Don't answer in English.

The Below is an instruction that describes a task. Write a response that appropriately completes the request.
# User Input:
{{input}}
"""


def main(args: argparse.Namespace) -> None:
    runtime_rank = tensorrt_llm.mpi_rank()
    logger.set_level(args.log_level)

    model_name, model_version = read_model_name(args.engine_dir)
    if args.hf_model_dir is None:
        logger.warning(
            "hf_model_dir is not specified. Try to infer from model_name, but this may be incorrect."
        )
        if model_name in DEFAULT_HF_MODEL_DIRS:
            args.hf_model_dir = DEFAULT_HF_MODEL_DIRS[model_name]
        else:
            args.hf_model_dir = None
    if args.tokenizer_dir is None:
        args.tokenizer_dir = args.hf_model_dir

    profiler.start('load tokenizer')
    tokenizer, pad_id, end_id = load_tokenizer(
        tokenizer_dir=args.tokenizer_dir,
        vocab_file=args.vocab_file,
        model_name=model_name,
        model_version=model_version,
        tokenizer_type=args.tokenizer_type,
    )
    profiler.stop('load tokenizer')
    logger.info(
        f'Load tokenizer takes: {profiler.elapsed_time_in_sec("load tokenizer")} sec'
    )
    dataset = load_jsonl(args.jsonl_path)
    max_batch_size = args.batch_size

    # runtime parameters
    top_k = args.top_k
    top_p = args.top_p
    output_len = args.output_len
    test_token_num = args.max_input_length
    max_attention_window_size = args.max_attention_window_size
    sink_token_length = args.sink_token_length

    if args.end_id:
        end_id = args.end_id
    end_id = 1
    print(f"pad_id: {pad_id}, end_id: {end_id}")
    assert pad_id == 0, "pad_id must be 0"
    assert end_id == 1, "end_id must be 1"

    stop_words_list = None
    if args.stop_words:
        stop_words_list = tensorrt_llm.runtime.decode_words_list(
            args.stop_words, tokenizer
        )

    bad_words_list = None
    if args.bad_words:
        bad_words_list = tensorrt_llm.runtime.decode_words_list(
            args.bad_words, tokenizer
        )

    random_seed = args.random_seed
    temperature = args.temperature
    num_beams = args.num_beams
    length_penalty = args.length_penalty
    early_stopping = args.early_stopping
    repetition_penalty = args.repetition_penalty
    presence_penalty = args.presence_penalty
    frequency_penalty = args.frequency_penalty
    torch.manual_seed(random_seed)

    def _prepare_inputs(
        batch_input_texts: list,
        add_special_tokens: bool = True,
        min_input_length: int = 0
    ) -> list[torch.Tensor]:
        batch_size: int = len(batch_input_texts)
        batch_input_ids = []
        for i in range(batch_size):
            curr_text = batch_input_texts[i][args.json_input_key]
            text = SELF_INSTRUCT.format(input=curr_text)
            text: str = tokenizer.apply_chat_template(  # type: ignore
                [{"role": "user", "content": text}],
                tokenize=False,
                add_generation_prompt=True,
            ) + "# Assistant Response\n"

            input_ids = tokenizer.encode(
                text,
                return_tensors='pt',
                add_special_tokens=add_special_tokens,
                truncation=True,
                max_length=test_token_num).squeeze(0)  # type: ignore

            if input_ids.numel() > min_input_length:
                batch_input_ids.append(input_ids)
        return batch_input_ids

    def generate_trl_llm(
        data_point: list[dict[str, Any]],
        add_special_tokens=True,
        min_input_length=0
    ):
        batch_size = len(data_point)
        batch_input_ids = _prepare_inputs(
            data_point,
            add_special_tokens=add_special_tokens,
            min_input_length=min_input_length
        )
        batch_size = len(batch_input_ids)
        if batch_size == 0:
            return []
        input_lengths = [x.size(0) for x in batch_input_ids]

        with torch.no_grad():
            outputs = runner.generate(
                batch_input_ids,
                max_new_tokens=output_len,
                max_attention_window_size=max_attention_window_size,
                sink_token_length=sink_token_length,
                end_id=end_id,
                pad_id=pad_id,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                stop_words_list=stop_words_list,
                bad_words_list=bad_words_list,
                num_beams=num_beams,
                length_penalty=length_penalty,
                early_stopping=early_stopping,
                repetition_penalty=repetition_penalty,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                output_sequence_lengths=True,
                return_dict=True,
            )
            torch.cuda.synchronize()

        # Extract a list of tensors of shape beam_width x output_ids.
        if runtime_rank == 0:
            output_ids: torch.Tensor = outputs['output_ids']  # type: ignore
            output_beams_list = [
                tokenizer.batch_decode(
                    output_ids[batch_idx, :,input_lengths[batch_idx]:],
                    skip_special_tokens=True
                )
                for batch_idx in range(batch_size)
            ]

            return output_beams_list
        return []

    if not supports_inflight_batching(args.engine_dir):
        logger.warning(
            "The given engine does not support in-flight batching, fallback to python session"
        )
        args.use_py_session = True

    if not PYTHON_BINDINGS and not args.use_py_session:
        logger.warning(
            "Python bindings of C++ session is unavailable, fallback to Python session."
        )
        args.use_py_session = True
    if args.return_all_generated_tokens:
        raise ValueError(
            "Returning all the generated tokens at each step is not supported in summarize.py"
        )
    runner_cls = ModelRunner if args.use_py_session else ModelRunnerCpp
    runner_kwargs = dict(
        engine_dir=args.engine_dir,
        rank=runtime_rank,
        debug_mode=args.debug_mode,
        gpu_weights_percent=args.gpu_weights_percent
    )

    if not args.use_py_session:
        runner_kwargs.update(
            max_batch_size=max_batch_size,
            max_input_len=test_token_num,
            max_output_len=output_len,
            max_beam_width=num_beams,
            max_attention_window_size=max_attention_window_size,
            sink_token_length=sink_token_length,
            max_tokens_in_paged_kv_cache=args.max_tokens_in_paged_kv_cache,
            kv_cache_enable_block_reuse=args.kv_cache_enable_block_reuse,
            kv_cache_free_gpu_memory_fraction=args.
            kv_cache_free_gpu_memory_fraction,
            enable_chunked_context=args.enable_chunked_context,
            multi_block_mode=args.multi_block_mode
        )
    runner_kwargs.update(
        enable_context_fmha_fp32_acc=args.enable_context_fmha_fp32_acc
    )
    runner = runner_cls.from_dir(**runner_kwargs)

    SAVE_INTERVAL = 10

    data_point_idx = 0
    total_batches = (len(dataset) + max_batch_size - 1) // max_batch_size
    un_saved_data = []
    with tqdm(total=total_batches, desc="Processing Data Points") as pbar:
        while data_point_idx < len(dataset):
            if runtime_rank == 0:
                logger.debug(
                    f"run data_point {data_point_idx} ~ {data_point_idx + max_batch_size}"
                )

            data_point = dataset[
                data_point_idx:(data_point_idx + max_batch_size)
            ]

            profiler.start('tensorrt_llm')
            output_tensorrt_llm = generate_trl_llm(
                data_point,
                add_special_tokens=args.add_special_tokens,
                min_input_length=args.min_input_length
            )
            profiler.stop('tensorrt_llm')

            if runtime_rank == 0:
                logger.info(
                    f"TensorRT-LLM Generated : {output_tensorrt_llm}"
                )
                for i, output in enumerate(output_tensorrt_llm):
                    data_point[i]["generated_response"] = output
                un_saved_data.extend(data_point)

                if data_point_idx % SAVE_INTERVAL == 0:
                    write_results(un_saved_data, args.output_path, mode='a')
                    un_saved_data = []

            empty_batch = (runtime_rank == 0 and len(output_tensorrt_llm) == 0)
            empty_batch = mpi_broadcast(empty_batch, 0)
            if empty_batch:
                # No valid samples in the current batch, skip this iteration
                data_point_idx += max_batch_size
                pbar.update(1)  # 進捗バーを更新
                continue

            data_point_idx += max_batch_size
            pbar.update(1)  # 進捗バーを更新

    del runner


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--jsonl-path', help='Path to the input JSONL file')
    parser.add_argument('--json-input-key', type=str, default='input')
    parser.add_argument('--output-path', type=str, default='output.jsonl')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--output_len', type=int, default=1096)
    parser.add_argument('--max-input-length', type=int, default=3000)
    parser.add_argument(
        '--min_input_length',
        type=int,
        default=0,
        help='skip the sentences which are shorter than min_input_length.')
    parser = add_common_args(parser)
    args = parser.parse_args()

    main(args)
