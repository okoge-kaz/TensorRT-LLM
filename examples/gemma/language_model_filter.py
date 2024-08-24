import json
import time
import os
import re
import torch
import argparse

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


JUDGE_PROMPT = "[指示]\n公平な判断者として行動し、以下に表示されるユーザーの質問に対するAIアシスタントの応答の品質を評価してください。あなたの評価は、応答の有用性、関連性、正確性、深さ、創造性、詳細度などの要素を考慮してそれぞれの点数と総合評価を示してください。AIアシスタントの返答の言語は、ユーザーが使用している言語と一致しているべきで、そうでない場合は減点されるべきです。評価は短い説明から始めてください。できるだけ客観的であること。説明を提供した後、**総合評価: 8/10** のようなフォーマットに厳密に従って1から10までのスケールで応答を評価する必要があります：\n\n[評価対象]\n"


LABEL_MAPPING = {
    "有用性": "usefulness",
    "関連性": "relevance",
    "正確性": "accuracy",
    "深さ": "depth",
    "創造性": "creativity",
    "詳細度": "detail",
    "総合評価": "overall"
}


def extract_scores(text: str) -> dict[str, float]:
    line_pattern = re.compile(r"(有用性|関連性|正確性|深さ|創造性|詳細度|総合評価).*?(\d+\.?\d*)/10")

    # 評価項目を抽出
    matches = line_pattern.findall(text)

    # スコアを辞書形式で保存
    score_dict = {}
    for match in matches:
        label, score = match
        score_dict[LABEL_MAPPING[label]] = float(score) if '.' in score else int(score)

    return score_dict


def main(args) -> None:
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

    # Load and process the JSONL file
    data = load_jsonl(args.jsonl_path)
    for item in data:
        start = time.perf_counter()
        conversations: list[dict[str, str]] = item[args.json_conversation_key]
        text = ""
        for conversation in conversations:
            text += conversation["role"] + ": " + conversation["content"] + "\n"
        judged_text = JUDGE_PROMPT + text
        input: str = tokenizer.apply_chat_template(  # type: ignore
            [{"role": "user", "content": judged_text}],
            tokenize=False,
            add_generation_prompt=True,
        )
        input += "[評価]\n"

        # tensorRT
        inputs = tokenizer.encode(
            input,
            return_tensors="pt",
            add_special_tokens=False
        ).squeeze(0)  # type: ignore
        batch_input_ids = [inputs]

        output_len = 4096 - inputs.shape[0]
        top_k = 1
        top_p = 0.0

        input_lengths = [x.size(0) for x in batch_input_ids]

        with torch.no_grad():
            if isinstance(model, torch.nn.Module):
                # Left padding for HF
                max_length = max(input_lengths)
                paddings = [
                    torch.ones(max_length - l, dtype=torch.int32) * pad_id
                    for l in input_lengths
                ]
                batch_input_ids = [
                    torch.cat([pad, x])
                    for x, pad in zip(batch_input_ids, paddings)
                ]
                batch_input_ids = torch.stack(batch_input_ids)
                batch_input_ids = batch_input_ids.cuda()
                with torch.no_grad():
                    # Use default temperature and top_k
                    outputs = model.generate(
                        batch_input_ids,  # type: ignore
                        max_new_tokens=output_len,
                        top_k=top_k
                    )
                    output_ids = outputs[0, input_lengths[0]:]

            elif isinstance(model, ModelRunnerCpp) or isinstance(model, ModelRunner):
                outputs = model.generate(
                    batch_input_ids,
                    max_new_tokens=output_len,
                    max_attention_window_size=args.max_attention_window_size,
                    end_id=end_id,
                    pad_id=pad_id,
                    top_k=top_k,
                    top_p=top_p,
                )
                torch.cuda.synchronize()
                if runtime_rank == 0:
                    output_ids = outputs[0, 0, input_lengths[0]:]

        if runtime_rank == 0:
            output_text = tokenizer.decode(output_ids, skip_special_tokens=True)

        if args.verbose and runtime_rank == 0:
            print(output_text, flush=True)
        scores = extract_scores(output_text)
        if args.verbose and runtime_rank == 0:
            print(scores, flush=True)
        item["scores"] = scores
        item["generated_text"] = output_text
        print(f"Processed in {time.perf_counter() - start:.2f}s", flush=True)

    # Save the filtered entries
    with open(args.output_path, 'w', encoding='utf-8') as file:
        for entry in data:
            json.dump(entry, file, ensure_ascii=False)
            file.write('\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='scoring dataset by language model'
    )
    parser.add_argument('--jsonl-path', help='Path to the input JSONL file')
    parser.add_argument(
        '--json-conversation-key', default='conversations', help='Key to access the conversations'
    )
    parser.add_argument('--output-path', help='Path to save the output JSONL file with Japanese entries')
    parser.add_argument('--verbose', action='store_true', help='Print verbose output')
    parser = add_common_args(parser)

    args = parser.parse_args()
    main(args)
