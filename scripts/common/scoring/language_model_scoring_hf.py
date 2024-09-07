import json
import time
import os
import re
import torch
import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator


def load_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return [json.loads(line) for line in file]


JUDGE_PROMPT = "[指示]\n公平な判断者として行動し、以下に表示されるユーザーの質問に対するAIアシスタントの応答の品質を評価してください。あなたの評価は、応答の有用性、関連性、正確性、深さ、創造性、詳細度などの要素を考慮してそれぞれの点数と総合評価を示してください。AIアシスタントの返答の言語は、ユーザーが使用している言語と一致しているべきで、そうでない場合は減点されるべきです。評価は短い説明から始めてください。できるだけ客観的であること。説明を提供した後、**総合評価: 7/10** のようなフォーマットに厳密に従って1から10までのスケールで応答を評価する必要があります：\n\n[評価対象]\n"  # noqa:


LABEL_MAPPING = {
    "有用性": "usefulness",
    "関連性": "relevance",
    "正確性": "accuracy",
    "深さ": "depth",
    "創造性": "creativity",
    "詳細度": "detail",
    "総合評価": "overall",
}


def extract_scores(text: str) -> dict[str, float]:
    line_pattern = re.compile(r"(有用性|関連性|正確性|深さ|創造性|詳細度|総合評価).*?(\d+\.?\d*)/10")

    # 評価項目を抽出
    matches = line_pattern.findall(text)

    # スコアを辞書形式で保存
    score_dict = {}
    for match in matches:
        label, score = match
        score_dict[LABEL_MAPPING[label]] = float(score) if "." in score else int(score)

    return score_dict


def write_results(data, output_path, mode="w"):
    with open(output_path, mode, encoding="utf-8") as file:
        for entry in data:
            json.dump(entry, file, ensure_ascii=False)
            file.write("\n")


def main(args) -> None:

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
    )

    # Load and process the JSONL file
    data = load_jsonl(args.jsonl_path)

    # Determine the starting index
    start_index = 0
    if args.resume and os.path.exists(args.output_path):
        with open(args.output_path, "r", encoding="utf-8") as file:
            for line in file:
                last_processed = json.loads(line)
                start_index = last_processed.get("index", 0) + 1
        print(f"Resuming from index {start_index}")
    else:
        # Clear the output file if not resuming
        with open(args.output_path, "w", encoding="utf-8") as file:
            file.write("")

    processed_data = []
    for idx, item in enumerate(data[start_index:], start=start_index + 1):
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
        input_ids: torch.Tensor = tokenizer.encode(  # type: ignore
            input, return_tensors="pt", add_special_tokens=False
        )

        if input_ids.shape[0] > 3000:
            print(f"input length is too long: {input_ids.shape[0]}")
            continue

        output_len = 4096 - input_ids.shape[0]
        output_ids = model.generate(
            input_ids.to(device=model.device),
            temperature=0.2,
            top_p=0.7,
            max_new_tokens=output_len,
            do_sample=True,
        )

        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        if args.verbose:
            print(output_text, flush=True)
        scores = extract_scores(output_text)
        item["scores"] = scores
        item["generated_text"] = output_text
        item["index"] = idx - 1  # Adjust index to match the original data
        processed_data.append(item)
        print(f"Processed item {idx} in {time.perf_counter() - start:.2f}s", flush=True)

        if len(processed_data) == 10:
            write_results(processed_data, args.output_path, mode="a")
            processed_data = []

    # Write any remaining processed data
    if processed_data:
        write_results(processed_data, args.output_path, mode="a")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="scoring dataset by language model")
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--jsonl-path", help="Path to the input JSONL file")
    parser.add_argument("--json-conversation-key", default="conversations", help="Key to access the conversations")
    parser.add_argument("--output-path", help="Path to save the output JSONL file with Japanese entries")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")
    parser.add_argument("--resume", action="store_true", help="Resume from the last processed index")

    args = parser.parse_args()
    main(args)
