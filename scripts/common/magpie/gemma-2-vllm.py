import json
import time
import random
import re
import argparse

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def load_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return [json.loads(line) for line in file]


JUDGE_PROMPT = "[指示]\n公平な判断者として行動し、以下に表示されるユーザーの質問に対するAIアシスタントの応答の品質を評価してください。あなたの評価は、応答の有用性、関連性、正確性、深さ、創造性、詳細度などの要素を考慮してそれぞれの点数と総合評価を示してください。AIアシスタントの返答の言語は、ユーザーが使用している言語と一致しているべきで、そうでない場合は減点されるべきです。評価は短い説明から始めてください。できるだけ客観的であること。説明を提供した後、**総合評価: 7/10** のようなフォーマットに厳密に従って1から10までのスケールで応答を評価する必要があります：\n\n[評価対象]\n"  # noqa:

INSTRUCTION_PROMPT = f"""
あなたは優秀な日本語のアシスタントです。{{category}}に関する話題を1つ考えて、それに関する質問を作成してください。
質問は、「Xとは何ですか？説明してください。」でも良いですし、「XとYはAという点で異なっていますが、YとZではどうですか？」、「AとBの関係性についてCであると理解していますが、これは正しいでしょうか？」のようなスタイルでも構いません。質問として有意義であり、適切な日本語表現で書かれた質問を1つ作成してください。
"""

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
    matches = line_pattern.findall(text)
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

def main(args):
    # Initialize the LLM
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel,
        dtype="bfloat16",
        gpu_memory_utilization=0.9,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
    )

    random.seed(int(time.time()))
    torch.manual_seed(random.randint(1, 10000))

    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=0.7,
        max_tokens=1024,
    )

    data = []
    for _ in range(args.num_samples):
        start = time.perf_counter()
        text = INSTRUCTION_PROMPT.format(category=args.category)

        instruction = llm.generate([text], sampling_params)
        instruction_text = instruction[0].outputs[0].text
        print(instruction_text, flush=True)

        assistant_output = llm.generate(
            [instruction_text], sampling_params
        )
        assistant_text = assistant_output[0].outputs[0].text

        # judge
        judge_text = JUDGE_PROMPT + "user:" + instruction_text + "\n\n" + "assistant:" + assistant_text
        judge_input: str = tokenizer.apply_chat_template(  # type: ignore
            conversation=[{"role": "user", "content": judge_text}],
            tokenize=False,
            add_generation_prompt=True,
        )
        judge_input += "[評価]\n"  # type: ignore
        judged_output = llm.generate(
            [judge_input], sampling_params
        )
        judged_text = judged_output[0].outputs[0].text

        scores = extract_scores(judged_text)

        item = {
            "scores": scores,
            "scoring_model": args.model_path,
            "input": {"role": "user", "content": instruction_text},
            "output": {"role": "assistant", "content": assistant_text},
            "judge": judge_text,
            "text": "user: " + instruction_text + "\n\n" + "assistant: " + assistant_text,
        }
        data.append(item)
        print(f"Processed item in {time.perf_counter() - start:.2f}s", flush=True)

        if len(data) == 10:
            write_results(data, args.output_path, mode="a")
            processed_data = []

    # Write any remaining processed data
    if processed_data:
        write_results(processed_data, args.output_path, mode="a")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="scoring dataset by language model")
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--category", type=str)
    parser.add_argument("--difficulty", type=str, choices=["easy", "medium", "difficult"])
    parser.add_argument("--style", type=str, choices=["qa", "article", "wiki"])
    parser.add_argument("--output-path", help="Path to save the output JSONL file with Japanese entries")
    parser.add_argument("--tensor-parallel", type=int, default=1, help="Tensor parallel size")

    args = parser.parse_args()
    main(args)
