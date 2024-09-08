import argparse
import json
import hashlib
import re


def hash_item(item):
    """項目のハッシュ値を計算する"""
    return hashlib.md5(json.dumps(item, sort_keys=True).encode()).hexdigest()


def clean_instruction(instruction):
    """指示文から末尾の \n<|im_start|> を削除する"""
    return re.sub(r"\n<\|im_start\|>$", "", instruction)


def is_valid_item(instruction, response):
    """項目が有効かどうかをチェックする"""
    if instruction == response:
        return False
    if len(instruction.strip()) <= 1 or len(response.strip()) <= 1:
        return False
    return True


def process_files(input_files: list[str], output_file: str):
    seen_hashes = set()
    total_items = 0
    valid_items = 0
    cleaned_items = 0

    with open(output_file, "w", encoding="utf-8") as out_f:
        for input_file in input_files:
            with open(input_file, "r", encoding="utf-8") as f:
                for line in f:
                    total_items += 1
                    item = json.loads(line)
                    processed_translated_instruction = item.get("processed_translated_instruction", "")
                    gemma_2_response = item.get("gemma_2_response", "")

                    # 指示文をクリーニング
                    cleaned_instruction = clean_instruction(processed_translated_instruction)
                    if cleaned_instruction != processed_translated_instruction:
                        cleaned_items += 1

                    if is_valid_item(cleaned_instruction, gemma_2_response):
                        processed_data = {
                            "input": {"role": "user", "content": cleaned_instruction},
                            "output": {"role": "assistant", "content": gemma_2_response},
                            "conversations": [
                                {"role": "user", "content": cleaned_instruction},
                                {"role": "assistant", "content": gemma_2_response},
                            ],
                            "text": f"user: {cleaned_instruction}\n\nassistant: {gemma_2_response}"
                        }

                        # ハッシュ値を計算し、重複チェック
                        item_hash = hash_item(processed_data)
                        if item_hash not in seen_hashes:
                            seen_hashes.add(item_hash)
                            json.dump(processed_data, out_f, ensure_ascii=False)
                            out_f.write("\n")
                            valid_items += 1

    return total_items, valid_items, cleaned_items


def main():
    parser = argparse.ArgumentParser(
        description="Process multiple JSON files, remove duplicates and invalid items, clean instructions, and format the output into a single JSONL file."
    )
    parser.add_argument("-i", "--input", nargs="+", required=True, help="Input JSONL file(s)")
    parser.add_argument("-o", "--output", required=True, help="Output JSONL file")

    args = parser.parse_args()

    total_items, valid_items, cleaned_items = process_files(args.input, args.output)
    print(f"Processed {len(args.input)} file(s)")
    print(f"Total items processed: {total_items}")
    print(f"Items with cleaned instructions: {cleaned_items}")
    print(f"Valid unique items after filtering and cleaning: {valid_items}")
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
