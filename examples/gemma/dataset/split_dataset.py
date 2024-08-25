import json
import os
import argparse
from typing import List

def split_jsonl_file(input_file: str, output_prefix: str, samples_per_file: int):
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    with open(input_file, 'r', encoding='utf-8') as f:
        chunk = []
        file_number = 1

        for i, line in enumerate(f, 1):
            chunk.append(json.loads(line))

            if i % samples_per_file == 0:
                write_chunk(chunk, output_prefix, base_name, file_number)
                chunk = []
                file_number += 1

        if chunk:
            write_chunk(chunk, output_prefix, base_name, file_number)

def write_chunk(chunk: List[dict], output_prefix: str, base_name: str, file_number: int):
    output_file = f"{output_prefix}_{base_name}_{file_number}.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in chunk:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')

def main():
    parser = argparse.ArgumentParser(description='Split JSONL files by sample count.')
    parser.add_argument('--input_files', nargs='+', help='Input JSONL file(s)')
    parser.add_argument('-o', '--output_prefix', default='output', help='Output files prefix')
    parser.add_argument('-n', '--samples_per_file', type=int, default=10000, help='Number of samples per output file')

    args = parser.parse_args()

    for input_file in args.input_files:
        print(f"Processing file: {input_file}")
        split_jsonl_file(input_file, args.output_prefix, args.samples_per_file)
        print(f"Finished processing: {input_file}")

if __name__ == "__main__":
    main()
