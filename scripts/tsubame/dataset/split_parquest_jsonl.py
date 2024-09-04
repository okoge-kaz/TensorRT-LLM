import argparse
import json
import os
from datasets import load_dataset

def split_parquet_to_jsonl(input_file: str, output_prefix: str, samples_per_file: int):
    base_name = os.path.splitext(os.path.basename(input_file))[0]

    # Load the Parquet file using Hugging Face datasets
    dataset = load_dataset('parquet', data_files=input_file, split='train')

    chunk = []
    file_number = 1
    total_rows = 0

    for idx, example in enumerate(dataset):
        chunk.append(example)
        total_rows += 1

        if len(chunk) == samples_per_file:
            write_chunk(chunk, output_prefix, base_name, file_number)
            chunk = []
            file_number += 1

        if total_rows % 100000 == 0:
            print(f"Processed {total_rows} rows...")

    if chunk:
        write_chunk(chunk, output_prefix, base_name, file_number)

    print(f"Total rows processed: {total_rows}")

def write_chunk(chunk, output_prefix: str, base_name: str, file_number: int):
    output_file = f"{output_prefix}_{base_name}_{file_number}.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in chunk:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')
    print(f"Written {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Split Parquet files into JSONL files by sample count using Hugging Face datasets.')
    parser.add_argument('--input_files', nargs='+', help='Input Parquet file(s)')
    parser.add_argument('-o', '--output_prefix', default='output', help='Output files prefix')
    parser.add_argument('-n', '--samples_per_file', type=int, default=10000, help='Number of samples per output file')

    args = parser.parse_args()

    for input_file in args.input_files:
        print(f"Processing file: {input_file}")
        split_parquet_to_jsonl(input_file, args.output_prefix, args.samples_per_file)
        print(f"Finished processing: {input_file}")

if __name__ == "__main__":
    main()
