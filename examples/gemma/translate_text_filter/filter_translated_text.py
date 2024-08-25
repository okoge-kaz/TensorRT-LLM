import json
import re
import argparse
import tqdm

def process_text(text):
    # Remove specified strings
    text = text.replace('\n\n<|im_end|><end_of_turn>', '')
    text = text.replace('\n<|im_end|><end_of_turn>', '')
    text = text.replace('\n\n<end_of_turn>', '')
    text = text.replace('\n<end_of_turn>', '')
    text = text.replace('\n\n<|im_end|>', '')
    text = text.replace('\n<|im_end|>', '')
    text = text.replace('<|im_end|><end_of_turn>', '')
    text = text.replace('<end_of_turn>', '')
    # Remove everything after '\n\nA:<|im_end|>'
    if '\n\nA:<|im_end|>' in text:
        text = text.split('\n\nA:<|im_end|>')[0]
    if '\n\nこの質問に対する答えは：' in text:
        text = text.split('\n\nこの質問に対する答えは：')[0]
    if '\n\n[Answer]:' in text:
        text = text.split('\n\n[Answer]:')[0]
    if '\nAnswer:' in text:
        text = text.split('\nAnswer:')[0]
    if '\n\n答え: ' in text:
        text = text.split('\n\n答え: ')[0]
    if '\n\n質問と回答：' in text:
        text = text.split('\n\n質問と回答：')[0]
    if '\n\n回答は：' in text:
        text = text.split('\n\n回答は：')[0]
    if '\n\n回答：' in text:
        text = text.split('\n\n回答：')[0]
    if '\n\n回答: ' in text:
        text = text.split('\n\n回答: ')[0]
    if '\n回答：<|im_end|>' in text:
        text = text.split('\n回答：<|im_end|>')[0]
    if "\n\n質問:" in text:
        text = text.split("\n\n質問:")[0]
    if '\nA: <|im_end|> ' in text:
        text = text.split('\nA: <|im_end|> ')[0]
    if '\n\n私は答えが<|im_end|> ' in text:
        text = text.split('\n\n私は答えが<|im_end|> ')[0]
    if '\n\n説明:\n\n' in text:
        text = text.split('\n\n説明:\n\n')[0]
    if '\n出力：' in text:
        text = text.split('\n出力：')[0]
    if '\n\n答えは：' in text:
        text = text.split('\n\n答えは：')[0]
    if '\n\n生徒：<|im_end|>' in text:
        text = text.split('\n\n生徒：<|im_end|>')[0]
    if '：<|im_end|>' in text:
        text = text.split('：<|im_end|>')[0]
    if '\n\n[English text]:' in text:
        text = text.split('\n\n[English text]:')[0]

    text = text.replace('...\n\n<end_of_turn>', '')
    text = text.replace('<|im_end|>', '')
    if len(text) >= 4 and text[-4:] == '\n\nA:':
        text = text[:-4]
    if len(text) >= 7 and text[-7] == '\nA: はい':
        text = text[:-7]
    if len(text) >= 5 and text[-5] == '\n回答:':
        text = text[:-5]
    text = text.replace('\n\n回答:\nA:', '')

    return text.strip()

def main():
    parser = argparse.ArgumentParser(description='Process JSONL file to modify translated_text field.')
    parser.add_argument('--input', help='Input JSONL file path')
    parser.add_argument('--output', help='Output JSONL file path')
    args = parser.parse_args()

    with open(args.input, 'r', encoding='utf-8') as infile, open(args.output, 'w', encoding='utf-8') as outfile:
        for line in tqdm.tqdm(infile):
            data = json.loads(line)
            if 'translated_text' in data:
                data['processed_translated_instruction'] = process_text(data['translated_text'])
            json.dump(data, outfile, ensure_ascii=False)
            outfile.write('\n')

if __name__ == "__main__":
    main()
