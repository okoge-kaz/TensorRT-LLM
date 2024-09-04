import argparse
import json

def convert_format(input_data):
    conversations = []

    # 入力データからユーザーメッセージとアシスタントの応答を取得
    for item in input_data['input']:
        conversations.append({
            "role": item['role'],
            "content": item['text']
        })

    # 最後のアシスタントの応答を追加
    conversations.append({
        "role": "assistant",
        "content": input_data['output']
    })

    # 新しい形式のデータを作成
    new_format = {
        "conversations": conversations
    }

    return new_format

def process_jsonl(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            # 各行をJSONオブジェクトとしてパース
            input_data = json.loads(line.strip())

            # フォーマットを変換
            output_data = convert_format(input_data)

            # 変換されたデータをJSONL形式で書き込む
            json.dump(output_data, outfile, ensure_ascii=False)
            outfile.write('\n')  # 各JSONオブジェクトの後に改行を追加

def main():
    parser = argparse.ArgumentParser(description="Convert JSONL conversation format")
    parser.add_argument('-i', '--input', required=True, help="Input JSONL file path")
    parser.add_argument('-o', '--output', required=True, help="Output JSONL file path")

    args = parser.parse_args()

    # JSONL ファイルを処理
    process_jsonl(args.input, args.output)

    print(f"Conversion complete. Output written to {args.output}")

if __name__ == "__main__":
    main()
