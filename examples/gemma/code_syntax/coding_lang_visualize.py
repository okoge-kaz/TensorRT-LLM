import json
import argparse
from collections import Counter
import matplotlib.pyplot as plt  # type: ignore
from tqdm import tqdm


def parse_jsonl(file_path):
    """Parse the JSONL file and return a list of languages."""
    languages = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            data = json.loads(line)
            language = data.get('language')
            if language:
                languages.append(language)
    return languages


def visualize_language_ratios(languages):
    """Visualize the language ratios using a pie chart, grouping languages with less than 3% as 'Other'."""
    language_counts = Counter(languages)
    total_count = sum(language_counts.values())
    threshold = 0.03 * total_count

    # Separate languages into main languages and 'Other'
    main_languages = {lang: count for lang, count in language_counts.items() if count >= threshold}
    other_count = sum(count for count in language_counts.values() if count < threshold)

    if other_count > 0:
        main_languages['Other'] = other_count

    # Sort languages by count (descending) for better visual representation
    sorted_languages = sorted(main_languages.items(), key=lambda x: x[1], reverse=True)
    labels, sizes = zip(*sorted_languages)

    plt.figure(figsize=(12, 8))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)  # type: ignore
    plt.axis('equal')  # Equal aspect ratio ensures that the pie is drawn as a circle.
    plt.title('Language Ratios in JSONL File')

    plt.savefig("examples/gemma/code_syntax/language_ratios.png", bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Visualize language ratios from a JSONL file.')
    parser.add_argument('--file-path', type=str, help='Path to the JSONL file')
    args = parser.parse_args()

    languages = parse_jsonl(args.file_path)
    visualize_language_ratios(languages)

if __name__ == '__main__':
    main()
