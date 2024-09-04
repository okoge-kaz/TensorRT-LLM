import argparse
import json
import matplotlib.pyplot as plt
from collections import Counter

def parse_arguments():
    parser = argparse.ArgumentParser(description="Analyze quality key in multiple JSONL files.")
    parser.add_argument('--file-paths', type=str, nargs='+', help='Paths to the JSONL files.')
    return parser.parse_args()

def count_quality_values(file_paths):
    quality_counts = Counter()
    total_count = 0

    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                record = json.loads(line)
                quality = record.get('quality')
                if quality is not None:
                    quality_counts[quality] += 1
                    total_count += 1

    return quality_counts, total_count

def plot_quality_distribution(quality_counts, total_count):
    # Absolute counts
    qualities = list(quality_counts.keys())
    counts = list(quality_counts.values())

    plt.figure(figsize=(12, 6))

    # Plotting absolute counts
    plt.subplot(1, 2, 1)
    bars = plt.bar(qualities, counts, color='skyblue')
    plt.title('Absolute Count of Quality Values')
    plt.xlabel('Quality')
    plt.ylabel('Count')

    # Adding numbers on top of bars for absolute counts
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, int(yval), va='bottom')  # type: ignore

    # Percentage distribution
    percentages = [(count / total_count) * 100 for count in counts]

    plt.subplot(1, 2, 2)
    bars = plt.bar(qualities, percentages, color='lightgreen')
    plt.title('Percentage Distribution of Quality Values')
    plt.xlabel('Quality')
    plt.ylabel('Percentage')

    # Adding numbers on top of bars for percentages
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.1f}%', va='bottom')

    plt.tight_layout()
    plt.savefig("examples/gemma/visualize/magpie_ultra_quality_distribution.png")
    plt.show()

def main():
    args = parse_arguments()
    quality_counts, total_count = count_quality_values(args.file_paths)
    plot_quality_distribution(quality_counts, total_count)

if __name__ == '__main__':
    main()
