import argparse
import json
import matplotlib.pyplot as plt
from collections import Counter

def parse_arguments():
    parser = argparse.ArgumentParser(description="Analyze quality and scores in multiple JSONL files.")
    parser.add_argument('--file-paths', type=str, nargs='+', help='Paths to the JSONL files.')
    return parser.parse_args()

def count_values(file_paths):
    score_counts = Counter()
    total_score_count = 0

    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                record = json.loads(line)

                # Count 'scores.overall' values
                scores = record.get('scores', {})
                overall_score = scores.get('overall')
                if overall_score is not None:
                    score_counts[overall_score] += 1
                    total_score_count += 1

    return score_counts, total_score_count

def plot_distribution(score_counts, total_score_count):
    plt.figure(figsize=(12, 6))

    # Plotting overall scores
    scores = list(score_counts.keys())
    score_counts_values = list(score_counts.values())

    plt.subplot(2, 2, 3)
    bars = plt.bar(scores, score_counts_values, color='coral')
    plt.title('Absolute Count of Overall Scores')
    plt.xlabel('Overall Score')
    plt.ylabel('Count')

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, int(yval), va='bottom')  # type: ignore

    score_percentages = [(count / total_score_count) * 100 for count in score_counts_values]

    plt.subplot(2, 2, 4)
    bars = plt.bar(scores, score_percentages, color='lightcoral')
    plt.title('Percentage Distribution of Overall Scores')
    plt.xlabel('Overall Score')
    plt.ylabel('Percentage')

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.1f}%', va='bottom')

    plt.tight_layout()
    plt.savefig("examples/gemma/visualize/nemotron_score_distribution_conversations.png")
    plt.show()

def main():
    args = parse_arguments()
    score_counts, total_score_count = count_values(args.file_paths)
    plot_distribution(score_counts, total_score_count)

if __name__ == '__main__':
    main()
