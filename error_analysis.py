"""
Usage: error_analysis.py [OPTIONS] OUTPUT

  Produce metrics and figures for error analysis.

Options:
  --help  Show this message and exit.
"""
import csv
from collections import Counter

import click

from nltk.metrics import ConfusionMatrix
from prettytable import PrettyTable

def get_output(file):
    csv_reader = csv.reader(file)
    next(csv_reader, None)
    return list(csv_reader)

# taken from https://stackoverflow.com/a/23715286/4557739
def precesion_and_recall(labels, cm):
    # precision and recall
    true_positives = Counter()
    false_negatives = Counter()
    false_positives = Counter()

    for i in labels:
        for j in labels:
            if i == j:
                true_positives[i] += cm[i, j]
            else:
                false_negatives[i] += cm[i, j]
                false_positives[j] += cm[i, j]

    results = []
    table = PrettyTable()
    table.field_names = ["label", "precision", "recall", "f-score"]
    for each in sorted(labels):
        if true_positives[each] == 0:
            fscore = 0
            results.append({"label": each, "f_score": fscore})
            table.add_row([each, None, None, fscore])
        else:
            precision = true_positives[each] / float(true_positives[each] + false_positives[each])
            recall = true_positives[each] / float(true_positives[each] + false_negatives[each])
            fscore = 2 * (precision * recall) / float(precision + recall)
            results.append(
                {"label": each, "precision": precision, "recall": recall, "f_score": fscore}
            )
            table.add_row([each, precision, recall, fscore])
    return results, table


@click.command()
@click.argument("output", type=click.File("r"))
def main(output):
    """
    Produce metrics and figures for error analysis.
    """
    zipped = list(zip(*get_output(output)))
    gold_labels = zipped[0]
    predicted_labels = zipped[1]

    confusion_matrix = ConfusionMatrix(gold_labels, predicted_labels)

    print(confusion_matrix.pretty_format(sort_by_count=True, show_percents=True, truncate=9))

    stats, table = precesion_and_recall(set(gold_labels + predicted_labels), confusion_matrix)
    print(table)

if __name__ == "__main__":
    main()
