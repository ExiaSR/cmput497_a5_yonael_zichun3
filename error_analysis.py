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
from sklearn import metrics
from prettytable import PrettyTable
from mlxtend.plotting import plot_confusion_matrix
from tabulate import tabulate
import matplotlib.pyplot as plt


def get_output(file):
    csv_reader = csv.reader(file)
    next(csv_reader, None)
    return list(csv_reader)


@click.command()
@click.argument("output", type=click.File("r"))
def main(output):
    """
    Produce metrics and figures for error analysis.
    """
    zipped = list(zip(*get_output(output)))
    gold_labels = zipped[0]
    predicted_labels = zipped[1]
    classes = list(set(gold_labels + predicted_labels))

    confusion_matrix = ConfusionMatrix(gold_labels, predicted_labels, sort_by_count=True)
    sorted_labels = confusion_matrix._values
    cm = metrics.confusion_matrix(gold_labels, predicted_labels, labels=sorted_labels)

    fig, ax = plot_confusion_matrix(cm, class_names=sorted_labels)
    fig.subplots_adjust(bottom=0.24)
    plt.savefig("cm.png")

    # get confusion matrix and precision/recall
    global_precision, global_recall, global_f_score, _ = metrics.precision_recall_fscore_support(
        gold_labels, predicted_labels, labels=sorted_labels
    )
    global_table = zip(sorted_labels, global_precision, global_recall, global_f_score)
    print(tabulate(global_table, headers=["Label", "Precision", "Recall", "F Score"]))

    print("\n")

    # aggregated precision
    pooled_stat = metrics.precision_recall_fscore_support(
        gold_labels, predicted_labels, average="micro", labels=sorted_labels
    )
    print("Aggregated microaverage precision: {:.5f}".format(pooled_stat[0]))

    # macro-averaged precision
    macro_stat = metrics.precision_recall_fscore_support(
        gold_labels, predicted_labels, average="macro", labels=sorted_labels
    )
    print("Microaverage precision: {:.5f}".format(macro_stat[0]))


if __name__ == "__main__":
    main()
