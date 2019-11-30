import csv
import os
import errno
import random
from collections import Counter

import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download("stopwords")
nltk.download("punkt")

flatten = lambda l: [item for sublist in l for item in sublist]


class Dataset:
    def __init__(self, raw_data, file_path, preprocesser=None, shuffle=False):
        self._path = file_path
        self._filename = os.path.splitext(os.path.basename(file_path))[0]
        self._predicted_labels = []  # assigned predicted labels

        if shuffle:
            random.shuffle(raw_data)

        zipped = list(zip(*raw_data))
        self._labels = zipped[0]  # original labels
        self._text = zipped[1]

        if preprocesser:
            self._documents = list(map(preprocesser, self._text))
        else:
            # dummpy tokenizer for testing purpose
            self._documents = list(map(lambda x: x.split(), self._text))

    def __repr__(self):
        return "<Dataset name={} N={} C={}>".format(self._filename, len(self._labels), len(self.classes()))

    def text(self):
        return self._text

    def labels(self):
        return self._labels

    def classes(self):
        return set(self._labels)

    def documents(self):
        return self._documents

    def labeled_documents(self):
        return zip(self._documents, self._labels)

    def predicted_labels(self):
        return self._predicted_labels

    def get_labeled_featuresets(self):
        """
        :return: A list of labeled word.
        :rtype: ``[(word, label), (word, label)]``
        """
        labeled_featuresets = []
        for tokens, label in zip(self._documents, self._labels):
            labeled_featuresets.extend([(token, label) for token in tokens])
        return labeled_featuresets

    def _set_predicted_labels(self, labels):
        self._predicted_labels = labels


def tokenizer(document: str):
    """
    Tokenized document and other neccesary preprocessing.

    :param document: One single sentence
    :return: Tokenized document.
    :rtype: ``["You", "re", "awesome"]``
    """
    stop_words = set(stopwords.words("english"))
    word_tokens = word_tokenize(document)
    tokens = [w for w in word_tokens if not w in stop_words]

    return tokens


def get_dataset(file_path, **kwargs) -> "Dataset":
    """
    Load datasets
    :param dir: Path to datasets directory.
    :return: A dataset
    :rtype: ``Dataset``
    """
    with open(file_path) as dataset_f:
        dataset_reader = csv.reader(dataset_f)
        next(dataset_reader, None)
        dataset = Dataset(list(dataset_reader), file_path, preprocesser=tokenizer, **kwargs)
        return dataset


def accuracy(predicted_labels, gold_labels):
    """
    Get accuracy of the classifier
    """
    correct = 0
    for pred, gold in zip(predicted_labels, gold_labels):
        if pred == gold:
            correct += 1

    return correct / len(predicted_labels)


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


# https://stackoverflow.com/a/23794010
def safe_open_w(path, mode="wt"):
    mkdir_p(os.path.dirname(path))
    return open(path, mode)


def save_output(dataset: Dataset, dir="output/"):
    """
    Save predicted output to disk
    :param dataset: The dataset object.
    :param dir: Path to output directory.
    """
    if len(dataset._predicted_labels) <= 0:
        raise Exception("No predicted assigned label.")

    with safe_open_w(os.path.join(dir, "output_{}.csv".format(dataset._filename)), "w") as output_f:
        output_writer = csv.writer(output_f)

        output_rows = list(zip(dataset.labels(), dataset.predicted_labels(), dataset.text()))
        output_rows.insert(0, ["category", "predictedCategory", "text"])
        output_writer.writerows(output_rows)


# Inspire by http://code.activestate.com/recipes/521906-k-fold-cross-validation-partition/
def k_fold(n, k):
    indices = np.arange(n)
    k_folds = [each.tolist() for each in np.array_split(indices, k)]
    for test_index in k_folds:
        train_info = ["{}-{}".format(fold[0], fold[-1]) for fold in k_folds if fold != test_index]
        train_index = [i for i in indices if i not in test_index]
        folds_info = "Validation fold: {}-{}, Training fold: {}".format(
            test_index[0], test_index[-1], ",".join(train_info)
        )
        yield train_index, test_index, folds_info
