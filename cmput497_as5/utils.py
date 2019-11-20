import csv
import os
import errno

from collections import Counter

flatten = lambda l: [item for sublist in l for item in sublist]


class Dataset:
    def __init__(self, raw_data, file_path, preprocesser=None):
        zipped = list(zip(*raw_data))
        self._path = file_path
        self._filename = os.path.splitext(os.path.basename(file_path))[0]
        self._labels = zipped[0]  # original labels
        self._predicted_labels = []  # assigned predicted labels
        self._text = zipped[1]
        if preprocesser:
            self._documents = list(map(preprocesser, zipped[1]))
        else:
            # dummpy tokenizer for testing purpose
            self._documents = list(map(lambda x: x.split(), zipped[1]))
        self._document_by_class_counter = Counter([c for c in self._labels])

    def text(self):
        return self._text

    def labels(self):
        return self._labels

    def classes(self):
        return set(self._labels)

    def tokens(self):
        return flatten(self._documents)

    def vocabs(self):
        return set(flatten(self._documents))

    def documents(self):
        return self._documents

    def predicted_labels(self):
        return self._predicted_labels

    def num_documents(self):
        return len(self._text)

    def num_documents_by_class(self, class_name):
        return self._document_by_class_counter[class_name]

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

    :param document: One single document
    :return: Tokenized document.
    :rtype: ``["You", "re", "awesome"]``
    """
    # TODO
    pass


def get_dataset(file_path) -> Dataset:
    """
    Load datasets
    :param dir: Path to datasets directory.
    :return: A dataset
    :rtype: ``Dataset``
    """
    with open(file_path) as dataset_f:
        dataset_reader = csv.reader(dataset_f)
        next(dataset_reader, None)
        dataset = Dataset(list(dataset_reader), file_path)
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
