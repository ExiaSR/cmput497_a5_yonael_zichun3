import csv
import os


class Dataset:
    def __init__(self, raw_data, file_path, preprocesser=None):
        zipped = list(zip(*raw_data))
        self._path = file_path
        self._filename = os.path.splitext(os.path.basename(file_path))[0]
        self._labels = zipped[0]  # original labels
        self._predicted_labels = [] # assigned predicted labels
        if preprocesser:
            self._text = list(map(preprocesser, zipped[1]))
        else:
            # dummpy tokenizer for testing purpose
            self._text = list(map(lambda x: x.split(), zipped[1]))

    def text(self):
        return self._text

    def lables(self):
        return self._labels

    def predicted_labels(self):
        return self._predicted_labels

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


def save_output(dataset: Dataset, dir="output/"):
    """
    Save predicted output to disk
    :param dataset: The dataset object.
    :param dir: Path to output directory.
    """
    if len(dataset._predicted_labels) <= 0:
        raise Exception("No predicted assigned label.")

    with open(os.path.join(dir, "output_{}.csv"), "w") as output_f:
        output_writer = csv.writer(output_f)

        output_rows = list(zip(dataset.lables(), dataset.predicted_labels(), dataset.text()))
        output_rows.insert(0, ["category", "predictedCategory", "text"])
        output_writer.writerows(output_rows)
