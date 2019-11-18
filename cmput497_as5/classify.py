from collections import defaultdict

from nltk.probability import FreqDist


class NaiveBayesClassifier:
    def __init__(self, label_probdist, word_probdist):
        self._label_probdist = label_probdist
        self._word_probdist = word_probdist

    @classmethod
    def train(cls, dataset):
        """
        c -> class/label
        w -> word
        """
        label_freqdist = FreqDist()  # P(c) or log_prior
        word_freqdist = defaultdict(FreqDist)  # P(w|c) or log_likelihood

        # TODO
        # implement algorithm descripes in Fig 4.2 on p4 ch4

        return cls(label_freqdist, word_freqdist)

    def classify(self, datasets: list) -> list:
        """
        :param datasets: A list of text for testing.
        :return: Return a list of predicted labels.
        :rtype: ``list``
        """
        pass
