from operator import itemgetter
from collections import defaultdict, OrderedDict, Counter
from functools import reduce

import numpy as np


class NaiveBayesClassifier:
    def __init__(self, log_prior, log_likelihood, classes, vocabs):
        self._log_prior = log_prior
        self._log_likelihood = log_likelihood
        self._vocabs = vocabs
        self._classes = classes

    @classmethod
    def train(cls, dataset: "Dataset"):
        """
        c -> class/label
        w -> word

        :param dataset: A Dataset object
        """
        # count number of occurences of each word given their originated label
        labled_featuresets = dataset.get_labeled_featuresets()
        big_doc = defaultdict(Counter)
        for word, label in labled_featuresets:
            big_doc[label][word] += 1
        # get vocabs
        vocabs: set = dataset.vocabs()
        # get labels
        classes = list(dataset.classes())
        # count number of documents in dataset in each class
        label_freqdist = Counter(dataset.labels())
        # total number of documents in dataset
        num_docs = sum(label_freqdist.values())

        log_prior = dict()
        log_likelihood = defaultdict(dict)  # avoid key error
        for class_name in classes:
            num_docs_in_class = label_freqdist[class_name]
            log_prior[class_name] = np.log2(label_freqdist[class_name] / num_docs)
            denominator = sum(big_doc[class_name].values()) + len(vocabs)
            for word in vocabs:
                count = big_doc[class_name][word]  # number of occurrences of w in bigdoc[c]
                numerator = count + 1
                log_likelihood[class_name][word] = np.log2(numerator / denominator)

        return cls(log_prior, log_likelihood, classes, vocabs)

    def classify(self, datasets: list) -> list:
        """
        :param datasets: A list of text for testing.
        :return: Return a list of predicted labels.
        :rtype: ``list``
        """
        predicted_labels = []
        for document in datasets:
            log_sum = {}
            for class_name in self._classes:
                log_sum[class_name] = self._log_prior[class_name]
                for word in document:
                    if word in self._vocabs:
                        log_sum[class_name] += self._log_likelihood[class_name][word]

            predicted_labels.append(max(log_sum, key=log_sum.get))

        return predicted_labels
