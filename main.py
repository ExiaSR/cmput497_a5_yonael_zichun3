"""
Usage: main.py [OPTIONS]

Options:
  --train TEXT     Path to dataset to train the model.  [required]
  --test TEXT      Path to dataset to test the model.  [required]
  --evaluate TEXT  Path to dataset to evaluate the model.
  --epoch INTEGER  Number of training iterations.
  --help           Show this message and exit.
"""
import operator

import click

from cmput497_as5.classify import NaiveBayesClassifier
from cmput497_as5.utils import *


@click.command()
@click.option("--train", help="Path to dataset to train the model.", required=True)
@click.option("--test", help="Path to dataset to test the model.", required=True)
@click.option("--evaluate", help="Path to dataset to evaluate the model.")
@click.option("--epoch", type=int, help="Number of training iterations, default to 3.", default=3)
def main(train, test, evaluate, epoch):
    train_dataset = get_dataset(train, shuffle=True)
    test_dataset = get_dataset(test)

    if evaluate:
        evaluate_dataset = get_dataset(evaluate)

    # train model using k-fold cross validation
    best_validation_accuracy = 0.0
    best_classifier = None
    labeled_documents = list(test_dataset.labeled_documents())
    training_accuracy_acc = 0
    for iteration, (train_index, validation_index, fold_info) in enumerate(k_fold(len(labeled_documents), epoch)):
        # get training and validation sets by indices
        train_documents = operator.itemgetter(*train_index)(labeled_documents)
        validation_documents = operator.itemgetter(*validation_index)(labeled_documents)

        # train model
        classifier = NaiveBayesClassifier.train(train_documents)

        # validation and calculate accuracy
        zipped = list(zip(*validation_documents))
        predicted_labels = classifier.classify(zipped[0])
        validation_accuracy = accuracy(predicted_labels, zipped[1])
        training_accuracy_acc += validation_accuracy

        # Record validation accuracy and persist
        # the classifier that has the highest validation accuracy
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_classifier = classifier

        print(fold_info)
        print("Iteration: {}, Validation Accuracy: {:.3f}%\n".format(iteration + 1, 100 * validation_accuracy))

    # classify model and save output
    classifier = best_classifier
    test_predicited_labels = classifier.classify(test_dataset.documents())
    test_dataset._set_predicted_labels(test_predicited_labels)
    save_output(test_dataset)
    test_accuracy = accuracy(test_predicited_labels, test_dataset.labels())
    print(
        "Total: {}, Testing Accuracy: {:.3f}%, Training Accuracy: {:.3f}%".format(
            len(test_predicited_labels), 100 * test_accuracy, 100 * training_accuracy_acc / epoch
        )
    )

    if evaluate:
        evaluate_predicited_labels = classifier.classify(evaluate_dataset.documents())
        evaluate_dataset._set_predicted_labels(evaluate_predicited_labels)
        save_output(evaluate_dataset)


if __name__ == "__main__":
    main()
