# cmput497_a5_yonael_zichun3

## Prerequisites

-   Python3
-   [virtualenv](https://virtualenv.pypa.io/en/latest/)

## Setup

```bash
# Setup python virtual environment
$ virtualenv venv --python=python3
$ source venv/bin/activate

# Install python dependencies
$ pip install -r requirements.txt
```

## How to run

### Classifier

```bash
$ python main.py --train data/trainBBC.csv --test data/testBBC.csv
```

```
Usage: main.py [OPTIONS]

Options:
  --train TEXT     Path to dataset to train the model.  [required]
  --test TEXT      Path to dataset to test the model.  [required]
  --evaluate TEXT  Path to dataset to evaluate the model.
  --epoch INTEGER  Number of training iterations, default to 3.
  --help           Show this message and exit.
```

### Error Analysis

```bash
$ python error_analysis.py output/output_testBBC.csv
```

```
Usage: error_analysis.py [OPTIONS] OUTPUT

  Produce metrics and figures for error analysis.

Options:
  --help  Show this message and exit.
```

## Output

### Classifier

The classifier uses k-fold cross-validation to find the best model. Each iteration the program prints out the range of validation dataset and training dataset, along with the validation accuracy.

The final output consists of total number of test cases, testing accruacy and average training (validation) accuracy.

e.g.

The output below can be interpreted as in the third iteration, within the training documents, documents from index `446` to `667` are used as validation dataset, and documents with index `0-222` and `223-445` are used as training dataset.

```
Validation fold: 446-667, Training fold: 0-222,223-445
Iteration: 3, Validation Accuracy: 94.595%
```

### Error Analysis

The script prints out the confusion matrix and precision/recall/f-score table.

## Authors

-   Yonael Bekele
-   Michael Lin
