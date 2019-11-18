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

```bash
$ python main.py --train data/trainBBC.csv --test data/testBBC.csv
```

```
Usage: main.py [OPTIONS]

Options:
  --train TEXT     Path to dataset to train the model.  [required]
  --test TEXT      Path to dataset to test the model.
  --evaluate TEXT  Path to dataset to evaluate the model.
  --help           Show this message and exit.
```

## Error Analysis

```bash
$ python error_analysis.py output/output_testBBC.csv
```

```
Usage: error_analysis.py [OPTIONS] OUTPUT

  Produce metrics and figures for error analysis.

Options:
  --help  Show this message and exit.
```

## Authors

-   Yonael Bekele
-   Michael Lin
