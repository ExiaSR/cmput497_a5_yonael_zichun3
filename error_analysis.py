"""
Usage: error_analysis.py [OPTIONS] OUTPUT

  Produce metrics and figures for error analysis.

Options:
  --help  Show this message and exit.
"""

import click


@click.command()
@click.argument("output", type=click.File("r"))
def main(output):
    """
    Produce metrics and figures for error analysis.
    """
    pass


if __name__ == "__main__":
    main()
