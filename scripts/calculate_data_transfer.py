"""
This script computes the main metrics for Data Transfer from one language (source or initialization)
to another languague (target).

It is a new version of `calculate_metrics.py` that acknowledges for different
model initializations (i.e., different pretrainings) and finetune setups
by calculating the metrics for each experiment (row) in the input file.

It takes, as input, a CSV file with the following required columns:

- model_size: a name for the size of the model used.
- init_language: the language in which the pretrained model was trained on.
- target: the target language
- data_size: the size of the finetune dataset (6M, 19M,60M, 189M, 600M, 6B)
- loss: the resulting loss of the experiment

For the random-initialized models, init_language MUST be *scratch*.
The script *aggregate_results.py* downloads results from GCS with naming convention and produces such an input file.

The output is a file with D_T, D_E, D_T / D_E calculated.
"""

import argparse
import numpy as np
import pandas as pd

from functools import partial
from pathlib import Path


MULTIPLIERS = {
    "M": 1000000,
    "B": 1000000000,
}


FINETUNE_SIZES = [
    6000000,
    19000000,
    60000000,
    189000000,
    600000000,
    6000000000,
]


def convert_size_to_number(number_repr):
    """
    Converts a numeric representation (`number_repr`), such as 6M,
    to a actual number: 6_000_000.
    """
    suffix = number_repr[-1]
    multiplier = MULTIPLIERS[suffix]
    number_part = number_repr[:-1]

    return int(number_part) * multiplier


def format_in_mega(v):
    """
    Format a number as MB.
    """
    return "{:,.2f} MB".format(v / 1024 / 1024)


def apply_defaults(raw_data: pd.DataFrame) -> pd.DataFrame:
    """
    Apply default values for required fields.
    """
    raw_data.loc[raw_data["model_size"].isna(), "model_size"] = "small"


def preprocess(raw_data: pd.DataFrame) -> pd.DataFrame:
    """
    Perform preprocessing routines to a raw dataset
    in order to be able to calculate the metrics D_e and D_t.
    """
    # Apply Defaults
    apply_defaults(raw_data)

    # Compute Columns
    raw_data.loc[:, "size"] = raw_data["data_size"].apply(convert_size_to_number)
    raw_data.loc[:, "perplexity"] = np.exp(raw_data["loss"])

    # Remove zero-shot experiments. We'll not consider them at this time
    raw_data = raw_data[raw_data["size"] > 0]

    return raw_data


def postprocess(results: pd.DataFrame) -> pd.DataFrame:
    """
    Apply post-processing routines to the dataset after metrics calculation.
    Mostly add formatted and indicator columns.
    """
    results.loc[:, "cross_lingual"] = results["init_language"] != results["target"]
    results.loc[:, "data_transfer_mb"] = results["data_transfer"].map(format_in_mega)
    results.loc[:, "data_finetune_mb"] = results["size"].map(format_in_mega)
    results.loc[:, "data_effective_mb"] = results["data_effective"].map(format_in_mega)

    return results


def compute_data_effective(
    experiment_row: pd.Series, scratch_data: pd.DataFrame
) -> pd.Series:
    """
    For each experiment with pretrained models, compute the Data Effective metric.
    The associated experiment with a model from scratch is selected from `scratch_data`.

    We follow the method described in https://aclanthology.org/2024.naacl-long.418.pdf and use
    Linear interpolation (`np.interp`) to compute the estimations.

    It is noteworthy that, if multiple scratch experiments were performed for a combination
    of (model size, target language, finetune size) - for comparing setups, for instance -
    the minimum perplexity will be selected, as if it was a result of Hyperparameter search.

    Args:
        * experiment_row: A row for any given experiment with a pretrained model
        * scratch_data: A data frame with all results for models initializaed from scratch.
                        The Dataframe must be indexed by (model_size, target).

    Returns:
        * The estimated Data Effective metric for the experiment.
    """
    index = (experiment_row["model_size"], experiment_row["target"])
    scratch_related = (
        scratch_data.loc[index, ["size", "perplexity"]]
        .groupby("size")
        .min()
        .reset_index()
    )

    x_values = scratch_related["size"]
    y_values = scratch_related["perplexity"]
    y_value_dotted = experiment_row["perplexity"]

    # Estimate the Data Effective through interpolation.
    # For more information, see: https://aclanthology.org/2024.naacl-long.418.pdf
    estimated_de = np.interp(x=y_value_dotted, xp=y_values, fp=x_values, period=10)

    return estimated_de


def calculate_data_transfer(results_file: Path, output_file: Path):
    """
    For a given `results_file` containing results from all experiments performed,
    calculate the Data Effect and Data Transfer Metrics.

    The resulting dataset only contains data for experiments with pretrained models.
    Scratch models are used to derivate the metrics and, then, discarded.

    The resulting dataset is saved to `output_file`.
    """
    raw_data = pd.read_csv(results_file)
    raw_data = preprocess(raw_data)

    scratch_mask = raw_data["init_language"] == "scratch"
    scratch, pretrained = raw_data[scratch_mask].copy(), raw_data[~scratch_mask].copy()

    # Define an index in scratch experiments that allows me to find the related experiment
    # for any given results with pretrained model
    scratch = scratch.set_index(["model_size", "target"]).sort_index(level=[0, 1])

    compute_de = partial(compute_data_effective, scratch_data=scratch)
    pretrained["data_effective"] = pretrained.apply(compute_de, axis=1)

    pretrained["data_transfer"] = pretrained["data_effective"] - pretrained["size"]
    pretrained["fraction_of_effective_dt"] = np.maximum(
        pretrained["data_transfer"] / pretrained["data_effective"], 0
    )

    pretrained = postprocess(pretrained)
    pretrained.to_csv(output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("results_file", type=Path)
    parser.add_argument(
        "--output_file", type=Path, default=Path(".") / "estimations.csv"
    )

    args = parser.parse_args()
    calculate_data_transfer(args.results_file, args.output_file)
