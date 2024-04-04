"""
This script computes the main metrics for Data Transfer from one language (source or initialization)
to another languague (target).

It takes, as input, a CSV file with the following required columns:

- initialization: the source language
- target: the target language
- data_size: the size of the finetune dataset (6M, 600M, 6B)
- loss: the resulting loss of the experiment

For the random-initialized models, initialization MUST be *scratch*.
the script *aggregate_results.py* downloads results from GCS with naming convention and produces such an input file.

The output is a file with D_T, D_E, D_T / D_E calculated.
"""

import argparse

import numpy as np
import pandas as pd

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
    suffix = number_repr[-1]
    multiplier = MULTIPLIERS[suffix]
    number_part = number_repr[:-1]

    return int(number_part) * multiplier


def format_in_mega(v):
    return "{:,.2f} MB".format(v / 1024 / 1024)


def preprocess(raw_data: pd.DataFrame) -> pd.DataFrame:
    raw_data["size"] = raw_data["data_size"].apply(convert_size_to_number)
    raw_data["cross_lingual"] = raw_data["initialization"] != raw_data["target"]
    raw_data["perplexity"] = np.exp(raw_data["loss"])

    return raw_data


def prepare_by_source_language(raw_data: pd.DataFrame) -> pd.DataFrame:
    source_languages = raw_data["initialization"].unique()

    target_v_initialization = pd.pivot_table(
        raw_data,
        index=["target", "size"],
        columns="initialization",
        values="perplexity",
    )

    by_lang = {
        lang: target_v_initialization[["scratch", lang]]
        for lang in source_languages
        if lang != "scratch"
    }

    return by_lang


def compute_data_transfer(by_lang_data):
    estimations = {
        "source": [],
        "target": [],
        "size": [],
        "perplexity": [],
        "scratch_perplexity": [],
        "dt": [],
        "de": [],
        "df": [],
        "fraction_of_effective_dt": [],
    }

    for lang, lang_data in by_lang_data.items():
        languages = by_lang_data[lang].index.get_level_values(0).unique().to_list()

        if lang in languages:
            languages.remove(
                lang
            )  # removing itself, since we are interested in cross lingual experiments

        for target in languages:
            target_data = lang_data.loc[target]
            target_data = target_data[
                target_data.index.isin(FINETUNE_SIZES)
            ]  # We do not consider, at first, zero-shot experiments

            x_values = target_data.index.to_numpy()

            y_values = target_data["scratch"].to_numpy()
            y_values_dotted = target_data[lang].to_numpy()

            estimated_de = np.interp(
                x=y_values_dotted, xp=y_values, fp=x_values, period=10
            )
            estimated_dt = estimated_de - x_values
            fraction_of_effective_dt = np.maximum(estimated_dt / estimated_de, 0)

            estimations["source"].extend([lang] * len(x_values))
            estimations["target"].extend([target] * len(x_values))
            estimations["size"].extend(x_values)
            estimations["perplexity"].extend(y_values_dotted)
            estimations["scratch_perplexity"].extend(y_values)
            estimations["de"].extend(estimated_de)
            estimations["df"].extend(x_values)
            estimations["dt"].extend(estimated_dt)
            estimations["fraction_of_effective_dt"].extend(fraction_of_effective_dt)

    return pd.DataFrame.from_dict(estimations)


def post_process_estimations(estimations: pd.DataFrame) -> pd.DataFrame:
    estimations["dt_formatted"] = estimations["dt"].map(format_in_mega)
    estimations["df_formatted"] = estimations["df"].map(format_in_mega)
    estimations["de_formatted"] = estimations["de"].map(format_in_mega)

    return estimations


def calculate_metrics(results_file: Path, output_file: Path):
    raw_data = pd.read_csv(results_file)
    raw_data = preprocess(raw_data)

    data_by_source = prepare_by_source_language(raw_data)

    estimations = compute_data_transfer(data_by_source)
    estimations = post_process_estimations(estimations)

    estimations.to_csv(output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("results_file", type=Path)
    parser.add_argument(
        "--output_file", type=Path, default=Path(".") / "estimations.csv"
    )

    args = parser.parse_args()
    calculate_metrics(args.results_file, args.output_file)
