import argparse
import logging
import pandas as pd

from pathlib import Path


LOGGER = logging.getLogger("build_tidy_dataset")


BASE_DATA_COLUMNS = {
    "model_size": "model_size",
    "init_language": "source",
    "init_data_size": "source_data_size",
    "init_train_epochs": "source_train_epochs",
    "target": "target",
    "data_size": "target_data_size",
    "finetune_epochs": "target_train_epochs",
    "size": "target_data_size_as_int",
    "perplexity": "perplexity",
    "data_effective": "data_effective",
    "data_transfer": "data_transfer",
    "fraction_of_effective_dt": "fraction_of_effective_dt",
    "cross_lingual": "cross_lingual",
    "data_transfer_mb": "data_transfer_formatted",
    "data_finetune_mb": "data_finetune_formatted",
    "data_effective_mb": "data_effective_formatted",
    "syntactic_distance": "syntactic_distance",
    "geographic_distance": "geographic_distance",
    "phonological_distance": "phonological_distance",
    "genetic_distance": "genetic_distance",
    "inventory_distance": "inventory_distance",
    "featural_distance": "featural_distance",
}


DATASET_STATS_COLUMNS = {
    "size_name": "data_size",
    "language": "language",
    "split": "data_split",
    "examples": "num_of_examples",
    "original_text_length": "original_text_length",
    "text_length_after_truncation": "text_length_after_truncation",
    "tokens": "num_of_tokens",
    "original_tokens_length": "original_num_of_tokens",
    "token2text_rate": "token2text_rate",
}


CONTAMINATION_DATA_COLUMNS = {
    "source_language": "primary_language",
    "sentences": "sentence_count",
    "target_language": "detected_language_label",
    "size": "dataset_size",
}


MULTIPLIERS = {
    "M": 1000000,
    "B": 1000000000,
}


def convert_size_to_number(number_repr):
    suffix = number_repr[-1]
    multiplier = MULTIPLIERS[suffix]
    number_part = number_repr[:-1]

    return int(number_part) * multiplier


def read_contamination_data(lang_contamination_dir: Path) -> pd.DataFrame:
    files = [
        str(file)
        for file in lang_contamination_dir.glob("languages_in_*_*_byline_dataset.csv")
    ]
    language_contamination_data = pd.concat([pd.read_csv(f) for f in files])

    return language_contamination_data


def preprocess_base_data(base_data: pd.DataFrame) -> pd.DataFrame:
    base_data.loc[base_data["init_train_epochs"].isna(), "init_train_epochs"] = 1.0
    base_data.loc[base_data["init_data_size"].isna(), "init_data_size"] = "6B"

    # Ensure Scratch Models do not have init params
    base_data.loc[base_data["init_language"] == "scratch", "init_train_epochs"] = 0.0
    base_data.loc[base_data["init_language"] == "scratch", "init_data_size"] = "0M"

    mask1 = (base_data["finetune_epochs"].isna()) & (base_data["data_size"] != "6B")
    mask2 = (base_data["finetune_epochs"].isna()) & (base_data["data_size"] == "6B")

    base_data.loc[mask1, "finetune_epochs"] = 10.0
    base_data.loc[mask2, "finetune_epochs"] = 3.0

    # Select desired columns and adjust column names
    base_data = base_data[list(BASE_DATA_COLUMNS.keys())].rename(
        columns=BASE_DATA_COLUMNS
    )

    return base_data


def preprocess_stats_file(dataset_stats_data: pd.DataFrame) -> pd.DataFrame:
    dataset_stats_data = dataset_stats_data[list(DATASET_STATS_COLUMNS.keys())].rename(
        columns=DATASET_STATS_COLUMNS
    )

    # For now, we only care about the information used for training
    # So, we discard validation data
    dataset_stats_data = dataset_stats_data[dataset_stats_data.data_split == "train"]

    # Setup and index for joining
    dataset_stats_data = dataset_stats_data.set_index(["language", "data_size"])

    # Remove duplicates, since we may have info of a dataset in multiple buckets.
    dataset_stats_data = dataset_stats_data.drop_duplicates()

    return dataset_stats_data


def preprocess_contamination_data(contamination_data: pd.DataFrame) -> pd.DataFrame:
    contamination_data = contamination_data[
        list(CONTAMINATION_DATA_COLUMNS.keys())
    ].rename(columns=CONTAMINATION_DATA_COLUMNS)

    all_sentences_per_language = contamination_data.groupby("primary_language")[
        "sentence_count"
    ].sum()

    contamination_data["detected_language_ratio"] = contamination_data.apply(
        lambda r: r["sentence_count"]
        / all_sentences_per_language[r["primary_language"]],
        axis=1,
    )

    contamination_data["detected_language"] = contamination_data[
        "detected_language_label"
    ].apply(lambda e: e.split("_")[-1])

    contamination_data = contamination_data.set_index(
        ["primary_language", "dataset_size", "detected_language"]
    )

    return contamination_data.drop(columns=["detected_language_label"])


def combine_with_dataset_stats(
    base_data: pd.DataFrame, stats_data: pd.DataFrame
) -> pd.DataFrame:
    tidy_data = base_data.join(
        stats_data, on=["source", "source_data_size"], how="inner", rsuffix="_source"
    )

    tidy_data = tidy_data.join(
        stats_data, on=["target", "target_data_size"], how="inner", rsuffix="_target"
    )

    # We also ackowledge the difference between source and target token2text_rate
    tidy_data["token2text_rate_square_difference"] = (
        tidy_data["token2text_rate"] - tidy_data["token2text_rate_target"]
    ) ** 2

    return tidy_data


def combine_with_language_contamination(
    base_data: pd.DataFrame, contamination_data: pd.DataFrame
) -> pd.DataFrame:

    tidy_data = base_data.join(
        contamination_data,
        on=["source", "source_data_size", "target"],
        how="left",
        rsuffix="_on_source",
    )

    tidy_data = tidy_data.join(
        contamination_data,
        on=["target", "target_data_size", "source"],
        how="left",
        rsuffix="_on_target",
    )

    # Fill with 0 the cases in which there is no contamination
    # or, at least, it is so few sentences that is shows as 'others'
    tidy_data.loc[
        tidy_data["detected_language_ratio"].isna(), "detected_language_ratio"
    ] = 0
    tidy_data.loc[
        tidy_data["detected_language_ratio_on_target"].isna(),
        "detected_language_ratio_on_target",
    ] = 0

    return tidy_data


def build_tidy(
    base_file: Path,
    dataset_stats_file: Path,
    lang_contamination_dir: Path,
    output_file_path: Path,
):
    base_data: pd.DataFrame = pd.read_csv(base_file)
    dataset_stats_data = pd.read_csv(dataset_stats_file)
    contamination_data = read_contamination_data(lang_contamination_dir)

    LOGGER.info(
        "Successfuly loaded all datasets. Sizes are: base=%d, stats=%d, contamination=%d",
        len(base_data),
        len(dataset_stats_data),
        0,
        # len(contamination_data),
    )

    base_data = preprocess_base_data(base_data)
    dataset_stats_data = preprocess_stats_file(dataset_stats_data)
    contamination_data = preprocess_contamination_data(contamination_data)

    tidy_data = combine_with_dataset_stats(base_data, dataset_stats_data)
    tidy_data = combine_with_language_contamination(tidy_data, contamination_data)

    LOGGER.info("Finished building tidy dataset. Final size is %d", len(tidy_data))

    tidy_data.to_csv(output_file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("base_file", type=Path)
    parser.add_argument("dataset_stats_file", type=Path)
    parser.add_argument("lang_contamination_dir", type=Path)

    parser.add_argument("--output_file", type=Path, default=Path(".") / "tidy_data.csv")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    build_tidy(
        args.base_file,
        args.dataset_stats_file,
        args.lang_contamination_dir,
        args.output_file,
    )
