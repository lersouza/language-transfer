import argparse
import pandas as pd

from pathlib import Path


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


def read_contamination_data(lang_contamination_dir: Path) -> pd.DataFrame:
    return pd.DataFrame()


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


def preprocess_contamination_data(contamindation_data: pd.DataFrame) -> pd.DataFrame:
    return contamindation_data


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
    return base_data


def build_tidy(
    base_file: Path,
    dataset_stats_file: Path,
    lang_contamination_dir: Path,
    output_file_path: Path,
):
    base_data: pd.DataFrame = pd.read_csv(base_file)
    dataset_stats_data = pd.read_csv(dataset_stats_file)
    contamination_data = read_contamination_data(lang_contamination_dir)

    base_data = preprocess_base_data(base_data)
    dataset_stats_data = preprocess_stats_file(dataset_stats_data)
    contamination_data = preprocess_contamination_data(contamination_data)

    tidy_data = combine_with_dataset_stats(base_data, dataset_stats_data)
    tidy_data = combine_with_language_contamination(tidy_data, contamination_data)

    tidy_data.to_csv(output_file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("base_file", type=Path)
    parser.add_argument("dataset_stats_file", type=Path)
    parser.add_argument("lang_contamination_dir", type=Path)

    parser.add_argument("--output_file", type=Path, default=Path(".") / "tidy_data.csv")

    args = parser.parse_args()
    build_tidy(
        args.base_file,
        args.dataset_stats_file,
        args.lang_contamination_dir,
        args.output_file,
    )
