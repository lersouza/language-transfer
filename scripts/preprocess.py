import argparse
import pandas as pd

from pathlib import Path


MULTIPLIERS = {
    "M": 1000000,
    "B": 1000000000,
}


BASE_DIMENSION_COLUMNS = [
    "initialization",
    "target",
    "model_size",
    "data_size",
    "finetune_epochs",
]


def convert_size_to_number(number_repr):
    suffix = number_repr[-1]
    multiplier = MULTIPLIERS[suffix]
    number_part = number_repr[:-1]

    return int(number_part) * multiplier


def fill_missing_data(base_data: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing data in the base dataset with default values.
    """
    base_data.loc[base_data["init_train_epochs"].isna(), "init_train_epochs"] = 1.0
    base_data.loc[base_data["init_data_size"].isna(), "init_data_size"] = "6B"

    # Ensure Scratch Models do not have init params
    base_data.loc[base_data["init_language"] == "scratch", "init_train_epochs"] = 0.0
    base_data.loc[base_data["init_language"] == "scratch", "init_data_size"] = "0M"

    mask1 = (base_data["finetune_epochs"].isna()) & (base_data["data_size"] != "6B")
    mask2 = (base_data["finetune_epochs"].isna()) & (base_data["data_size"] == "6B")

    base_data.loc[mask1, "finetune_epochs"] = 10.0
    base_data.loc[mask2, "finetune_epochs"] = 3.0

    return base_data


def determine_type_of_experiment(base_data: pd.DataFrame) -> pd.DataFrame:
    """
    Determine the type of experiment based on the initialization language.
    """
    def _define_typeof_experiment(row):
        if row["source"].endswith("2") and row["source"][:-1] == row["target"]:
            return "upper_bound"

        if row["source"] == "nonsense":
            return "lower_bound"
        
        if (
            row["init_data_size"] != "6B"
            or row["init_train_epochs"] != 1.0
            or (row["data_size"] == "6B" and row["finetune_epochs"] != 3.0)
            or (row["data_size"] != "6B" and row["finetune_epochs"] != 10.0)
        ):
            return "ablation"

        if row["init_language"] != row["target"]:
            return "cross_lingual"
        
        return "unknown"

    base_data["experiment_type"] = base_data.apply(_define_typeof_experiment, axis=1)


def add_integer_size_columns(base_data: pd.DataFrame) -> pd.DataFrame:
    """
    Add integer size columns to the base dataset.
    """
    base_data["init_data_size_int"] = base_data["init_data_size"].apply(
        convert_size_to_number
    )
    base_data["data_size_int"] = base_data["data_size"].apply(
        convert_size_to_number
    )

    return base_data


def dedup_experiments(base_data: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicate experiments in the base dataset.
    """
    # Order by loss
    base_data = base_data.sort_values(by=["filename", "loss"], ascending=True).drop_duplicates(
        subset=["filename"], keep="first", inplace=False
    )

    # Reset index after deduplication
    base_data.reset_index(drop=True, inplace=True)

    return base_data


def preprocess(args):
    """
    Perform preprocessing routines to a raw dataset
    in order to be able to calculate the metrics D_e and D_t.
    """
    dataset = pd.read_csv(args.results_file)

    # Transforms
    dataset = fill_missing_data(dataset)

    # Add features
    dataset = add_integer_size_columns(dataset)

    # Deduplicate Experiments
    dataset = dedup_experiments(dataset)

    # Save the preprocessed dataset
    dataset.to_csv(args.output_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("results_file", type=Path)
    parser.add_argument(
        "--output_file",
        type=Path,
        default=Path(".") / "results/finetune_preprocessed.csv",
    )

    preprocess(parser.parse_args())
