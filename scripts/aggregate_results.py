import argparse
import json
import logging
import os
import re
import tempfile

from pathlib import Path
from typing import Dict, List

import tensorflow as tf
import pandas as pd

from collections import defaultdict

from google.cloud import storage

from tensorflow.python.lib.io import tf_record
from tensorflow.core.util import event_pb2
from tensorflow.python.framework import tensor_util
from tensorflow.python.summary.summary_iterator import summary_iterator


# Gets the root folder for reference
ROOT_FOLDER = Path(__file__).parent.parent.resolve()

# We consolidate experiments from multiple buckets
BUCKETS = ["lang_agnostic", "lang_agnostic_europe"]

# Some experiment metadata can be extracted from file names
EXP_PATTERN = r"\.*\/(?P<filename>(?P<initialization>(?P<init_language>\w{2,8})(_from_(?P<init_data_size>[0-9]+[MB]+)(_(?P<init_train_epochs>\d+)epochs){0,1}){0,1})_(?P<target>\w{2})_(?P<model_size>(small|gadre_1.4B|550M))_(?P<data_size>[0-9]+[MB]+)(_(?P<finetune_epochs>\d+)epoch[s]*){0,1})\/$"

# We remove some experiments based on patterns over metadata
EXPERIMENTS_TO_REMOVE = [
    {
        "bucket": "lang_agnostic",
        "target": "es",
        "data_size": "6B",
        "init_language": "en",
    },  # Results are inconsitent. Experiment was made again in Europe Bucket.
    {"target": "es", "data_size": "0M"},
    {"target": "pt"},
    {"url_metadata": "no"}
]

# The metric of interest for LOSS.
LOSS_METRIC = "loss_per_all_target_tokens"

# We create a global client for connecting to GCS
client = storage.Client()

# Logger definition
LOGGER = logging.getLogger(__file__)


def extract_metadata_from_name(
    experiment_path: str, bucket_name: str
) -> Dict[str, str]:
    """
    Extracts metadata information from URL.
    If no metadata could be extracted, property "url_metadata" will be "no".
    """
    matched = re.search(EXP_PATTERN, experiment_path)
    base_metadata = {"url": experiment_path, "bucket": bucket_name}

    if matched:
        return {**base_metadata, "url_metadata": "yes", **matched.groupdict()}

    return {**base_metadata, "url_metadata": "no"}


def list_experiments():
    """
    List all avaliable experiments in all `BUCKETS`.
    """
    experiments_in_storage = []

    for bucket in BUCKETS:
        storage_response = client.list_blobs(
            bucket, prefix="models/finetune/", delimiter="/"
        )
        _, prefixes = all(storage_response), list(
            storage_response.prefixes
        )  # Force request execution

        experiments_in_storage.extend(
            [extract_metadata_from_name(i, bucket) for i in prefixes]
        )

    return experiments_in_storage


def log_ignored_experiments(ignored_experiments: List):
    """
    Log, as a JSON file, the experiments that were ignored when compiling results.
    """
    target_path = str(Path(ROOT_FOLDER / f"results/ignored_experiments.json"))

    with open(target_path, "w+", encoding="utf-8") as ignore_file:
        json.dump(ignored_experiments, ignore_file)


def download_dataset_stats(target_dir):
    """
    Lists all Datasets available for experiments.
    """
    for bucket in BUCKETS:
        files = client.list_blobs(
            bucket, prefix="dataset/", match_glob="**.tfrecord.stats"
        )

        for stats_file in files:
            target_path = os.path.join(target_dir, os.path.split(stats_file.name)[1])
            stats_file.download_to_filename(target_path)

            yield target_path


def should_take_out(exp):
    """
    Determined whether an experiments should be ignored based on  `EXPERIMENTS_TO_REMOVE`.
    Returns `True` if it should be ignored and `False` otherwise.
    """
    for r in EXPERIMENTS_TO_REMOVE:
        if all([k in exp and exp[k] == r[k] for k in r.keys()]):
            return True
    return False


def retrieve_results(experiment, target_dir):
    """
    For any given (existing) `experiment`, retrieve all its results.
    The results are saved in `target_dir` under the name it was found in the bucket.
    """
    prefix = f"{experiment['url']}training_eval/langagnostic.{experiment['target']}.validation/events.out.tfevents"
    files_it = client.list_blobs(experiment["bucket"], prefix=prefix)

    local_file_paths = []

    for i, remote_file in enumerate(files_it):
        local_file = os.path.join(target_dir, f"{experiment['filename']}.{i}")
        local_file_paths.append(local_file)

        remote_file.download_to_filename(local_file)

    return local_file_paths


def read_all_recorded_metrics(local_files, metric_of_interest):
    """
    Returns a dictionary with `metric_of_interest` extracted from experiment results.
    """
    recorded = {metric_of_interest: [], "steps": [], "event_files": []}

    for event_file in local_files:
        for event_string in tf_record.tf_record_iterator(event_file):
            event = event_pb2.Event.FromString(event_string)
            for value in event.summary.value:
                if value.tag in recorded:
                    tensor_value = tensor_util.MakeNdarray(value.tensor)

                    recorded[value.tag].append(tensor_value)
                    recorded["steps"].append(event.step)
                    recorded["event_files"].append(event_file)

    return recorded


def process_experiment_results(experiment, local_files):
    """
    For a given experiment and its local files (where results were downloaded), returns a dictionary with experiment results.
    """
    all_losses = read_all_recorded_metrics(local_files, LOSS_METRIC)

    if not all_losses[LOSS_METRIC]:
        print("Loss not found for experiment. Experiment Details:", experiment)
        return {}

    best_metrics_idx = min(
        range(len(all_losses[LOSS_METRIC])), key=lambda i: all_losses[LOSS_METRIC][i]
    )

    final_metrics = {
        "loss": all_losses[LOSS_METRIC][best_metrics_idx],
        "step": all_losses["steps"][best_metrics_idx],
        **experiment,
    }

    return final_metrics


def aggregate_results(experiments):
    """
    Aggregates the results of all experiments and return them as a dictionary,
    where each key represents a column for a list of values.
    """
    all_experiment_results = defaultdict(list)

    with tempfile.TemporaryDirectory() as tempdir:
        for exp in experiments:
            local = retrieve_results(exp, tempdir)
            exp_metrics = process_experiment_results(exp, local)

            for metric, value in exp_metrics.items():
                all_experiment_results[metric].append(value)

    return all_experiment_results


def process_dataset_stats(dataset_stats_files):
    """
    Process a list of dataset stats files and return a dictionary with all attributes in those files.
    """
    dataset_info = defaultdict(list)
    dataset_stat_file_regex = r"(?P<origin>[\w\d]+)_(?P<language_split>[\w]{2,8})_(?P<train_split>[\w]+)_(?P<size_name>[\w\d]+)"

    for ds_stat in dataset_stats_files:
        with open(ds_stat, "r", encoding="utf-8") as stat_file:
            file_ms_name = os.path.split(str(stat_file))[1].split(".")[0]
            file_name_metadata = re.search(dataset_stat_file_regex, file_ms_name)

            LOGGER.debug("Processing stats: %s", file_ms_name)

            for k, v in file_name_metadata.groupdict().items():
                dataset_info[k].append(v)

            for line in stat_file:
                k, v = line.split(":")
                dataset_info[k.strip()].append(v.strip())

    LOGGER.debug("Dataset information stats:")

    for k, v in dataset_info.items():
        LOGGER.debug("%s (%d)", k, len(v))

    # Some datasets do not have validation_percentage.
    # As this info is not important at this time, we remove it.
    dataset_info.pop("validation_percentage")

    return dataset_info


def retrieve_finetune_results():
    """
    Retrieve results from finetune experiments in GCS and return a Data Frame consolidating its data.
    """
    all_experiments = list_experiments()
    filtered_experiments = [a for a in all_experiments if not should_take_out(a)]

    # Log ignored experiments for troubleshooting purposes.
    log_ignored_experiments([a for a in all_experiments if should_take_out(a)])

    LOGGER.info(
        f"Retrieved {len(all_experiments)}. After filtering:"
        f" {len(filtered_experiments)}"
    )

    all_experiment_results = aggregate_results(filtered_experiments)
    return pd.DataFrame.from_dict(all_experiment_results)


def retrieve_dataset_stats():
    """
    Retrive statistics from the datasets used for the experiments in a DataFrame.
    """
    with tempfile.TemporaryDirectory() as tempdir:
        dataset_info = process_dataset_stats(download_dataset_stats(tempdir))

    return pd.DataFrame.from_dict(dataset_info).drop_duplicates()


def retrieve_pretrain_results():
    """
    Retrieve results from pretraining experiments with source languages in GCS
    and return a Data Frame consolidating its data.
    """
    pass


def export_experiments_to_csv(info_to_retrieve: str):
    """
    Export experiments from GCS Tensorboard files to a CSV file.
    """
    target_path = str(Path(ROOT_FOLDER / f"results/{info_to_retrieve}_data.csv"))

    LOGGER.info(
        "About to retrieve %s experiments and save them to %s",
        info_to_retrieve,
        target_path,
    )

    method = AVAILABLE_RESULTS.get(info_to_retrieve, None)

    if not method:
        LOGGER.error("Method for %s is not implemented!", info_to_retrieve)
        return

    data: pd.DataFrame = method()

    with open(target_path, "w+", encoding="utf-8") as target_file:
        data.to_csv(target_file)


# Possible results to retrieve
AVAILABLE_RESULTS = {
    "finetune": retrieve_finetune_results,
    "datasets": retrieve_dataset_stats,
    "pretrain": retrieve_pretrain_results,
}


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("info_to_retrieve", choices=AVAILABLE_RESULTS.keys())
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    export_experiments_to_csv(args.info_to_retrieve)
