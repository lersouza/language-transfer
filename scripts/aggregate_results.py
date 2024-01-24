import argparse
import logging
import os
import re
import tempfile

from pathlib import Path

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
EXP_PATTERN = r"\.*\/(?P<filename>(?P<initialization>\w+)_(?P<target>\w+)_(?P<model_size>\w+)_(?P<data_size>[0-9]+[MB]+))\/$"

# We remove some experiments based on patterns over metadata
EXPERIMENTS_TO_REMOVE = [
    {
        "bucket": "lang_agnostic",
        "target": "es",
        "data_size": "6B",
        "initialization": "en",
    },  # Results are inconsitent. Experiment was re-
    {"target": "es", "data_size": "0M"},
    {"target": "pt"},
]

# The metric of interest for LOSS.
LOSS_METRIC = "loss_per_all_target_tokens"

# We create a global client for connecting to GCS
client = storage.Client()

# Logger definition
LOGGER = logging.getLogger(__file__)


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
            [
                {"url": i, "bucket": bucket, **re.search(EXP_PATTERN, i).groupdict()}
                for i in prefixes
                if re.search(EXP_PATTERN, i)
            ]
        )

    return experiments_in_storage


def should_take_out(exp):
    """
    Determined whether an experiments should be ignored based on  `EXPERIMENTS_TO_REMOVE`.
    Returns `True` if it should be ignored and `False` otherwise.
    """
    for r in EXPERIMENTS_TO_REMOVE:
        if all([exp[k] == r[k] for k in r.keys()]):
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


def export_experiments_to_csv():
    """
    Export experiments from GCS Tensorboard files to a CSV file.
    """
    target_path = str(Path(ROOT_FOLDER / "results/experiments.csv"))

    LOGGER.info("About to retrieve experiments and save them to %s", target_path)

    all_experiments = list_experiments()
    filtered_experiments = [a for a in all_experiments if not should_take_out(a)]

    LOGGER.info(
        f"Retrieved {len(all_experiments)}. After filtering:"
        f" {len(filtered_experiments)}"
    )

    all_experiment_results = aggregate_results(filtered_experiments)
    df = pd.DataFrame.from_dict(all_experiment_results)

    with open(target_path, "w+", encoding="utf-8") as target_file:
        df.to_csv(target_file)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    export_experiments_to_csv()
