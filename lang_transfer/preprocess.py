import argparse
import functools
import os
import seqio
import tasks

import tensorflow as tf

from tqdm import tqdm
from typing import Dict


def _int64_feature_list(values):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _get_file_name(target_dir: str, task_name: str):
    return os.path.join(target_dir, f"{task_name}.preprocessed")


@seqio.utils.map_over_dataset
def cast(x, features=["targets"],dtype=tf.int32):
  for feat in features:
    x[feat]=tf.cast(x[feat],dtype)
  return x


def build_preprocessed_dataset(task_name: str, target_length: int, target_dir: str):
    target_file_name = _get_file_name(target_dir, task_name)

    print("About to process", task_name, "to", target_file_name)

    dataset = seqio.get_dataset(
        mixture_or_task_name=task_name,
        task_feature_lengths={"targets": target_length},
        feature_converter=seqio.PassThroughFeatureConverter(),
        dataset_split="train",
        batch_size=1,
    )

    print("Got dataset", dataset)

    with tf.io.TFRecordWriter(target_file_name) as file_writer, tqdm() as pbar:
        for example in dataset:
            file_writer.write(
                tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            "tokenized_text": _int64_feature_list(example["targets"].numpy().tolist()[0])
                        }
                    )
                ).SerializeToString()
            )

            pbar.update(1)


def test(file_path, target_length):
    task = seqio.Task(
        f"langagnostic.es.6M.preprocessed",
        source=seqio.TFExampleDataSource(
            {
                "train": file_path,
            },
            feature_description={
                "tokenized_text": tf.io.RaggedFeature(tf.int64)
            },
        ),
        preprocessors=[
            functools.partial(
                seqio.preprocessors.rekey, key_map={"inputs": None, "targets": "tokenized_text"}
            ),
            functools.partial(
                cast, features=["targets"]
            ),
        ],
        output_features=tasks.DEFAULT_PASSTHROUGH_OUTPUT_FEATURES,
        metric_fns=[],
    )

    ds = seqio.get_dataset(
        task,
        task_feature_lengths={"targets": target_length},
        feature_converter=seqio.DecoderFeatureConverter(),
        dataset_split="train",
        batch_size=1,
    )

    for i, ex in enumerate(ds):
        print("example", i)

        if i % 100 == 0:
            print("samples=", ex)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("task_name")
    parser.add_argument("--target_length", default=4096)
    parser.add_argument("--target_dir", default="./data")
    parser.add_argument("--test_only", action="store_true")

    args = parser.parse_args()

    if args.test_only:
        test(_get_file_name(args.target_dir, args.task_name), args.target_length)
    else:
        build_preprocessed_dataset(args.task_name, args.target_length, args.target_dir)
