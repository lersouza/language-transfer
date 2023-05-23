import argparse
import functools

from pathlib import Path
from typing import Any, Dict

import preprocessing
import seqio
import tensorflow as tf



def print_stats(
    stats: Dict[str, Any], additional_info: Dict[str, Any] = None
):
    """Print statistics for followup"""
    status_message = "Done."

    print("=" * 100)
    print(status_message)
    print("Stats:")

    for k, v in stats.items():
        print(k, "\t\t=", v)

    if additional_info:
        for info_name, info_value in additional_info.items():
            print(info_name, ":", info_value)

    print("=" * 100)


def check(file: Path, number_of_tokens: int, batch_size: int, seq_length: int):

    seqio.TaskRegistry.add(
        "check_task",
        source=seqio.TFExampleDataSource({
            "train": str(file),
        },
        feature_description={
                "text": tf.io.FixedLenFeature([], tf.string, default_value=""),
            },
        ),
        preprocessors=[
            functools.partial(
                seqio.preprocessors.rekey, key_map={"inputs": None, "targets": "text"}
            ),
            seqio.preprocessors.tokenize,
            preprocessing.group_texts,
            seqio.preprocessors.append_eos_after_trim,
            functools.partial(
                preprocessing.take_n_tokens,
                n=number_of_tokens
            )
        ],
        output_features={
            "inputs": seqio.Feature(vocabulary=seqio.ByteVocabulary(), required=False),
            "targets": seqio.Feature(vocabulary=seqio.ByteVocabulary()),
        },
        metric_fns=[],
    )

    dataset = seqio.get_dataset(
        mixture_or_task_name="check_task",
        task_feature_lengths={"targets": seq_length},
        feature_converter=seqio.LMFeatureConverter(pack=True),
        dataset_split="train",
        batch_size=batch_size
    )

    stats = {
        "tokens": 0,
        "expected": number_of_tokens,
        "min_seq_length": 999999999,
        "max_seq_length": -1,
        "num_of_examples": 0,
        "num_of_steps": 0,
        "num_of_examples_out_of_seq_length": 0,
        "min_batch_size": 99999999,
    }

    tokens_found = 0

    for i, item in enumerate(dataset):
        bsize, slen = item["decoder_target_tokens"].shape

        stats["tokens"] += (bsize * slen)
        stats["min_seq_length"] = min(stats["min_seq_length"], slen)
        stats["max_seq_length"] = max(stats["max_seq_length"], slen)
        stats["num_of_examples"] += bsize
        stats["num_of_examples"] += bsize
        stats["num_of_steps"] += 1
        stats["min_batch_size"] = min(stats["min_batch_size"] , bsize)

        if slen != seq_length:
            stats["num_of_examples_out_of_seq_length"] += bsize
            print("The shape for example", i, "is", f"({bsize}, {slen})", ".")

    print_stats(stats)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("file", type=Path)
    parser.add_argument("number_of_tokens", type=int)
    
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--seq_length", type=int, default=1024)

    args = parser.parse_args()

    check(args.file, args.number_of_tokens, args.batch_size, args.seq_length)