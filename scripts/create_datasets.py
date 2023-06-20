"""
This script truncates mc4's language subset to a specified number of Byte-level tokens.
"""
import argparse
import os

from pathlib import Path
from typing import Any, Counter, Dict, List

import tensorflow as tf

from seqio.vocabularies import ByteVocabulary
from tqdm.auto import tqdm
from datasets import load_dataset


DATASET_SIZES = {
    "6M": 6815744,
    "19M": 19398656,
    "60M": 60817408,
    "189M": 189267968,
    "600M": 600834048,
    "6B": 6001000448,
}


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def save_stats(target_file_name: Path, stats: Dict[str, Any]):
    """
    Save a stats file for the `target_file_name`, using the `stats` value provided.
    """
    with open(str(target_file_name) + ".stats", "w+", encoding="utf-8") as stats_file:
        for key, value in stats.items():
            stats_file.write(f"{key}: {value}\n")


def print_stats(
    stats: Dict[str, Any], additional_info: Dict[str, Any] = None, complete: bool = True
):
    """Print statistics for followup"""
    status_message = "Done truncating." if complete else "Intermediary Stats."

    print("=" * 100)
    print(status_message)
    print("Stats:", stats)

    if additional_info:
        for info_name, info_value in additional_info.items():
            print(info_name, ":", info_value)

    print("=" * 100)


def truncate(
    language: str,
    split: str,
    max_train_tokens: int,
    keep_full_doc: bool,
    validation_percentage: float,
    output_directory: Path,
    size_name: str = None,
    overwrite: bool = False,
):
    """
    Truncate the specified mC4's `language` subset to a maximum of `max_tokens`
    based on the result of `seqio.vocabularies.ByteVocabulary.encode`

    The output is saved as a TFRecord File to `output_directory`.
    A stats file is also saved in this directory.
    """
    tokens_to_process = (
        max_train_tokens
        if split == "train"
        else int(
            (validation_percentage * max_train_tokens) / (1.0 - validation_percentage)
        )
    )
    target_file_name = (
        output_directory
        / f"mc4_{language}_{split}_{size_name or tokens_to_process}.tfrecord"
    )

    if os.path.exists(target_file_name) and not overwrite:
        print(f"File {target_file_name} already exists. Skipping...")
        return

    original_dataset = load_dataset("mc4", language, split=split, streaming=True)
    vocabulary = ByteVocabulary()  # No special tokens are added for ByT5
    processed_urls = []

    stats = {
        "language": language,
        "split": split,
        "examples": 0,
        "original_text_length": 0,
        "text_length_after_truncation": 0,
        "tokens": 0,
        "original_tokens_length": 0,
        "max_tokens": tokens_to_process,
        "max_train_tokens": max_train_tokens,
        "validation_percentage": validation_percentage,
        "token2text_rate": None,
        "dropped_text_length": 0,
        "dropped_tokens_length": 0,
    }

    with tf.io.TFRecordWriter(str(target_file_name)) as file_writer, tqdm(
        total=tokens_to_process
    ) as pbar:
        for example in original_dataset:
            raw_text = example["text"]
            in_bytes = vocabulary.encode(raw_text)

            processed_urls.append(example["url"])

            stats["original_text_length"] += len(raw_text)
            stats["original_tokens_length"] += len(in_bytes)

            if (
                len(in_bytes) + stats["tokens"] > tokens_to_process
                and keep_full_doc is False
            ):
                remaining = int(tokens_to_process - stats["tokens"])

                # Truncate at the UTF-8 Byte level here
                # The main issue is that we may loose valid characters, since UTF-8 can take up to
                # 4 bytes for representing a single character and we may truncate in the middle.
                # However, since the ByteVocabulary has a fallback mechanism,
                # we may not have errors, just less characters in the end and model that
                # sees invalid bytes at the end of an epoch (if it comes to that, anyway)
                #
                # TODO Come up with a better way for that
                in_bytes = in_bytes[:remaining]
                raw_text = vocabulary.decode(in_bytes)

            file_writer.write(
                tf.train.Example(
                    features=tf.train.Features(
                        feature={"text": _bytes_feature(raw_text.encode("utf-8"))}
                    )
                ).SerializeToString()
            )

            stats["text_length_after_truncation"] += len(raw_text)
            stats["examples"] += 1
            stats["tokens"] += len(in_bytes)

            pbar.update(len(in_bytes))

            if stats["tokens"] >= tokens_to_process:
                break

    stats["token2text_rate"] = stats["tokens"] / stats["text_length_after_truncation"]
    stats["dropped_tokens_length"] = stats["original_tokens_length"] - stats["tokens"]
    stats["dropped_text_length"] = (
        stats["original_text_length"] - stats["text_length_after_truncation"]
    )

    save_stats(target_file_name, stats)
    print_stats(
        stats,
        additional_info={
            "Top 3 Processed URLS": Counter(processed_urls).most_common(n=3)
        },
    )


def generate_datasets(
    language: str, validation_pct: float, keep_full_doc: bool, output_directory: str, overwrite: bool
):
    """
    Generate datasets with different `sizes` for the selected `language`.
    All resulting files are saved to `output_directory`.
    """
    os.makedirs(output_directory, exist_ok=True)

    for size_name, size in DATASET_SIZES.items():
        truncate(
            language=language,
            split="train",
            max_train_tokens=size,
            keep_full_doc=keep_full_doc,
            validation_percentage=validation_pct,
            output_directory=output_directory,
            size_name=size_name,
            overwrite=overwrite,
        )

    if validation_pct is not None:
        base_size = DATASET_SIZES["6B"] * validation_pct
        validation_size_name = "6B-slice"

        truncate(
            language=language,
            split="validation",
            max_train_tokens=base_size,
            keep_full_doc=keep_full_doc,
            validation_percentage=validation_pct,
            output_directory=output_directory,
            size_name=validation_size_name,
            overwrite=overwrite,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("language", type=str)
    parser.add_argument("--validation_pct", type=float, default=None)
    parser.add_argument("--keep_full_doc", action="store_true", default=False)
    parser.add_argument("--output_dir", type=Path, default=Path("./"))
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    generate_datasets(
        args.language,
        args.validation_pct,
        args.keep_full_doc,
        args.output_dir,
        args.overwrite,
    )
