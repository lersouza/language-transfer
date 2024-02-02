"""
This script truncates mc4's language subset to a specified number of Byte-level tokens.
"""
import argparse
import json
import os
import pickle

from datetime import datetime
from pathlib import Path
from typing import Any, Counter, Dict, List

import tensorflow as tf

from seqio.vocabularies import ByteVocabulary
from tqdm.auto import tqdm
from datasets import load_dataset

# sizes found for points between 6e6 and 6e9 plus 6B
#   points = np.logspace(start, end, num=7, base=10)
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


def create_checkpoint_file_name(target_file_path: str):
    return f"{target_file_path}.ckpt"


def retrieve_checkpoint(target_file_path: str):
    checkpoint_path = create_checkpoint_file_name(target_file_path)

    if not os.path.exists(checkpoint_path):
        return None

    with open(checkpoint_path, "rb") as ckpt_file:
        return pickle.load(ckpt_file)


def save_checkpoint(checkpoint_object, related_file_path):
    checkpoint_path = create_checkpoint_file_name(related_file_path)

    with open(checkpoint_path, "wb") as ckpt_file:
        return pickle.dump(checkpoint_object, ckpt_file)


def remove_checkpoint(related_file_path):
    os.remove(create_checkpoint_file_name(related_file_path))


def flush_and_checkpoint_if_needed(
    target_file_name: str,
    checkpoint_every_n_examples: int,
    current_idx: int,
    examples_buffer: List[tf.train.Example],
    state: Dict[str, Any],
    record_writer: tf.io.TFRecordWriter,
    last_saved_key: str = "last_saved_example_idx",
    force_flush: bool = False,
):
    if checkpoint_every_n_examples % current_idx != 0 and not force_flush:
        return

    for example in examples_buffer:
        record_writer.write(example.SerializeToString())

    examples_buffer.clear()
    state[last_saved_key] = current_idx

    save_checkpoint(state, target_file_name)


def truncate(
    language: str,
    split: str,
    max_train_tokens: int,
    keep_full_doc: bool,
    validation_percentage: float,
    output_directory: Path,
    size_name: str = None,
    overwrite: bool = False,
    checkpoint_every_n_examples: int = 10_000,
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

    checkpoint = retrieve_checkpoint(target_file_name)

    if checkpoint is None and os.path.exists(target_file_name) and not overwrite:
        print(f"File {target_file_name} already exists. Skipping...")
        return

    original_dataset = load_dataset("mc4", language, split=split, streaming=True)
    vocabulary = ByteVocabulary()  # No special tokens are added for ByT5

    if not checkpoint:
        state = {
            "processed_urls": [],
            "last_saved_example_idx": 0,
            "stats": {
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
            },
        }
    else:
        state = checkpoint

        print(
            "Restoring checkpoint. Last Processed Example Index:",
            state["last_saved_example_idx"],
            "Stats was:",
        )
        print_stats(state["stats"])

        os.rename(target_file_name, f"{target_file_name}.{datetime.now().timestamp()}")

    stats = state["stats"]
    record_buffer = []

    with tf.io.TFRecordWriter(str(target_file_name)) as file_writer, tqdm(
        total=tokens_to_process
    ) as pbar:
        for idx, example in enumerate(original_dataset):
            if idx < stats["last_saved_example_idx"]:
                pbar.update(stats["tokens"] // stats["examples"])
                continue

            raw_text = example["text"]
            in_bytes = vocabulary.encode(raw_text)

            state["processed_urls"].append(example["url"])

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

            record_buffer.append(
                tf.train.Example(
                    features=tf.train.Features(
                        feature={"text": _bytes_feature(raw_text.encode("utf-8"))}
                    )
                )
            )

            stats["text_length_after_truncation"] += len(raw_text)
            stats["examples"] += 1
            stats["tokens"] += len(in_bytes)

            flush_and_checkpoint_if_needed(
                target_file_name,
                checkpoint_every_n_examples,
                idx,
                record_buffer,
                state,
                file_writer,
            )

            pbar.update(len(in_bytes))

            if stats["tokens"] >= tokens_to_process:
                flush_and_checkpoint_if_needed(
                    target_file_name,
                    checkpoint_every_n_examples,
                    idx,
                    record_buffer,
                    state,
                    file_writer,
                    force_flush=True,
                )
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
            "Top 3 Processed URLS": Counter(state["processed_urls"]).most_common(n=3)
        },
    )

    remove_checkpoint(target_file_name)


def generate_datasets(
    language: str,
    validation_pct: float,
    keep_full_doc: bool,
    output_directory: str,
    overwrite: bool,
    sizes: Dict[str, int] = None,
):
    """
    Generates train and validation datasets for a given `language`
    and saves them to `output_directory`.

    The sizes for training can be provided as a dictionary of <size_name, size>
    or the default `DATASET_SIZES` dictionary will be used.
    """
    sizes_to_generate = sizes or DATASET_SIZES

    os.makedirs(output_directory, exist_ok=True)

    for size_name, size in sizes_to_generate.items():
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
    parser.add_argument(
        "--sizes", type=str, help="A dictionary with <size_name, size> in tokens."
    )

    args = parser.parse_args()
    sizes_to_generate = None if not args.sizes else json.loads(args.sizes)

    generate_datasets(
        args.language,
        args.validation_pct,
        args.keep_full_doc,
        args.output_dir,
        args.overwrite,
        sizes_to_generate,
    )
