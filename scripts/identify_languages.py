import argparse
import csv
import itertools
import logging

from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable

import fasttext
import tensorflow as tf

from tqdm import tqdm


# Gets the root folder for reference
ROOT_FOLDER = Path(__file__).parent.parent.resolve()


LOGGER = logging.getLogger("language_classifier")


class ClassificationStrategy(ABC):

    def __init__(self) -> None:
        self.name = ""

    @abstractmethod
    def load_dataset(self, language, size, bucket) -> tf.data.Dataset:
        pass

    @abstractmethod
    def prepare_sequences(self, example) -> Iterable[str]:
        pass


class ClassifyByLineStrategy(ClassificationStrategy):
    def __init__(self) -> None:
        super(ClassifyByLineStrategy).__init__()
        self.name = "byline"

    @staticmethod
    def decode_tf_example(record_bytes):
        return tf.io.parse_single_example(
            record_bytes, {"text": tf.io.FixedLenFeature([], dtype=tf.string)}
        )

    def load_dataset(self, language, size, bucket):
        dataset_location = (
            f"gs://{bucket}/dataset/{language}/mc4_{language}_train_{size}.tfrecord"
        )

        return tf.data.TFRecordDataset(dataset_location).map(
            ClassifyByLineStrategy.decode_tf_example
        )

    def prepare_sequences(self, example):
        doc_lines = bytes.decode(example["text"].numpy()).splitlines()
        valid_lines = list(filter(lambda l: len(l) > 20, doc_lines))

        return valid_lines


__DATASET_SIZES = {
    "6M",
    "19M",
    "60M",
    "189M",
    "600M",
    "6B",
}

__ALL_LANGUAGES = (
    "ar",
    "en",
    "es",
    "pt",
    "zh",
    "fi",
    "de",
    "ko",
    "id",
    "ja",
    "ru",
)


__STRATEGIES = {
    "byline": ClassifyByLineStrategy,
}


def save_stats(target_file_name, stats):
    """
    Save a stats file for the `target_file_name`, using the `stats` value provided.
    """
    with open(str(target_file_name) + ".stats", "w+", encoding="utf-8") as stats_file:
        for key, value in stats.items():
            stats_file.write(f"{key}: {value}\n")


def format_stats_into_lines(source_language, size, stats):
    overall = {
        "min_line_size": stats.pop("min_line_size"),
        "max_line_size": stats.pop("max_line_size"),
        "min_confidence_interval": stats.pop("min_confidence_interval"),
        "max_confidence_interval": stats.pop("max_confidence_interval"),
        "total_sentences": stats.pop("total_sentences"),
        "avg_line_size": stats.pop("avg_line_size"),
        "threshold": stats.pop("threshold"),
    }

    rows = []

    for lang in stats.keys():
        rows.append([source_language, size, lang, stats[lang]])

    return rows, overall


def classify_dataset(
    language, size, bucket, model, strategy: ClassificationStrategy, threshold: float
):
    stats: Dict[str, Any] = defaultdict(lambda: 0.0)

    stats["min_line_size"] = 999
    stats["max_line_size"] = 0
    stats["min_confidence_interval"] = 100
    stats["max_confidence_interval"] = 0
    stats["line_sizes"] = []

    dataset = strategy.load_dataset(language, size, bucket)

    for doc in tqdm(dataset):
        valid_sentences = strategy.prepare_sequences(doc)

        try:
            classification = model.predict(valid_sentences)
        except:
            LOGGER.error(
                "Error classifying sequences. Sequences were: %s.", valid_sentences
            )

            return None

        for sentence, clazz, probs in zip(
            valid_sentences, classification[0], classification[1]
        ):
            if probs[0] >= threshold:
                stats[clazz[0]] += 1
            else:
                stats["__label__other"] += 1

            stats["total_sentences"] += 1
            stats["min_line_size"] = min(stats["min_line_size"], len(sentence))
            stats["max_line_size"] = max(stats["max_line_size"], len(sentence))
            stats["min_confidence_interval"] = min(
                stats["min_confidence_interval"], probs[0]
            )
            stats["max_confidence_interval"] = max(
                stats["max_confidence_interval"], probs[0]
            )
            stats["line_sizes"].append(len(sentence))

    stats["avg_line_size"] = sum(stats["line_sizes"]) / len(stats["line_sizes"])
    stats["threshold"] = threshold

    del stats["line_sizes"]

    return stats


def classify(
    model_path,
    language,
    size,
    bucket,
    strategy: ClassificationStrategy,
    threshold: float,
):
    model = fasttext.load_model(model_path)
    target_file = Path(ROOT_FOLDER / f"results/languages_in_{language}_{size}_{strategy.name}_dataset.csv")

    if target_file.exists():
        LOGGER.warning("Target file %s already exists. Skipping.", target_file)
        return

    with open(str(target_file), "w+", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["source_language", "size", "target_language", "sentences"])

        classifications = classify_dataset(
            language, size, bucket, model, strategy, threshold
        )

        if not classifications:
            LOGGER.warning(
                "Could not perform classifications for language %s, size: %s with strategy %s",
                language,
                size,
                strategy.name,
            )
            return

        rows, overall = format_stats_into_lines(language, size, classifications)

        writer.writerows(rows)

    save_stats(target_file, overall)

    LOGGER.info("Done! Language: %s, Size: %s, Stats: %s", language, size, overall)


def run_for_languages(
    model_path,
    languages,
    sizes,
    bucket,
    strategy: ClassificationStrategy,
    threshold: float,
):
    LOGGER.info(
        "Running for languages: %s and sizes %s. Model is %s. Thresold is %f. Strategy is %s.",
        languages,
        sizes,
        model_path,
        threshold,
        strategy.name,
    )

    for lang, size in itertools.product(languages, sizes):
        try:
            classify(model_path, lang, size, bucket, strategy, threshold)
        except:
            LOGGER.warning("Error processing language %s in size %d. Bucket: %s", lang, size, bucket)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("model_path")
    parser.add_argument("language")
    parser.add_argument("size")

    parser.add_argument(
        "--bucket",
        default="lang_agnostic",
        help="The bucket name where the dataset is.",
    )

    parser.add_argument("--strategy", default="byline")
    parser.add_argument("--threshold", default=0.6)

    args = parser.parse_args()
    strategy = __STRATEGIES.get(args.strategy, ClassifyByLineStrategy)()

    languages = __ALL_LANGUAGES if args.language == "all" else [args.language]
    sizes = __DATASET_SIZES if args.size == "all" else [args.size]

    logging.basicConfig(level=logging.INFO)

    run_for_languages(
        args.model_path, languages, sizes, args.bucket, strategy, args.threshold
    )
