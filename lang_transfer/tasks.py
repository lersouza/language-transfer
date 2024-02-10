import functools
import itertools
import os

import seqio
import tensorflow as tf

import preprocessing

from pathlib import Path


VOCAB_SIZE = seqio.ByteVocabulary().vocab_size


DEFAULT_BYTE_OUTPUT_FEATURES = {
    "inputs": seqio.Feature(vocabulary=seqio.ByteVocabulary(), required=False),
    "targets": seqio.Feature(vocabulary=seqio.ByteVocabulary()),
}


DEFAULT_PASSTHROUGH_OUTPUT_FEATURES = {
    "inputs": seqio.Feature(vocabulary=seqio.PassThroughVocabulary(VOCAB_SIZE), required=False,  dtype=tf.string, rank=0),
    "targets": seqio.Feature(vocabulary=seqio.PassThroughVocabulary(VOCAB_SIZE), dtype=tf.int32, rank=1),
}


DEFAULT_PRE_PROCESSORS = [
    functools.partial(
        seqio.preprocessors.rekey, key_map={"inputs": None, "targets": "text"}
    ),
    seqio.preprocessors.tokenize,
    preprocessing.group_texts,
    seqio.preprocessors.append_eos_after_trim,
]


ALL_LANGUAGES = (
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

DATASET_SIZES = [
    "6M",
    "19M",
    "60M",
    "189M",
    "600M",
    "6B",
    "150B",
]


# ---------------- Bucket Config  ----------------
BUCKET_NAME=os.environ.get("BUCKET_NAME", "lang_agnostic_europe")
print("Using Bucket", BUCKET_NAME, "as source of data.")

# ---------------- Local Files support Config  ----------------
LOCAL_DATA_DIR=os.environ.get("LOCAL_DATA_DIR", Path.home())
print("LOCAL_DATA_DIR=", LOCAL_DATA_DIR)


# ---------------- Task Functions  ----------------
def _register_gcs_task(lang, size_name):
    seqio.TaskRegistry.add(
        f"langagnostic.{lang}.{size_name}",
        source=seqio.TFExampleDataSource(
            {
                "train": f"gs://{BUCKET_NAME}/dataset/{lang}/mc4_{lang}_train_{size_name}.tfrecord",
            },
            feature_description={
                "text": tf.io.FixedLenFeature([], tf.string, default_value=""),
            },
        ),
        preprocessors=DEFAULT_PRE_PROCESSORS,
        output_features=DEFAULT_BYTE_OUTPUT_FEATURES,
        metric_fns=[],
    )


def _register_local_task(lang, size_name):
    local_folder = os.path.join(LOCAL_DATA_DIR, lang)
    file_base_name = f"mc4_{lang}_train_{size_name}.tfrecord"
    files_or_file = None

    index_file_path = os.path.join(local_folder, f"{file_base_name}.index")

    if os.path.exists(index_file_path):
        files_or_file = []

        with open(index_file_path, "r", encoding="UTF-8") as idx_file:
            files_or_file.extend(idx_file.readlines())
    else:
        files_or_file = os.path.join(local_folder, file_base_name)

    seqio.TaskRegistry.add(
        f"langagnostic.{lang}.{size_name}.local",
        source=seqio.TFExampleDataSource(
            {
                "train": files_or_file,
            },
            feature_description={
                "text": tf.io.FixedLenFeature([], tf.string, default_value=""),
            },
        ),
        preprocessors=DEFAULT_PRE_PROCESSORS,
        output_features=DEFAULT_BYTE_OUTPUT_FEATURES,
        metric_fns=[],
    )


def _register_preprocessed_task(lang, size_name):
    seqio.TaskRegistry.add(
        f"langagnostic.{lang}.{size_name}.preprocessed",
        source=seqio.TFExampleDataSource(
            {
                "train": f"gs://{BUCKET_NAME}/dataset/{lang}/langagnostic.{lang}.{size_name}.preprocessed",
            },
            feature_description={
                "tokenized_text": tf.io.RaggedFeature(tf.int64),
            },
        ),
        preprocessors=[
            functools.partial(
                seqio.preprocessors.rekey, key_map={"inputs": None, "targets": "tokenized_text"}
            ),
            functools.partial(
                preprocessing.cast, features=["targets"]
            ),
        ],
        output_features=DEFAULT_PASSTHROUGH_OUTPUT_FEATURES,
        metric_fns=[],
    )


# ---------------- Language tasks -----------------
# ADD TRAIN datasets for all languages and sizes

for lang, size_name in itertools.product(ALL_LANGUAGES, DATASET_SIZES):
    _register_gcs_task(lang, size_name)
    _register_local_task(lang, size_name)
    _register_preprocessed_task(lang, size_name)

# ADD VALIDATION datasets for all languages
for lang in ALL_LANGUAGES:
    seqio.TaskRegistry.add(
        f"langagnostic.{lang}.validation",
        source=seqio.TFExampleDataSource(
            {
                "validation": f"gs://{BUCKET_NAME}/dataset/{lang}/mc4_{lang}_validation_6B-slice.tfrecord",
            },
            feature_description={
                "text": tf.io.FixedLenFeature([], tf.string, default_value=""),
            },
        ),
        preprocessors=DEFAULT_PRE_PROCESSORS,
        output_features=DEFAULT_BYTE_OUTPUT_FEATURES,
        metric_fns=[],
    )
