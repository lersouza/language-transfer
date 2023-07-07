import functools
import itertools

import gin
import seqio
import tensorflow as tf

from lang_transfer import preprocessing


DEFAULT_BYTE_OUTPUT_FEATURES = {
    "inputs": seqio.Feature(vocabulary=seqio.ByteVocabulary(), required=False),
    "targets": seqio.Feature(vocabulary=seqio.ByteVocabulary()),
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
)

DATASET_SIZES = [
    "6M",
    "19M",
    "60M",
    "189M",
    "600M",
    "6B",
]

# ---------------- Language tasks -----------------
@gin.configurable
def register_training_tasks(languages, sizes, bucket_name="<<bucket>>"):
    print("Using Bucket", bucket_name, "as source of data.")

    for lang, size_name in itertools.product(languages, sizes):
        seqio.TaskRegistry.add(
            f"langagnostic.{lang}.{size_name}",
            source=seqio.TFExampleDataSource(
                {
                    "train": f"gs://{bucket_name}/dataset/{lang}/mc4_{lang}_train_{size_name}.tfrecord",
                },
                feature_description={
                    "text": tf.io.FixedLenFeature([], tf.string, default_value=""),
                },
            ),
            preprocessors=DEFAULT_PRE_PROCESSORS,
            output_features=DEFAULT_BYTE_OUTPUT_FEATURES,
            metric_fns=[],
        )

@gin.configurable
def register_validation_tasks(languages, bucket_name="<<bucket>>"):
    for lang in languages:
        seqio.TaskRegistry.add(
            f"langagnostic.{lang}.validation",
            source=seqio.TFExampleDataSource(
                {
                    "validation": f"gs://{bucket_name}/dataset/{lang}/mc4_{lang}_validation_6B-slice.tfrecord",
                },
                feature_description={
                    "text": tf.io.FixedLenFeature([], tf.string, default_value=""),
                },
            ),
            preprocessors=DEFAULT_PRE_PROCESSORS,
            output_features=DEFAULT_BYTE_OUTPUT_FEATURES,
            metric_fns=[],
        )


# ADD TRAINING datasets for all languages AND dataset sizes
register_training_tasks(ALL_LANGUAGES, DATASET_SIZES)

# ADD VALIDATION datasets for all languages
register_validation_tasks(ALL_LANGUAGES)
