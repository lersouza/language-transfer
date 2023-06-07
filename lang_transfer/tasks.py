import functools
import itertools
from typing import Callable, Optional

import gin
import seqio
import tensorflow as tf

from datasets import load_dataset
from lang_transfer import preprocessing

from t5x.utils import DatasetConfig, get_dataset


@seqio.utils.map_over_dataset
def cast(x, features=["targets"], dtype=tf.int32):
    for feat in features:
        x[feat] = tf.cast(x[feat], dtype)
    return x


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

PASSTHROUGH_BYTE_OUTPUT_FEATURES = {
    "inputs": seqio.Feature(
        vocabulary=seqio.PassThroughVocabulary(259),
        add_eos=False,
        required=False,
        dtype=tf.string,
        rank=0,
    ),
    "targets": seqio.Feature(
        vocabulary=seqio.PassThroughVocabulary(259),
        add_eos=False,
        dtype=tf.int64,
        rank=1,
    ),
}

PASSTHROUGH_PREPROCESSORS = [
    functools.partial(
        seqio.preprocessors.rekey, key_map={"inputs": None, "targets": "targets"}
    ),
    functools.partial(cast, features=["targets"]),
]

ALL_LANGUAGES = (
    "en",
    "es",
)

DATASET_SIZES = [
    "6M",
    "60M",
    "600M",
    "6B",
]

# ---------------- Language tasks -----------------

# ADD TRAIN datasets for all languages and sizes

for lang, size_name in itertools.product(ALL_LANGUAGES, DATASET_SIZES):
    seqio.TaskRegistry.add(
        f"langagnostic.{lang}.{size_name}",
        source=seqio.TFExampleDataSource(
            {
                "train": f"gs://lang_agnostic/dataset/{lang}.{size_name}.examples",
            },
            feature_description={
                "targets": tf.io.FixedLenFeature([], tf.int64, default_value=None),
            },
        ),
        preprocessors=PASSTHROUGH_PREPROCESSORS,
        output_features=PASSTHROUGH_BYTE_OUTPUT_FEATURES,
        metric_fns=[],
    )

# ADD VALIDATION datasets for all languages
for lang in ALL_LANGUAGES:
    seqio.TaskRegistry.add(
        f"langagnostic.{lang}.validation",
        source=seqio.TFExampleDataSource(
            {
                "validation": f"gs://lang_agnostic/dataset/mc4_{lang}_validation_309278350.tfrecord",
            },
            feature_description={
                "text": tf.io.FixedLenFeature([], tf.string, default_value=""),
            },
        ),
        preprocessors=DEFAULT_PRE_PROCESSORS,
        output_features=DEFAULT_BYTE_OUTPUT_FEATURES,
        metric_fns=[],
    )
