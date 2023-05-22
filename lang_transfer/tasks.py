import functools
import itertools

import seqio
import tensorflow as tf

from datasets import load_dataset
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
    "en",
    "es",
)

DATASET_SIZES = {
    "6M": 6291456,
    "60M": 60293120,
    "600M": 600309760,
    "6B": 6000476160,
}

# ---------------- Language tasks -----------------

for lang, (size_name, size) in itertools.product(ALL_LANGUAGES, DATASET_SIZES.items()):
    seqio.TaskRegistry.add(
        f"langagnostic.{lang}.{size_name}",
        source=seqio.TFExampleDataSource(
            {
                "train": (
                    f"gs://lang_agnostic/dataset/mc4_{lang}_train_10000000000.tfrecord"
                ),
                "validation": f"gs://lang_agnostic/dataset/mc4_{lang}_validation_309278350.tfrecord",
            },
            feature_description={
                "text": tf.io.FixedLenFeature([], tf.string, default_value=""),
            },
        ),
        preprocessors=DEFAULT_PRE_PROCESSORS
        + [functools.partial(preprocessing.take_n_tokens, n=size)],
        output_features=DEFAULT_BYTE_OUTPUT_FEATURES,
        metric_fns=[],
    )
