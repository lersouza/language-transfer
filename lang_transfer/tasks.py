import functools
import itertools
from typing import Callable, Optional

import gin
import seqio
import tensorflow as tf

from datasets import load_dataset
from lang_transfer import preprocessing

from t5x.utils import DatasetConfig, get_dataset


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
    "6M": 6815744,
    "60M": 60817408,
    "600M": 600834048,
    "6B": 6001000448,
}

# ---------------- Language tasks -----------------

# ADD TRAIN datasets for all languages and sizes

for lang, (size_name, size) in itertools.product(ALL_LANGUAGES, DATASET_SIZES.items()):
    seqio.TaskRegistry.add(
        f"langagnostic.{lang}.{size_name}",
        source=seqio.TFExampleDataSource(
            {
                "train": (
                    f"gs://lang_agnostic/dataset/mc4_{lang}_train_10000000000.tfrecord"
                ),
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


# Define a custom dataset function
@gin.configurable(allowlist=["num_epochs", "continue_from_last_checkpoint"])
def get_train_dataset_wrapper(
    cfg: DatasetConfig,
    shard_id: int,
    num_shards: int,
    feature_converter_cls: Callable[..., seqio.FeatureConverter],
    num_epochs: Optional[int] = None,
    continue_from_last_checkpoint: bool = False,
) -> tf.data.Dataset:
    return get_dataset(
        cfg=cfg,
        shard_id=shard_id,
        num_shards=num_shards,
        feature_converter_cls=feature_converter_cls,
        num_epochs=num_epochs,
        continue_from_last_checkpoint=continue_from_last_checkpoint,
    )
