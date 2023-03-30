import functools
import itertools

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

PRETRAIN_LANGUAGES = ("en",)
PRETRAIN_SIZES = ("6B",)

FINETUNE_LANGUAGES = ("es",)
FINETUNE_SIZES = (60e5, 60e6, 60e7, 60e8)


# ---------------- Pretrain tasks -----------------
for lang in PRETRAIN_LANGUAGES:
    seqio.TaskRegistry.add(
        f"langagnostic.pretrain.{lang}.small",
        source=seqio.TFExampleDataSource(
            split_to_filepattern={
                "train": f"gs://lang_agnostic/dataset/pretrain/mc4_{lang}_train_6000000000.tfrecord",
                "validation": f"gs://lang_agnostic/dataset/pretrain/mc4_{lang}_validation_315789473.tfrecord",
            },
            feature_description={
                "text": tf.io.FixedLenFeature([], tf.string, default_value=""),
            },
        ),
        preprocessors=DEFAULT_PRE_PROCESSORS,
        output_features=DEFAULT_BYTE_OUTPUT_FEATURES,
        metric_fns=[],
    )

# ---------------- Finetune tasks -----------------
for lang, size in itertools.product(FINETUNE_LANGUAGES, FINETUNE_SIZES):
    seqio.TaskRegistry.add(
        f"langagnostic.finetune.{lang}.small.{int(size)}",
        source=seqio.TFExampleDataSource(
            split_to_filepattern={
                "train": f"gs://lang_agnostic/dataset/data/finetune/{lang}/mc4_{lang}_train_{size}.tfrecord",
                "validation": f"gs://lang_agnostic/dataset/pretrain/mc4_{lang}_validation_315789473.tfrecord",
            },
            feature_description={
                "text": tf.io.FixedLenFeature([], tf.string, default_value=""),
            },
        ),
        preprocessors=DEFAULT_PRE_PROCESSORS,
        output_features=DEFAULT_BYTE_OUTPUT_FEATURES,
        metric_fns=[],
    )
