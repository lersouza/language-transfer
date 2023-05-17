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
    preprocessing.take_n_tokens,
]

ALL_LANGUAGES = (
    "en",
    "es",
)

def get_dataset(split, shuffle_files=False, seed=None, language="en"):
    """
    Returns a `tf.data.Dataset` object with the mC4 data for the specified `language`.

    This method leverages 2 strategies:

    * It uses Huggingface's streaming capability to avoid download the whole mc4, which
      is useful when building smaller datasets from it

    * It encapsulates the HF's dataset library inside a tf.data.Dataset created from a generator.
    """
    dataset = load_dataset("mc4", language, split=split, streaming=True)

    def _get_iterator():
        """
        This inner function only yields examples from the created `dataset`.
        """
        for item in dataset:
            # print("one more")
            yield item

    return tf.data.Dataset.from_generator(
        _get_iterator,
        output_types={"text": tf.string, "timestamp": tf.string, "url": tf.string},
    )


# ---------------- Language tasks -----------------
for lang in ALL_LANGUAGES:
    seqio.TaskRegistry.add(
        f"langagnostic.{lang}",
        source=seqio.FunctionDataSource(
            dataset_fn=functools.partial(get_dataset, language=lang),
            splits=["train", "validation"],
        ),
        preprocessors=DEFAULT_PRE_PROCESSORS,
        output_features=DEFAULT_BYTE_OUTPUT_FEATURES,
        metric_fns=[],
    )
