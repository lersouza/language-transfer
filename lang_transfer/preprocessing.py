""" Preprocessing functions for pretraining our models. """
import math

import tensorflow as tf
import gin
import seqio

from typing import Dict, Optional, Sequence
from t5.data.preprocessors import (
    select_random_chunk,
    reduce_concat_tokens,
    split_tokens,
)


def group_texts(
    dataset,
    sequence_length,
    output_features,
    input_feature_key="targets",
    merge_examples_to_reduce_padding=True,
    reserved_for_packing=None,
    passthrough_feature_keys: Optional[Sequence[str]] = None,
):
    """
    Group texts into chunks to avoid loosing tokens in a simple sequence trimming.
    This function is a copy of t5.data.preprocessors.span_corruption,
    but without the denoising part.

    Args:
      dataset: A tf.data.Dataset with dictionaries containing the key
        `input_feature_key`.
      sequence_length: dict mapping of feature key to int length for that feature.
      output_features: mapping of keys to features.
      input_feature_key: which feature to use from the dataset as the input text
        tokens. Default set to `targets`for Decoder Only models.
      merge_examples_to_reduce_padding: if True, combines multiple input examples
        to reduce padding.
      reserved_for_packing: if specified, reduces the desired inputs length by the
        specified amount to enable multiple examples to be packed together
        downstream.
      passthrough_feature_keys: a sequence of feature names that should be passed
        through to the output of this preprocessor. eg: ["tokens"]. Only
        supported if `merge_examples_to_reduce_padding` is set to False.
    Returns:
      a dataset
    """
    input_length = sequence_length[input_feature_key]

    if reserved_for_packing:
        input_length -= reserved_for_packing

    preprocessed_dataset = dataset
    preprocessed_dataset = select_random_chunk(
        preprocessed_dataset,
        output_features=output_features,
        feature_key="targets",
        max_length=65536,
        passthrough_feature_keys=passthrough_feature_keys,
    )
    if merge_examples_to_reduce_padding:
        if passthrough_feature_keys:
            raise ValueError(
                "passthrough_feature_keys not supported with "
                "merge_examples_to_reduce_padding=True. "
                f"Got: {passthrough_feature_keys}"
            )
        preprocessed_dataset = reduce_concat_tokens(
            preprocessed_dataset, feature_key="targets", batch_size=128
        )

    preprocessed_dataset = split_tokens(
        preprocessed_dataset,
        feature_key="targets",
        min_tokens_per_segment=None,
        max_tokens_per_segment=input_length,
        passthrough_feature_keys=passthrough_feature_keys,
    )

    return preprocessed_dataset


@gin.configurable()
def take_n_tokens(
    dataset: tf.data.Dataset,
    sequence_length: Dict[str, int],
    n: int = 6_029_312_000,
    feature_key: str = "targets",
):
    """
    Limit the `dataset` to a specified number of tokens described by `n`.

    This method is meant to be used at the end of the pipeline in order to
    accurately limit the number of tokens to be seen by the model.
    """
    target_length = sequence_length.get(feature_key)
    number_of_examples = math.ceil(n / target_length)

    print("Taking", n, "tokens. Considering", number_of_examples, "to do so.")

    return dataset.take(number_of_examples)


# custom preprocessor that cast certain features to another dtype
# necessary to cast the tokenized data from tf.int64 (stored in the tfrecords)
# to tf.int32, expected output for target features.
@seqio.map_over_dataset
def cast(x, features=["targets"], dtype=tf.int32):
    for feat in features:
        x[feat] = tf.cast(x[feat], dtype)
    return x
