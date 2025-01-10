"""
This script is based in the paper:  'Does Pretraining for Summarization Require Knowledge Transfer?'
Available at: https://arxiv.org/abs/2109.04953

It generates synthetic data for Pretraining.
The data consists of n-grams randomly selected to form sentences and paragraphs.
"""

import argparse
import numpy as np
import tensorflow as tf

from random import choices, randint
from tqdm import tqdm
from typing import Any, Dict


# The number of tokens to include in the dataset of size 6B
DATASET_SIZE_6B = 6001000448


class RandomVocab:
    """
    This class was extracted from https://github.com/acmi-lab/pretraining-with-nonsense/
    It generates a vocabulary of n-grams, where n depends on the vocabulary size.
    Examples of words (considering 3-grams): aaa, aab, aac, and so on.
    """

    def __init__(self, vocab_size=5000) -> None:
        char_slots = np.ceil(np.log(vocab_size) / np.log(26))
        char_slots = int(char_slots)
        assert char_slots > 0

        vocab = []
        for i in range(vocab_size):
            out = ""
            val = i
            for _ in range(char_slots):
                rem = val % 26
                character = chr(97 + rem)
                out = out + character
                val = int(val / 26)

            vocab.append(out)

        assert len(set(vocab)) == vocab_size
        self.tokens = vocab


class BaseGenerator:
    """
    This class was extracted and adapted from https://github.com/acmi-lab/pretraining-with-nonsense/
    It generates paragraphs with a specificed average number of sentences and words.

    The adaptation includes the following:
    (1) Use `RandomVocab` as a fixed vocab, not enabling the T5Vocab as in the original class
    (2) Added a `get_example` method for generating an entire document (set of paragraphs)
    (3) Joined `gen_sent` and `get_para` methods to improve code performance by leveraging matrix operations in numpy.
    """

    def __init__(
        self,
        numsent_tolerance: int = 3,
        sentlen_tolerance: int = 5,
        paralen_tolerance: int = 3,
    ) -> None:
        self.numsent_tolerance = numsent_tolerance
        self.sentlen_tolerance = sentlen_tolerance
        self.paralen_tolerance = paralen_tolerance

        self.vocab = RandomVocab()
        self.tokens = [tok for tok in self.vocab.tokens if "." not in tok]

    def get_para(self, mean_numsents, mean_sentlen):
        num_sentences = mean_numsents + np.random.randint(
            -self.numsent_tolerance, self.numsent_tolerance + 1
        )
        max_num_tokens = mean_sentlen + self.sentlen_tolerance
        min_num_tokens = mean_sentlen - self.sentlen_tolerance

        random_sentences = [
            " ".join(
                choices(
                    row,
                    k=randint(min_num_tokens, max_num_tokens),
                )
            )
            + "."
            for row in np.random.choice(
                self.tokens, (num_sentences, max_num_tokens), replace=True
            )
        ]

        return random_sentences

    def get_example(
        self, mean_numpara: int = 10, mean_numsent: int = 10, mean_sentlen: int = 10
    ):
        """
        This method was added to handle examples (documents) with many paragraphs.
        It relies on a very similar implementation as `self.get_para`.

        Parameters:

        * mean_numpara (int, default: 10):
            The mean number of paragraphs to include in the document.
            The number will be something in the interval [mean_numpara - self.paralen_tolerance, mean_numpara + self.paralen_tolerance]

        * mean_numsent (int, default: 10):
            The mean number of sentences in each paragraph.

        * mean_sentlen (int, default: 10):
            The mean number of tokens to include in each sentence.

        Both `mean_numsent` and `mean_sentlen` are passed as argments to `self.get_para`, which generates the paragraphs.
        """
        num_paragraphs = mean_numpara + np.random.randint(
            -self.paralen_tolerance, self.paralen_tolerance + 1
        )
        num_paragraphs = max(1, num_paragraphs)
        paragraphs = []

        for _ in range(num_paragraphs):
            sentences = self.get_para(mean_numsent, mean_sentlen)
            paragraphs.append(
                " ".join(sentences) + "\n"
            )  # Every paragraph ends with EOL char

        return paragraphs


def create_byte_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.

    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def save_stats(target_file_name: str, stats: Dict[str, Any]):
    """
    Save a stats file for the `target_file_name`, using the `stats` value provided.
    """
    with open(target_file_name + ".stats", "w+", encoding="utf-8") as stats_file:
        for key, value in stats.items():
            stats_file.write(f"{key}: {value}\n")


def print_stats(
    stats: Dict[str, Any],
    additional_info: Dict[str, Any] | None = None,
    complete: bool = True,
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


def create_nonsense_data(
    tokens: int, split: str, output_path: str, write_every_n_examples: int = 100_000
):
    """
    Creates a dataset with non-sense data. The number of tokens to include is specified by `tokens`.
    The final TF Examples file is output at `output_path`.
    """
    generator = BaseGenerator()
    buffer = []
    size, num_of_writes = 0, 0

    stats = {
        "language": "nonsense",
        "split": split,
        "examples": 0,
        "original_text_length": 0,
        "text_length_after_truncation": 0,
        "tokens": 0,
        "original_tokens_length": 0,
        "max_tokens": tokens,
        "max_train_tokens": tokens,
        "token2text_rate": 1.0,  # We use only ASCII chars
        "dropped_text_length": 0,
        "dropped_tokens_length": 0,
    }

    with tf.io.TFRecordWriter(output_path) as out_file, tqdm(
        total=tokens
    ) as pbar, tqdm(total=0, position=1, bar_format="{desc}") as sbar:
        sbar.set_description_str(f"No writes. Buffer size is {write_every_n_examples}.")
        while size < tokens:
            document = " ".join(generator.get_example())
            document_size = len(document)

            stats["original_text_length"] += document_size
            stats["original_tokens_length"] += document_size

            if size + document_size > tokens:
                # Ensure the final text meets the max number of tokens constraint
                document = document[: tokens - size]
                document_size = len(document)

            size += document_size  # Since we only use ASCII Characters, assuming each char as a byte is reasoneable.

            stats["text_length_after_truncation"] += document_size
            stats["examples"] += 1
            stats["tokens"] += document_size

            buffer.append(
                tf.train.Example(
                    features=tf.train.Features(
                        feature={"text": create_byte_feature(document.encode("utf-8"))}
                    )
                )
            )

            if len(buffer) % write_every_n_examples == 0:
                num_of_writes += 1
                sbar.set_description_str(
                    f"Number of writes: {num_of_writes}. Last write: Writing {len(buffer)} examples to file"
                )
                for example in buffer:
                    out_file.write(example.SerializeToString())

                buffer.clear()

            pbar.update(document_size)

    # Flush the final examples
    if buffer:
        for example in buffer:
            out_file.write(example.SerializeToString())

    stats["token2text_rate"] = stats["tokens"] / stats["text_length_after_truncation"]
    stats["dropped_tokens_length"] = stats["original_tokens_length"] - stats["tokens"]
    stats["dropped_text_length"] = (
        stats["original_text_length"] - stats["text_length_after_truncation"]
    )

    save_stats(output_path, stats)
    print_stats(stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--tokens",
        type=int,
        default=DATASET_SIZE_6B,
        help="Number of tokens to generate. Default is the same used for datasets with 6B.",
    )

    parser.add_argument(
        "--split", type=str, default="train", help="The split to generate"
    )

    parser.add_argument(
        "--output_path",
        type=str,
        help="The path of the file to be generated with the Non-Sense content",
    )

    args = parser.parse_args()

    create_nonsense_data(args.tokens, args.split, args.output_path)
