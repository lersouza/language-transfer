"""
This script computes multiple distances between language A and language B.

The input is a file containing:

- init_language: The language A. Usually a language used to pretrain a model.
- target: The language B. Usually the language to transfer to.

The output file is the same as the input, with additional columns with the computations for each language.
"""

import argparse
from functools import partial
import lang2vec.lang2vec as l2v
import pandas as pd

from pathlib import Path


DEFAULT_DISTANCES = [
    "syntactic",
    "geographic",
    "phonological",
    "genetic",
    "inventory",
    "featural",
]

# Imported from http://www-01.sil.org/iso639-3/download.asp on July 12, 2014
ISO_639_1_TO_3 = {
    "aa": "aar",
    "ab": "abk",
    "ae": "ave",
    "af": "afr",
    "ak": "aka",
    "am": "amh",
    "an": "arg",
    "ar": "ara",
    "as": "asm",
    "av": "ava",
    "ay": "aym",
    "az": "aze",
    "ba": "bak",
    "be": "bel",
    "bg": "bul",
    "bi": "bis",
    "bm": "bam",
    "bn": "ben",
    "bo": "bod",
    "br": "bre",
    "bs": "bos",
    "ca": "cat",
    "ce": "che",
    "ch": "cha",
    "co": "cos",
    "cr": "cre",
    "cs": "ces",
    "cu": "chu",
    "cv": "chv",
    "cy": "cym",
    "da": "dan",
    "de": "deu",
    "dv": "div",
    "dz": "dzo",
    "ee": "ewe",
    "el": "ell",
    "en": "eng",
    "eo": "epo",
    "es": "spa",
    "et": "est",
    "eu": "eus",
    "fa": "fas",
    "ff": "ful",
    "fi": "fin",
    "fj": "fij",
    "fo": "fao",
    "fr": "fra",
    "fy": "fry",
    "ga": "gle",
    "gd": "gla",
    "gl": "glg",
    "gn": "grn",
    "gu": "guj",
    "gv": "glv",
    "ha": "hau",
    "he": "heb",
    "hi": "hin",
    "ho": "hmo",
    "hr": "hrv",
    "ht": "hat",
    "hu": "hun",
    "hy": "hye",
    "hz": "her",
    "ia": "ina",
    "id": "ind",
    "ie": "ile",
    "ig": "ibo",
    "ii": "iii",
    "ik": "ipk",
    "io": "ido",
    "is": "isl",
    "it": "ita",
    "iu": "iku",
    "ja": "jpn",
    "jv": "jav",
    "ka": "kat",
    "kg": "kon",
    "ki": "kik",
    "kj": "kua",
    "kk": "kaz",
    "kl": "kal",
    "km": "khm",
    "kn": "kan",
    "ko": "kor",
    "kr": "kau",
    "ks": "kas",
    "ku": "kur",
    "kv": "kom",
    "kw": "cor",
    "ky": "kir",
    "la": "lat",
    "lb": "ltz",
    "lg": "lug",
    "li": "lim",
    "ln": "lin",
    "lo": "lao",
    "lt": "lit",
    "lu": "lub",
    "lv": "lav",
    "mg": "mlg",
    "mh": "mah",
    "mi": "mri",
    "mk": "mkd",
    "ml": "mal",
    "mn": "mon",
    "mr": "mar",
    "ms": "msa",
    "mt": "mlt",
    "my": "mya",
    "na": "nau",
    "nb": "nob",
    "nd": "nde",
    "ne": "nep",
    "ng": "ndo",
    "nl": "nld",
    "nn": "nno",
    "no": "nor",
    "nr": "nbl",
    "nv": "nav",
    "ny": "nya",
    "oc": "oci",
    "oj": "oji",
    "om": "orm",
    "or": "ori",
    "os": "oss",
    "pa": "pan",
    "pi": "pli",
    "pl": "pol",
    "ps": "pus",
    "pt": "por",
    "qu": "que",
    "rm": "roh",
    "rn": "run",
    "ro": "ron",
    "ru": "rus",
    "rw": "kin",
    "sa": "san",
    "sc": "srd",
    "sd": "snd",
    "se": "sme",
    "sg": "sag",
    "sh": "hbs",
    "si": "sin",
    "sk": "slk",
    "sl": "slv",
    "sm": "smo",
    "sn": "sna",
    "so": "som",
    "sq": "sqi",
    "sr": "srp",
    "ss": "ssw",
    "st": "sot",
    "su": "sun",
    "sv": "swe",
    "sw": "swa",
    "ta": "tam",
    "te": "tel",
    "tg": "tgk",
    "th": "tha",
    "ti": "tir",
    "tk": "tuk",
    "tl": "tgl",
    "tn": "tsn",
    "to": "ton",
    "tr": "tur",
    "ts": "tso",
    "tt": "tat",
    "tw": "twi",
    "ty": "tah",
    "ug": "uig",
    "uk": "ukr",
    "ur": "urd",
    "uz": "uzb",
    "ve": "ven",
    "vi": "vie",
    "vo": "vol",
    "wa": "wln",
    "wo": "wol",
    "xh": "xho",
    "yi": "yid",
    "yo": "yor",
    "za": "zha",
    "zh": "zho",
    "zu": "zul",
}


def compute_distance_for_pair(row: pd.Series, distance_name: str):
    lang_a = ISO_639_1_TO_3.get(row["init_language"])
    lang_b = ISO_639_1_TO_3.get(row["target"])

    return l2v.distance(distance_name, lang_a, lang_b)


def compute_distance(pairs: pd.DataFrame, distance_name: str):
    compute_distance_fn = partial(
        compute_distance_for_pair, distance_name=distance_name
    )
    pairs[f"{distance_name}_distance"] = pairs.apply(compute_distance_fn, axis=1)


def calculate_language_distances(
    results_file: Path, output_file: Path, distances: list
):
    results = pd.read_csv(results_file)
    language_pairs = results[["init_language", "target"]].copy().drop_duplicates()

    # Compute all distances within the language_pairs data frame.
    [compute_distance(language_pairs, distance_name) for distance_name in distances]

    results = results.join(
        language_pairs.set_index(["init_language", "target"]),
        how="inner",
        on=["init_language", "target"],
        lsuffix=None,
        rsuffix=None,
    )

    results.to_csv(output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("results_file", type=Path)
    parser.add_argument("--distances", nargs="+", default=DEFAULT_DISTANCES)
    parser.add_argument(
        "--output_file", type=Path, default=Path(".") / "estimations_with_distances.csv"
    )

    args = parser.parse_args()
    calculate_language_distances(args.results_file, args.output_file, args.distances)
