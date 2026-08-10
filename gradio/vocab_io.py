"""Save/restore the three vocabularies that define OTRec's input encoding.

Why this exists: build_two_tower_model() re-adapts its TextVectorization and
StringLookup layers from whatever frame it is given. TextVectorization orders
its vocabulary by token frequency, so adapting on any frame other than the
original full training frame (e.g. the deduplicated df_learn_sub the Space
ships) produces the same-SIZED vocabulary in a different ORDER -- the trained
weights then load without error but are applied to permuted features, and
predictions degrade to chance. Serializing the vocabularies alongside the
weights removes the dependency on reproducing exact token frequencies.

Usage (serving):
    model = build_two_tower_model(df_learn_sub)      # any frame with the right columns
    apply_vocabularies(model, load_vocabularies(p))  # BEFORE load_weights
    model.load_weights(weights_path)

Usage (after training):
    save_vocabularies(extract_vocabularies(model), p)
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path


def _text_vectorizer(feature_space):
    return feature_space.preprocessors["text"]


def extract_vocabularies(model) -> dict[str, list[str]]:
    """Vocabularies WITHOUT special tokens ('[UNK]'/padding) -- exactly the
    form set_vocabulary() expects, so apply_vocabularies needs no slicing."""
    return {
        "disease_text": [str(t) for t in
                         _text_vectorizer(model.q_fs).get_vocabulary(include_special_tokens=False)],
        "target_text": [str(t) for t in
                        _text_vectorizer(model.k_fs).get_vocabulary(include_special_tokens=False)],
        "diseaseId": [str(t) for t in model.dise_lookup.get_vocabulary(include_special_tokens=False)],
    }


def save_vocabularies(vocabs: dict[str, list[str]], path: str | Path) -> None:
    with gzip.open(Path(path), "wt", encoding="utf-8") as fh:
        json.dump(vocabs, fh, ensure_ascii=False)


def load_vocabularies(path: str | Path) -> dict[str, list[str]]:
    with gzip.open(Path(path), "rt", encoding="utf-8") as fh:
        return json.load(fh)


def apply_vocabularies(model, vocabs: dict[str, list[str]]) -> None:
    """Overwrite the adapted vocabularies with the saved ones. Call after
    build_two_tower_model() and before load_weights().

    Vocabularies are stored WITHOUT special tokens (see extract_vocabularies),
    which is the exact form set_vocabulary() expects.
    """
    _text_vectorizer(model.q_fs).set_vocabulary(vocabs["disease_text"])
    _text_vectorizer(model.k_fs).set_vocabulary(vocabs["target_text"])
    model.dise_lookup.set_vocabulary(vocabs["diseaseId"])
