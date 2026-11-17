"""Contamination risk audit: overlap between a dataset and a reference corpus (v0.5.0, #47).

If test questions appeared in a model's training data, its score is inflated. OpenBench cannot
see training data, but it can flag **risk**: records whose prompt or reference answer shares
long word sequences with a corpus you supply (a public benchmark, a forum dump, your own
fine-tuning data).

Method (standard library only, read-only):

* Text is normalised (Unicode NFKC, case-folded, punctuation removed, whitespace collapsed) and
  split into words.
* Every word n-gram (default ``n = 13``, the length commonly used in contamination studies) of
  the corpus is hashed into a set.
* For each record and field (``prompt``, ``expected_answer``): the share of the field's n-grams
  found in the corpus, and the longest run of consecutive shared n-grams (reported in words).
  Fields shorter than ``n`` words are checked with ``n`` reduced to their length when they have
  at least ``min_words`` words; shorter fields are skipped and counted.
* A record is flagged when either field's overlap reaches the threshold (default 0.5).

A low overlap does not prove a dataset is clean: paraphrases, translations and partial copies
below the n-gram length are not detected, and the corpus is only what you supplied.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from openbench.dataset import Dataset
from openbench.exceptions import ConfigurationError

DEFAULT_NGRAM = 13
DEFAULT_THRESHOLD = 0.5
DEFAULT_MIN_WORDS = 8
CORPUS_SUFFIXES = (".txt", ".md", ".jsonl", ".json", ".csv")
MAX_CORPUS_BYTES = 2 * 1024**3
DISCLAIMER = (
    "A low overlap does not prove the dataset is clean: paraphrases, translations, partial copies "
    "shorter than the n-gram length and anything outside the supplied corpus are not detected. A high "
    "overlap is a risk signal to review, not proof that a model saw the data."
)
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_WS = re.compile(r"\s+")


def normalise_words(text: str) -> list[str]:
    folded = unicodedata.normalize("NFKC", text).casefold()
    return _WS.sub(" ", _PUNCT.sub(" ", folded)).strip().split()


def _hash(words: tuple[str, ...]) -> bytes:
    return hashlib.blake2b(" ".join(words).encode("utf-8"), digest_size=8).digest()


def _ngrams(words: list[str], n: int) -> list[bytes]:
    return [_hash(tuple(words[i:i + n])) for i in range(len(words) - n + 1)]


def _strings_in_json(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _strings_in_json(v)
    elif isinstance(value, list):
        for v in value:
            yield from _strings_in_json(v)


def iter_corpus_texts(path: str | Path) -> Iterator[tuple[str, str]]:
    """Yield ``(file, text)`` for supported files under ``path`` (sorted, read-only)."""
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"Corpus path does not exist: {root}")
    files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in CORPUS_SUFFIXES)
    if not files:
        raise ConfigurationError(f"No corpus files ({', '.join(CORPUS_SUFFIXES)}) found under {root}")
    total = 0
    for f in files:
        total += f.stat().st_size
        if total > MAX_CORPUS_BYTES:
            raise ConfigurationError("Corpus exceeds 2 GiB; audit a sample or split it")
        text = f.read_text(encoding="utf-8-sig", errors="replace")
        suffix = f.suffix.lower()
        if suffix == ".jsonl":
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    yield str(f), "\n".join(_strings_in_json(json.loads(line)))
                except json.JSONDecodeError:
                    yield str(f), line
        elif suffix == ".json":
            try:
                yield str(f), "\n".join(_strings_in_json(json.loads(text)))
            except json.JSONDecodeError:
                yield str(f), text
        else:
            yield str(f), text


class NgramIndex:
    """Hashed n-gram sets of a corpus for every n that the audit needs."""

    def __init__(self, texts: Iterable[tuple[str, str]], sizes: Iterable[int]) -> None:
        self.sizes = sorted(set(sizes))
        self.sets: dict[int, set[bytes]] = {n: set() for n in self.sizes}
        self.files: set[str] = set()
        self.documents = 0
        self.words = 0
        for name, text in texts:
            words = normalise_words(text)
            self.files.add(name)
            self.documents += 1
            self.words += len(words)
            for n in self.sizes:
                self.sets[n].update(_ngrams(words, n))


def _field_overlap(words: list[str], n: int, index: NgramIndex) -> dict[str, Any]:
    grams = _ngrams(words, n)
    hits = [g in index.sets[n] for g in grams]
    longest = run = 0
    for h in hits:
        run = run + 1 if h else 0
        longest = max(longest, run)
    return {
        "words": len(words),
        "ngram": n,
        "ngrams": len(grams),
        "shared_ngrams": sum(hits),
        "overlap": (sum(hits) / len(grams)) if grams else 0.0,
        "longest_shared_span_words": (longest + n - 1) if longest else 0,
    }


def audit_contamination(
    dataset: Dataset,
    corpus_path: str | Path,
    *,
    ngram: int = DEFAULT_NGRAM,
    threshold: float = DEFAULT_THRESHOLD,
    min_words: int = DEFAULT_MIN_WORDS,
    fields: tuple[str, ...] = ("prompt", "expected_answer"),
) -> dict[str, Any]:
    if ngram < 2:
        raise ConfigurationError("ngram must be at least 2")
    if not 0.0 < threshold <= 1.0:
        raise ConfigurationError("threshold must lie in (0, 1]")
    if min_words < 2 or min_words > ngram:
        raise ConfigurationError("min_words must lie in [2, ngram]")
    texts = {}
    sizes = {ngram}
    for rec in dataset.records:
        for f in fields:
            words = normalise_words(getattr(rec, f) or "")
            texts[(rec.id, f)] = words
            if min_words <= len(words) < ngram:
                sizes.add(len(words))
    index = NgramIndex(iter_corpus_texts(corpus_path), sizes)
    records: list[dict[str, Any]] = []
    skipped = {f: 0 for f in fields}
    for rec in dataset.records:
        entry: dict[str, Any] = {"id": rec.id, "category": rec.category, "fields": {}}
        flagged_by = []
        for f in fields:
            words = texts[(rec.id, f)]
            if len(words) < min_words:
                skipped[f] += 1
                entry["fields"][f] = {"words": len(words), "skipped": "fewer_than_min_words"}
                continue
            res = _field_overlap(words, min(ngram, len(words)), index)
            entry["fields"][f] = res
            if res["overlap"] >= threshold:
                flagged_by.append(f)
        entry["flagged"] = bool(flagged_by)
        entry["flagged_by"] = flagged_by
        records.append(entry)
    flagged = [r for r in records if r["flagged"]]
    max_entry = max(
        ((r["id"], f, v["overlap"]) for r in records for f, v in r["fields"].items() if "overlap" in v),
        key=lambda t: (t[2], t[0]),
        default=None,
    )
    return {
        "schema_version": "0.5.0",
        "analysis": "contamination_audit",
        "dataset": {"name": dataset.name, "version": dataset.version, "checksum": dataset.checksum, "records": len(dataset)},
        "corpus": {"path": str(corpus_path), "files": len(index.files), "documents": index.documents, "words": index.words},
        "parameters": {"ngram": ngram, "threshold": threshold, "min_words": min_words, "fields": list(fields),
                       "normalisation": "NFKC, casefold, punctuation removed, whitespace collapsed; word n-grams"},
        "summary": {
            "records_flagged": len(flagged),
            "share_flagged": (len(flagged) / len(records)) if records else 0.0,
            "fields_skipped_too_short": skipped,
            "max_overlap": {"id": max_entry[0], "field": max_entry[1], "overlap": max_entry[2]} if max_entry else None,
        },
        "flagged_ids": [r["id"] for r in flagged],
        "records": records,
        "disclaimer": DISCLAIMER,
        "read_only": True,
    }


def render_contamination(payload: dict[str, Any], fmt_name: str) -> str:
    from openbench.reporting_v040 import _csv, _table, fmt, report_doc_markdown, to_json

    if fmt_name == "json":
        return to_json(payload)
    rows = []
    for r in payload["records"]:
        p = r["fields"].get("prompt", {})
        a = r["fields"].get("expected_answer", {})
        rows.append([r["id"], r["category"], fmt(p.get("overlap")), p.get("longest_shared_span_words", "n/a"),
                     fmt(a.get("overlap")), a.get("longest_shared_span_words", "n/a"), "yes" if r["flagged"] else "no"])
    headers = ["id", "category", "prompt_overlap", "prompt_longest_span_words", "answer_overlap", "answer_longest_span_words", "flagged"]
    if fmt_name == "csv":
        return _csv(headers, rows)
    s = payload["summary"]
    lines = [
        "# OpenBench AI contamination risk audit",
        "",
        f"> {payload['disclaimer']}",
        "",
        f"- Dataset: `{payload['dataset']['name']}` ({payload['dataset']['records']} records); corpus: {payload['corpus']['documents']} document(s), {payload['corpus']['words']} words",
        f"- Word {payload['parameters']['ngram']}-grams; flag threshold: overlap >= {payload['parameters']['threshold']}",
        f"- Flagged: **{s['records_flagged']}** record(s) ({100 * s['share_flagged']:.1f}%)",
        "",
    ]
    shown = [row for row in rows if row[-1] == "yes"] or rows[:20]
    lines += _table(["ID", "Category", "Prompt overlap", "Longest span (words)", "Answer overlap", "Longest span (words)", "Flagged"], shown)
    lines += ["", report_doc_markdown("contamination")]
    return "\n".join(lines) + "\n"
