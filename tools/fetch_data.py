"""Fetch and verify the source corpora that `data/v2/` holds.

The corpora are downloads, not project source, so they are excluded from git
(see `.gitignore`). This module is the replacement: it documents where each
file came from, what shape it must have, and how to put it back.

    python -m tools.fetch_data --check      # verify what is on disk
    python -m tools.fetch_data --download    # rebuild from Hugging Face

The contract for each file is its **record count and field schema**, both of
which are checked on arrival. A SHA-256 of our copy is recorded as well and is
reported when it differs, but a hash mismatch is a warning rather than a
failure: a rebuilt file can differ byte-for-byte in key order or float
formatting while being the same data.

The ``--download`` path needs the optional ``datasets`` package, which is not
a project dependency:

    pip install datasets
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "v2"


@dataclass(frozen=True)
class Corpus:
    """One file in `data/v2/`, with the upstream it was built from."""

    filename: str
    description: str
    upstream: str
    #: Number of records. ``None`` for a plain-text file.
    records: int | None = None
    #: Fields every record must carry, in order.
    fields: tuple[str, ...] = ()
    #: SHA-256 of our copy. Informational.
    sha256: str | None = None
    #: HF dataset id / config / split, for the download path.
    hf: tuple[str, str, str] | None = None
    #: Extra notes for the reader.
    notes: str = ""
    #: For plain-text files: how many lines.
    lines: int | None = None
    _extra: tuple[str, ...] = field(default=(), repr=False)


CORPORA: tuple[Corpus, ...] = (
    Corpus(
        filename="c4_en_shard_0.jsonl",
        description="C4 English, shard 0 - the corpus every ladder rung trains on",
        upstream="https://huggingface.co/datasets/allenai/c4 (en, train, shard 0)",
        records=356_317,
        fields=("text", "timestamp", "url"),
        sha256="128458c4a88f8aa209bb7db6bf6056e1ab245efd1f2101fc4ffd21b31d9a91d7",
        hf=("allenai/c4", "en", "train"),
        notes=(
            "One shard of the 356,317-document English C4 split. "
            "src/scaling/preprocess.py tokenises this into "
            "artifacts/scaling/corpus_tokens.mmap, which is what the ladder reads."
        ),
    ),
    Corpus(
        filename="glue_mnli_train.jsonl",
        description="GLUE MNLI train - the 300-item entailment set behind Act 6c",
        upstream="https://huggingface.co/datasets/nyu-mll/glue (mnli, train)",
        records=392_702,
        fields=("premise", "hypothesis", "label", "idx"),
        sha256="418263b02725939e30e5da273b81ab40d4b975cde38c4dd4fb3fd594374cc3e9",
        hf=("nyu-mll/glue", "mnli", "train"),
        notes="Label 0 = entailment, 1 = neutral, 2 = contradiction.",
    ),
    Corpus(
        filename="glue_cola_train.jsonl",
        description="GLUE CoLA train - grammaticality judgements",
        upstream="https://huggingface.co/datasets/nyu-mll/glue (cola, train)",
        records=8_551,
        fields=("sentence", "label", "idx"),
        sha256="87edb36afb84ac8c1028fb88fc9b2b59faba12fe0b9fddcb70c6edc7c1a414e0",
        hf=("nyu-mll/glue", "cola", "train"),
    ),
    Corpus(
        filename="glue_sst2_train.jsonl",
        description="GLUE SST-2 train - sentiment",
        upstream="https://huggingface.co/datasets/nyu-mll/glue (sst2, train)",
        records=67_349,
        fields=("sentence", "label", "idx"),
        sha256="911d50b17508f9d89535beca38dc78f990b80e027c10fe6bc057764bfc48f0a0",
        hf=("nyu-mll/glue", "sst2", "train"),
    ),
    Corpus(
        filename="squad_v1.1_train.jsonl",
        description="SQuAD v1.1 train - the reading-comprehension pool",
        upstream="https://huggingface.co/datasets/rajpurkar/squad (train)",
        records=87_599,
        fields=("id", "title", "context", "question", "answers"),
        sha256="0d6ca6ddd83d8f1358cc4d00374cc93d110fdfc1dea658d672ddb0b352474ebc",
        hf=("rajpurkar/squad", None, "train"),
    ),
    Corpus(
        filename="indiccorp_hi_sample.txt",
        description="IndicCorp Hindi sample - non-English text, one document per line",
        upstream="IndicCorpV2 (ai4bharat), Hindi - a local sample; the exact upstream "
        "subset is not pinned",
        lines=277_749,
        sha256="51f7991e2bb720a162e28136018f9bb2b467cd5b2b6cfa86ca234f4c94de5cd6",
        notes=(
            "Plain text, newline-separated documents, 82 MB. The upstream subset is not "
            "pinned, so this file is reproducible only in shape, not byte-for-byte. "
            "Nothing in the published results depends on it."
        ),
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_corpus(corpus: Corpus) -> list[str]:
    """Return a list of problems with the on-disk copy. Empty means it is fine."""
    path = DATA_DIR / corpus.filename
    if not path.exists():
        return [f"{corpus.filename}: missing"]
    problems: list[str] = []

    if corpus.records is not None:
        count = 0
        bad_schema: str | None = None
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                count += 1
                if bad_schema is None:
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        bad_schema = f"line {count} is not valid JSON"
                        continue
                    keys = tuple(record.keys())
                    if keys != corpus.fields:
                        bad_schema = (
                            f"line {count} has fields {keys}, expected {corpus.fields}"
                        )
        if count != corpus.records:
            problems.append(
                f"{corpus.filename}: {count:,} records, expected {corpus.records:,}"
            )
        if bad_schema is not None:
            problems.append(f"{corpus.filename}: {bad_schema}")
    elif corpus.lines is not None:
        with path.open(encoding="utf-8") as handle:
            count = sum(1 for line in handle if line.strip())
        if count != corpus.lines:
            problems.append(
                f"{corpus.filename}: {count:,} lines, expected {corpus.lines:,}"
            )

    if corpus.sha256 is not None:
        actual = _sha256(path)
        if actual != corpus.sha256:
            problems.append(
                f"{corpus.filename}: sha256 {actual[:16]}..., recorded {corpus.sha256[:16]}... "
                "(shape is still checked above; a rebuild may differ in key order)"
            )
    return problems


def download_corpus(corpus: Corpus) -> None:
    if corpus.hf is None:
        raise SystemExit(
            f"{corpus.filename} has no pinned upstream; restore it from your own copy"
        )
    try:
        from datasets import load_dataset
    except ModuleNotFoundError:
        raise SystemExit(
            "the 'datasets' package is required for --download:\n"
            "    pip install datasets"
        )
    dataset_id, config, split = corpus.hf
    print(f"[fetch] {corpus.filename} <- {dataset_id}"
          f"{'/' + config if config else ''} [{split}]")
    dataset = load_dataset(dataset_id, config, split=split)
    if corpus.filename.startswith("c4_en_shard"):
        # C4 publishes one file per shard; the JSONL conversion keeps
        # text/timestamp/url in that order.
        dataset = dataset.select(range(min(corpus.records or len(dataset), len(dataset))))
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with (DATA_DIR / corpus.filename).open("w", encoding="utf-8") as handle:
        for record in dataset:
            if corpus.fields:
                ordered = {key: record[key] for key in corpus.fields}
            else:
                ordered = record
            handle.write(json.dumps(ordered, ensure_ascii=False) + "\n")
    print(f"[fetch] wrote {corpus.filename}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="verify the corpora on disk (default)")
    parser.add_argument("--download", action="store_true",
                        help="rebuild the corpora from Hugging Face")
    args = parser.parse_args()

    if args.download:
        for corpus in CORPORA:
            download_corpus(corpus)

    print(f"data/v2 -> {DATA_DIR}")
    failures = 0
    for corpus in CORPORA:
        problems = check_corpus(corpus)
        size = ""
        path = DATA_DIR / corpus.filename
        if path.exists():
            size = f"{path.stat().st_size / 1e6:8.1f} MB"
        status = "ok" if not problems else "PROBLEM"
        print(f"  [{status:7}] {size}  {corpus.filename:32} {corpus.description}")
        for problem in problems:
            print(f"             - {problem}")
        failures += bool(problems)
    print(f"\n{len(CORPORA) - failures}/{len(CORPORA)} corpora verified.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
