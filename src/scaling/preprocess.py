"""C4 pre-tokenization: fit tokenizer on 10M-char sample, tokenize full corpus.

Produces:
- A uint16 memmapped token array (the training corpus)
- A manifest JSON with SHA-256, byte counts, doc counts, chars/token
- A frozen tokenizer checkpoint

The memmapped array can be loaded instantly without re-tokenizing.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from src.tokenizer.bpe import BPETokenizer
from src.scaling.fast_bpe import train_bpe_fast


def read_c4_sample(path: str, max_chars: int = 10_000_000) -> str:
    """Read up to max_chars characters from a C4 JSONL file."""
    chunks = []
    total = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = doc.get("text", "")
            if not text:
                continue
            chunks.append(text)
            total += len(text)
            if total >= max_chars:
                break
    return "\n".join(chunks)


def read_c4_full(path: str) -> tuple[str, int, int]:
    """Read the full C4 corpus. Returns (text, doc_count, byte_count)."""
    chunks = []
    doc_count = 0
    byte_count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            byte_count += len(line.encode("utf-8"))
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = doc.get("text", "")
            if not text:
                continue
            chunks.append(text)
            doc_count += 1
    return "\n".join(chunks), doc_count, byte_count


def tokenize_corpus(
    input_path: str = "data/v2/c4_en_shard_0.jsonl",
    output_dir: str = "artifacts/scaling",
    sample_chars: int = 10_000_000,
    target_vocab_size: int = 4_000,
    max_docs: int | None = None,
) -> dict:
    """Fit tokenizer on sample, tokenize full corpus, save memmap + manifest."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()

    # Step 1: Read sample and fit tokenizer
    print(f"[preprocess] Reading {sample_chars:,} char sample ...")
    sample_text = read_c4_sample(input_path, max_chars=sample_chars)
    print(f"[preprocess] Sample: {len(sample_text):,} chars")

    print(f"[preprocess] Fitting BPE (fast) with target vocab={target_vocab_size:,} on {len(sample_text):,} chars ...")
    vocab, merges = train_bpe_fast(sample_text, max_merges=target_vocab_size)
    tokenizer = BPETokenizer(tuple(vocab), [tuple(m) for m in merges])
    print(f"[preprocess] Tokenizer fitted: vocab_size={tokenizer.vocabulary_size:,}, merges={len(tokenizer.merges)}")

    # Save tokenizer
    tokenizer.save(output_dir / "tokenizer.json")

    # Step 2: Read full corpus
    print(f"[preprocess] Reading full corpus ...")
    full_text, doc_count, byte_count = read_c4_full(input_path)
    print(f"[preprocess] Full corpus: {len(full_text):,} chars, {doc_count:,} docs, {byte_count:,} bytes")

    # Step 3: Tokenize full corpus
    print(f"[preprocess] Tokenizing full corpus ...")
    t1 = time.perf_counter()
    token_ids = tokenizer.encode(full_text)
    t2 = time.perf_counter()
    print(f"[preprocess] Tokenized {len(token_ids):,} tokens in {t2 - t1:.1f}s")

    chars_per_token = len(full_text) / len(token_ids)
    print(f"[preprocess] Chars/token: {chars_per_token:.4f}")

    # Step 4: Save as uint16 memmap
    print(f"[preprocess] Saving memmap ...")
    token_array = np.array(token_ids, dtype=np.uint16)
    mmap_path = output_dir / "corpus_tokens.mmap"
    mmap = np.memmap(mmap_path, dtype=np.uint16, mode="w+", shape=token_array.shape)
    mmap[:] = token_array[:]
    mmap.flush()
    del mmap

    # Step 5: Compute SHA-256 of source
    print(f"[preprocess] Computing SHA-256 of source ...")
    sha256 = hashlib.sha256()
    with open(input_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            sha256.update(chunk)
    source_hash = sha256.hexdigest()

    # Step 6: Write manifest
    manifest = {
        "source_url": "https://huggingface.co/datasets/allenai/c4",
        "source_file": input_path,
        "source_sha256": source_hash,
        "source_bytes": byte_count,
        "source_doc_count": doc_count,
        "source_char_count": len(full_text),
        "tokenizer_type": "bpe",
        "tokenizer_vocab_size": tokenizer.vocabulary_size,
        "tokenizer_fit_sample_chars": len(sample_text),
        "total_tokens": len(token_ids),
        "chars_per_token": chars_per_token,
        "token_dtype": "uint16",
        "memmap_path": str(mmap_path),
        "manifest_version": "1.0",
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[preprocess] Manifest written to {manifest_path}")

    elapsed = time.perf_counter() - t0
    print(f"[preprocess] Done in {elapsed:.1f}s")

    return manifest


def load_corpus_memmap(manifest_path: str = "artifacts/scaling/manifest.json") -> np.memmap:
    """Load the tokenized corpus as a memory-mapped array."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    return np.memmap(manifest["memmap_path"], dtype=np.uint16, mode="r")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="C4 pre-tokenization")
    parser.add_argument("--input", default="data/v2/c4_en_shard_0.jsonl")
    parser.add_argument("--output", default="artifacts/scaling")
    parser.add_argument("--sample-chars", type=int, default=10_000_000)
    parser.add_argument("--target-vocab", type=int, default=16_000)
    args = parser.parse_args()

    tokenize_corpus(
        input_path=args.input,
        output_dir=args.output,
        sample_chars=args.sample_chars,
        target_vocab_size=args.target_vocab,
    )
