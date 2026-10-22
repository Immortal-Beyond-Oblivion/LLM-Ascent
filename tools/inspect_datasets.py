#!/usr/bin/env python3
"""Inspect JSONL and TXT datasets in a directory, including basic duplicate checks."""

from pathlib import Path
import hashlib
import json
import sys

CONTENT_KEYS = (
    "text", "sentence", "sentence1", "sentence2", "premise", "hypothesis",
    "context", "question", "answer", "answers", "article", "paragraph",
    "document", "review", "content", "body", "title", "summary",
)
META_KEYS = {
    "id", "idx", "label", "labels", "guid", "url", "timestamp", "date",
    "metadata", "source", "split", "version", "answer_start",
}
DUPLICATE_SAMPLE_LIMIT = 200_000
TEXT_SAMPLE_CHARS = 400


def collect_content_strings(value):
    """Extract likely natural-language content from a JSON record."""
    if isinstance(value, dict):
        preferred_keys = [key for key in CONTENT_KEYS if key in value]
        if preferred_keys:
            result = []
            for key in preferred_keys:
                result.extend(collect_content_strings(value[key]))
            return result

        result = []
        for key, child in value.items():
            if str(key).lower() not in META_KEYS:
                result.extend(collect_content_strings(child))
        return result

    if isinstance(value, list):
        result = []
        for child in value:
            result.extend(collect_content_strings(child))
        return result

    if isinstance(value, str):
        return [value]

    return []


def text_hash(text):
    # Used only for duplicate detection, not for security.
    return hashlib.blake2b(text.strip().encode("utf-8"), digest_size=16).digest()


def print_common_summary(path, file_size, records, empty, chars, samples,
                         hashes_checked, duplicate_count):
    print(f"\n{'=' * 80}")
    print(f"FILE: {path}")
    print(f"Size: {file_size / (1024 ** 2):,.2f} MiB ({file_size:,} bytes)")
    print(f"Records/documents: {records:,}")
    print(f"Empty text records: {empty:,} ({empty / max(records, 1):.2%})")
    print(f"Content characters: {chars:,} ({chars / 1e6:.2f} million)")
    print(f"Estimated tokens: {chars / 4 / 1e6:.2f} million (~4 chars/token; rough estimate)")
    print(
        f"Exact duplicate rate (first {hashes_checked:,} non-empty text records): "
        f"{duplicate_count / max(hashes_checked, 1):.4%} "
        f"({duplicate_count:,} duplicates)"
    )

    for index, sample in enumerate(samples, start=1):
        preview = sample[:TEXT_SAMPLE_CHARS].replace("\n", " ").replace("\r", " ")
        print(f"\n--- Sample {index} (first {TEXT_SAMPLE_CHARS} chars max) ---")
        print(preview if preview else "(empty)")


def inspect_jsonl(path):
    records = 0
    empty = 0
    chars = 0
    blank_lines = 0
    parse_errors = 0
    keys = set()
    samples = []
    seen_hashes = set()
    hashes_checked = 0
    duplicate_count = 0

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                blank_lines += 1
                continue

            try:
                record = json.loads(raw_line)
            except Exception as exc:
                parse_errors += 1
                if parse_errors <= 3:
                    print(f"Warning: invalid JSON at {path}:{line_no}: {exc}")
                continue

            records += 1
            if isinstance(record, dict):
                keys.update(record.keys())

            strings = collect_content_strings(record)
            chars += sum(len(text) for text in strings)
            nonempty_strings = [text.strip() for text in strings if text.strip()]

            if not nonempty_strings:
                empty += 1
                continue

            combined_text = "\n".join(nonempty_strings)
            if len(samples) < 3:
                samples.append(combined_text)

            if hashes_checked < DUPLICATE_SAMPLE_LIMIT:
                digest = text_hash(combined_text)
                if digest in seen_hashes:
                    duplicate_count += 1
                else:
                    seen_hashes.add(digest)
                hashes_checked += 1

    print_common_summary(
        path, path.stat().st_size, records, empty, chars, samples,
        hashes_checked, duplicate_count
    )
    content_fields = sorted(key for key in keys if key in CONTENT_KEYS)
    print(f"Top-level fields present: {sorted(keys)}")
    print(f"Recognized content fields: {content_fields or '(none; generic string extraction was used)'}")
    print(f"Invalid JSON lines: {parse_errors:,}")
    print(f"Blank physical lines: {blank_lines:,}")


def inspect_txt(path):
    total_lines = 0
    nonempty_records = 0
    empty = 0
    chars = 0
    samples = []
    seen_hashes = set()
    hashes_checked = 0
    duplicate_count = 0

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            total_lines += 1
            line = raw_line.rstrip("\r\n")
            chars += len(line)

            if not line.strip():
                empty += 1
                continue

            nonempty_records += 1
            clean_text = line.strip()
            if len(samples) < 3:
                samples.append(clean_text)

            if hashes_checked < DUPLICATE_SAMPLE_LIMIT:
                digest = text_hash(clean_text)
                if digest in seen_hashes:
                    duplicate_count += 1
                else:
                    seen_hashes.add(digest)
                hashes_checked += 1

    print_common_summary(
        path, path.stat().st_size, total_lines, empty, chars, samples,
        hashes_checked, duplicate_count
    )
    print(f"Non-empty text records: {nonempty_records:,}")
    print(f"Physical lines: {total_lines:,}")
    print("Record definition for duplicate checking: each non-empty line is one text record.")


def main():
    data_dir = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else Path("data/v2")
    data_dir = data_dir.resolve()

    if not data_dir.is_dir():
        print(f"ERROR: Directory not found: {data_dir}")
        print("Run this script from the project root, or pass the data directory as an argument.")
        sys.exit(1)

    files = sorted(
        path for path in data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".jsonl", ".txt"}
    )

    if not files:
        print(f"No .jsonl or .txt files found under: {data_dir}")
        sys.exit(1)

    print(f"Dataset directory: {data_dir}")
    print(f"Found {len(files)} .jsonl/.txt file(s):")
    for path in files:
        print(f"  - {path.relative_to(data_dir)}")

    for path in files:
        try:
            if path.suffix.lower() == ".jsonl":
                inspect_jsonl(path)
            elif path.suffix.lower() == ".txt":
                inspect_txt(path)
        except OSError as exc:
            print(f"\nERROR reading {path}: {exc}")

    print(f"\nDone. Inspected {len(files)} file(s).")


if __name__ == "__main__":
    main()
