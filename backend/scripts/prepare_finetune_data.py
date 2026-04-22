"""
Convert the SheriaPlex FAQ corpus into a JSONL fine-tune dataset.

Each S3 key under ``faq-chunks/`` is transformed into one training record:

    {
      "instruction": "Answer the Kenyan legal question in a clear, layperson-friendly tone ...",
      "input": "<question_title> — <question_body>",
      "output": "<merged lawyer answers>"
    }

The JSONL is uploaded to::

    s3://{data_bucket}/models/haki-ai-finetune/dataset/train.jsonl

That matches the ``training_dataset_prefix`` output on the
``infra/modules/ml`` Terraform module. ``scripts/run-finetune.sh`` picks
up the same path and kicks off a SageMaker JumpStart qLoRA training job.

Usage
-----
    # Prod (pulls FAQ chunks from the real S3 bucket):
    uv run python -m scripts.prepare_finetune_data

    # Local smoke test (reads a directory of .txt + .txt.metadata.json
    # files, writes the JSONL to stdout without touching S3):
    uv run python -m scripts.prepare_finetune_data --local /tmp/sheriaplex-dryrun

The ``--local`` path lets the Phase 4a dry-run output flow straight into
a fine-tune dataset without a round-trip through S3, which is handy for
reviewing the training data in a PR.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from dataclasses import dataclass

from clients import make_s3
from app.config import load_config


FAQ_PREFIX = "faq-chunks/"
DATASET_KEY = "models/haki-ai-finetune/dataset/train.jsonl"

_INSTRUCTION = (
    "You are Haki AI, a Kenyan legal aid assistant. Answer the user's "
    "question about Kenyan law in clear, layperson-friendly language. "
    "Cite the Act, Chapter/Part and Section number when the relevant "
    "statute is known. Keep the response concise and accurate."
)


@dataclass
class FAQRecord:
    chunk_id: str
    title: str
    question: str
    answer: str
    category: str
    url: str


def _split_question(body: str) -> tuple[str, str]:
    """Pulls `Question: <title>\n<body>\n\nLawyer answers:\n<answer>`."""
    try:
        question_block, answer_block = body.split("\nLawyer answers:\n", 1)
    except ValueError:
        return body.strip(), ""
    title = question_block.removeprefix("Question:").strip()
    question = title
    if "\n" in title:
        title_line, rest = title.split("\n", 1)
        question = f"{title_line.strip()} — {rest.strip()}" if rest.strip() else title_line.strip()
    return question, answer_block.strip()


def _to_record(chunk_id: str, text: str, metadata: dict) -> FAQRecord | None:
    question, answer = _split_question(text)
    if not question or not answer or answer == "(no answers yet)":
        return None
    return FAQRecord(
        chunk_id=chunk_id,
        title=metadata.get("section") or question,
        question=question,
        answer=answer,
        category=metadata.get("category", "general"),
        url=metadata.get("url", ""),
    )


def _record_to_json(rec: FAQRecord) -> dict:
    return {
        "instruction": _INSTRUCTION,
        "input": rec.question,
        "output": rec.answer,
        "metadata": {"category": rec.category, "url": rec.url, "chunkId": rec.chunk_id},
    }


# ── S3 loader ─────────────────────────────────────────────────────────────────

def _load_from_s3(bucket: str) -> list[FAQRecord]:
    config = load_config()
    s3 = make_s3(config)
    records: list[FAQRecord] = []
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=FAQ_PREFIX)
    for page in pages:
        for obj in page.get("Contents", []) or []:
            key = obj["Key"]
            if not key.endswith(".txt"):
                continue
            meta_obj = s3.get_object(Bucket=bucket, Key=f"{key}.metadata.json")
            meta_body = meta_obj["Body"].read().decode("utf-8")
            metadata = json.loads(meta_body).get("metadataAttributes", {})
            text_obj = s3.get_object(Bucket=bucket, Key=key)
            text = text_obj["Body"].read().decode("utf-8")
            chunk_id = metadata.get("chunkId") or os.path.splitext(os.path.basename(key))[0]
            record = _to_record(chunk_id, text, metadata)
            if record:
                records.append(record)
    return records


def _upload_to_s3(bucket: str, records: list[FAQRecord]) -> str:
    config = load_config()
    s3 = make_s3(config)
    buf = io.StringIO()
    for rec in records:
        buf.write(json.dumps(_record_to_json(rec), ensure_ascii=False))
        buf.write("\n")
    body = buf.getvalue().encode("utf-8")
    s3.put_object(Bucket=bucket, Key=DATASET_KEY, Body=body,
                  ContentType="application/jsonl")
    return f"s3://{bucket}/{DATASET_KEY}"


# ── Local (filesystem) loader for dry-run data ───────────────────────────────

def _load_from_dir(path: str) -> list[FAQRecord]:
    records: list[FAQRecord] = []
    for name in os.listdir(path):
        if not name.endswith(".txt") or name.endswith(".metadata.json"):
            continue
        text_path = os.path.join(path, name)
        meta_path = f"{text_path}.metadata.json"
        if not os.path.exists(meta_path):
            continue
        with open(text_path, "r", encoding="utf-8") as f:
            text = f.read()
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f).get("metadataAttributes", {})
        chunk_id = metadata.get("chunkId") or os.path.splitext(name)[0]
        record = _to_record(chunk_id, text, metadata)
        if record:
            records.append(record)
    return records


# ── Entrypoint ────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local", metavar="DIR", help="Read chunks from a local directory (dry-run output).")
    parser.add_argument("--no-upload", action="store_true", help="Print JSONL to stdout instead of uploading.")
    parser.add_argument("--bucket", help="Override the data bucket (defaults to config.s3_bucket).")
    args = parser.parse_args(argv)

    config = load_config()
    bucket = args.bucket or config.s3_bucket

    if args.local:
        print(f"[finetune-prep] reading chunks from {args.local}")
        records = _load_from_dir(args.local)
    else:
        print(f"[finetune-prep] listing s3://{bucket}/{FAQ_PREFIX} …")
        records = _load_from_s3(bucket)

    print(f"[finetune-prep] built {len(records)} training records")
    if not records:
        print("[finetune-prep] no records produced; is the FAQ corpus populated?")
        return 1

    if args.no_upload or args.local:
        for rec in records:
            print(json.dumps(_record_to_json(rec), ensure_ascii=False))
        return 0

    uri = _upload_to_s3(bucket, records)
    print(f"[finetune-prep] uploaded → {uri}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
