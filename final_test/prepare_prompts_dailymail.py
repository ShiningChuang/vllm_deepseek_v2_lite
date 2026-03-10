#!/usr/bin/env python3
"""
Download CNN/DailyMail dataset and build summarization prompts.

Each prompt is:
  "Summarize the following news article in a few sentences.\n\nArticle:\n{article}\n\nSummary:"

Truncates articles so the prompt stays within a token budget.

Usage:
  python3 prepare_prompts_dailymail.py \
      --output dailymail_prompts.txt \
      --max-prompts 2000 \
      --max-chars 3000
"""
import argparse
import json
import os
import urllib.request
from pathlib import Path

# CNN/DailyMail 1.0.0 train split on HuggingFace datasets-server
# We use the raw parquet file directly (no datasets library needed).
HF_PARQUET_URL = (
    "https://huggingface.co/datasets/abisee/cnn_dailymail/resolve/main/"
    "1.0.0/train-00000-of-00003.parquet"
)
PARQUET_FILE = "cnn_dailymail_train.parquet"

INSTRUCTION = (
    "Summarize the following news article in a few sentences.\n\n"
    "Article:\n{article}\n\nSummary:"
)


def download_if_needed(out_dir: str) -> str:
    path = os.path.join(out_dir, PARQUET_FILE)
    if os.path.exists(path):
        print(f"[✓] Parquet already exists: {path}")
        return path
    print(f"[↓] Downloading CNN/DailyMail parquet to {path} ...")
    urllib.request.urlretrieve(HF_PARQUET_URL, path)
    print(f"[✓] Downloaded ({os.path.getsize(path)//1024//1024} MB)")
    return path


def load_parquet(path: str):
    """Load parquet using pyarrow (available in most ML environments)."""
    try:
        import pyarrow.parquet as pq
        table = pq.read_table(path, columns=["article"])
        return table.to_pydict()["article"]
    except ImportError:
        pass

    # Fallback: use pandas
    try:
        import pandas as pd
        df = pd.read_parquet(path, columns=["article"])
        return df["article"].tolist()
    except ImportError:
        raise RuntimeError(
            "Neither pyarrow nor pandas is installed. "
            "Run: pip install pyarrow"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="dailymail_prompts.txt")
    parser.add_argument("--max-prompts", type=int, default=2000)
    parser.add_argument(
        "--max-chars", type=int, default=3000,
        help="Max characters of article body (truncate if longer). "
             "~3000 chars ≈ 750 tokens, well within 8192 limit."
    )
    args = parser.parse_args()

    out_dir = str(Path(args.output).parent)
    parquet_path = download_if_needed(out_dir)

    articles = load_parquet(parquet_path)
    print(f"[✓] Loaded {len(articles)} articles from parquet.")

    prompts = []
    for art in articles:
        if not isinstance(art, str) or not art.strip():
            continue
        # Truncate to max_chars
        art_trunc = art[: args.max_chars].strip()
        # Remove incomplete trailing sentence
        last_period = max(art_trunc.rfind(". "), art_trunc.rfind(".\n"))
        if last_period > len(art_trunc) * 0.5:
            art_trunc = art_trunc[: last_period + 1]
        prompt = INSTRUCTION.format(article=art_trunc)
        prompt = prompt.replace("\n", " ")  # keep single-line for prompts.txt
        prompts.append(prompt)
        if len(prompts) >= args.max_prompts:
            break

    with open(args.output, "w", encoding="utf-8") as f:
        for p in prompts:
            f.write(p + "\n")

    avg_len = sum(len(p) for p in prompts) / max(len(prompts), 1)
    print(f"[✓] Wrote {len(prompts)} prompts → {args.output}")
    print(f"    avg prompt length: {avg_len:.0f} chars  "
          f"(≈ {avg_len/4:.0f} tokens)")


if __name__ == "__main__":
    main()
