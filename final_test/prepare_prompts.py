#!/usr/bin/env python3
"""
Download ShareGPT dataset and extract first-turn human prompts into prompts.txt.
Usage: python3 prepare_prompts.py [--output prompts.txt] [--max-prompts 1000]
"""
import argparse
import json
import os
import urllib.request
from pathlib import Path

SHAREGPT_URL = (
    "https://huggingface.co/datasets/anon8231489123/"
    "ShareGPT_Vicuna_unfiltered/resolve/main/"
    "ShareGPT_V3_unfiltered_cleaned_split.json"
)
DEFAULT_JSON = "ShareGPT_V3_unfiltered_cleaned_split.json"


def download_if_needed(json_path: str):
    if os.path.exists(json_path):
        print(f"[✓] Dataset already exists: {json_path}")
        return
    print(f"[↓] Downloading ShareGPT dataset to {json_path} ...")
    urllib.request.urlretrieve(SHAREGPT_URL, json_path)
    print(f"[✓] Downloaded.")


def extract_prompts(json_path: str, max_prompts: int) -> list:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    prompts = []
    for item in data:
        convs = item.get("conversations", [])
        for c in convs:
            if c.get("from") == "human":
                text = c["value"].replace("\n", " ").strip()
                if text:
                    prompts.append(text)
                break  # first human turn only
        if len(prompts) >= max_prompts:
            break
    return prompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="prompts.txt")
    parser.add_argument("--max-prompts", type=int, default=2000)
    parser.add_argument("--json", default=DEFAULT_JSON)
    args = parser.parse_args()

    out_dir = Path(args.output).parent
    json_path = str(out_dir / args.json) if not os.path.isabs(args.json) else args.json

    download_if_needed(json_path)
    prompts = extract_prompts(json_path, args.max_prompts)
    with open(args.output, "w", encoding="utf-8") as f:
        for p in prompts:
            f.write(p + "\n")
    print(f"[✓] Wrote {len(prompts)} prompts to {args.output}")


if __name__ == "__main__":
    main()
