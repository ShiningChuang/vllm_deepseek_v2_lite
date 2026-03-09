#!/usr/bin/env python3
"""
Async concurrent client for vLLM OpenAI-compatible server.
Sends prompts from prompts.txt with configurable concurrency.

Usage:
  python3 run_client.py --concurrency 8 --total-requests 200 --output results.json
"""
import argparse
import asyncio
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List

import aiohttp

URL = os.environ.get("VLLM_URL", "http://localhost:8000/v1/completions")
MODEL = os.environ.get("VLLM_MODEL", "deepseek-ai/DeepSeek-V2-Lite")


def load_prompts(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    if not lines:
        raise ValueError(f"No prompts found in {path}")
    return lines


async def send_request(
    session: aiohttp.ClientSession, prompt: str, max_tokens: int
) -> Dict[str, Any]:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.2,
    }
    async with session.post(URL, json=payload) as resp:
        return await resp.json()


async def worker(
    name: str,
    queue: asyncio.Queue,
    sem: asyncio.Semaphore,
    session: aiohttp.ClientSession,
    results: List[Dict[str, Any]],
):
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            break
        prompt, max_tokens, req_id = item
        async with sem:
            t0 = time.perf_counter()
            try:
                out = await send_request(session, prompt, max_tokens)
                ok = True
                err = None
            except Exception as e:
                out = None
                ok = False
                err = repr(e)
            elapsed = time.perf_counter() - t0
        results.append(
            {
                "req_id": req_id,
                "worker": name,
                "latency_s": elapsed,
                "ok": ok,
                "err": err,
            }
        )
        if (req_id + 1) % 10 == 0:
            total = len(results)
            ok_cnt = sum(1 for r in results if r["ok"])
            print(
                f"  [{name}] {total} done, {ok_cnt} ok, last_lat={elapsed:.2f}s",
                flush=True,
            )
        queue.task_done()


async def run_load(
    prompts: List[str],
    concurrency: int,
    total_requests: int,
    qps: float,
) -> Dict[str, Any]:
    sem = asyncio.Semaphore(concurrency)
    queue: asyncio.Queue = asyncio.Queue()
    results: List[Dict[str, Any]] = []

    timeout = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        num_workers = max(concurrency * 2, 4)
        workers = [
            asyncio.create_task(
                worker(f"w{i}", queue, sem, session, results)
            )
            for i in range(num_workers)
        ]

        t0 = time.perf_counter()
        for i in range(total_requests):
            prompt = random.choice(prompts)
            max_tokens = random.choice([64, 128, 256])
            await queue.put((prompt, max_tokens, i))
            # Poisson inter-arrival if qps > 0
            if qps > 0:
                interval = random.expovariate(qps)
                await asyncio.sleep(interval)

        await queue.join()
        for _ in workers:
            await queue.put(None)
        await asyncio.gather(*workers)

    elapsed = time.perf_counter() - t0
    ok_cnt = sum(1 for r in results if r["ok"])
    fail_cnt = len(results) - ok_cnt
    lats = [r["latency_s"] for r in results]
    mean_lat = sum(lats) / max(len(lats), 1)
    lats_sorted = sorted(lats)
    p50 = lats_sorted[int(len(lats_sorted) * 0.50)] if lats_sorted else 0
    p95 = lats_sorted[int(len(lats_sorted) * 0.95)] if lats_sorted else 0

    return {
        "concurrency": concurrency,
        "total_requests": total_requests,
        "qps_target": qps,
        "elapsed_s": elapsed,
        "actual_rps": total_requests / elapsed,
        "completed": len(results),
        "ok": ok_cnt,
        "fail": fail_cnt,
        "mean_latency_s": mean_lat,
        "p50_latency_s": p50,
        "p95_latency_s": p95,
        "per_request": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", default="prompts.txt")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--total-requests", type=int, default=200)
    parser.add_argument(
        "--qps",
        type=float,
        default=0,
        help="Target QPS (Poisson). 0 = send as fast as concurrency allows.",
    )
    parser.add_argument("--output", default="client_results.json")
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducible prompt/max_tokens selection.",
    )
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    prompts = load_prompts(args.prompts)
    print(
        f"[run_client] concurrency={args.concurrency} "
        f"total={args.total_requests} qps={args.qps or 'unlimited'} "
        f"prompts={len(prompts)}"
    )

    summary = asyncio.run(
        run_load(prompts, args.concurrency, args.total_requests, args.qps)
    )

    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)

    print(
        f"[done] elapsed={summary['elapsed_s']:.1f}s "
        f"rps={summary['actual_rps']:.2f} "
        f"ok={summary['ok']} fail={summary['fail']} "
        f"mean_lat={summary['mean_latency_s']:.2f}s "
        f"p50={summary['p50_latency_s']:.2f}s "
        f"p95={summary['p95_latency_s']:.2f}s"
    )


if __name__ == "__main__":
    main()
