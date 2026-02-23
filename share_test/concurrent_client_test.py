import asyncio
import aiohttp
import random
import time
import math
import os
from typing import List, Dict, Any

os.environ["RANK"] = "4"
URL = os.environ.get("URL", "http://localhost:8000/v1/completions")
MODEL = os.environ.get("MODEL", "deepseek-ai/DeepSeek-V2-Lite")

with open("prompts.txt", "r", encoding="utf-8") as f:
    PROMPTS = f.readlines()

def expovariate_delay(qps: float) -> float:
    """Sample inter-arrival time for a Poisson process: Exp(rate=qps)."""
    if qps <= 0:
        raise ValueError("qps must be > 0")
    # random.expovariate expects lambda (rate)
    return random.expovariate(qps)

async def send_request(session: aiohttp.ClientSession, prompt: str) -> Dict[str, Any]:
    payload = {
        "model": MODEL,
        "prompt": prompt.strip(),
        "max_tokens": random.choice([32, 128, 256]),
        "temperature": 0.2,
    }
    # Send POST request and parse JSON response asynchronously.
    async with session.post(URL, json=payload) as resp:
        # If server returns non-JSON on errors, this may throw; keep it simple.
        return await resp.json()

async def worker(
    name: str,
    queue: asyncio.Queue,
    sem: asyncio.Semaphore,
    session: aiohttp.ClientSession,
    results: List[Dict[str, Any]],
):
    """Consume arrivals from the queue and execute requests with correct concurrency limiting."""
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            break

        prompt, scheduled_ts = item
        # Concurrency limit must wrap the *actual request execution*.
        async with sem:
            t_start = time.time()
            try:
                out = await send_request(session, prompt)
                ok = True
                err = None
            except Exception as e:
                out = None
                ok = False
                err = repr(e)
            t_end = time.time()

        results.append(
            {
                "worker": name,
                "scheduled_ts": scheduled_ts,
                "start_ts": t_start,
                "end_ts": t_end,
                "latency_s": t_end - t_start,
                "ok": ok,
                "err": err,
            }
        )
        queue.task_done()

async def run_load_poisson(
    qps: float = 5.0,
    concurrency: int = 8,
    total_requests: int = 200,
):
    """
    Generate arrivals with Poisson process (Exp inter-arrival) and process them
    with a strict concurrency cap.
    """
    sem = asyncio.Semaphore(concurrency)
    queue: asyncio.Queue = asyncio.Queue()
    results: List[Dict[str, Any]] = []

    async with aiohttp.ClientSession() as session:
        # Start a few workers; concurrency is *still* governed by the semaphore.
        # Worker count can be >= concurrency; it just helps keep the pipeline busy.
        num_workers = max(concurrency, 1)
        workers = [
            asyncio.create_task(worker(f"w{i}", queue, sem, session, results))
            for i in range(num_workers)
        ]

        # Arrival generator (Poisson): push requests into queue with Exp gaps.
        t0 = time.time()
        for i in range(total_requests):
            prompt = random.choice(PROMPTS)
            scheduled_ts = time.time()
            await queue.put((prompt, scheduled_ts))
            await asyncio.sleep(expovariate_delay(qps))

        # Wait until all queued jobs are processed
        await queue.join()

        # Stop workers
        for _ in workers:
            await queue.put(None)
        await asyncio.gather(*workers)

    elapsed = time.time() - t0
    # Basic summary (optional but useful)
    oks = sum(1 for r in results if r["ok"])
    fails = len(results) - oks
    mean_lat = sum(r["latency_s"] for r in results) / max(len(results), 1)

    return {
        "qps": qps,
        "concurrency": concurrency,
        "total_requests": total_requests,
        "elapsed_s": elapsed,
        "completed": len(results),
        "ok": oks,
        "fail": fails,
        "mean_latency_s": mean_lat,
        "results": results,  # per-request records for p50/p95 etc.
    }

if __name__ == "__main__":
    t0 = time.time()
    summary = asyncio.run(run_load_poisson(qps=5.0, concurrency=8, total_requests=200))
    print("elapsed:", time.time() - t0)
    print(
        f"done={summary['completed']} ok={summary['ok']} fail={summary['fail']} "
        f"mean_lat={summary['mean_latency_s']:.3f}s "
        f"(qps={summary['qps']} conc={summary['concurrency']})"
    )