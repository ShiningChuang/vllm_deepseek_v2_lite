import json
import statistics
from tqdm import tqdm

def compute_goodput(jsonl_path):
    goodputs = []
    bad_rows = 0

    with open(jsonl_path, "r") as f:
        for line in tqdm(f, desc="Processing"):
            line = line.strip()
            if not line:
                continue

            rec = json.loads(line)

            if rec.get("where") != "after_align_before_gemm1":
                continue

            tpe = rec["tokens_per_expert_assign"]
            tokens_real = sum(tpe)
            tokens_post_padded = rec["num_tokens_post_padded"]

            if tokens_post_padded <= 0:
                continue

            goodput = tokens_real / tokens_post_padded
            goodputs.append(goodput)

            # sanity check
            expected = rec["tokens_in_chunk"] * rec["topk"]
            if tokens_real != expected:
                bad_rows += 1

    if not goodputs:
        print("No valid records found.")
        return

    avg = sum(goodputs) / len(goodputs)
    p50 = statistics.quantiles(goodputs, n=100)[49]
    p90 = statistics.quantiles(goodputs, n=100)[89]
    p99 = statistics.quantiles(goodputs, n=100)[98]

    print(f"Total samples: {len(goodputs)}")
    print(f"Avg padding goodput: {avg:.4f}")
    print(f"P50: {p50:.4f}, P90: {p90:.4f}, P99: {p99:.4f}")
    print(f"Bad rows (sanity check failed): {bad_rows}")

    return goodputs


if __name__ == "__main__":
    path = "/data/lxzhong_home/result/moe_shapes_rank3.jsonl"
    compute_goodput(path)