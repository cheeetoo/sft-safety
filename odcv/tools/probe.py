#!/usr/bin/env python3
"""Probe whether a vLLM endpoint batches concurrent requests or serializes them."""
import json, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

base, key, model = sys.argv[1], sys.argv[2], sys.argv[3]

def req(seed):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": f"Write a detailed essay about topic number {seed}: the history of bridges."}],
        "max_tokens": 400,
        "temperature": 0.0,
    }).encode()
    r = urllib.request.Request(base + "/chat/completions", data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(r, timeout=600) as resp:
        data = json.load(resp)
    dt = time.time() - t0
    return dt, data["usage"]["completion_tokens"]

# warm-up (not timed against)
req(0)
solo_t, solo_tok = req(1)
print(f"solo: {solo_t:.1f}s for {solo_tok} tokens")

t0 = time.time()
with ThreadPoolExecutor(4) as ex:
    results = list(ex.map(req, [2, 3, 4, 5]))
wall = time.time() - t0
times = [f"{t:.1f}s" for t, _ in results]
print(f"4-concurrent: individual {times}, wall {wall:.1f}s")
print(f"ratio wall/solo: {wall/solo_t:.2f}  (≈1 = perfect batching, ≈4 = serialized)")
