#!/usr/bin/env python3
"""Preflight: verify an OpenAI-compatible endpoint returns PARSED tool calls.

If the server dumps tool-call JSON into `content` instead of `tool_calls`,
the ODCV harness produces empty trajectories that fake a "safe" result —
this must pass before any run is launched.

Usage: toolcall_check.py <base_url> <api_key> <model>
Exit 0 = OK, exit 1 = endpoint unusable (details printed).
"""
import json, ssl, sys, urllib.error, urllib.request

base, key, model = sys.argv[1], sys.argv[2], sys.argv[3]

body = json.dumps({
    "model": model,
    "messages": [{"role": "user", "content": "run ls"}],
    "tools": [{"type": "function", "function": {"name": "bash", "parameters": {
        "type": "object", "properties": {"command": {"type": "string"}},
        "required": ["command"]}}}],
    "tool_choice": "auto",
}).encode()
req = urllib.request.Request(base + "/chat/completions", data=body,
    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})

try:
    with urllib.request.urlopen(req, timeout=420) as resp:
        r = json.load(resp)
except (ssl.SSLError, urllib.error.URLError) as e:
    print(f"FAIL: cannot reach {base}: {e}")
    if base.startswith("https://"):
        print("HINT: a TLS 'protocol version' alert usually means the pod serves "
              "plain HTTP — retry with http:// in the URL.")
    sys.exit(1)

msg = r["choices"][0]["message"]
tc = msg.get("tool_calls")
if not tc:
    print("FAIL: no parsed tool_calls in response — harness would record empty trajectories.")
    print("content head:", repr((msg.get("content") or "")[:300]))
    sys.exit(1)

print(f"OK: model={r.get('model')} finish_reason={r['choices'][0].get('finish_reason')}")
print("tool_calls:", json.dumps(tc)[:200])
