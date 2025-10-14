from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import httpx


def main() -> None:
    p = argparse.ArgumentParser(description="Minimal SSE client for /execute/stream")
    p.add_argument("url", nargs="?", default="http://localhost:8000/execute/stream")
    p.add_argument("config_path", nargs="?", default="examples/configs/single_agent_with_tool.json")
    p.add_argument("text", nargs="?", default="hello")
    args = p.parse_args()

    with open(args.config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    body: dict[str, Any] = {"config": cfg, "input": {"text": args.text}}

    with httpx.Client(timeout=None) as client:
        with client.stream("POST", args.url, json=body) as resp:
            if resp.status_code != 200:
                print(f"HTTP {resp.status_code}", file=sys.stderr)
                print(resp.text, file=sys.stderr)
                sys.exit(2)
            for line in resp.iter_lines():
                if not line:
                    continue
                s = line.decode("utf-8") if isinstance(line, (bytes, bytearray)) else str(line)
                if s.startswith("data: "):
                    payload = s[len("data: ") :]
                    try:
                        obj = json.loads(payload)
                    except Exception:
                        print(s)
                        continue
                    print(f"event: {obj.get('type')} -> {json.dumps(obj, ensure_ascii=False)}")


if __name__ == "__main__":
    main()

