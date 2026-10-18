"""Print teacher chats readably:  python tests/show.py out/pilot/dialogs.jsonl [n]"""
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
for d in rows[: int(sys.argv[2]) if len(sys.argv) > 2 else len(rows)]:
    print(f"\n=== seed {d['seed']} | {d['persona'][:50]} | {d['scenario'][:60]}")
    for m in d["messages"]:
        if m["role"] == "user":
            print(f"  you:  {m['content']}")
        elif m["role"] == "assistant":
            for c in m.get("tool_calls") or []:
                print(f"        [{c['name']} {c['arguments'].get('command') or c['arguments'].get('path')}]")
            if m.get("content"):
                print(f"  moth: {m['content']}")
        else:
            for r in m["results"]:
                print("        > " + r[:150].replace("\n", " | "))
