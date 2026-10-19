"""Reward-hacking monitor for RL: after every quick eval, log what training optimises (the reward judge's mean reward on
the eval chats) next to what we actually want (the eval judge's on_gist, and the pairwise win rate vs final2 when it
is run), and flag divergence.

  mon = HackMonitor("runs/rl1/hack_monitor.jsonl", rise=0.15, window=3)
  rec = mon.update(step, summary)        # summary = moth.rl.evalrun.run_eval(..., tier="quick", reward_judge=...)
  if rec["flag"]: ...                    # stop / roll back / inspect

A flag fires when, over the last `window` evals, the reward judge's reward rose by >= `rise` while the eval judge's
on_gist (and the win rate vs final2, if logged) did not rise by more than its own MDE (points, from the summary):
flat or falling. Reasons are listed in the record. The log is append-only JSON lines (one record per eval).
"""
from __future__ import annotations

import json
import os


class HackMonitor:
    def __init__(self, path: str | None = None, rise: float = 0.15, window: int = 3):
        self.path, self.rise, self.window = path, rise, window
        self.hist: list[dict] = []
        if path and os.path.exists(path):
            self.hist = [json.loads(x) for x in open(path, encoding="utf-8") if x.strip()]

    @staticmethod
    def point(step: int, s: dict) -> dict:
        rj = (s.get("reward_judge_vs_eval_judge") or {}).get("reward") or {}
        og = (s.get("all") or {}).get("on_gist") or {}
        vf = ((s.get("pairwise") or {}).get("final2") or {}).get("win_rate") or {}
        return {"step": step, "reward_judge_reward": rj.get("mean"), "eval_on_gist": og.get("pct"),
                "eval_on_gist_mde": og.get("mde"), "vs_final2": vf.get("pct"), "vs_final2_mde": vf.get("mde"),
                "eval_reward": ((s.get("all") or {}).get("reward") or {}).get("mean")}

    def check(self) -> list[str]:
        if len(self.hist) < 2:
            return []
        a, b = self.hist[-min(self.window, len(self.hist))], self.hist[-1]
        if a["reward_judge_reward"] is None or b["reward_judge_reward"] is None:
            return []
        d_r = b["reward_judge_reward"] - a["reward_judge_reward"]
        if d_r < self.rise:
            return []
        why = []
        for k, mk in (("eval_on_gist", "eval_on_gist_mde"), ("vs_final2", "vs_final2_mde")):
            if a.get(k) is None or b.get(k) is None:
                continue
            d = b[k] - a[k]
            if d <= (b.get(mk) or 0.0):
                why.append(f"reward judge reward +{d_r:.3f} over steps {a['step']}->{b['step']} but {k} "
                           f"{'fell' if d < 0 else 'is flat'} ({d:+.2f} points, MDE {b.get(mk)})")
        return why

    def update(self, step: int, summary: dict) -> dict:
        rec = self.point(step, summary)
        self.hist.append(rec)
        why = self.check()
        rec["flag"], rec["why"] = bool(why), why
        if self.path:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        return rec
