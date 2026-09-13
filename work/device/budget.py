#!/usr/bin/env python3
"""Estimate how much of the 5-hour Claude Code quota this session has burned.

Written 2026-08-23 (session 7) after the standing request to "pause at ~85%" could not
be answered any other way: there is no `claude usage` subcommand, `/usage` is
interactive-only and not callable as a tool, and `~/.claude/stats-cache.json` holds only
stale daily counts.

## What this is NOT

It is **not** context-window usage. Those are different resources and conflating them is
the trap: a `PreToolUse` hook reading `CLAUDE_CONTEXT_TOKEN_COUNT` would report how full
the *context* is, which in this session sits near zero while the *quota* is at 50%.
(Those two env vars also simply do not exist -- verified with `env` in this harness; the
real ones are `CLAUDE_CODE_SESSION_ID`, `CLAUDE_PID`, ...)

## What it is

Claude Code appends every assistant message to a transcript at
`~/.claude/projects/<slug>/<session>.jsonl`, and each carries a real `usage` block:
`input_tokens`, `output_tokens`, `cache_creation_input_tokens`, `cache_read_input_tokens`.
Summing those over the current rolling window gives a quantity proportional to quota
burn. The constant of proportionality is unknown -- the server weights models and token
kinds differently -- so it is CALIBRATED against one observed `/usage` reading.

Anchor from this session: at 04:42 local, `/usage` reported **50%** with the window
resetting **08:19**, i.e. a window that opened at **03:19**.

Re-anchor whenever a fresh `/usage` reading is available; the estimate is only as good as
its most recent anchor.

  usage: py -3.10 work/device/budget.py [--anchor-pct 50 --anchor-time 04:42]
"""
import argparse
import glob
import json
import os
from datetime import datetime, timedelta, timezone

PROJECTS = os.path.expanduser("~/.claude/projects")


def load(window_start, window_end):
    """Sum usage across every transcript touching the window."""
    tot = dict(inp=0, out=0, cc=0, cr=0, msgs=0)
    series = []
    for f in glob.glob(os.path.join(PROJECTS, "*", "*.jsonl")):
        if datetime.fromtimestamp(os.path.getmtime(f), timezone.utc) < window_start:
            continue
        with open(f, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                ts = d.get("timestamp")
                m = d.get("message")
                if not ts or not isinstance(m, dict):
                    continue
                u = m.get("usage")
                if not isinstance(u, dict):
                    continue
                try:
                    t = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                except ValueError:
                    continue
                if not (window_start <= t <= window_end):
                    continue
                tot["inp"] += u.get("input_tokens", 0) or 0
                tot["out"] += u.get("output_tokens", 0) or 0
                tot["cc"] += u.get("cache_creation_input_tokens", 0) or 0
                tot["cr"] += u.get("cache_read_input_tokens", 0) or 0
                tot["msgs"] += 1
                series.append((t, u.get("output_tokens", 0) or 0))
    return tot, sorted(series)


def weighted(t):
    """A single scalar standing in for quota burn.

    Cache READS are the cheapest thing on the API price list and dominate the raw counts
    by ~100x here, so counting them equally would make the number all-but-constant and
    useless. Output is the most expensive. These weights mirror the published price
    ratios (output 5x input, cache write 1.25x input, cache read 0.1x input) -- they are
    a guess at the quota formula, which is why the result is calibrated, not absolute.
    """
    return t["out"] * 5.0 + t["inp"] * 1.0 + t["cc"] * 1.25 + t["cr"] * 0.1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", default="08:19", help="local time the window resets")
    ap.add_argument("--hours", type=float, default=5.0)
    ap.add_argument("--anchor-pct", type=float, default=50.0)
    ap.add_argument("--anchor-time", default="04:42",
                    help="local time the anchor /usage reading was taken")
    a = ap.parse_args()

    now = datetime.now().astimezone()
    hh, mm = (int(x) for x in a.reset.split(":"))
    reset = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if reset < now:
        reset += timedelta(days=1)
    start = reset - timedelta(hours=a.hours)

    ah, am = (int(x) for x in a.anchor_time.split(":"))
    anchor = now.replace(hour=ah, minute=am, second=0, microsecond=0)
    if anchor > now:
        anchor -= timedelta(days=1)

    tot_now, _ = load(start.astimezone(timezone.utc), now.astimezone(timezone.utc))
    tot_anc, _ = load(start.astimezone(timezone.utc), anchor.astimezone(timezone.utc))

    w_now, w_anc = weighted(tot_now), weighted(tot_anc)
    if w_anc <= 0:
        print("no usage found before the anchor time -- cannot calibrate")
        return 1

    pct = a.anchor_pct * w_now / w_anc
    left_min = (reset - now).total_seconds() / 60

    # Two rates, because they can differ by 3x and the wrong one misleads badly.
    # The WINDOW-AVERAGE is dominated by however busy the start of the window was; the
    # RECENT rate (anchor -> now) is what actually predicts the next hour. On this
    # session the average said 22.5%/h and forecast 85% before the reset, while the
    # recent rate was 7.4%/h and put 85% an hour past it -- opposite conclusions.
    avg_rate = pct / max((now - start).total_seconds() / 60, 1)
    since_min = max((now - anchor).total_seconds() / 60, 1)
    recent_rate = (pct - a.anchor_pct) / since_min
    use = recent_rate if since_min >= 10 and recent_rate > 0 else avg_rate
    to85 = (85 - pct) / use if use > 0 else 9e9

    print(f"window   {start:%H:%M} -> {reset:%H:%M}   now {now:%H:%M}"
          f"   ({left_min:.0f} min left)")
    print(f"anchor   {a.anchor_pct:.0f}% at {anchor:%H:%M}")
    print(f"messages {tot_now['msgs']}   out {tot_now['out']:,}"
          f"   cache_read {tot_now['cr']:,}   cache_write {tot_now['cc']:,}")
    print(f"\nESTIMATE {pct:.0f}% used")
    print(f"  window average {avg_rate*60:5.1f} %/h")
    print(f"  since anchor   {recent_rate*60:5.1f} %/h   <- the one that predicts")
    if pct >= 85:
        print("  >>> AT/OVER 85% -- checkpoint SESSION_STATE.md and stop starting new work")
    elif to85 < left_min:
        print(f"  reaches 85% in ~{to85:.0f} min "
              f"({(now + timedelta(minutes=to85)):%H:%M}), before the {reset:%H:%M} reset")
    else:
        print(f"  stays under 85% through the {reset:%H:%M} reset at the recent rate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
