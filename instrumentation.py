import functools, json, time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

LOG_PATH = Path('timings.jsonl')

_records: list[tuple[str, float]] = []

# recording
@contextmanager
def stage(name: str):
    """time an arbitrary block of code"""
    t0 = time.perf_counter()

    try:
        yield
    finally:
        _records.append((name, time.perf_counter() - t0))

def timed(name: str | None = None):
    """decorator form"""
    def decorator(fn):
        label = name or fn.__name__
 
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with stage(label):
                return fn(*args, **kwargs)
        return wrapper
    
    return decorator

def reset() -> None:
    _records.clear()

# aggregation & reporting
def aggregate() -> dict:
    """group repeated calls to the same stage"""

    agg: dict[str, dict] = {}

    for name, dur in _records:
        entry = agg.setdefault(name, {"calls": 0, "total": 0.0, "durations": []})
        entry["calls"] += 1
        entry["total"] += dur
        entry["durations"].append(dur)

    for entry in agg.values():
        entry["mean"] = entry["total"] / entry["calls"]
        entry["max"] = max(entry["durations"])

    return agg

def report(title: str = "Timing") -> float:
    """print stage-by-stage table sorted by total cost"""

    agg = aggregate()

    if not agg:
        print("No timings recorded")
        return 0.0

    total = sum(e["total"] for e in agg.values())
    rows = sorted(agg.items(), key=lambda kv:kv[1]['total'], reverse=True)
    width = max(max(len(n) for n in agg), 5)

    print(f"\n{title}")
    print(f"{'stage':<{width}} {'calls':>5} {'total':>9} {'mean':>9} {'share':>6}")
    print("-" * (width + 36))

    for name, e in rows:
        print(
            f"{name:<{width}} {e['calls']:>5} {e['total']:>8.2f}s"
            f"{e['mean']:>8.2f}s {e['total'] / total * 100:5.1f}%"
        )

    print("-" * (width + 36))
    print(f"{'TOTAL':<{width}} {'':>5} {total:>8.2f}s")

    return total

# persistence
def save_run(label: str, notes: str="", path: Path | str = LOG_PATH) -> None:
    """append current run to JSONL log under a label"""

    agg = aggregate()

    if not agg:
        print("Nothing to save")
        return

    record = {
        "label": label,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "notes": notes,
        "total": sum(e["total"] for e in agg.values()),
        "stages": {n: {"calls": e["calls"], "total": round(e["total"], 3)}
                   for n, e in agg.items()}
    }

    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")

    print(f"Saved run '{label}' ({record['total']:.1f}s total) to {path}")

def _load(path: Path | str) -> list[dict]:
    p = Path(path)

    if not p.exists():
        return []
    
    with open(p, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]

def compare(before: str, after: str, path: Path | str = LOG_PATH) -> None:
    """print before-after table"""

    runs = _load(path)
    picks = {}

    for r in runs:
        if r["label"] in (before, after):
            picks[r["label"]] = r

    missing = [lbl for lbl in (before, after) if lbl not in picks]
    if missing:
        print(f"No saved runs labeled: {', '.join(missing)}")
        return

    b, a = picks[before], picks[after]
    names = sorted(set(b["stages"]) | set(a["stages"]))
    width = max(max(len(n) for n in names), 5)

    print(f"\n{before} -> {after}")
    print(f"{'stage':<{width}} {'before':>9} {'after':>9} {'change':>8}")
    print("-" * (width + 32))

    for n in names:
        bt = b["stages"].get(n, {}).get("total", 0.0)
        at = a["stages"].get(n, {}).get("total", 0.0)
        delta = f"{(at - bt) / bt * 100:+.0f}%" if bt else "new"
        print(f"{n:<{width}} {bt:>8.2f}s {at:8.2f}s {delta:>8}")

    print("-" * (width + 32))
    change = (a["total"] - b["total"]) / b["total"] * 100 if b["total"] else 0

    print(f"{'TOTAL':<{width}} {b['total']:>8.2f}s {a['total']:>8.2f}s {change:>7.0f}%")
