#!/usr/bin/env python3
"""
ОГЛЯД ДАТАСЕТУ БЕЗ ПЕРЕЗАПУСКУ ФІЛЬТРА

Робить data/preview_by_month.csv (по PREVIEW_PER_MONTH випадкових рядків за кожен місяць)
із уже готового data/reddit_ukraine_support.csv. Корисно, якщо filter_reddit.py не встиг
зберегти огляд (наприклад, обірвався на останньому кроці) або якщо хочете нову випадкову вибірку.

Запуск (у тій самій папці, що й filter_reddit.py / filter_redit.py):  python make_preview.py
"""

from __future__ import annotations

import importlib
import random
import time

import pandas as pd

for _module in ("filter_reddit", "filter_redit"):
    try:
        F = importlib.import_module(_module)
        break
    except ModuleNotFoundError:
        continue
else:
    raise ModuleNotFoundError("Put make_preview.py in the same folder as filter_reddit.py")

SEED = int(time.time())        # new sample every run; a fixed number repeats it

COLS = ["id", "type", "subreddit", "date", "topic_source", "thread_id",
        "countries", "countries_focus", "organizations", *F.FLAG_COLUMNS,
        "us_domestic", "is_lifestyle", "text"]


def main():
    src = F.OUTPUT_CSV
    if not src.exists():
        raise FileNotFoundError(f"{src} not found")
    k = F.PREVIEW_PER_MONTH
    rng = random.Random(SEED)
    seen: dict[str, int] = {}
    keep: dict[str, list[dict]] = {}
    has_title = "thread_title" in F.read_header(src)     # missing in newer datasets
    cols = COLS + (["thread_title"] if has_title else [])
    bar = F.FileProgress(src, "Preview")
    for chunk in F.iter_csv_chunks(src, cols, bar):
        for rec in chunk.to_dict("records"):
            rec["month"] = m = rec["date"][:7]           # month from the date
            seen[m] = seen.get(m, 0) + 1
            bucket = keep.setdefault(m, [])
            if len(bucket) < k:
                bucket.append(rec)
            else:
                j = rng.randrange(seen[m])
                if j < k:
                    bucket[j] = rec
    bar.close()

    v = pd.DataFrame([r for rows in keep.values() for r in rows])
    v["support"] = [",".join(F.SUPPORT_SHORT[f] for f in F.FLAG_COLUMNS if str(r[f]) in ("1", "True"))
                    for r in v.to_dict("records")]
    v["link"] = [f"https://www.reddit.com/r/{r['subreddit']}/comments/"
                 + (f"{r['id']}/" if r["type"] == "submission" else f"{r['thread_id']}/_/{r['id']}/")
                 for r in v.to_dict("records")]
    v = v.sort_values(["month", "date", "subreddit"])
    cols = ["month", "date", "subreddit", "type", "topic_source", "countries",
            "countries_focus", "organizations", "support", "us_domestic", "is_lifestyle",
            *(["thread_title"] if has_title else []), "text", "link", "id"]
    out = F.safe_to_csv(v[cols], F.PREVIEW_FILE, index=False, encoding="utf-8-sig")
    print(f"Saved: {out} ({len(v)} rows, {v['month'].nunique()} months)")


if __name__ == "__main__":
    main()