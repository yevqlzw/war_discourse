#!/usr/bin/env python3
"""
РЕАКЦІЯ → КОЛОНКИ ДАТАСЕТУ

Дописує в data/reddit_ukraine_support.csv реакцію, яку модель визначила в Colab:
    reaction_ukraine   — реакція на Україну в цьому тексті: 1 позитив, 0 нейтрал, −1 негатив,
                         порожньо — якщо Україну в тексті не згадано;
    reaction_russia    — те саме для Росії;
    reaction           — для всіх адресатів рядка (Україна, Росія, країни з countries_focus,
                         NATO / EU / UN), JSON: {"Poland":1,"Russia":-1,"Ukraine":1}.
Рядки us_domestic / is_lifestyle реакції не мають (їх виключають з аналізу).
Середнє колонки — це одразу баланс реакції (частка позитиву − частка негативу).

І рахує підсумки за всіма парами (вага треду 1/√thread_n, як у ноутбуці):
    data/reactions/summary_by_period.csv, data/reactions/summary_by_month.csv

Потрібно: data/reactions/reactions_all.csv — з Диска (результат reaction_finetune_colab.ipynb
          для pairs_all.csv.gz). Якщо його немає, але є reactions_ft2.csv / reactions_ft.csv
          (лише вибірка), колонки заповняться тільки для рядків із вибірки.

Запуск (у тій самій папці, що й filter_reddit.py / filter_redit.py):  python add_reactions.py
"""

from __future__ import annotations

import importlib
import json
import time
from collections import defaultdict

import numpy as np
import pandas as pd

for _module in ("filter_reddit", "filter_redit"):
    try:
        F = importlib.import_module(_module)
        break
    except ModuleNotFoundError:
        continue
else:
    raise ModuleNotFoundError("Put add_reactions.py in the same folder as filter_reddit.py")

REACT_DIR = F.OUTPUT_DIR / "reactions"
SOURCES = [REACT_DIR / "reactions_all.csv",          # all pairs
           REACT_DIR / "reactions_ft2.csv",          # sample only
           REACT_DIR / "reactions_ft.csv"]
SUMMARY_PERIOD = REACT_DIR / "summary_by_period.csv"
SUMMARY_MONTH = REACT_DIR / "summary_by_month.csv"
MIN_PAIRS = 30                                       # smaller cells are left out of the summaries
USE_THREAD_WEIGHT = True
LABEL_CODE = {"negative": -1, "neutral": 0, "positive": 1}
# older filter versions don't have this list
NEW_COLUMNS = list(getattr(F, "REACTION_COLUMNS", ["reaction_ukraine", "reaction_russia", "reaction"]))


def load_reactions(path) -> tuple[dict[str, dict[str, int]], int]:
    """Record id -> {target: -1/0/1}; pair_id = "<id>_<target, spaces as _>"."""
    by_id: dict[str, dict[str, int]] = defaultdict(dict)
    n = 0
    cols = F.read_header(path)
    code_col = "reaction" if "reaction" in cols else "label"
    bar = F.FileProgress(path, f"Reading {path.name}")
    for chunk in F.iter_csv_chunks(path, ["pair_id", code_col], bar):
        codes = (F.to_num(chunk[code_col]) if code_col == "reaction"
                 else pd.Series(chunk[code_col].map(LABEL_CODE), index=chunk.index))
        for pid, code in zip(chunk["pair_id"], codes):
            if pd.isna(code) or "_" not in pid:
                continue
            rid, target = pid.split("_", 1)
            by_id[rid][target.replace("_", " ")] = int(code)
            n += 1
    bar.close()
    return by_id, n


def main():
    t0 = time.time()
    src = next((p for p in SOURCES if p.exists()), None)
    if src is None:
        raise FileNotFoundError("data/reactions/reactions_all.csv not found. Download it from Drive "
                                "(reddit_reactions folder) after reaction_finetune_colab.ipynb")
    full = src == SOURCES[0]
    if not full:
        print(f"Warning: using {src.name}, which is only a sample of pairs, so not every row gets a reaction.")
    reactions, n_pairs = load_reactions(src)
    print(f"Reactions: {n_pairs:,} for {len(reactions):,} records")

    dataset = F.OUTPUT_CSV
    header = [c for c in F.read_header(dataset) if c not in NEW_COLUMNS]
    tmp = dataset.with_name(dataset.name + ".tmp")
    # (target, period/month) -> weighted sums for -1/0/1 and pair count
    sums: dict[tuple, np.ndarray] = defaultdict(lambda: np.zeros(4))
    first, n_rows, n_with, matched = True, 0, 0, 0
    bar = F.FileProgress(dataset, "Adding reactions")
    for chunk in F.iter_csv_chunks(dataset, header, bar):
        chunk = chunk.loc[:, header]
        recs = [reactions.get(i) for i in chunk["id"]]
        chunk["reaction_ukraine"] = [str(r["Ukraine"]) if r and "Ukraine" in r else "" for r in recs]
        chunk["reaction_russia"] = [str(r["Russia"]) if r and "Russia" in r else "" for r in recs]
        chunk["reaction"] = [json.dumps(r, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
                             if r else "" for r in recs]
        n_rows += len(chunk)
        n_with += sum(1 for r in recs if r)
        matched += sum(len(r) for r in recs if r)
        if full:
            parts = F.add_date_parts(pd.DataFrame({"date": chunk["date"]}), ["month", "period"])
            tn = F.to_num(chunk["thread_n"]).fillna(1).clip(lower=1)
            weights = (1 / np.sqrt(tn)) if USE_THREAD_WEIGHT else pd.Series(1.0, index=tn.index)
            for r, per, mon, w in zip(recs, parts["period"], parts["month"], weights):
                if not r:
                    continue
                for target, code in r.items():
                    for key in ((target, "period", per), (target, "month", mon)):
                        acc = sums[key]
                        acc[code + 1] += w
                        acc[3] += 1
        chunk.to_csv(tmp, mode="w" if first else "a", header=first, index=False, encoding="utf-8")
        first = False
    bar.close()
    final = F.safe_replace(tmp, dataset)

    print("\n" + "=" * 70)
    print(f"Rows: {n_rows:,} | with reaction: {n_with:,} ({n_with / max(n_rows, 1):.0%})")
    if matched < n_pairs:
        print(f"Warning: {n_pairs - matched:,} reactions matched no row, reactions_all.csv "
              f"may come from an older version of the dataset")
    print(f"Saved: {final}  (+ columns {', '.join(NEW_COLUMNS)})")

    if full:
        for kind, path in (("period", SUMMARY_PERIOD), ("month", SUMMARY_MONTH)):
            rows = []
            for (target, k, value), acc in sums.items():
                if k != kind or acc[3] < MIN_PAIRS:
                    continue
                total = acc[:3].sum()
                neg, neu, pos = (acc[:3] / total) if total else (0.0, 0.0, 0.0)
                rows.append({"target": target, kind: value, "share_negative": round(neg, 3),
                             "share_neutral": round(neu, 3), "share_positive": round(pos, 3),
                             "net": round(pos - neg, 3), "n_pairs": int(acc[3])})
            table = pd.DataFrame(rows).sort_values(["target", kind])
            out = F.safe_to_csv(table, path, index=False, encoding="utf-8")
            print(f"Summary: {out}  ({len(table)} rows)")
    print(f"Time: {F.fmt_time(time.time() - t0)} (h:m:s)")


if __name__ == "__main__":
    main()