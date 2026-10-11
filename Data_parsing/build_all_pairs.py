#!/usr/bin/env python3
"""
РЕАКЦІЯ: УСІ ПАРИ «ТЕКСТ × АДРЕСАТ» (для колонки реакції в кожному рядку датасету)

build_pairs.py бере вибірку (до 500 пар на адресата за тиждень) — цього досить для підсумків.
Щоб реакція була в КОЖНОМУ рядку датасету, модель має розмітити всі пари. Цей скрипт робить
їх тими самими правилами, що й build_pairs.py (ті самі адресати, слово-аспект і речення-контекст),
але без вибірки й лише з потрібними колонками, стиснуто:

Вихід: data/reactions/pairs_all.csv.gz  (pair_id, aspect, context) — завантажити на Диск
       у папку reddit_reactions; ноутбук reaction_finetune_colab.ipynb розмітить його повністю.

Рядки us_domestic / is_lifestyle пропускаються (як і в build_pairs.py) — реакції в них не буде.

Запуск (у тій самій папці, що й build_pairs.py):  python build_all_pairs.py
"""

from __future__ import annotations

import csv
import gzip
import importlib
import time
from collections import Counter

import pandas as pd

B = importlib.import_module("build_pairs")
F = B.F

OUT_GZ = B.OUT_DIR / "pairs_all.csv.gz"
COLS = ["id", "text", "countries", "organizations", "countries_focus", "us_domestic",
        "is_lifestyle"]


def main():
    t0 = time.time()
    src = B.INPUT_CSV
    if not src.exists():
        raise FileNotFoundError(f"{src} not found")
    B.OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = OUT_GZ.with_name(OUT_GZ.name + ".tmp")
    n_rows = n_pairs = n_missing = 0
    by_target: Counter = Counter()
    with gzip.open(tmp, "wt", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["pair_id", "aspect", "context"])
        for chunk in B.read_chunks(B.available(COLS), "All pairs"):
            skip = B.excluded(chunk).tolist()
            for pos, rec in enumerate(chunk.itertuples(index=False)):
                n_rows += 1
                if skip[pos]:
                    continue
                targets = B.row_targets(rec.text, rec.countries, rec.organizations,
                                        rec.countries_focus)
                if not targets:
                    continue
                mentions = B.find_mentions(rec.text)
                for target in sorted(set(targets)):
                    spans = mentions.get(target)
                    if not spans:
                        n_missing += 1
                        continue
                    context, aspect = B.build_context(rec.text, spans)
                    writer.writerow([f"{rec.id}_{target.replace(' ', '_')}", aspect, context])
                    n_pairs += 1
                    by_target[target] += 1
    F.safe_replace(tmp, OUT_GZ)

    print("\n" + "=" * 70)
    print(f"Rows: {n_rows:,} | pairs: {n_pairs:,} | no context (skipped): {n_missing:,}")
    print("Top targets:", ", ".join(f"{t} {n:,}" for t, n in by_target.most_common(10)))
    size_mb = OUT_GZ.stat().st_size / 1024 ** 2
    hours = n_pairs / 1000 / 3600
    print(f"Saved: {OUT_GZ}  ({size_mb:,.0f} MB)")
    print(f"   About {hours:.1f} h of labelling on a Colab T4 (~1000 pairs/s)")
    print(f"Time: {F.fmt_time(time.time() - t0)} (h:m:s)")


if __name__ == "__main__":
    pd.set_option("mode.chained_assignment", None)
    main()