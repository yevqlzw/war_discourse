#!/usr/bin/env python3
"""
РЕАКЦІЯ: ВИБІРКА ДЛЯ ДОНАВЧАННЯ МОДЕЛІ

Готова модель (yangheng/deberta-v3-base-absa) навчена на відгуках про ноутбуки й ресторани
і на Reddit про війну систематично бачить негатив там, де його немає: «Russia shelled Kyiv»
для неї — негатив до України. Щоб це виправити, модель донавчають на наших власних
розмічених прикладах.

Скрипт відбирає з data/reactions/reaction_pairs.csv нові пари для розмітки
(ті, що вже є в gold_sample.csv, не беруться — еталон лишається окремою перевіркою):
  • 25% — 2019, 15% — 2022 до вторгнення, 60% — 2022 після (як в еталоні);
  • Україна і Росія — з більшою часткою, бо саме на них модель помиляється найбільше.

Вихід: data/reactions/train_sample.csv (колонки label і note порожні — для розмітки).

Запуск (у тій самій папці, що й build_pairs.py):  python make_train_sample.py
"""

from __future__ import annotations

import importlib
import random
from collections import Counter

import pandas as pd

for _module in ("filter_reddit", "filter_redit"):
    try:
        F = importlib.import_module(_module)
        break
    except ModuleNotFoundError:
        continue
else:
    raise ModuleNotFoundError("Put make_train_sample.py in the same folder as filter_reddit.py")

B = importlib.import_module("build_pairs")

N_TRAIN = 800                     # pairs to label
RANDOM_SEED = 7                   # different from the gold sample
TRAIN_CSV = B.OUT_DIR / "train_sample.csv"

# Target shares in the sample (within each period)
TARGET_SHARE = {
    "Ukraine": 0.15, "Russia": 0.10,
    "USA": 0.06, "Germany": 0.06, "Poland": 0.06, "UK": 0.06, "France": 0.06,
    "Hungary": 0.06, "Turkey": 0.06, "China": 0.06, "Belarus": 0.06,
    "other": 0.21,                # other countries, NATO, EU, UN
}
PERIOD_SHARE = B.GOLD_PERIOD_SHARE


def main():
    src = B.PAIRS_CSV
    if not src.exists():
        raise FileNotFoundError(f"{src} not found, run build_pairs.py first")
    gold_ids: set[str] = set()
    if B.GOLD_CSV.exists():
        gold = pd.read_csv(B.GOLD_CSV, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        gold_ids = set(gold["pair_id"])

    # 1) list of pairs without text (saves memory)
    index = []
    bar = F.FileProgress(src, "Pass 1/2: pairs")
    for chunk in F.iter_csv_chunks(src, ["pair_id", "period", "target"], bar):
        index.append(chunk)
    bar.close()
    pairs: pd.DataFrame = pd.concat(index, ignore_index=True)
    pairs = pairs.loc[~pairs["pair_id"].isin(list(gold_ids))].copy()
    pairs["group"] = pairs["target"].where(pairs["target"].isin(list(TARGET_SHARE)), "other")

    rng = random.Random(RANDOM_SEED)
    present = {p: s for p, s in PERIOD_SHARE.items() if (pairs["period"] == p).any()}
    total = sum(present.values()) or 1.0
    chosen: set[str] = set()
    for period, p_share in present.items():
        part = pairs[pairs["period"] == period]
        for group, g_share in TARGET_SHARE.items():
            ids = part.loc[part["group"] == group, "pair_id"].tolist()
            k = min(len(ids), round(N_TRAIN * p_share / total * g_share))
            chosen.update(rng.sample(ids, k))

    # 2) texts only for the selected pairs
    cols = ["pair_id", "period", "date", "subreddit", "target", "aspect", "context"]
    has_title = "thread_title" in F.read_header(src)
    rows = []
    bar = F.FileProgress(src, "Pass 2/2: texts")
    for chunk in F.iter_csv_chunks(src, cols + (["thread_title"] if has_title else []), bar):
        rows.append(chunk[chunk["pair_id"].isin(list(chosen))])
    bar.close()
    out: pd.DataFrame = pd.concat(rows, ignore_index=True).sample(frac=1, random_state=RANDOM_SEED)
    if not has_title:
        out["thread_title"] = ""
    out["label"] = ""
    out["note"] = ""
    out = out.loc[:, ["pair_id", "period", "date", "subreddit", "target", "aspect",
                      "thread_title", "context", "label", "note"]]
    path = F.safe_to_csv(out, TRAIN_CSV, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {path} ({len(out)} pairs to label)")
    print("Pairs by period:", dict(Counter(out["period"].tolist())))
    print("Top targets:", dict(Counter(out["target"].tolist()).most_common(12)))


if __name__ == "__main__":
    main()