#!/usr/bin/env python3
"""
РЕАКЦІЯ: ДРУГА ВИБІРКА ДЛЯ ДОНАВЧАННЯ (там, де модель вагається)

Перша донавчена модель (reactions_ft.csv) дала macro-F1 0.66 на еталоні: найчастіше вона
пропускає позитив і плутає негатив із нейтралом. Найкорисніші нові приклади — ті, де модель
не впевнена. Тому скрипт відбирає 800 нових пар:
  • половина — пари з найменшою різницею між двома найімовірнішими мітками (модель «вагається»);
  • половина — випадкові, щоб навчальна вибірка лишалася схожою на весь датасет.
Пари з gold_sample.csv і train_sample.csv не беруться. Частки періодів і адресатів — як у
першій вибірці (Україна й Росія — з більшою часткою).

Потрібно: data/reactions/reaction_pairs.csv (той самий, що на Диску)
          data/reactions/reactions_ft.csv   (завантажте з Диска, папка reddit_reactions)
Вихід:    data/reactions/train_sample2.csv (label і note порожні — для розмітки)

Запуск (у тій самій папці, що й build_pairs.py):  python make_train_sample2.py
"""

from __future__ import annotations

import importlib
import random
from collections import Counter

import numpy as np
import pandas as pd

for _module in ("filter_reddit", "filter_redit"):
    try:
        F = importlib.import_module(_module)
        break
    except ModuleNotFoundError:
        continue
else:
    raise ModuleNotFoundError("Put make_train_sample2.py in the same folder as filter_reddit.py")

B = importlib.import_module("build_pairs")
M = importlib.import_module("make_train_sample")

N_TRAIN = 800
UNCERTAIN_SHARE = 0.5              # share of pairs where the model is unsure
RANDOM_SEED = 11
REACTIONS_CSV = B.OUT_DIR / "reactions_ft.csv"
TRAIN2_CSV = B.OUT_DIR / "train_sample2.csv"


def labelled_ids(path) -> set[str]:
    if not path.exists():
        return set()
    return set(pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig",
                           usecols=["pair_id"])["pair_id"])


def main():
    src = B.PAIRS_CSV
    for path, hint in ((src, "run build_pairs.py first"),
                       (REACTIONS_CSV, "download reactions_ft.csv from Drive into data/reactions/")):
        if not path.exists():
            raise FileNotFoundError(f"{path} not found: {hint}")
    used = labelled_ids(B.GOLD_CSV) | labelled_ids(M.TRAIN_CSV)

    # 1) pairs + model confidence
    index = []
    bar = F.FileProgress(src, "Pass 1/2: pairs")
    for chunk in F.iter_csv_chunks(src, ["pair_id", "period", "target"], bar):
        index.append(chunk)
    bar.close()
    pairs: pd.DataFrame = pd.concat(index, ignore_index=True)
    probs = pd.read_csv(REACTIONS_CSV, usecols=["pair_id", "p_neg", "p_neu", "p_pos"],
                        dtype={"pair_id": str}).drop_duplicates("pair_id", keep="last")
    pairs = pairs.merge(probs, on="pair_id", how="inner")
    pairs = pairs.loc[~pairs["pair_id"].isin(list(used))].copy()
    p = np.sort(pairs[["p_neg", "p_neu", "p_pos"]].to_numpy(dtype=float), axis=1)
    pairs["margin"] = p[:, 2] - p[:, 1]              # small gap = model is unsure
    pairs["group"] = pairs["target"].where(pairs["target"].isin(list(M.TARGET_SHARE)), "other")
    if pairs.empty:
        raise SystemExit("No pairs with predictions found. Check that reactions_ft.csv "
                         "was made from this reaction_pairs.csv")

    # 2) in each period x target cell: most uncertain + random
    rng = random.Random(RANDOM_SEED)
    present = {k: v for k, v in M.PERIOD_SHARE.items() if (pairs["period"] == k).any()}
    total = sum(present.values()) or 1.0
    chosen: set[str] = set()
    why: dict[str, str] = {}
    for period, p_share in present.items():
        part = pd.DataFrame(pairs.loc[pairs["period"] == period])
        for group, g_share in M.TARGET_SHARE.items():
            cell = pd.DataFrame(part.loc[part["group"] == group])
            k = min(len(cell), round(N_TRAIN * p_share / total * g_share))
            k_unc = round(k * UNCERTAIN_SHARE)
            unc = cell.sort_values("margin")["pair_id"].head(k_unc).tolist()
            unc_set = set(unc)
            rest = [x for x in cell["pair_id"].tolist() if x not in unc_set]
            rnd = rng.sample(rest, min(len(rest), k - len(unc)))
            chosen.update(unc + rnd)
            why.update({x: "uncertain" for x in unc})
            why.update({x: "random" for x in rnd})

    # 3) texts for the selected pairs
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
    path = F.safe_to_csv(out, TRAIN2_CSV, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {path} ({len(out)} pairs to label)")
    print("Why selected:", dict(Counter(why[x] for x in out["pair_id"])))
    print("Pairs by period:", dict(Counter(out["period"].tolist())))
    print("Top targets:", dict(Counter(out["target"].tolist()).most_common(12)))


if __name__ == "__main__":
    main()