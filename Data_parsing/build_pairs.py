#!/usr/bin/env python3
"""
РЕАКЦІЯ: КРОК 1 — ПАРИ «ТЕКСТ × АДРЕСАТ»

Читає готовий датасет (data/reddit_ukraine_support.csv) і готує вхід для моделі тональності.

Адресат — кого стосується реакція: країна (USA, Germany, Poland…), організація (NATO, EU, UN),
а також Україна та Росія. Країни й організації беруться лише тоді, коли їх згадано в одному
реченні з Україною, підтримкою або російською агресією (COUNTRY_PAIRS = "focus"). Один коментар може містити кілька адресатів:
«Thank you Poland, but Hungary keeps blocking sanctions» → дві пари: Poland і Hungary.

Для кожної пари зберігається:
  aspect   — слово, яким адресата названо в тексті («Scholz», «Germans», «Putin», «Kyiv»…);
             його отримує модель, а в аналізі можна розділяти, наприклад, «Putin» і «Russians»;
  context  — речення (до двох), де адресата згадано; саме їх оцінює модель.

Вибірка. Усіх пар — кілька мільйонів, а для статистики досить кількох сотень на адресата
за тиждень. Тому для кожної клітинки «адресат × тиждень» береться випадкова вибірка до
SAMPLE_PER_TARGET_WEEK пар, а колонка weight показує, скільки пар представляє кожна відібрана
(у підсумках рахуйте зважені частки). SAMPLE_PER_TARGET_WEEK = None — узяти всі пари.

Вихід (папка data/reactions/):
  reaction_pairs.csv — пари для моделі (завантажити в Google Drive для Colab)
  gold_sample.csv    — ~400 пар для ручної розмітки (еталон для перевірки моделі);
                       якщо він уже розмічений (є мітки в label), скрипт його не чіпає

Запуск (у тій самій папці, що й filter_reddit.py / filter_redit.py):  python build_pairs.py
"""

from __future__ import annotations

import json
import random
import re
import time
from bisect import bisect_right
from collections import Counter, defaultdict
from pathlib import Path

import importlib

import pandas as pd

# the filter can be called filter_reddit.py or filter_redit.py
for _module in ("filter_reddit", "filter_redit"):
    try:
        F = importlib.import_module(_module)
        break
    except ModuleNotFoundError:
        continue
else:
    raise ModuleNotFoundError("Put build_pairs.py in the same folder as filter_reddit.py")

# ============================================================
# SETTINGS
# ============================================================

INPUT_CSV = F.OUTPUT_CSV                              # data/reddit_ukraine_support.csv
OUT_DIR = F.OUTPUT_DIR / "reactions"
PAIRS_CSV = OUT_DIR / "reaction_pairs.csv"
GOLD_CSV = OUT_DIR / "gold_sample.csv"

SAMPLE_PER_TARGET_WEEK: int | None = 500  # None = all pairs
GOLD_SIZE = 400                           # pairs to label by hand
EXCLUDE_US_DOMESTIC = True                # skip US domestic politics
EXCLUDE_LIFESTYLE = True                  # skip lifestyle topics
# "focus" = only countries from countries_focus (same sentence as Ukraine, support
# or aggression); "all" = every mention, but many are off topic (Brexit, WWII etc.).
# Ukraine and Russia are always included.
COUNTRY_PAIRS = "focus"
MAX_SENTENCES = 2                         # sentences with the mention to use as context
CONTEXT_MAX_CHARS = 400
RANDOM_SEED = 42

# Ukraine and Russia as targets: only the country, government, army, people. No cities or
# regions (Mariupol, Crimea), since those are places, not targets.
UKRAINE_TARGET_TERMS = [
    "ukraine", "ukrainian", "ukrainians", "ukraina", "kyiv", "kiev",
    "zelensky", "zelenskyy", "zelenskiy", "zelenskyi", "zelenski",
    "verkhovna rada", "zsu", "afu",
]
RUSSIA_TARGET_TERMS = [
    "russia", "russian", "russians", "russian federation", "putin", "putler",
    "kremlin", "moscow", "lavrov", "shoigu", "peskov", "medvedev",
    "ruzzia", "ruzzian", "ruzzians", "rashist", "rashists", "wagner", "wagner group",
    "kadyrov", "kadyrovites",
]

# groups for the gold sample, so all key targets are covered
GOLD_KEY_TARGETS = ["Ukraine", "Russia", "USA", "Germany", "Poland", "UK", "France",
                    "Hungary", "Turkey", "China", "Belarus"]
GOLD_PERIOD_SHARE = {"2019": 0.25, "2022_pre": 0.15, "2022_war": 0.60}

PAIR_COLUMNS = [
    "pair_id", "id", "type", "subreddit", "date", "week", "month", "period",
    "days_since_invasion", "target", "target_kind", "aspect", "context", "thread_title",
    "topic_source", "support", "score", "thread_n", "weight",
]


# ============================================================
# FINDING TARGETS IN TEXT
# ============================================================

UKRAINE_RE = F.compile_terms(UKRAINE_TARGET_TERMS)
RUSSIA_RE = F.compile_terms(RUSSIA_TARGET_TERMS)


def self_spans(text: str) -> list[tuple[int, int]]:
    """Self-identification ("As an American", "I'm Ukrainian") is the author, not a target."""
    if not F.SELF_HINT_RE.search(text):
        return []
    return [m.span() for m in F.SELF_ID_RE.finditer(text)]


def _outside(spans: list[tuple[int, int]], pos: int) -> bool:
    return not any(s <= pos < e for s, e in spans)


def has_target_term(rx: re.Pattern, text: str, spans: list[tuple[int, int]]) -> bool:
    return any(_outside(spans, m.start()) for m in rx.finditer(text))


def row_targets(text: str, countries: str, organizations: str, focus: str) -> list[str]:
    """Targets of a row: countries from the dataset, Ukraine and Russia by search."""
    targets = [t for t in (countries.split("|") + organizations.split("|")) if t]
    if COUNTRY_PAIRS == "focus":
        in_focus = set(focus.split("|"))
        targets = [t for t in targets if t in in_focus]
    spans = self_spans(text)
    if has_target_term(UKRAINE_RE, text, spans):
        targets.append("Ukraine")
    if has_target_term(RUSSIA_RE, text, spans):
        targets.append("Russia")
    return targets


def find_mentions(text: str) -> dict[str, list[tuple[int, int, str]]]:
    """Target -> [(start, end, word in text)]."""
    spans = self_spans(text)
    found: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    _cats, hits = F.scan(text)
    for a, b, _c, actors, term in hits:
        if not actors or term.startswith("flag:") or not _outside(spans, a):
            continue
        for key in actors:
            found[F.ACTOR_NAME[key]].append((a, b, text[a:b]))
    for m in F.ACTOR_SPECIAL_RE.finditer(text):
        if m.lastgroup and _outside(spans, m.start()):
            found[F.ACTOR_NAME[m.lastgroup]].append((m.start(), m.end(), m.group(0)))
    for name, rx in (("Ukraine", UKRAINE_RE), ("Russia", RUSSIA_RE)):
        for m in rx.finditer(text):
            if _outside(spans, m.start()):
                found[name].append((m.start(), m.end(), m.group(0)))
    return found


def target_kind(name: str) -> str:
    if name in ("Ukraine", "Russia"):
        return name.lower()
    key = name.replace(" ", "_")
    return F.ACTOR_KIND.get(key, "country")


def build_context(text: str, spans: list[tuple[int, int, str]]) -> tuple[str, str]:
    """Sentences with the target (up to MAX_SENTENCES) and the word used for it."""
    starts = [0] + [m.end() for m in F.SENT_SPLIT_RE.finditer(text)]
    seen: list[int] = []
    parts: list[str] = []
    for a, b, _t in sorted(spans):
        i = bisect_right(starts, a) - 1
        if i in seen:
            continue
        seen.append(i)
        parts.append(F._sentence(text, starts, i, a, b))
        if len(parts) >= MAX_SENTENCES:
            break
    aspect = sorted(spans)[0][2]
    return " … ".join(parts)[:CONTEXT_MAX_CHARS], aspect


# ============================================================
# READING
# ============================================================

# week, month, period and days_since_invasion are derived from date
PASS1_COLS = ["date", "text", "countries", "organizations", "countries_focus",
              "us_domestic", "is_lifestyle"]
PASS2_COLS = ["id", "type", "subreddit", "date", "text", "topic_source",
              "country_support", "score", "thread_n", "countries", "organizations",
              "countries_focus", "us_domestic", "is_lifestyle"]
OPTIONAL_COLS = ["thread_title"]      # missing in newer datasets


def available(cols: list[str]) -> list[str]:
    """Required columns plus optional ones if the file has them."""
    header = set(F.read_header(INPUT_CSV))
    missing = [c for c in cols if c not in header]
    if missing:
        raise ValueError(f"{INPUT_CSV.name} has no columns {missing}, rerun the filter")
    return cols + [c for c in OPTIONAL_COLS if c in header]


def read_chunks(usecols: list[str], desc: str):
    """Dataset chunks with progress."""
    bar = F.FileProgress(INPUT_CSV, desc)
    try:
        yield from F.iter_csv_chunks(INPUT_CSV, usecols, bar)
    finally:
        bar.close()


def excluded(chunk: pd.DataFrame) -> pd.Series:
    mask = pd.Series(False, index=chunk.index)
    if EXCLUDE_US_DOMESTIC:
        mask |= chunk["us_domestic"].isin(["1", "True"])
    if EXCLUDE_LIFESTYLE:
        mask |= chunk["is_lifestyle"].isin(["1", "True"])
    return mask


# ============================================================
# PASS 1: COUNTS AND SAMPLING
# ============================================================

def pass1(rng: random.Random):
    """Counts pairs per target x week cell and samples rows (reservoir sampling)."""
    counts: Counter = Counter()
    reservoirs: dict[tuple[str, str], list[int]] = {}
    cap = SAMPLE_PER_TARGET_WEEK
    row_no = 0
    for chunk in read_chunks(available(PASS1_COLS), "Pass 1/2: counting pairs"):
        skip = excluded(chunk).tolist()
        weeks = F.add_date_parts(chunk, ["week"], as_text=True)["week"]
        for skip_row, week, text, cs, orgs, foc in zip(skip, weeks, chunk["text"],
                                                      chunk["countries"], chunk["organizations"],
                                                      chunk["countries_focus"]):
            if not skip_row:
                for target in row_targets(text, cs, orgs, foc):
                    cell = (target, week)
                    counts[cell] += 1
                    if cap is None:
                        continue
                    r = reservoirs.setdefault(cell, [])
                    if len(r) < cap:
                        r.append(row_no)
                    else:
                        j = rng.randrange(counts[cell])
                        if j < cap:
                            r[j] = row_no
            row_no += 1

    selected: dict[int, set[str]] | None = None
    if cap is not None:
        selected = defaultdict(set)
        for (target, _week), rows in reservoirs.items():
            for r in rows:
                selected[r].add(target)
    return counts, selected, row_no


# ============================================================
# PASS 2: CONTEXTS AND WRITING PAIRS
# ============================================================

def support_for(country_support: str, target: str) -> str:
    if not country_support:
        return ""
    try:
        return ",".join(json.loads(country_support).get(target, []))
    except json.JSONDecodeError:
        return ""


def pass2(counts: Counter, selected: dict[int, set[str]] | None) -> tuple[int, int]:
    cap = SAMPLE_PER_TARGET_WEEK
    weight = {cell: (n / min(n, cap) if cap is not None else 1.0) for cell, n in counts.items()}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = PAIRS_CSV.with_name(PAIRS_CSV.name + ".tmp")
    header, n_pairs, n_missing, row_no = True, 0, 0, 0
    for chunk in read_chunks(available(PASS2_COLS), "Pass 2/2: contexts"):
        F.add_date_parts(chunk, as_text=True)        # week, month, period, days_since_invasion
        skip = excluded(chunk).tolist()
        rows = []
        for pos, rec in enumerate(chunk.itertuples(index=False)):
            r_no = row_no + pos
            if skip[pos]:
                continue
            if selected is not None:
                wanted = selected.get(r_no)
                if not wanted:
                    continue
            else:
                wanted = None
            mentions = find_mentions(rec.text)
            targets = wanted if wanted is not None else set(
                row_targets(rec.text, rec.countries, rec.organizations, rec.countries_focus))
            for target in sorted(targets):
                spans = mentions.get(target)
                if not spans:
                    n_missing += 1
                    continue
                context, aspect = build_context(rec.text, spans)
                rows.append({
                    "pair_id": f"{rec.id}_{target.replace(' ', '_')}",
                    "id": rec.id, "type": rec.type, "subreddit": rec.subreddit,
                    "date": rec.date, "week": rec.week, "month": rec.month,
                    "period": rec.period, "days_since_invasion": rec.days_since_invasion,
                    "target": target, "target_kind": target_kind(target),
                    "aspect": aspect, "context": context,
                    "thread_title": getattr(rec, "thread_title", ""),
                    "topic_source": rec.topic_source,
                    "support": support_for(rec.country_support, target),
                    "score": rec.score, "thread_n": rec.thread_n,
                    "weight": round(weight.get((target, rec.week), 1.0), 4),
                })
        row_no += len(chunk)
        if rows:
            pd.DataFrame(rows, columns=PAIR_COLUMNS).to_csv(
                tmp, mode="w" if header else "a", header=header, index=False, encoding="utf-8")
            header = False
            n_pairs += len(rows)
    if header:                                   # no pairs, write the header only
        pd.DataFrame(columns=PAIR_COLUMNS).to_csv(tmp, index=False, encoding="utf-8")
    F.safe_replace(tmp, PAIRS_CSV)
    return n_pairs, n_missing


# ============================================================
# GOLD SAMPLE FOR MANUAL LABELLING
# ============================================================

def gold_is_labelled() -> bool:
    """Whether a labelled gold sample already exists (it is never overwritten)."""
    if not GOLD_CSV.exists():
        return False
    try:
        old = pd.read_csv(GOLD_CSV, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError):
        return True                      # unreadable, better leave it alone
    return "label" in old.columns and bool(old["label"].str.strip().ne("").any())


def make_gold(rng_seed: int):
    pairs = pd.read_csv(PAIRS_CSV, dtype=str, keep_default_na=False,
                        usecols=["pair_id", "period", "target", "aspect", "context",
                                 "thread_title", "subreddit", "date"])
    if pairs.empty:
        return 0
    pairs["group"] = pairs["target"].where(pairs["target"].isin(GOLD_KEY_TARGETS),
                                           "other countries / NATO / EU / UN")
    parts = []
    present = {k: v for k, v in GOLD_PERIOD_SHARE.items() if (pairs["period"] == k).any()}
    total_share = sum(present.values()) or 1.0
    for period, share in present.items():
        share = share / total_share          # split among the periods that exist
        p = pairs[pairs["period"] == period]
        groups = sorted(set(p["group"]))
        per_group = max(1, round(GOLD_SIZE * share / len(groups)))
        for _g, grp in p.groupby("group"):
            parts.append(grp.sample(n=min(per_group, len(grp)), random_state=rng_seed))
    gold = pd.concat(parts).sample(frac=1, random_state=rng_seed)    # shuffle
    gold = gold.drop(columns=["group"])
    gold["label"] = ""
    gold["note"] = ""
    gold = gold[["pair_id", "period", "date", "subreddit", "target", "aspect",
                 "thread_title", "context", "label", "note"]]
    F.safe_to_csv(gold, GOLD_CSV, index=False, encoding="utf-8-sig")   # utf-8-sig for Excel
    return len(gold)


# ============================================================
# MAIN
# ============================================================

def main():
    t0 = time.time()
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"{INPUT_CSV} not found, run filter_reddit.py first")
    print("=" * 70)
    print("REACTIONS, STEP 1: TEXT x TARGET PAIRS")
    print("=" * 70)
    print(f"Input:    {INPUT_CSV}")
    cap = SAMPLE_PER_TARGET_WEEK
    print(f"Sample:   {'all pairs' if cap is None else f'up to {cap} pairs per target per week'}")

    rng = random.Random(RANDOM_SEED)
    counts, selected, n_rows = pass1(rng)
    total_pairs = sum(counts.values())
    print(f"  rows: {n_rows:,} | pairs: {total_pairs:,} | cells: {len(counts):,}")

    n_pairs, n_missing = pass2(counts, selected)
    keep_gold = gold_is_labelled()
    n_gold = 0 if keep_gold else make_gold(RANDOM_SEED)

    by_target = Counter()
    for (target, _w), n in counts.items():
        by_target[target] += n
    print("\nTop targets (all pairs, before sampling):")
    for target, n in by_target.most_common(20):
        print(f"  {target:<15} {n:>10,}")

    print("\n" + "=" * 70)
    print(f"Pairs in file:   {n_pairs:,}  (of {total_pairs:,})")
    if n_missing:
        print(f"No context found: {n_missing:,} (skipped)")
    size_mb = PAIRS_CSV.stat().st_size / 1024 ** 2
    print(f"Saved: {PAIRS_CSV}  ({size_mb:,.1f} MB)")
    if keep_gold:
        print(f"{GOLD_CSV.name} is already labelled, keeping it (delete it to make a new one)")
    else:
        print(f"Gold sample: {GOLD_CSV}  ({n_gold} pairs to label)")
    print(f"Time: {F.fmt_time(time.time() - t0)} (h:m:s)")


if __name__ == "__main__":
    main()