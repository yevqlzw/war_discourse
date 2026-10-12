"""
Helpers for eda.ipynb (Homework 3): plotting style, figure frame, data loading with a cache,
tone statistics, word statistics and maps. Settings and reference data (colours, events,
keywords, labels) are in eda_config.json next to this file; country shapes in countries.json.

The notebook does `from eda_utils import *`, so every name below is available there.
"""
import json
import pickle
import re
import textwrap
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import font_manager, patheffects
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, PathPatch
from matplotlib.path import Path as MPath
from matplotlib.ticker import FuncFormatter, MaxNLocator
from IPython.display import Markdown, clear_output, display

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 160)
pd.set_option("display.max_colwidth", None)

HERE = Path(__file__).resolve().parent
CFG = json.load(open(HERE / "eda_config.json", encoding="utf-8"))

# ---------------------------------------------------------------- colours
COL = CFG["colors"]
BLUE, ORANGE, AQUA, YELLOW = COL["blue"], COL["orange"], COL["aqua"], COL["yellow"]
MAGENTA, GREEN, VIOLET, RED = COL["magenta"], COL["green"], COL["violet"], COL["red"]
PINK, PINK_LIGHT, BLUE_LIGHT = COL["pink"], COL["pink_light"], COL["blue_light"]
INK, INK2, MUTED = COL["ink"], COL["ink2"], COL["muted"]
GRID, AXIS, SURFACE = COL["grid"], COL["axis"], COL["surface"]
CONTEXT, OTHER, C19, MAP_EMPTY = COL["context"], COL["other"], COL["year2019"], COL["map_empty"]
CAT = [COL[c] for c in CFG["categorical_order"]]
NEG, NEU, POS = PINK, COL["neutral"], BLUE          # tone: pink = negative, blue = positive
RUSSIA_COLOR = PINK
SUB_COLORS = [COL[c] for c in CFG["subreddit_colors"]]

_ramp = {k: LinearSegmentedColormap.from_list(k, v) for k, v in CFG["ramps"].items()}
SEQ, DIV = _ramp["seq"], _ramp["div"]                     # blue ramp; pink-grey-blue (tone)
SEQ_PINK, SEQ_GREEN, SEQ_VIOLET = _ramp["pink"], _ramp["green"], _ramp["violet"]
DIV_PG = _ramp["pinkgreen"]                               # green-grey-pink

PERIODS = list(CFG["periods"])
PERIOD_LABEL = CFG["periods"]
PERIOD_COLOR = {p: COL[c] for p, c in CFG["period_colors"].items()}

# ---------------------------------------------------------------- reference data
def _events(rows):
    return [tuple(r) for r in rows]


EV2019, EV2022 = _events(CFG["events_2019"]), _events(CFG["events_2022"])
ENTITY_EVENTS = {k: _events(v) for k, v in CFG["country_events"].items()}
SUPPORT_EVENTS = {k: _events(v) for k, v in CFG["support_events"].items()}
THEME_EVENTS = {k: _events(v) for k, v in CFG["theme_events"].items()}
SUPPORT_COL = CFG["support_columns"]
SUPPORT = list(SUPPORT_COL)
SUP_LABEL = CFG["support_labels"]
SRC_LABEL = CFG["topic_source_labels"]
SRC_COLOR = {k: COL[c] for k, c in CFG["topic_source_colors"].items()}
LEADERS = {k: (name, COL[c]) for k, (name, c) in CFG["leaders"].items()}
THEMES, WEAPONS = CFG["themes"], CFG["weapons"]
KEYWORDS = {k: (v["stems"], v["regex"]) for k, v in CFG["keywords"].items()}
ORGS = set(CFG["organisations"])
ENT_A3 = CFG["entity_iso3"]
EUROPE_VIEW = CFG["europe_map"]
STOP = set(CFG["stopwords"].split())
INVASION = pd.Timestamp(CFG["data"]["invasion_date"])
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTHS_FULL = ["January", "February", "March", "April", "May", "June", "July", "August",
               "September", "October", "November", "December"]

# ---------------------------------------------------------------- matplotlib style
_fonts = {f.name for f in font_manager.fontManager.ttflist}
FONT = next((f for f in ["Segoe UI", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"] if f in _fonts), "DejaVu Sans")
mpl.rcParams.update({
    "font.family": FONT, "font.size": 10, "text.color": INK,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "figure.dpi": 130, "savefig.dpi": 130,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.7,
    "axes.axisbelow": True, "axes.labelcolor": INK2, "axes.labelsize": 9.5,
    "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.titlecolor": INK, "axes.titlepad": 7,
    "axes.prop_cycle": mpl.cycler(color=CAT),
    "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "xtick.labelsize": 8.8, "ytick.labelsize": 8.8, "xtick.major.size": 3, "ytick.major.size": 0,
    "legend.frameon": False, "legend.fontsize": 9.2, "legend.labelcolor": INK2,
    "lines.linewidth": 2, "lines.solid_capstyle": "round", "lines.markersize": 5,
    "patch.linewidth": 0,
})

# ---------------------------------------------------------------- formatting
FIG = {"n": 0}       # running figure number
FIGREF = {}          # figure numbers by key, so the text can refer to them
FACTS = {}           # numbers reused in the conclusions
FIG_SCALE = 0.86     # < 1 makes text larger relative to the figure (figures are scaled to page width in the PDF)


def human(x, pos=None):
    """12345 -> 12k, 1234567 -> 1.2M."""
    x = float(x)
    for div, suffix in ((1e6, "M"), (1e3, "k")):
        if abs(x) >= div:
            v = x / div
            return f"{v:.0f}{suffix}" if abs(v) >= 10 or v == int(v) else f"{v:.1f}{suffix}"
    return f"{x:.0f}" if x == int(x) else f"{x:.1f}"


def pct_fmt(decimals=None):
    """Percent ticks: 12% / 2.5% / 0.8% (decimals only where the tick needs them)."""
    return FuncFormatter(lambda v, p: f"{round(v, 6):.3g}%")


SIGNED = FuncFormatter(lambda v, p: "0" if abs(v) < 1e-9 else f"{v:+.0f}")


def signed(v):
    return "0" if round(v) == 0 else f"{v:+.0f}"


def log_ticks(axis, fmt=human, subs=(1, 2, 5)):
    """Log axis with ticks at 1-2-5 steps, so short ranges still get labels."""
    axis.set_major_locator(mpl.ticker.LogLocator(base=10, subs=subs))
    axis.set_major_formatter(FuncFormatter(fmt))
    axis.set_minor_formatter(FuncFormatter(lambda v, p: ""))


def day_month(ts):
    ts = pd.Timestamp(ts)
    return f"{ts.day} {ts:%b}"


SHOW_NUMBERS = False   # True: print the exact numbers behind each figure (useful while working, hidden in the report)


def say(*lines, title="Exact numbers"):
    """Exact numbers behind a figure; shown only when SHOW_NUMBERS is True."""
    if not SHOW_NUMBERS:
        return
    body = "\n".join(f"- {line}" for line in lines if line)
    display(Markdown(f"**{title}**\n\n{body}"))


def spread(values, gap):
    """Nudge label positions apart so that end-of-line labels do not overlap."""
    vals = np.asarray(values, float)
    order = np.argsort(vals)
    out = vals.copy()
    for a, b in zip(order[:-1], order[1:]):
        if out[b] - out[a] < gap:
            out[b] = out[a] + gap
    return out


# ---------------------------------------------------------------- figure building blocks
def frame(title, subtitle="", nrows=1, ncols=1, w=10.0, h=4.2, note="", legend=None,
          legend_cols=None, key=None, **kw):
    """Figure with a left-aligned header (number, title, subtitle, legend) and an optional note."""
    FIG["n"] += 1
    if key:
        FIGREF[key] = FIG["n"]
    w, h = w * FIG_SCALE, h * FIG_SCALE
    titles = textwrap.wrap(title, int(w * 10))
    sub = textwrap.wrap(subtitle, int(w * 14.5)) if subtitle else []
    notes = [line for para in note.split("\n") for line in textwrap.wrap(para, int(w * 15))] if note else []
    extra = 0.25 * (len(titles) - 1)
    head = 0.6 + extra + 0.19 * len(sub) + (0.36 if legend else 0.04)
    foot = 0.16 + 0.19 * len(notes) if notes else 0.05
    fig = plt.figure(figsize=(w, head + h + foot), layout="constrained")
    top, body, bottom = fig.subfigures(3, 1, height_ratios=[head, h, foot])
    top.text(0.012, 1 - 0.07 / head, f"FIGURE {FIG['n']}", fontsize=7.8, color=MUTED, weight="bold", va="top")
    top.text(0.012, 1 - 0.24 / head, "\n".join(titles), fontsize=14, color=INK, weight="bold", va="top",
             linespacing=1.2)
    if sub:
        top.text(0.012, 1 - (0.55 + extra) / head, "\n".join(sub), fontsize=9.6, color=INK2, va="top", linespacing=1.38)
    if legend:
        handles, labels = legend
        top.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.004, 0.0),
                   ncol=legend_cols or len(labels), handlelength=1.8, columnspacing=1.8, fontsize=9.2)
    if notes:
        bottom.text(0.012, 1 - 0.07 / foot, "\n".join(notes), fontsize=9.2, color=INK2, va="top", linespacing=1.45)
    axes = body.subplots(nrows, ncols, **kw)
    return fig, axes


def line_h(color, lw=2.2, ls="-", marker=None):
    return Line2D([], [], color=color, lw=lw, ls=ls, marker=marker, markersize=6)


def dot_h(color, marker="o"):
    return Line2D([], [], color=color, marker=marker, ls="none", markersize=7)


def patch_h(color):
    return Patch(facecolor=color, edgecolor="none")


def period_legend(kind="dot"):
    if kind == "dot":
        handles = [Line2D([], [], color=PERIOD_COLOR[p], marker="o", ls="none", markersize=7,
                          markerfacecolor=SURFACE if p == "2022_pre" else PERIOD_COLOR[p], markeredgewidth=2)
                   for p in PERIODS]
        return handles, [PERIOD_LABEL[p] for p in PERIODS]
    make = patch_h if kind == "patch" else line_h
    return [make(PERIOD_COLOR[p]) for p in PERIODS], [PERIOD_LABEL[p] for p in PERIODS]


def period_dot(ax, x, y, p, ms=7.5):
    """One period marker: filled dot for 2019 and wartime, a ring for 2022 before the invasion."""
    if p == "2022_pre":
        ax.plot(x, y, "o", ms=ms - 0.5, mfc="none", mec=PERIOD_COLOR[p], mew=2, zorder=5)
    else:
        ax.plot(x, y, "o", color=PERIOD_COLOR[p], ms=ms, mec=SURFACE, mew=1.2, zorder=3)


def month_axis(ax, year=2022, step=1, fmt="%b"):
    ax.set_xlim(pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-12-31"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=step))
    ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt))


def month_index_axis(ax, short=False):
    ax.set_xlim(0.5, 12.5)
    if short:
        ax.set_xticks([1, 4, 7, 10], ["Jan", "Apr", "Jul", "Oct"])
    else:
        ax.set_xticks(range(1, 13), MONTHS)


def mark_events(ax, events, mode="number", start=1, min_gap_days=10, fontsize=7.4):
    """Vertical hairlines at event dates; numbered circles above the plot or short text labels."""
    lo, hi = ax.get_xlim()
    level, last, last_frac = 0, None, None
    for i, (d, label) in enumerate(events, start):
        x = pd.Timestamp(d)
        ax.axvline(x, color="#a9a79e", lw=0.8, ls=(0, (3, 2)), zorder=1.5)
        if mode == "number":
            level = (1 - level) if last is not None and (x - last).days < min_gap_days else 0
            ax.annotate(str(i), xy=(x, 1), xycoords=("data", "axes fraction"), xytext=(0, 4 + 15 * level),
                        textcoords="offset points", ha="center", va="bottom", fontsize=fontsize, color=INK2,
                        bbox=dict(boxstyle="circle,pad=0.25", fc=SURFACE, ec="#a9a79e", lw=0.8),
                        annotation_clip=False)
            last = x
        else:
            frac = (mdates.date2num(x) - lo) / (hi - lo)
            right = frac > 0.62
            level = 0 if last_frac is None else 1 - level     # several labels in one panel: alternate rows
            last_frac = frac
            ax.annotate(label, xy=(x, 1), xycoords=("data", "axes fraction"), xytext=(-3 if right else 3, -3 - 11 * level),
                        textcoords="offset points", ha="right" if right else "left", va="top",
                        fontsize=fontsize, color=INK2, annotation_clip=False)


def event_key(events, start=1):
    return "Events: " + "   ".join(f"({i}) {day_month(d)} — {lab}" for i, (d, lab) in enumerate(events, start))


def event_effect(daily, events):
    """Records per day in the 3 days from the event / the 7 days before it."""
    out = []
    for d, label in events:
        d = pd.Timestamp(d)
        before = daily.loc[d - pd.Timedelta(days=7): d - pd.Timedelta(days=1)].mean()
        after = daily.loc[d: d + pd.Timedelta(days=2)].mean()
        out.append((f"{day_month(d)} — {label}", after / before if before > 0 else np.nan))
    return pd.DataFrame(out, columns=["event", "ratio"]).dropna().sort_values("ratio")


def weekly_panels(axes, weekly, baseline, keys, color, events=None, titles=None, ylabel="% of core records"):
    """Small multiples: one weekly 2022 line per key, the 2019 average as a dashed line, key events labelled."""
    for ax, k in zip(np.ravel(axes), keys):
        s = weekly[k] if k in weekly else pd.Series(0.0, index=weekly.index)
        b = baseline.get(k, np.nan)
        ax.plot(s.index, s.values, color=color, lw=1.8)
        if np.isfinite(b) and b > 0:
            ax.axhline(b, color=C19, lw=1.3, ls="--")
        top = max(s.max(), b if np.isfinite(b) else 0)
        ax.set_ylim(0, top * 1.3 if top > 0 else 1)
        month_axis(ax, step=3)
        ax.yaxis.set_major_formatter(pct_fmt())
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.set_title(titles.get(k, k) if titles else k)
        mark_events(ax, (events or {}).get(k, []), mode="label")
    for ax in np.ravel(axes)[len(keys):]:
        ax.set_visible(False)
    for ax in np.atleast_2d(axes)[:, 0]:
        ax.set_ylabel(ylabel)


def period_arrows(ax, table, label_fmt="{:.1f}", log=False):
    """One row per item: dots for the three periods, an arrow from 2019 to wartime 2022, value label at the end."""
    for i, (name, r) in enumerate(table.iterrows()):
        v = r.astype(float).replace(0, np.nan) if log else r.astype(float)
        if np.isfinite(v["2019"]) and np.isfinite(v["2022_war"]):
            ax.annotate("", xy=(v["2022_war"], i), xytext=(v["2019"], i),
                        arrowprops=dict(arrowstyle="-|>", color=CONTEXT, lw=1.6, shrinkA=4, shrinkB=5, mutation_scale=10))
        for p in PERIODS:
            period_dot(ax, v[p], i, p)
        if np.isfinite(r["2022_war"]):           # the wartime value, in its own column at the right edge
            ax.annotate(label_fmt.format(r["2022_war"]), xy=(1, i), xycoords=("axes fraction", "data"),
                        xytext=(8, 0), textcoords="offset points", ha="left", va="center", fontsize=8.3, color=INK)
    ax.annotate("from\n24 Feb", xy=(1, 1), xycoords="axes fraction", xytext=(8, 2), textcoords="offset points",
                ha="left", va="bottom", fontsize=7.8, color=INK2, weight="bold", linespacing=1.1)
    ax.set_yticks(range(len(table)), table.index)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)


def heat_colorbar(ax, im, label, fraction=0.03):
    cb = ax.figure.colorbar(im, ax=ax, fraction=fraction, pad=0.01)
    cb.set_label(label, color=INK, fontsize=9.8)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=9, colors=INK2, labelcolor=INK)
    return cb


# ---------------------------------------------------------------- data loading
class Codes:
    """Stable string -> int codes across chunks."""
    def __init__(self):
        self.ids, self.names = {}, []

    def id(self, name):
        i = self.ids.get(name)
        if i is None:
            i = self.ids[name] = len(self.names)
            self.names.append(name)
        return i

    def encode(self, values):
        local, uniq = pd.factorize(pd.Series(values).fillna(""))
        lut = np.array([self.id(u) for u in uniq], dtype=np.int32)
        return lut[local] if len(lut) else np.zeros(len(local), np.int32)


def _num(s, dtype):
    return pd.to_numeric(s, errors="coerce").fillna(0).to_numpy().astype(dtype)


def _flag(s):
    return s.isin(["1", "1.0", "True", "true"]).to_numpy().astype(np.int8)


USECOLS = ["type", "subreddit", "date", "author_country", "score", "thread_id", "text", "topic_source",
           "us_domestic", "is_lifestyle", "countries", "organizations", "countries_focus",
           *SUPPORT_COL.values(), "country_support", "thread_n", "subreddit_day_total",
           "subreddit_month_total", "reaction_ukraine", "reaction_russia", "reaction"]


def prepare(csv_path, chunk=CFG["data"]["chunk_rows"], sample_size=CFG["data"]["text_sample_per_period"]):
    """One pass over the CSV: compact record table + long tables (record × country) + a text sample."""
    t0 = time.time()
    types, subs, srcs, auth, threads, ent = (Codes() for _ in range(6))
    stype = {s: i for i, s in enumerate(SUPPORT)}
    frames, focus, ment, react, supp = [], [], [], [], []
    sample, rng, offset = None, np.random.default_rng(42), 0
    reader = pd.read_csv(csv_path, usecols=USECOLS, dtype=str, keep_default_na=False,
                         chunksize=chunk, encoding="utf-8")
    for ch in reader:
        n = len(ch)
        ch.index = pd.RangeIndex(offset, offset + n)
        date = pd.to_datetime(ch["date"], format="%Y-%m-%d")
        per = np.where(date.dt.year.to_numpy() == 2019, 0,
                       np.where((date < INVASION).to_numpy(), 1, 2)).astype(np.int8)
        f = pd.DataFrame({
            "date": date.to_numpy(), "per": per,
            "type": types.encode(ch["type"]).astype(np.int8),
            "sub": subs.encode(ch["subreddit"]).astype(np.int16),
            "src": srcs.encode(ch["topic_source"]).astype(np.int8),
            "auth": auth.encode(ch["author_country"]).astype(np.int16),
            "thread": threads.encode(ch["thread_id"]).astype(np.int32),
            "score": _num(ch["score"], np.int32),
            "tn": np.maximum(_num(ch["thread_n"], np.int32), 1),
            "smt": _num(ch["subreddit_month_total"], np.int32),
            "sdt": _num(ch["subreddit_day_total"], np.int32),
            "usd": _flag(ch["us_domestic"]), "life": _flag(ch["is_lifestyle"]),
            "nw": (ch["text"].str.count(" ") + 1).to_numpy().astype(np.int32),
            "r_ua": pd.to_numeric(ch["reaction_ukraine"], errors="coerce").to_numpy().astype(np.float32),
            "r_ru": pd.to_numeric(ch["reaction_russia"], errors="coerce").to_numpy().astype(np.float32),
        }, index=ch.index)
        for name, col in SUPPORT_COL.items():
            f["s_" + name] = _flag(ch[col])
        low = ch["text"].str.lower()
        has = {}
        for name, (stems, pat) in KEYWORDS.items():
            cand = np.zeros(n, bool)
            for st in stems:
                if st not in has:
                    has[st] = low.str.contains(st, regex=False).to_numpy(dtype=bool)
                cand |= has[st]
            hit = np.zeros(n, np.int8)
            if cand.any():
                hit[cand] = low[cand].str.contains(pat, regex=True).to_numpy().astype(np.int8)
            f["kw_" + name] = hit
        frames.append(f)

        for col, out in (("countries_focus", focus), ("countries", ment), ("organizations", ment)):
            s = ch[col]
            s = s[s != ""]
            if len(s):
                ex = s.str.split("|").explode()
                out.append((ex.index.to_numpy().astype(np.int32), ent.encode(ex.to_numpy()).astype(np.int16)))

        rr, rt, rv, sr, se, st = [], [], [], [], [], []
        for i, js in zip(range(offset, offset + n), ch["reaction"].to_numpy()):
            if len(js) > 2:
                for t, v in json.loads(js).items():
                    rr.append(i); rt.append(ent.id(t)); rv.append(v)
        for i, js in zip(range(offset, offset + n), ch["country_support"].to_numpy()):
            if len(js) > 2:
                for t, kinds in json.loads(js).items():
                    for k in kinds:
                        k = k.replace("aid_", "")
                        if k in stype:
                            sr.append(i); se.append(ent.id(t)); st.append(stype[k])
        react.append((np.array(rr, np.int32), np.array(rt, np.int16), np.array(rv, np.int8)))
        supp.append((np.array(sr, np.int32), np.array(se, np.int16), np.array(st, np.int8)))

        core = (f["usd"].to_numpy() == 0) & (f["life"].to_numpy() == 0)
        cand = pd.DataFrame({"u": rng.random(n), "per": per, "tn": f["tn"].to_numpy(),
                             "text": ch["text"].to_numpy()})[core]
        sample = cand if sample is None else pd.concat([sample, cand], ignore_index=True)
        sample = sample.sort_values("u").groupby("per", sort=False).head(sample_size).reset_index(drop=True)
        offset += n
        print(f"\r{offset:,} records read · {time.time() - t0:.0f} s", end="")

    R = pd.concat(frames)
    auth.names = [a if a else "(unknown)" for a in auth.names]
    for col, codes in (("type", types), ("sub", subs), ("src", srcs), ("auth", auth)):
        R[col] = pd.Categorical.from_codes(R[col].to_numpy().astype(np.int32), categories=codes.names)

    def long(parts, names):
        cols = [np.concatenate([p[k] for p in parts]) for k in range(len(names))]
        return pd.DataFrame(dict(zip(names, cols)))

    out = {"R": R, "ENT": ent.names,
           "FOCUS": long(focus, ["row", "ent"]), "MENT": long(ment, ["row", "ent"]).drop_duplicates(),
           "REACT": long(react, ["row", "ent", "v"]), "SUPP": long(supp, ["row", "ent", "stype"]),
           "SAMPLE": sample}
    print(f"\nprepared in {time.time() - t0:.0f} s")
    return out


def _attach(R, t, cols):
    """Copy record-level columns onto a long table (row = record position)."""
    rows = t["row"].to_numpy()
    for c in cols:
        src = R[c]
        if isinstance(src.dtype, pd.CategoricalDtype):
            t[c] = pd.Categorical.from_codes(src.cat.codes.to_numpy()[rows], categories=src.cat.categories)
        else:
            t[c] = src.to_numpy()[rows]
    return t


def load_data():
    """Find the CSV, read it (or the cache) and return all tables used in the notebook."""
    dcfg = CFG["data"]
    data_dir = next((Path(p) for p in dcfg["search_dirs"] if (Path(p) / dcfg["csv_name"]).exists()), None)
    if data_dir is None:
        raise FileNotFoundError(f"{dcfg['csv_name']} not found: run the notebook from the repo root "
                                f"(next to Data_parsing/) or edit 'search_dirs' in eda_config.json")
    csv, cache = data_dir / dcfg["csv_name"], data_dir / dcfg["cache_name"]
    stat = csv.stat()
    key = {"size": stat.st_size, "mtime": stat.st_mtime_ns, "version": dcfg["prep_version"]}
    data = None
    if cache.exists():
        try:
            with open(cache, "rb") as fh:
                data = pickle.load(fh)
            if data.get("key") != key:
                data = None
        except Exception:
            data = None
    if data is None:
        data = prepare(csv)
        data["key"] = key
        with open(cache, "wb") as fh:
            pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)

    R, names = data["R"], data["ENT"]
    R["period"] = pd.Categorical.from_codes(R["per"].to_numpy().astype(np.int32), categories=PERIODS)
    R["year"] = R["date"].dt.year.astype(np.int16)
    R["mon"] = R["date"].dt.month.astype(np.int8)
    R["month"] = R["date"] - pd.to_timedelta(R["date"].dt.day - 1, unit="D")
    R["week"] = R["date"] - pd.to_timedelta(R["date"].dt.dayofweek, unit="D")
    R["core"] = (R["usd"] == 0) & (R["life"] == 0)
    R["w"] = (1.0 / np.sqrt(R["tn"].clip(lower=1))).astype(np.float32)   # megathreads count less

    base = ("per", "core", "w", "mon", "year")
    tables = {"FOCUS": base, "MENT": base, "REACT": base + ("period", "sub", "type"), "SUPP": base}
    out = {}
    for name, cols in tables.items():
        t = _attach(R, data[name], cols)
        t["ent"] = pd.Categorical.from_codes(t["ent"].to_numpy().astype(np.int32), categories=names)
        out[name] = t[t["core"]].copy()
    out["SUPP"]["kind"] = pd.Categorical.from_codes(out["SUPP"]["stype"].to_numpy().astype(np.int32), categories=SUPPORT)
    core = R[R["core"]]
    clear_output()   # drop the progress lines from the report
    print(f"Data: {csv}\n{len(R):,} records · {len(core):,} core records · {len(out['FOCUS']):,} country mentions "
          f"in focus · {len(out['REACT']):,} labelled (record, country) tone pairs")
    return dict(R=R, CORE=core, FOCUS=out["FOCUS"], MENT=out["MENT"], RC=out["REACT"], SUPP=out["SUPP"],
                SAMPLE=data["SAMPLE"], N_CORE=core.groupby("per").size(), CSV=csv, REACTIONS_DIR=data_dir / "reactions")


# ---------------------------------------------------------------- statistics
def net_table(df, keys):
    """Weighted share of negative/positive statements and net tone with a standard error, by keys."""
    v = df["v"].to_numpy().astype(float)
    w = df["w"].to_numpy().astype(float)
    t = df[keys].reset_index(drop=True)
    t["sw"], t["swv"], t["sw2"], t["sw2v"], t["sw2v2"] = w, w * v, w * w, w * w * v, w * w * v * v
    t["neg"], t["pos"], t["n"] = w * (v < 0), w * (v > 0), 1
    g = t.groupby(keys, observed=True).sum()
    g["net"] = g["swv"] / g["sw"]
    var = (g["sw2v2"] - 2 * g["net"] * g["sw2v"] + g["net"] ** 2 * g["sw2"]) / g["sw"] ** 2
    g["se"] = np.sqrt(var.clip(lower=0))
    g["neg_share"], g["pos_share"] = g["neg"] / g["sw"], g["pos"] / g["sw"]
    g["neu_share"] = 1 - g["neg_share"] - g["pos_share"]
    return g[["n", "net", "se", "neg_share", "neu_share", "pos_share"]]


def scores(cm):
    """Accuracy, macro-F1 and per-class recall from a confusion matrix (rows = truth)."""
    tp = np.diag(cm).astype(float)
    prec = tp / np.maximum(cm.sum(axis=0), 1)
    rec = tp / np.maximum(cm.sum(axis=1), 1)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-9), 0)
    return tp.sum() / cm.sum(), f1.mean(), rec


TOKEN = re.compile(r"[a-z][a-z'\-]+")


def doc_terms(text):
    """Words and two-word phrases of one text, without stop words, each counted once."""
    toks = [t.strip("'-") for t in TOKEN.findall(text.lower())]
    toks = [t[:-2] if t.endswith("'s") else t for t in toks]
    words = [t for t in toks if len(t) > 2 and t not in STOP]
    bigrams = [f"{a} {b}" for a, b in zip(toks, toks[1:]) if a not in STOP and b not in STOP and len(a) > 2 and len(b) > 2]
    return set(words) | set(bigrams)


def term_counts(texts):
    c = Counter()
    for t in texts:
        c.update(doc_terms(t))
    return c


def log_odds(ca, cb, alpha0=1000, min_count=25):
    """Weighted log-odds ratio with an informative Dirichlet prior (Monroe et al., 2008); z-scores, + = typical of ca."""
    pooled = ca + cb
    vocab = [w for w, n in pooled.items() if n >= min_count]
    na, nb, n0 = sum(ca.values()), sum(cb.values()), sum(pooled.values())
    ya, yb = np.array([ca[w] for w in vocab], float), np.array([cb[w] for w in vocab], float)
    al = np.array([pooled[w] for w in vocab], float) / n0 * alpha0
    d = np.log((ya + al) / (na + alpha0 - ya - al)) - np.log((yb + al) / (nb + alpha0 - yb - al))
    z = d / np.sqrt(1 / (ya + al) + 1 / (yb + al))
    return pd.Series(z, index=vocab)


# ---------------------------------------------------------------- maps (Natural Earth shapes; Crimea is part of Ukraine)
GEO_FILE = HERE / "countries.json"
GEO = json.load(open(GEO_FILE, encoding="utf-8")) if GEO_FILE.exists() else None


def laea(lon, lat, lon0=EUROPE_VIEW["lon0"], lat0=EUROPE_VIEW["lat0"]):
    """Lambert azimuthal equal-area projection (good for a Europe-sized map)."""
    lam, phi = np.radians(lon) - np.radians(lon0), np.radians(lat)
    p0 = np.radians(lat0)
    k = np.sqrt(2 / (1 + np.sin(p0) * np.sin(phi) + np.cos(p0) * np.cos(phi) * np.cos(lam)))
    return k * np.cos(phi) * np.sin(lam), k * (np.cos(p0) * np.sin(phi) - np.sin(p0) * np.cos(phi) * np.cos(lam))


def equal_earth(lon, lat):
    """Equal Earth projection (Šavrič, Patterson & Jenny, 2018) for the world map."""
    lam, phi = np.radians(lon), np.radians(lat)
    th = np.arcsin(np.sqrt(3) / 2 * np.sin(phi))
    a1, a2, a3, a4 = 1.340264, -0.081106, 0.000893, 0.003796
    x = 2 * np.sqrt(3) * lam * np.cos(th) / (3 * (a1 + 3 * a2 * th ** 2 + th ** 6 * (7 * a3 + 9 * a4 * th ** 2)))
    y = th * (a1 + a2 * th ** 2 + th ** 6 * (a3 + a4 * th ** 2))
    return x, y


def draw_map(ax, layer, values, cmap, norm, proj, labels=None, lw=0.45):
    """Choropleth: values = {ISO3: number}; countries without a value are light grey."""
    for f in GEO[layer]:
        verts, codes = [], []
        for poly in f["polys"]:
            for ring in poly:
                arr = np.asarray(ring, float)
                x, y = proj(arr[:, 0], arr[:, 1])
                verts += list(zip(x, y))
                codes += [MPath.MOVETO] + [MPath.LINETO] * (len(ring) - 2) + [MPath.CLOSEPOLY]
        v = values.get(f["a3"], np.nan)
        face = cmap(norm(v)) if np.isfinite(v) else MAP_EMPTY
        ax.add_patch(PathPatch(MPath(verts, codes), facecolor=face, edgecolor=SURFACE, lw=lw))
        if labels and f["a3"] in labels and f["lx"] is not None:
            lx, ly = proj(np.array([f["lx"]]), np.array([f["ly"]]))
            ax.text(lx[0], ly[0], labels[f["a3"]], ha="center", va="center", fontsize=7, color=INK,
                    path_effects=[patheffects.withStroke(linewidth=2.2, foreground=SURFACE)], zorder=5,
                    clip_on=True)
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.axis("off")


def map_colorbar(axes, cmap, norm, label, extend="neither", ticks=None, fmt=None):
    axes = list(np.atleast_1d(axes).ravel())
    cb = axes[0].figure.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axes, orientation="horizontal",
                                 fraction=0.035, pad=0.02, aspect=55, extend=extend)
    if ticks is not None:
        cb.set_ticks(ticks, labels=[fmt(t) if fmt else str(t) for t in ticks])
    cb.set_label(label, color=INK, fontsize=9.8)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=9, colors=INK2, labelcolor=INK)
    return cb


def europe_axes(ax):
    lo1, lo2, la1, la2 = EUROPE_VIEW["extent"]
    lon = np.r_[np.linspace(lo1, lo2, 50), np.full(50, lo2), np.linspace(lo2, lo1, 50), np.full(50, lo1)]
    lat = np.r_[np.full(50, la1), np.linspace(la1, la2, 50), np.full(50, la2), np.linspace(la2, la1, 50)]
    x, y = laea(lon, lat)
    ax.set_xlim(np.percentile(x, 2), np.percentile(x, 98))
    ax.set_ylim(y.min(), y.max())


def in_europe_view(layer="europe"):
    """ISO3 codes whose label point lies inside the Europe map."""
    lo1, lo2, la1, la2 = EUROPE_VIEW["extent"]
    return {f["a3"] for f in GEO[layer] if f["lx"] is not None and lo1 <= f["lx"] <= lo2 and la1 <= f["ly"] <= la2}


def to_a3(series):
    """{entity name: value} -> {ISO3: value}."""
    return {ENT_A3[e]: v for e, v in series.items() if e in ENT_A3 and np.isfinite(v)}


print("matplotlib", mpl.__version__, "| pandas", pd.__version__, "| font:", FONT,
      "| map shapes:", "ok" if GEO else "not found (maps will be skipped)")