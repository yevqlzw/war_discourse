from pathlib import Path
from collections import Counter
import re
import sys
import traceback
import pandas as pd

DATA_DIR = Path("data")

OUTPUT_FILE = DATA_DIR / "reddit_countries_ukraine_support.csv"
MONTHLY_OUTPUT_FILE = DATA_DIR / "monthly_country_counts.csv"

CHUNK_SIZE = 500_000

TEXT_CANDIDATES = [
    "text",
    "body",
    "selftext",
    "title",
    "content",
]

DATE_CANDIDATES = [
    "created_utc",
    "created",
    "timestamp",
    "date",
    "datetime",
]

ID_CANDIDATES = [
    "id",
    "post_id",
    "comment_id",
]

COMPUTED_COLUMNS = [
    "countries",
    "is_ukraine_direct",
    "is_russia_direct",
    "is_core_war",
    "is_aid",
    "is_econ",
    "is_political",
    "is_critic",
    "is_noise",
    "relevance_score",
    "signal_count",
    "relevance_level",
    "matched_signals",
]


COUNTRY_PATTERNS = {
    "USA": (
        r"\b("
        r"usa|u\.s\.a\.|u\.s\.|"
        r"united states|"
        r"america|american|americans|"
        r"biden|blinken|austin|nuland|pelosi|"
        r"white house|washington|pentagon|"
        r"state department|"
        r"department of defense|department of defence|"
        r"congress|senate|"
        r"house of representatives|"
        r"capitol hill|"
        r"nato|cia|"
        r"us army|us military|"
        r"american troops|"
        r"american weapons|"
        r"american aid|"
        r"american government|"
        r"us government|"
        r"us military aid|"
        r"us weapons|"
        r"us sanctions"
        r")\b"
    ),
    "UK": (
        r"\b("
        r"uk|u\.k\.|"
        r"united kingdom|"
        r"britain|british|"
        r"england|english|"
        r"scotland|scottish|"
        r"wales|welsh|"
        r"johnson|truss|sunak|wallace|"
        r"downing street|london|westminster|"
        r"parliament|"
        r"house of commons|"
        r"house of lords|"
        r"foreign office|"
        r"uk government|"
        r"british government|"
        r"british army|"
        r"british military|"
        r"british weapons|"
        r"british aid"
        r")\b"
    ),
    "Poland": (
        r"\b("
        r"poland|polish|"
        r"duda|morawiecki|tusk|"
        r"warsaw|warszawa|"
        r"polish government|"
        r"polish parliament|"
        r"sejm|"
        r"polish army|"
        r"polish military|"
        r"polish border|"
        r"polish weapons|"
        r"polish aid|"
        r"polish defense|"
        r"polish defence"
        r")\b"
    ),
    "Germany": (
        r"\b("
        r"germany|german|"
        r"scholz|baerbock|steinmeier|"
        r"berlin|bundestag|bundesrat|"
        r"german government|"
        r"german parliament|"
        r"bundeswehr|"
        r"german army|"
        r"german military|"
        r"german weapons|"
        r"german aid|"
        r"german defense|"
        r"german defence|"
        r"leopard tanks|"
        r"leopard 2|"
        r"rheinmetall"
        r")\b"
    ),
    "France": (
        r"\b("
        r"france|french|"
        r"macron|paris|"
        r"elysee|"
        r"élysée|"
        r"french government|"
        r"french parliament|"
        r"national assembly|"
        r"french senate|"
        r"french army|"
        r"french military|"
        r"french weapons|"
        r"french aid|"
        r"french defense|"
        r"french defence|"
        r"leclerc tanks"
        r")\b"
    ),
    "Hungary": (
        r"\b("
        r"hungary|hungarian|"
        r"orban|orbán|budapest|"
        r"hungarian government|"
        r"hungarian parliament|"
        r"hungarian army|"
        r"hungarian military|"
        r"hungarian border|"
        r"hungarian weapons|"
        r"hungarian aid"
        r")\b"
    ),
    "Turkey": (
        r"\b("
        r"turkey|turkish|"
        r"erdogan|erdoğan|ankara|"
        r"turkish government|"
        r"turkish parliament|"
        r"turkish army|"
        r"turkish military|"
        r"turkish weapons|"
        r"turkish aid|"
        r"bayraktar|tb2|"
        r"turkish drones"
        r")\b"
    ),
    "Canada": (
        r"\b("
        r"canada|canadian|"
        r"trudeau|ottawa|"
        r"canadian government|"
        r"canadian parliament|"
        r"house of commons canada|"
        r"canadian army|"
        r"canadian military|"
        r"canadian weapons|"
        r"canadian aid|"
        r"canadian sanctions"
        r")\b"
    ),
    "China": (
        r"\b("
        r"china|chinese|"
        r"beijing|"
        r"xi jinping|"
        r"chinese communist party|"
        r"communist party of china|"
        r"ccp|"
        r"chinese government|"
        r"chinese military|"
        r"pla|"
        r"people's liberation army|"
        r"chinese weapons|"
        r"chinese aid|"
        r"chinese sanctions"
        r")\b"
    ),
    "Czechia": (
        r"\b("
        r"czechia|czech republic|czech|"
        r"prague|fiala|petr fiala|"
        r"czech government|"
        r"czech parliament|"
        r"czech army|"
        r"czech military|"
        r"czech weapons|"
        r"czech aid|"
        r"czech ammunition"
        r")\b"
    ),
    "Slovakia": (
        r"\b("
        r"slovakia|slovak|"
        r"bratislava|"
        r"slovak government|"
        r"slovak parliament|"
        r"slovak army|"
        r"slovak military|"
        r"slovak weapons|"
        r"slovak aid|"
        r"mig-29|mig 29"
        r")\b"
    ),
    "Romania": (
        r"\b("
        r"romania|romanian|"
        r"bucharest|"
        r"romanian government|"
        r"romanian parliament|"
        r"romanian army|"
        r"romanian military|"
        r"romanian weapons|"
        r"romanian aid"
        r")\b"
    ),
    "Lithuania": (
        r"\b("
        r"lithuania|lithuanian|"
        r"vilnius|"
        r"lithuanian government|"
        r"lithuanian parliament|"
        r"lithuanian army|"
        r"lithuanian military|"
        r"lithuanian weapons|"
        r"lithuanian aid"
        r")\b"
    ),
    "Latvia": (
        r"\b("
        r"latvia|latvian|"
        r"riga|"
        r"latvian government|"
        r"latvian parliament|"
        r"latvian army|"
        r"latvian military|"
        r"latvian weapons|"
        r"latvian aid"
        r")\b"
    ),
    "Estonia": (
        r"\b("
        r"estonia|estonian|"
        r"tallinn|"
        r"estonian government|"
        r"estonian parliament|"
        r"estonian army|"
        r"estonian military|"
        r"estonian weapons|"
        r"estonian aid"
        r")\b"
    ),
    "Baltics": (
        r"\b("
        r"baltic|baltic states|"
        r"baltic countries|"
        r"baltic military|"
        r"baltic aid"
        r")\b"
    ),
    "Italy": (
        r"\b("
        r"italy|italian|"
        r"rome|draghi|meloni|"
        r"italian government|"
        r"italian parliament|"
        r"italian army|"
        r"italian military|"
        r"italian weapons|"
        r"italian aid"
        r")\b"
    ),
    "Netherlands": (
        r"\b("
        r"netherlands|dutch|"
        r"holland|"
        r"hague|amsterdam|"
        r"dutch government|"
        r"dutch parliament|"
        r"dutch army|"
        r"dutch military|"
        r"dutch weapons|"
        r"dutch aid"
        r")\b"
    ),
    "Sweden": (
        r"\b("
        r"sweden|swedish|"
        r"stockholm|"
        r"swedish government|"
        r"swedish parliament|"
        r"swedish army|"
        r"swedish military|"
        r"swedish weapons|"
        r"swedish aid"
        r")\b"
    ),
    "Finland": (
        r"\b("
        r"finland|finnish|"
        r"helsinki|"
        r"finnish government|"
        r"finnish parliament|"
        r"finnish army|"
        r"finnish military|"
        r"finnish weapons|"
        r"finnish aid"
        r")\b"
    ),
}


UKRAINE_DIRECT_RE = (
    r"\b("
    r"ukraine|ukrainian|ukrainians|kyiv|kiev|"
    r"zelensky|zelenskyy|zelenskiy|volodymyr zelensky|"
    r"verkhovna rada|armed forces of ukraine|ukrainian armed forces|"
    r"afu|zsu|crimea|crimean|donbas|donbass|donetsk|luhansk|lugansk|"
    r"kharkiv|kherson|mariupol|zaporizhzhia|zaporizhya|odesa|odessa|"
    r"dnipro|bakhmut|avdiivka|bucha|irpin|melitopol|soledar|"
    r"chernihiv|sumy|mykolaiv|chernobyl|chornobyl"
    r")\b"
)

RUSSIA_DIRECT_RE = (
    r"\b("
    r"russia|russian|russians|russian federation|"
    r"putin|vladimir putin|moscow|kremlin|"
    r"russian government|russian military|russian army|"
    r"russian forces|russian armed forces|russian troops|"
    r"wagner|wagner group|prigozhin|shoigu|lavrov|medvedev|fsb|gru"
    r")\b"
)

CORE_WAR_RE = (
    r"\b("
    r"war|warfare|invasion|invade|invaded|invading|"
    r"occupation|occupied|occupying|conflict|hostilities|"
    r"battle|battles|frontline|front line|frontlines|"
    r"offensive|counteroffensive|counter-offensive|"
    r"retreat|advance|attack|attacked|attacking|assault|"
    r"shelling|shell|bombing|bombed|airstrike|air strike|airstrikes|"
    r"missile strike|missile strikes|drone strike|drone strikes|"
    r"artillery strike|artillery strikes|military|army|troops|soldiers|"
    r"armed forces|air defense|air defence|mobilization|mobilisation|"
    r"conscription|reservists|artillery|tanks|tank|missiles|missile|"
    r"drones|drone|himars|javelin|stinger|nlaw|patriot|nasams|"
    r"leopard|leopard 2|challenger|abrams|caesar|howitzer|"
    r"rocket launcher|mlrs|nuclear|nuclear power plant|radiation|"
    r"civilian|civilians|refugee|refugees|displaced|humanitarian|"
    r"massacre|war crime|war crimes|atrocity|atrocities|"
    r"peace talks|peace negotiations|ceasefire|truce|"
    r"annexation|annexed|military aid|weapons shipment|arms shipment"
    r")\b"
)

AID_RE = (
    r"\b("
    r"aid|military aid|financial aid|humanitarian aid|foreign aid|"
    r"weapons aid|arms aid|support package|aid package|"
    r"military assistance|financial assistance|humanitarian assistance|"
    r"funding|funds|donation|donations|sanctions|sanction package"
    r")\b"
)

ECON_RE = (
    r"\b("
    r"economy|economic|economics|gdp|inflation|"
    r"interest rates|interest rate|currency|exchange rate|"
    r"trade|exports|imports|energy prices|gas prices|oil prices|"
    r"oil|gas|natural gas|sanctions|financial markets|markets|"
    r"banking|bank|debt|bonds|budget|government spending|"
    r"unemployment|recession|energy crisis|food prices|"
    r"grain|wheat|fertilizer|fertiliser|grain exports"
    r")\b"
)

POLITICAL_RE = (
    r"\b("
    r"president|prime minister|government|parliament|congress|senate|"
    r"election|elections|political|politics|politician|politicians|"
    r"minister|ministry|foreign policy|foreign affairs|"
    r"diplomacy|diplomatic|sanctions|treaty|agreement|negotiations|"
    r"peace talks|ceasefire|referendum|annexation|"
    r"territorial integrity|sovereignty|nato|european union|"
    r"united nations|security council"
    r")\b"
)

CRITIC_RE = (
    r"\b("
    r"critic|criticism|criticize|criticise|criticized|criticised|"
    r"blame|blamed|accuse|accused|condemn|condemned|"
    r"propaganda|disinformation|misinformation|fake news|"
    r"corruption|controversy|controversial|failure|failed|"
    r"responsibility|responsible|support|oppose|opposition"
    r")\b"
)

NOISE_RE = (
    r"\b("
    r"football|soccer|basketball|baseball|hockey|tennis|cricket|"
    r"movie|movies|tv show|television|celebrity|music|concert|"
    r"gaming|video game|videogame|recipe|cooking|weather|"
    r"vacation|travel|tourism|restaurant|food|fashion|"
    r"unrelated|off topic|off-topic"
    r")\b"
)



COMPILED_COUNTRIES = {
    country: re.compile(pattern, re.IGNORECASE)
    for country, pattern in COUNTRY_PATTERNS.items()
}

UKRAINE_DIRECT_COMP = re.compile(UKRAINE_DIRECT_RE, re.IGNORECASE)
RUSSIA_DIRECT_COMP = re.compile(RUSSIA_DIRECT_RE, re.IGNORECASE)
CORE_WAR_COMP = re.compile(CORE_WAR_RE, re.IGNORECASE)
AID_COMP = re.compile(AID_RE, re.IGNORECASE)
ECON_COMP = re.compile(ECON_RE, re.IGNORECASE)
POLITICAL_COMP = re.compile(POLITICAL_RE, re.IGNORECASE)
CRITIC_COMP = re.compile(CRITIC_RE, re.IGNORECASE)
NOISE_COMP = re.compile(NOISE_RE, re.IGNORECASE)


def find_column(columns, candidates):
    lower_map = {str(col).lower(): col for col in columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    return None


def detect_countries(text):
    if not isinstance(text, str):
        return []
    return [country for country, pattern in COMPILED_COUNTRIES.items() if pattern.search(text)]


def matched_signals(row):
    signals = []
    if row["is_ukraine_direct"]:
        signals.append("ukraine_direct")
    if row["is_russia_direct"]:
        signals.append("russia_direct")
    if row["is_core_war"]:
        signals.append("war")
    if row["is_aid"]:
        signals.append("aid")
    if row["is_econ"]:
        signals.append("economic")
    if row["is_political"]:
        signals.append("political")
    if row["is_critic"]:
        signals.append("critic")
    if row["is_noise"]:
        signals.append("noise")
    return "|".join(signals)


def process_chunk(df):
    text_col = find_column(df.columns, TEXT_CANDIDATES)
    if text_col is None:
        raise ValueError(f"Could not find text column. Available: {list(df.columns)}")

    text = df[text_col].fillna("").astype(str)

    df["countries"] = text.apply(detect_countries)
    has_country = df["countries"].map(len) > 0

    df["is_ukraine_direct"] = text.str.contains(UKRAINE_DIRECT_COMP, na=False)
    df["is_russia_direct"] = text.str.contains(RUSSIA_DIRECT_COMP, na=False)
    df["is_core_war"] = text.str.contains(CORE_WAR_COMP, na=False)
    df["is_aid"] = text.str.contains(AID_COMP, na=False)
    df["is_econ"] = text.str.contains(ECON_COMP, na=False)
    df["is_political"] = text.str.contains(POLITICAL_COMP, na=False)
    df["is_critic"] = text.str.contains(CRITIC_COMP, na=False)
    df["is_noise"] = text.str.contains(NOISE_COMP, na=False)

    score = pd.Series(0, index=df.index, dtype="int16")
    score += df["is_ukraine_direct"].astype("int16") * 6
    score += df["is_russia_direct"].astype("int16") * 5
    score += df["is_core_war"].astype("int16") * 3
    score += df["is_aid"].astype("int16") * 3
    score += df["is_econ"].astype("int16") * 2
    score += df["is_political"].astype("int16") * 2
    score += df["is_critic"].astype("int16") * 1
    score -= df["is_noise"].astype("int16") * 4
    score -= (text.str.len() < 30).astype("int16") * 1
    df["relevance_score"] = score

    df["signal_count"] = (
        df["is_ukraine_direct"].astype(int)
        + df["is_russia_direct"].astype(int)
        + df["is_core_war"].astype(int)
        + df["is_aid"].astype(int)
        + df["is_econ"].astype(int)
        + df["is_political"].astype(int)
        + df["is_critic"].astype(int)
    )

    df["relevance_level"] = "weak"
    strong_mask = (
        ((df["is_ukraine_direct"] | df["is_russia_direct"]) & (df["relevance_score"] >= 5))
        | (df["relevance_score"] >= 8)
    )
    medium_mask = ~strong_mask & (df["relevance_score"] >= 4)
    df.loc[strong_mask, "relevance_level"] = "strong"
    df.loc[medium_mask, "relevance_level"] = "medium"

    noise_mask = df["is_noise"] & ~(df["is_ukraine_direct"] | df["is_russia_direct"])
    df.loc[noise_mask, "relevance_level"] = "noise"

    df["matched_signals"] = df.apply(matched_signals, axis=1)

    direct_signal = df["is_ukraine_direct"] | df["is_russia_direct"]
    strong_context = df["is_core_war"] | df["is_aid"] | df["is_econ"]
    political_context = df["is_political"] | df["is_critic"]

    keep_mask = (
        has_country
        & ~noise_mask
        & (
            direct_signal
            | (strong_context & (df["relevance_score"] >= 3))
            | (political_context & (df["relevance_score"] >= 4))
            | (political_context & strong_context & (df["relevance_score"] >= 3))
            | (df["relevance_score"] >= 6)
            | (df["signal_count"] >= 3)
        )
    )

    return df.loc[keep_mask].copy()


def postprocess_selected(selected, id_col, date_col):
    if id_col is not None and id_col in selected.columns and id_col != "id":
        selected = selected.rename(columns={id_col: "id"})

    if date_col is not None and date_col in selected.columns:
        if date_col == "created_utc":
            selected["date"] = pd.to_datetime(selected[date_col], unit="s", errors="coerce", utc=True)
        else:
            selected["date"] = pd.to_datetime(selected[date_col], errors="coerce", utc=True)

    selected["countries"] = selected["countries"].apply(
        lambda x: ", ".join(x) if isinstance(x, list) else str(x)
    )

    return selected


def build_global_columns(input_files):
    union_cols = []
    seen = set()
    id_col_global = None
    date_col_global = None

    for f in input_files:
        try:
            header_df = pd.read_csv(f, nrows=0)
        except Exception as e:
            print(f"  [WARN] Could not read header of {f.name}: {e}")
            continue

        cols = list(header_df.columns)

        if id_col_global is None:
            id_col_global = find_column(cols, ID_CANDIDATES)
        if date_col_global is None:
            date_col_global = find_column(cols, DATE_CANDIDATES)

        for c in cols:
            if c not in seen:
                seen.add(c)
                union_cols.append(c)

    if id_col_global is not None and id_col_global in union_cols and id_col_global != "id":
        idx = union_cols.index(id_col_global)
        union_cols[idx] = "id"
        if union_cols.count("id") > 1:
            first = union_cols.index("id")
            union_cols = [c for i, c in enumerate(union_cols) if c != "id" or i == first]

    final_columns = list(union_cols)

    if date_col_global is not None and "date" not in final_columns:
        final_columns.append("date")

    for c in COMPUTED_COLUMNS:
        if c not in final_columns:
            final_columns.append(c)

    return final_columns, id_col_global, date_col_global


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    input_files = sorted(DATA_DIR.glob("*_2022.csv"))
    if not input_files:
        raise FileNotFoundError(f"No *_2022.csv files found in {DATA_DIR.resolve()}")

    print("=" * 70)
    print("REDDIT 2022 FILTER — STREAMING VERSION")
    print("=" * 70)
    print(f"Input directory: {DATA_DIR.resolve()}")
    print(f"Files found: {len(input_files)}")
    for file in input_files:
        print(f"  - {file.name}")

    final_columns, id_col_global, date_col_global = build_global_columns(input_files)

    print()
    print(f"Global id column:   {id_col_global}")
    print(f"Global date column: {date_col_global}")
    print(f"Output columns:     {len(final_columns)}")

    total_rows = 0
    total_selected = 0
    header_written = False

    monthly_counter = Counter()
    relevance_level_counter = Counter()
    score_counter = Counter()
    country_counter = Counter()
    signal_totals = Counter()
    matched_signals_counter = Counter()

    signal_columns = [
        "is_ukraine_direct", "is_russia_direct", "is_core_war",
        "is_aid", "is_econ", "is_political", "is_critic", "is_noise",
    ]

    for input_file in input_files:
        print()
        print("-" * 70)
        print(f"Processing: {input_file.name}")
        print("-" * 70)

        file_total = 0
        file_selected = 0

        try:
            reader = pd.read_csv(
                input_file, chunksize=CHUNK_SIZE, low_memory=False,
                encoding="utf-8", encoding_errors="replace", on_bad_lines="skip",
            )
        except TypeError:
            reader = pd.read_csv(input_file, chunksize=CHUNK_SIZE, low_memory=False)
        except Exception as e:
            print(f"  [ERROR] Could not open {input_file.name}: {e}")
            continue

        chunk_number = 0
        while True:
            try:
                chunk = next(reader)
            except StopIteration:
                break
            except Exception as e:
                print(f"  [ERROR] Failed reading a chunk from {input_file.name}: {e}")
                break

            chunk_number += 1
            print(f"Chunk {chunk_number}: {len(chunk):,} rows")

            file_total += len(chunk)
            total_rows += len(chunk)

            try:
                selected = process_chunk(chunk)
            except Exception as e:
                print(f"  [ERROR] process_chunk failed on chunk {chunk_number}: {e}")
                traceback.print_exc(file=sys.stdout)
                continue

            if selected.empty:
                print("  selected: 0")
                continue

            try:
                selected = postprocess_selected(selected, id_col_global, date_col_global)
            except Exception as e:
                print(f"  [ERROR] postprocess_selected failed: {e}")
                traceback.print_exc(file=sys.stdout)
                continue

            file_selected += len(selected)
            total_selected += len(selected)
            print(f"  selected: {len(selected):,}")

            relevance_level_counter.update(selected["relevance_level"].tolist())
            score_counter.update(selected["relevance_score"].tolist())
            matched_signals_counter.update(selected["matched_signals"].tolist())

            for col in signal_columns:
                if col in selected.columns:
                    signal_totals[col] += int(selected[col].sum())

            countries_series = selected["countries"].astype(str)
            exploded_countries = countries_series.str.split(", ").explode()
            exploded_countries = exploded_countries[exploded_countries != ""]
            country_counter.update(exploded_countries.tolist())

            if "date" in selected.columns:
                months = selected["date"].dt.to_period("M").astype(str)
                per_row_countries = countries_series.str.split(", ")
                tmp = pd.DataFrame({"month": months, "country": per_row_countries})
                tmp = tmp.explode("country")
                tmp = tmp[tmp["country"].notna() & (tmp["country"] != "")]
                for (month, country), cnt in tmp.groupby(["month", "country"]).size().items():
                    monthly_counter[(month, country)] += int(cnt)

            out_df = selected.reindex(columns=final_columns)
            try:
                out_df.to_csv(
                    OUTPUT_FILE,
                    mode="a" if header_written else "w",
                    header=not header_written,
                    index=False,
                )
                header_written = True
            except Exception as e:
                print(f"  [ERROR] Failed writing chunk: {e}")

        print(f"\nFile total:    {file_total:,}")
        print(f"File selected: {file_selected:,}")

    if not header_written:
        print("\nNo matching rows found.")
        return

    if monthly_counter:
        monthly_rows = [
            {"month": month, "country": country, "count": cnt}
            for (month, country), cnt in monthly_counter.items()
        ]
        monthly_counts = pd.DataFrame(monthly_rows)
        monthly_counts = monthly_counts.sort_values(["month", "count"], ascending=[True, False])
        monthly_counts.to_csv(MONTHLY_OUTPUT_FILE, index=False)

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"Total input rows:    {total_rows:,}")
    print(f"Total selected rows: {total_selected:,}")
    if total_rows > 0:
        print(f"Selection rate:      {total_selected / total_rows:.2%}")
    print(f"\nOutput: {OUTPUT_FILE.resolve()}")
    if monthly_counter:
        print(f"Monthly: {MONTHLY_OUTPUT_FILE.resolve()}")

    print("\n" + "-" * 70)
    print("RELEVANCE LEVEL")
    print("-" * 70)
    for level, cnt in relevance_level_counter.most_common():
        print(f"{level:10s}{cnt:10,}")

    print("\n" + "-" * 70)
    print("COUNTRIES")
    print("-" * 70)
    for country, cnt in country_counter.most_common():
        print(f"{country:15s}{cnt:10,}")

    print("FINISHED")

if __name__ == "__main__":
    main()