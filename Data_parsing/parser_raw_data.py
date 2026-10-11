import zstandard as zstd
import csv
import re
from datetime import datetime, timezone
from pathlib import Path
import io
from collections import Counter

from tqdm import tqdm


try:
    import orjson
    def json_loads(line):
        return orjson.loads(line)
    JSONDecodeError = orjson.JSONDecodeError
    JSON_PARSER = "orjson"
except ImportError:
    import json
    def json_loads(line):
        return json.loads(line)
    JSONDecodeError = json.JSONDecodeError
    JSON_PARSER = "json (stdlib)"


BASE_DIR = Path(__file__).parent
INPUT_PATH = BASE_DIR / "data" / "reddit"
OUTPUT_DIR = BASE_DIR / "output"


SUBREDDITS = {
    "ukraine",
    "ukrainianconflict",
    "ukrainewarreport",
    "worldnews",
    "europe",
    "nato",
    "geopolitics",
    "credibledefense",
}


# Minimum text length
MIN_TEXT_LENGTH = 15

# How often to update the progress bar info
UPDATE_EVERY = 10_000


# ============================================================
# PROGRESS READER (for tqdm)
# ============================================================

class ProgressReader:
    """File wrapper that reports read progress to tqdm."""

    def __init__(self, file, progress_bar):
        self.file = file
        self.progress_bar = progress_bar

    def read(self, size=-1):
        data = self.file.read(size)
        if data:
            self.progress_bar.update(len(data))
        return data

    def readinto(self, buffer):
        count = self.file.readinto(buffer)
        if count:
            self.progress_bar.update(count)
        return count

    def readable(self):
        return True

    def seekable(self):
        return False

    def close(self):
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ============================================================
# READING .ZST
# ============================================================

def read_zst_lines(file_path: Path, progress_bar):
    """Streams a .zst file line by line (low RAM)."""
    with open(file_path, "rb") as raw_file:
        progress_file = ProgressReader(raw_file, progress_bar)
        dctx = zstd.ZstdDecompressor(max_window_size=2**31)

        with dctx.stream_reader(progress_file) as stream_reader:
            text_stream = io.TextIOWrapper(
                stream_reader,
                encoding="utf-8",
                errors="replace",
            )
            for line in text_stream:
                line = line.strip()
                if line:
                    yield line


# ============================================================
# MONTH FROM FILE NAME
# ============================================================

MONTH_PATTERN = re.compile(r"(\d{4}-\d{2})")

def extract_month(filename: str):
    """RS_2024-04.zst -> '2024-04'. Returns None if not found."""
    match = MONTH_PATTERN.search(filename)
    return match.group(1) if match else None


# ============================================================
# HELPER: clean text
# ============================================================

def clean_text(value):
    """Removes line breaks and returns str."""
    if not value:
        return ""
    return str(value).replace("\n", " ").replace("\r", " ").strip()


def safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# ============================================================
# PROCESSING ONE FILE
# ============================================================

def process_file(file_path: Path, writer, output_file):
    """Processes one .zst and writes the filtered rows to CSV."""

    is_submission = file_path.name.startswith("RS_")
    record_type = "submission" if is_submission else "comment"

    total_lines = 0
    matched = 0
    subreddit_counter = Counter()

    file_size = file_path.stat().st_size

    with tqdm(
        total=file_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=file_path.name[:30],
        dynamic_ncols=True,
    ) as progress_bar:

        for line in read_zst_lines(file_path, progress_bar):
            total_lines += 1

            # --- JSON ---
            try:
                obj = json_loads(line)
            except JSONDecodeError:
                continue

            # --- subreddit ---
            subreddit = (obj.get("subreddit") or "").lower()
            if subreddit not in SUBREDDITS:
                continue

            # --- text, title, selftext, body (separately) ---
            if is_submission:
                title = clean_text(obj.get("title") or "")
                selftext = clean_text(obj.get("selftext") or "")
                if selftext in ("[deleted]", "[removed]"):
                    selftext = ""
                body = ""
                text = (title + " " + selftext).strip()
            else:
                title = ""
                selftext = ""
                body = clean_text(obj.get("body") or "")
                if body in ("[deleted]", "[removed]"):
                    continue
                text = body

            if not text or len(text) < MIN_TEXT_LENGTH:
                continue

            # --- datetime ---
            try:
                ts = int(obj.get("created_utc", 0))
                dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                datetime_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                month = dt.strftime("%Y-%m")
            except Exception:
                continue

            # --- extra fields ---
            author = obj.get("author", "")
            author_flair = clean_text(obj.get("author_flair_text") or "")

            # Posts only
            if is_submission:
                num_comments = safe_int(obj.get("num_comments", 0))
                upvote_ratio = safe_float(obj.get("upvote_ratio", 0.0))
                url = obj.get("url", "")
                is_self = bool(obj.get("is_self", False))
                link_id = ""
                parent_id = ""
            else:
                num_comments = 0
                upvote_ratio = 0.0
                url = ""
                is_self = False
                link_id = obj.get("link_id", "")
                parent_id = obj.get("parent_id", "")

            permalink = obj.get("permalink", "")

            # --- row (19 columns) ---
            row = {
                # Basic
                "id": obj.get("id", ""),
                "type": record_type,
                "subreddit": subreddit,
                "datetime": datetime_str,
                "month": month,

                # Author
                "author": author,
                "author_flair": author_flair,

                # Score / engagement
                "score": safe_int(obj.get("score", 0)),
                "num_comments": num_comments,
                "upvote_ratio": upvote_ratio,

                # Text parts
                "title": title,
                "selftext": selftext,
                "body": body,
                "text": text,

                # Links
                "link_id": link_id,
                "parent_id": parent_id,
                "permalink": permalink,

                # Post-specific
                "url": url,
                "is_self": is_self,
            }

            writer.writerow(row)
            matched += 1
            subreddit_counter[subreddit] += 1

            if total_lines % UPDATE_EVERY == 0:
                progress_bar.set_postfix(
                    lines=f"{total_lines:,}",
                    matched=f"{matched:,}",
                )

    return total_lines, matched, subreddit_counter


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 19 columns, extended version
    fieldnames = [
        # Basic
        "id", "type", "subreddit", "datetime", "month",

        # Author
        "author", "author_flair",

        # Engagement
        "score", "num_comments", "upvote_ratio",

        # Text
        "title", "selftext", "body", "text",

        # Links
        "link_id", "parent_id", "permalink",

        # Post-specific
        "url", "is_self",
    ]

    # --- Find the files ---
    if INPUT_PATH.is_file():
        zst_files = [INPUT_PATH]
    elif INPUT_PATH.is_dir():
        zst_files = sorted(INPUT_PATH.rglob("*.zst"))
    else:
        print(f"Not found: {INPUT_PATH}")
        return

    if not zst_files:
        print(f"No .zst files in: {INPUT_PATH}")
        return

    print("=" * 70)
    print("Reddit parser, extended (19 columns)")
    print("=" * 70)
    print(f"JSON parser:      {JSON_PARSER}")
    print(f".zst files found: {len(zst_files)}")
    print(f"Output folder:    {OUTPUT_DIR}")
    print()

    total_lines_all = 0
    total_matched_all = 0
    total_skipped = 0
    total_processed = 0
    total_subreddits = Counter()
    errors = []

    for index, file_path in enumerate(zst_files, start=1):

        month = extract_month(file_path.name)
        rec_type = "submissions" if file_path.name.startswith("RS_") else "comments"

        if month is None:
            print(f"[{index}/{len(zst_files)}] Skipped (no month in file name): {file_path.name}")
            continue

        output_csv = OUTPUT_DIR / f"reddit_{month}_{rec_type}.csv"

        if output_csv.exists() and output_csv.stat().st_size > 0:
            print(f"[{index}/{len(zst_files)}] Already exists: {output_csv.name}")
            total_skipped += 1
            continue

        print()
        print(f"[{index}/{len(zst_files)}] Processing: {file_path.name}")
        print(f"                   -> {output_csv.name}")

        try:
            with open(output_csv, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                f.flush()

                lines, matched, subreddit_stats = process_file(
                    file_path, writer, f
                )

            total_lines_all += lines
            total_matched_all += matched
            total_processed += 1
            total_subreddits.update(subreddit_stats)

            print(f"  Lines: {lines:,} | kept: {matched:,}")
            for sub, count in sorted(subreddit_stats.items()):
                print(f"       r/{sub:<25} {count:,}")

        except zstd.ZstdError as e:
            print(f"  ZSTD error: {e}")
            errors.append((file_path.name, str(e)))
            if output_csv.exists():
                output_csv.unlink()
            continue

        except OSError as e:
            print(f"  File read error: {e}")
            errors.append((file_path.name, str(e)))
            if output_csv.exists():
                output_csv.unlink()
            continue

        except MemoryError:
            print("  Not enough RAM. Close other programs.")
            if output_csv.exists():
                output_csv.unlink()
            return

        except KeyboardInterrupt:
            print("\n  Interrupted by user.")
            if output_csv.exists():
                output_csv.unlink()
            break

    # ========================================================
    # FINAL STATS
    # ========================================================

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"Files processed:       {total_processed}")
    print(f"Skipped (exist):       {total_skipped}")
    print(f"Errors:                {len(errors)}")
    print()
    print(f"Total lines:           {total_lines_all:,}")
    print(f"Total kept:            {total_matched_all:,}")

    if total_subreddits:
        print()
        print("By subreddit:")
        for sub, count in sorted(total_subreddits.items(), key=lambda x: -x[1]):
            print(f"  r/{sub:<25} {count:,}")

    if errors:
        print()
        print("Files with errors:")
        for name, err in errors:
            print(f"  {name}: {err}")

    print()
    print(f"All CSV files saved to: {OUTPUT_DIR}")

    csv_files = sorted(OUTPUT_DIR.glob("reddit_*.csv"))
    if csv_files:
        print()
        print(f"CSV files: {len(csv_files)}")
        for csv_file in csv_files:
            size_mb = csv_file.stat().st_size / (1024 * 1024)
            print(f"  {csv_file.name:<40} {size_mb:>7.2f} MB")


if __name__ == "__main__":
    main()