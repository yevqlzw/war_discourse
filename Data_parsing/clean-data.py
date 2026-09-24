from pathlib import Path
import pandas as pd

INPUT_FILE = Path("../data/reddit_countries_ukraine_support.csv")
OUTPUT_FILE = Path("../data/reddit_countries_ukraine_support_strong_only.csv")

CHUNK_SIZE = 200_000


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE.resolve()}")

    print("=" * 70)
    print("FILTER: KEEP ONLY STRONG + REMOVE DUPLICATES")
    print("=" * 70)
    print(f"Input:  {INPUT_FILE.resolve()}")
    print(f"Output: {OUTPUT_FILE.resolve()}")
    print()

    total_rows = 0
    kept_rows = 0
    header_written = False
    seen_ids = set()          # для відстеження дублікатів між чанками

    for chunk_number, chunk in enumerate(
        pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE, low_memory=False),
        start=1
    ):
        total_rows += len(chunk)

        mask = chunk["relevance_level"] == "strong"
        cleaned = chunk[mask].copy()

        if "id" in cleaned.columns:
            cleaned = cleaned.drop_duplicates(subset=["id"], keep="first")

            before = len(cleaned)
            cleaned = cleaned[~cleaned["id"].isin(seen_ids)]
            seen_ids.update(cleaned["id"].tolist())
            removed_cross = before - len(cleaned)
        else:
            removed_cross = 0

        if "relevance_level" in cleaned.columns:
            cleaned = cleaned.drop(columns=["relevance_level"])

        kept_rows += len(cleaned)

        print(f"Chunk {chunk_number}: {len(chunk):,} → kept {len(cleaned):,}"
              + (f" (removed {removed_cross} cross-chunk duplicates)" if removed_cross else ""))

        if cleaned.empty:
            continue

        cleaned.to_csv(
            OUTPUT_FILE,
            mode="a" if header_written else "w",
            header=not header_written,
            index=False
        )
        header_written = True

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"Total rows processed: {total_rows:,}")
    print(f"Rows kept (strong only, unique): {kept_rows:,}")

    if total_rows > 0:
        print(f"Kept rate: {kept_rows / total_rows:.2%}")

    print()
    print(f"Clean file saved to:\n{OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()