import zstandard as zstd
import json
import csv
from datetime import datetime, timezone
from pathlib import Path
import io
from collections import Counter

from tqdm import tqdm


# ============================================================
# НАЛАШТУВАННЯ
# ============================================================

# Можна вказати:
# 1. конкретний .zst файл
# 2. папку з .zst файлами

INPUT_PATH = Path(
    r"D:\\procesing"
)

OUTPUT_CSV = Path(
    r"D:\Desktop\Computational Social Science\Project\reddit_clean.csv"
)


# Субреддити, які залишаємо
# ВАЖЛИВО: усі назви в нижньому регістрі,
# тому що нижче використовується .lower()
SUBREDDITS = {
    "ukraine",
    "ukrainianconflict",
    "ukrainewarreport",
    "worldnews",
    "europe",
    "nato",
}


# Розмір буфера для flush CSV
# Не потрібно flush після кожного рядка,
# бо це сильно сповільнить програму.
FLUSH_EVERY = 100_000


# ============================================================
# PROGRESS READER
# ============================================================

class ProgressReader:
    """
    Обгортка над файлом, яка дозволяє tqdm
    показувати, скільки стиснених байтів .zst
    вже було прочитано.
    """

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
# ЧИТАННЯ ZST
# ============================================================

def read_zst_lines(file_path: Path, progress_bar):
    """
    Streaming-читання .zst файлу по одному рядку.

    Важливо:
    тут НЕ створюється величезний список через split("\\n"),
    тому використання RAM значно менше.
    """

    with open(file_path, "rb") as raw_file:

        progress_file = ProgressReader(
            raw_file,
            progress_bar
        )

        dctx = zstd.ZstdDecompressor(
            max_window_size=2**31
        )

        with dctx.stream_reader(progress_file) as stream_reader:

            text_stream = io.TextIOWrapper(
                stream_reader,
                encoding="utf-8",
                errors="replace"
            )

            # TextIOWrapper сам читає файл поступово.
            # Ми отримуємо по одному рядку.
            for line in text_stream:

                line = line.strip()

                if line:
                    yield line


# ============================================================
# ОБРОБКА ФАЙЛУ
# ============================================================

def process_file(file_path: Path, writer, output_file):

    is_submission = file_path.name.startswith("RS_")

    record_type = (
        "submission"
        if is_submission
        else "comment"
    )

    total_lines = 0
    matched = 0

    subreddit_counter = Counter()

    file_size = file_path.stat().st_size

    # --------------------------------------------------------
    # Progress bar
    # --------------------------------------------------------

    with tqdm(
        total=file_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=file_path.name,
        dynamic_ncols=True,
    ) as progress_bar:

        for line in read_zst_lines(
            file_path,
            progress_bar
        ):

            total_lines += 1

            # ------------------------------------------------
            # JSON
            # ------------------------------------------------

            try:
                obj = json.loads(line)

            except json.JSONDecodeError:
                continue

            # ------------------------------------------------
            # SUBREDDIT
            # ------------------------------------------------

            subreddit = (
                obj.get("subreddit", "")
                .lower()
            )

            if subreddit not in SUBREDDITS:
                continue

            # ------------------------------------------------
            # TEXT
            # ------------------------------------------------

            if is_submission:

                title = obj.get("title") or ""
                selftext = obj.get("selftext") or ""

                if selftext in (
                    "[deleted]",
                    "[removed]"
                ):
                    selftext = ""

                text = (
                    title + " " + selftext
                ).strip()

            else:

                text = obj.get("body") or ""

                if text in (
                    "[deleted]",
                    "[removed]"
                ):
                    continue

            # Занадто короткі записи
            if not text or len(text) < 15:
                continue

            # ------------------------------------------------
            # DATETIME
            # ------------------------------------------------

            try:

                ts = int(
                    obj.get(
                        "created_utc",
                        0
                    )
                )

                dt = datetime.fromtimestamp(
                    ts,
                    tz=timezone.utc
                )

                datetime_str = dt.strftime(
                    "%Y-%m-%d %H:%M:%S"
                )

                month = dt.strftime(
                    "%Y-%m"
                )

            except Exception:
                continue

            # ------------------------------------------------
            # ROW
            # ------------------------------------------------

            row = {
                "id": obj.get("id", ""),
                "type": record_type,
                "subreddit": subreddit,
                "datetime": datetime_str,
                "month": month,
                "author": obj.get("author", ""),
                "score": obj.get("score", 0),
                "text": (
                    text
                    .replace("\n", " ")
                    .replace("\r", " ")
                ),
            }

            writer.writerow(row)

            matched += 1
            subreddit_counter[subreddit] += 1

            # ------------------------------------------------
            # Періодично оновлюємо CSV на диску
            # ------------------------------------------------

            if matched % FLUSH_EVERY == 0:
                output_file.flush()

            # ------------------------------------------------
            # Оновлюємо інформацію біля progress bar
            # ------------------------------------------------

            if total_lines % 10_000 == 0:

                progress_bar.set_postfix(
                    lines=f"{total_lines:,}",
                    matched=f"{matched:,}",
                )

    return total_lines, matched, subreddit_counter


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fieldnames = [
        "id",
        "type",
        "subreddit",
        "datetime",
        "month",
        "author",
        "score",
        "text",
    ]

    # --------------------------------------------------------
    # Визначаємо файли
    # --------------------------------------------------------

    if INPUT_PATH.is_file():

        if INPUT_PATH.suffix != ".zst":
            print(
                "Помилка: INPUT_PATH має бути .zst файлом."
            )
            return

        zst_files = [INPUT_PATH]

    elif INPUT_PATH.is_dir():

        zst_files = sorted(
            INPUT_PATH.glob("*.zst")
        )

    else:

        print(
            f"Не знайдено файл або папку:\n{INPUT_PATH}"
        )
        return

    if not zst_files:

        print(
            f"Не знайдено .zst файлів:\n{INPUT_PATH}"
        )
        return

    print("=" * 70)
    print("Reddit parser")
    print("=" * 70)

    print(f"Файлів для обробки: {len(zst_files)}")
    print(f"Output: {OUTPUT_CSV}")
    print()

    # --------------------------------------------------------
    # Загальна статистика
    # --------------------------------------------------------

    total_lines_all = 0
    total_matched_all = 0

    total_subreddits = Counter()

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    with open(
        OUTPUT_CSV,
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        # Щоб заголовок гарантовано записався
        f.flush()

        # ----------------------------------------------------
        # Файли
        # ----------------------------------------------------

        for index, file_path in enumerate(
            zst_files,
            start=1
        ):

            print()
            print(
                f"[{index}/{len(zst_files)}] "
                f"Обробляю: {file_path.name}"
            )

            try:

                (
                    lines,
                    matched,
                    subreddit_stats,
                ) = process_file(
                    file_path,
                    writer,
                    f
                )

            except zstd.ZstdError as e:

                print()
                print(
                    f"ПОМИЛКА ZSTD у файлі "
                    f"{file_path.name}:"
                )
                print(e)

                print(
                    "\nФайл може бути пошкоджений "
                    "або недокачаний."
                )

                continue

            except MemoryError:

                print()
                print(
                    "ПОМИЛКА: недостатньо RAM."
                )

                print(
                    "Спробуйте закрити інші програми "
                    "або зменшити навантаження."
                )

                return

            total_lines_all += lines
            total_matched_all += matched

            total_subreddits.update(
                subreddit_stats
            )

            print()
            print(
                f"Рядків прочитано: {lines:,}"
            )

            print(
                f"Відібрано: {matched:,}"
            )

            print("По subreddit:")

            for subreddit, count in sorted(
                subreddit_stats.items()
            ):

                print(
                    f"  {subreddit:<25} "
                    f"{count:,}"
                )

    # --------------------------------------------------------
    # ФІНАЛЬНА СТАТИСТИКА
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("ГОТОВО")
    print("=" * 70)

    print(
        f"Всього рядків прочитано: "
        f"{total_lines_all:,}"
    )

    print(
        f"Всього відібрано: "
        f"{total_matched_all:,}"
    )

    print()
    print("Всього по subreddit:")

    for subreddit, count in sorted(
        total_subreddits.items()
    ):

        print(
            f"  {subreddit:<25} "
            f"{count:,}"
        )

    print()
    print(
        f"CSV збережено: {OUTPUT_CSV}"
    )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    main()