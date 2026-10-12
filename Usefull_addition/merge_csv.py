from pathlib import Path
import shutil

file1 = Path(r"D:\\Desktop\\Computational Social Science\\Project\\Week_1\\data\\reddit_ukraine_support.csv")
file2 = Path(r"D:\\Desktop\\Computational Social Science\\Project\\data_processed.csv")

output = Path(r"D:\\Desktop\\Computational Social Science\\Project\\01_2022.csv")

with open(output, "wb") as out:

    # Перший CSV копіюємо повністю, разом із header
    with open(file1, "rb") as f:
        shutil.copyfileobj(f, out)

    # Другий CSV копіюємо без header
    with open(file2, "rb") as f:
        f.readline()
        shutil.copyfileobj(f, out)

print("Готово!")
print(f"Файл: {output}")