"""
Сканирует распакованный Speech Commands v2 и строит манифесты train/val/test.

- validation_list.txt и testing_list.txt определяют val и test split.
- Всё остальное идёт в train.
- Классы target (10 команд) остаются как есть.
- Все "другие" слова попадают в класс _unknown_.
- Для _silence_ записывается специальная метка в filepath = SILENCE_LABEL —
  сэмплы будут сгенерированы из _background_noise_ на лету в dataset.py.
"""

import csv
import random
import sys
from pathlib import Path

import config


def _read_list(path: Path) -> set[str]:
    if not path.exists():
        print(
            f"[preprocess] ERROR: нет файла {path}. Сначала запустите download.py",
            file=sys.stderr,
        )
        sys.exit(1)
    with open(path, "r") as f:
        return {line.strip() for line in f if line.strip()}


def _label_for_word(word: str) -> str:
    return word if word in config.TARGET_WORDS else config.UNKNOWN_LABEL


def _collect_all_wavs(root: Path) -> list[tuple[Path, str]]:
    """Возвращает [(relative_path, word)] для всех .wav кроме _background_noise_."""
    items: list[tuple[Path, str]] = []
    for word_dir in sorted(root.iterdir()):
        if not word_dir.is_dir():
            continue
        if word_dir.name == config.BG_NOISE_SUBDIR:
            continue
        if word_dir.name.startswith("_"):
            continue
        for wav in word_dir.glob("*.wav"):
            rel = wav.relative_to(root)
            items.append((rel, word_dir.name))
    return items


def _balanced_add_silence_unknown(
    rows: list[tuple[str, str]],
    target_count_per_class: int,
    split_name: str,
    rng: random.Random,
) -> list[tuple[str, str]]:
    """
    Добавляет записи _silence_ и подрезает _unknown_ до target_count.
    rows — уже отфильтрованные для split записи target+unknown.
    """
    target_rows = [r for r in rows if r[1] != config.UNKNOWN_LABEL]
    unknown_rows = [r for r in rows if r[1] == config.UNKNOWN_LABEL]

    n_silence = max(1, int(target_count_per_class * config.SILENCE_PERCENTAGE / 100.0))
    n_unknown = max(1, int(target_count_per_class * config.UNKNOWN_PERCENTAGE / 100.0))

    rng.shuffle(unknown_rows)
    unknown_subset = unknown_rows[:n_unknown]

    silence_rows = [
        (config.SILENCE_LABEL, config.SILENCE_LABEL) for _ in range(n_silence)
    ]

    print(
        f"[preprocess] split={split_name} | target={len(target_rows)} "
        f"unknown={len(unknown_subset)} silence={len(silence_rows)}"
    )
    return target_rows + unknown_subset + silence_rows


def _write_manifest(rows: list[tuple[str, str]], out: Path) -> None:
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filepath", "label"])
        for r in rows:
            w.writerow(r)
    print(f"[preprocess] записан {out} ({len(rows)} строк)")


def main() -> None:
    rng = random.Random(config.SEED)

    if not config.DATASET_ROOT.exists():
        print(
            f"[preprocess] ERROR: {config.DATASET_ROOT} не найден. Запустите download.py",
            file=sys.stderr,
        )
        sys.exit(1)

    val_list = _read_list(config.DATASET_ROOT / "validation_list.txt")
    test_list = _read_list(config.DATASET_ROOT / "testing_list.txt")

    all_items = _collect_all_wavs(config.DATASET_ROOT)
    print(f"[preprocess] всего .wav (кроме фона): {len(all_items)}")

    train, val, test = [], [], []
    for rel, word in all_items:
        lbl = _label_for_word(word)
        key = rel.as_posix()  # в val/test_list пути через /
        abs_path = str((config.DATASET_ROOT / rel).resolve())
        row = (abs_path, lbl)
        if key in val_list:
            val.append(row)
        elif key in test_list:
            test.append(row)
        else:
            train.append(row)

    # среднее количество примеров на target-класс в train (для балансировки silence/unknown)
    target_counts: dict[str, int] = {w: 0 for w in config.TARGET_WORDS}
    for _, lbl in train:
        if lbl in target_counts:
            target_counts[lbl] += 1
    avg_target = sum(target_counts.values()) // len(target_counts)
    print(f"[preprocess] среднее кол-во target-примеров в train: {avg_target}")

    val_target = len([r for r in val if r[1] != config.UNKNOWN_LABEL]) // len(
        config.TARGET_WORDS
    )
    test_target = len([r for r in test if r[1] != config.UNKNOWN_LABEL]) // len(
        config.TARGET_WORDS
    )

    train = _balanced_add_silence_unknown(train, avg_target, "train", rng)
    val = _balanced_add_silence_unknown(val, max(val_target, 100), "val", rng)
    test = _balanced_add_silence_unknown(test, max(test_target, 100), "test", rng)

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    _write_manifest(train, config.MANIFEST_DIR / "train.csv")
    _write_manifest(val, config.MANIFEST_DIR / "val.csv")
    _write_manifest(test, config.MANIFEST_DIR / "test.csv")

    print("[preprocess] Готово.")


if __name__ == "__main__":
    main()
