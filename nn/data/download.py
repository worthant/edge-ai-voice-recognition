"""
Скачивает и распаковывает Google Speech Commands v2.

Проверяет наличие архива и распакованной папки, чтобы не качать повторно.
Выводит прогресс-бар и размер датасета на диске.
"""

import hashlib
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

from tqdm import tqdm

import config


def _download_with_progress(url: str, dst: Path) -> None:
    """Скачивает url в dst с tqdm-прогрессбаром."""
    dst.parent.mkdir(parents=True, exist_ok=True)

    class _Hook(tqdm):
        def update_to(self, b: int = 1, bsize: int = 1, tsize: int | None = None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    with _Hook(unit="B", unit_scale=True, miniters=1, desc=dst.name) as t:
        urllib.request.urlretrieve(url, filename=str(dst), reporthook=t.update_to)


def _dir_size_gb(path: Path) -> float:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total / (1024**3)


def _md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main() -> None:
    target_marker = config.DATASET_ROOT / "yes"  # одна из папок с target-командой
    if target_marker.exists():
        size = _dir_size_gb(config.DATASET_ROOT)
        print(
            f"[download] Датасет уже распакован в {config.DATASET_ROOT} ({size:.2f} GB). Пропускаю."
        )
        return

    if not config.DATASET_ARCHIVE.exists():
        print(f"[download] Скачиваю {config.DATASET_URL}")
        _download_with_progress(config.DATASET_URL, config.DATASET_ARCHIVE)
    else:
        print(f"[download] Архив уже скачан: {config.DATASET_ARCHIVE}")

    print(f"[download] MD5: {_md5(config.DATASET_ARCHIVE)}")

    print(f"[download] Распаковываю в {config.DATASET_ROOT}")
    config.DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    with tarfile.open(config.DATASET_ARCHIVE, "r:gz") as tar:
        members = tar.getmembers()
        for m in tqdm(members, desc="extract"):
            tar.extract(m, path=config.DATASET_ROOT)

    size = _dir_size_gb(config.DATASET_ROOT)
    print(f"[download] Готово. Размер на диске: {size:.2f} GB")

    required = ["yes", "no", "_background_noise_"]
    for r in required:
        if not (config.DATASET_ROOT / r).exists():
            print(f"[download] ERROR: не найдена ожидаемая папка {r}", file=sys.stderr)
            sys.exit(1)
    print("[download] Структура датасета проверена.")


if __name__ == "__main__":
    main()
