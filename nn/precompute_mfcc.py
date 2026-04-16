"""
Предвычисление MFCC для всех сплитов (train/val/test).

Читает WAV-файлы один раз, вычисляет MFCC (49×10), сохраняет
в .npz файлы. При обучении dataset.py грузит готовые MFCC из RAM
вместо чтения 31000 WAV-файлов каждую эпоху.

Аугментации на уровне аудио (time shift, noise mixing) при этом
заменяются на аугментации на уровне MFCC (SpecAugment: time mask,
frequency mask). Это стандартная практика в production KWS.

Запуск:
    python precompute_mfcc.py

На выходе:
    data/cache/train_mfcc.npz  (~24 MB)
    data/cache/val_mfcc.npz    (~3 MB)
    data/cache/test_mfcc.npz   (~3 MB)
"""

from __future__ import annotations

import sys
import time

import numpy as np
import tensorflow as tf

import config
from data.dataset import (
    compute_mfcc,
    _ensure_bg_noise,
    _random_bg_slice,
    _load_wav_file,
)

CACHE_DIR = config.DATA_DIR / "cache"


def _load_audio(filepath: str, is_silence: bool) -> np.ndarray:
    """Загружает 1 секунду аудио как float32 numpy array."""
    if is_silence:
        # Для silence: случайный кусок фонового шума
        _ensure_bg_noise()
        return _random_bg_slice().numpy()
    else:
        return _load_wav_file(tf.constant(filepath)).numpy()


def _compute_mfcc_numpy(audio: np.ndarray) -> np.ndarray:
    """Считает MFCC для одного аудио. Возвращает (49, 10)."""
    tensor = tf.constant(audio, dtype=tf.float32)
    mfcc = compute_mfcc(tensor)  # (49, 10, 1)
    return mfcc.numpy()[:, :, 0]  # (49, 10) — убираем канал


def precompute_split(split_name: str) -> None:
    """Предвычисляет MFCC для одного сплита."""
    import pandas as pd

    manifest = config.MANIFEST_DIR / f"{split_name}.csv"
    if not manifest.exists():
        print(f"[precompute] ERROR: {manifest} не найден", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(manifest)
    n = len(df)
    print(f"[precompute] {split_name}: {n} примеров...")

    # Выходные массивы
    mfccs = np.zeros((n, config.NUM_FRAMES, config.NUM_MFCC), dtype=np.float32)
    labels = np.zeros(n, dtype=np.int32)

    t0 = time.time()
    for i, row in df.iterrows():
        filepath = row["filepath"]
        label_str = row["label"]
        is_sil = label_str == config.SILENCE_LABEL

        audio = _load_audio(filepath, is_sil)
        mfccs[i] = _compute_mfcc_numpy(audio)
        labels[i] = config.LABEL_TO_INDEX[label_str]

        if (i + 1) % 5000 == 0:
            elapsed = time.time() - t0
            speed = (i + 1) / elapsed
            eta = (n - i - 1) / speed
            print(f"  {i+1}/{n} ({speed:.0f} samples/sec, ETA {eta:.0f}s)")

    # Сохраняем
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CACHE_DIR / f"{split_name}_mfcc.npz"
    np.savez_compressed(out_path, mfccs=mfccs, labels=labels)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    elapsed = time.time() - t0
    print(
        f"[precompute] {split_name}: {n} примеров за {elapsed:.1f}s → {out_path} ({size_mb:.1f} MB)"
    )


def main() -> None:
    for split in ["train", "val", "test"]:
        precompute_split(split)
    print("[precompute] Готово.")


if __name__ == "__main__":
    main()
