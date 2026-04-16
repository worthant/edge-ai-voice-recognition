"""
tf.data.Dataset pipeline для Speech Commands v2.

Загружает WAV, нормализует длину до 1 секунды, применяет аугментации
(time shift ± 100 мс, подмешивание фонового шума), считает MFCC (49×10)
и формирует батчи.

MFCC реализация — через tf.signal, параметры согласованы с config.py и
повторяются 1:1 в C++ реализации на ESP32.

Совместимо с TensorFlow 2.14 — 2.19+.
Фоновый шум и mel-матрица хранятся как tf.Variable (не tf.constant),
потому что tf.data.map() на TF 2.16+ не может захватывать eager-тензоры
(tf.constant) в замыканиях — падает на convert_to_mixed_eager_tensors.
tf.Variable (ResourceVariable) обрабатывается корректно на всех версиях.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

import config


AUTOTUNE = tf.data.AUTOTUNE


# ============================================================================
# Background noise: один склеенный tf.Variable
# ============================================================================
_BG_VAR: tf.Variable | None = None  # [total_samples] float32, trainable=False
_BG_TOTAL: int = 0  # длина в сэмплах (Python int, не тензор)
_BG_READY: bool = False


def _load_bg_noises_numpy() -> list[np.ndarray]:
    """Читает все WAV из _background_noise_ как numpy float32."""
    bg_dir = config.DATASET_ROOT / config.BG_NOISE_SUBDIR
    if not bg_dir.exists():
        return []
    out: list[np.ndarray] = []
    for p in sorted(bg_dir.glob("*.wav")):
        contents = tf.io.read_file(str(p))
        audio, _ = tf.audio.decode_wav(contents, desired_channels=1)
        out.append(tf.squeeze(audio, axis=-1).numpy().astype(np.float32))
    return out


def _ensure_bg_noise() -> None:
    """
    Один раз склеивает все фоновые шумы в единый tf.Variable.

    Почему tf.Variable а не tf.constant:
    tf.data.Dataset.map() в TF 2.16+ трейсит функцию в граф.
    Когда функция захватывает tf.constant (EagerTensor) из внешнего scope,
    трейсер вызывает convert_to_mixed_eager_tensors() и падает с
    'SymbolicTensor has no attribute _datatype_enum'.
    tf.Variable — это ResourceVariable, у которого resource handle,
    и tf.data умеет его передавать через граф на всех версиях TF.
    """
    global _BG_VAR, _BG_TOTAL, _BG_READY
    if _BG_READY:
        return

    arrays = _load_bg_noises_numpy()

    if not arrays:
        print("[dataset] WARNING: _background_noise_ не найден, silence будет нулями")
        _BG_VAR = tf.Variable(
            tf.zeros([config.CLIP_SAMPLES], dtype=tf.float32),
            trainable=False,
        )
        _BG_TOTAL = config.CLIP_SAMPLES
        _BG_READY = True
        return

    concatenated = np.concatenate(arrays, axis=0)
    _BG_TOTAL = len(concatenated)
    _BG_VAR = tf.Variable(concatenated, trainable=False, dtype=tf.float32)

    total_mb = _BG_TOTAL * 4 / (1024 * 1024)
    print(
        f"[dataset] bg noise: {len(arrays)} файлов, "
        f"{_BG_TOTAL} samples ({total_mb:.1f} MB), склеено в один тензор"
    )
    _BG_READY = True


def _random_bg_slice() -> tf.Tensor:
    """Случайный кусок длиной CLIP_SAMPLES из склеенного фона."""
    max_start = max(_BG_TOTAL - config.CLIP_SAMPLES, 1)
    start = tf.random.uniform([], 0, max_start, dtype=tf.int32)
    return _BG_VAR[start : start + config.CLIP_SAMPLES]


# ============================================================================
# Загрузка WAV
# ============================================================================
def _decode_wav(contents: tf.Tensor) -> tf.Tensor:
    """Декодирует wav-файл в float32 [-1, 1] длиной CLIP_SAMPLES."""
    audio, _ = tf.audio.decode_wav(contents, desired_channels=1)
    audio = tf.squeeze(audio, axis=-1)
    audio = audio[: config.CLIP_SAMPLES]
    pad = config.CLIP_SAMPLES - tf.shape(audio)[0]
    audio = tf.cond(
        pad > 0,
        lambda: tf.pad(audio, [[0, pad]]),
        lambda: audio,
    )
    return audio


def _load_wav_file(path: tf.Tensor) -> tf.Tensor:
    contents = tf.io.read_file(path)
    return _decode_wav(contents)


# ============================================================================
# Аугментации
# ============================================================================
def _time_shift(audio: tf.Tensor) -> tf.Tensor:
    """Сдвигает аудио на случайное число сэмплов ±TIME_SHIFT_SAMPLES."""
    max_s = config.TIME_SHIFT_SAMPLES
    shift = tf.random.uniform([], -max_s, max_s + 1, dtype=tf.int32)
    pad_left = tf.maximum(shift, 0)
    pad_right = tf.maximum(-shift, 0)
    padded = tf.pad(audio, [[pad_left, pad_right]])
    start = tf.maximum(-shift, 0)
    return padded[start : start + config.CLIP_SAMPLES]


def _augment(audio: tf.Tensor, is_silence: tf.Tensor) -> tf.Tensor:
    """
    Для silence: чистый background slice.
    Для обычных: time shift + подмешивание фона с вероятностью BG_NOISE_PROB.
    """

    def silence_branch():
        return _random_bg_slice()

    def normal_branch():
        a = _time_shift(audio)

        def with_bg():
            vol = tf.random.uniform([], 0.0, config.BG_NOISE_VOLUME_MAX)
            bg = _random_bg_slice()
            return a + vol * bg

        return tf.cond(
            tf.random.uniform([]) < config.BG_NOISE_PROB,
            with_bg,
            lambda: a,
        )

    return tf.cond(is_silence, silence_branch, normal_branch)

def _spec_augment(mfcc: tf.Tensor) -> tf.Tensor:
    """
    SpecAugment: аугментации на уровне MFCC-"картинки".
    Заменяет audio-level аугментации (time shift, noise mixing),
    которые невозможны после предвычисления MFCC.

    Два типа масок:
    - Time mask: обнулить 1-5 случайных фреймов подряд (как будто часть слова пропала)
    - Frequency mask: обнулить 1-2 случайных MFCC-коэффициента (как будто часть частот пропала)
    """
    # Time mask: обнулить t случайных последовательных фреймов
    t = tf.random.uniform([], 1, 6, dtype=tf.int32)         # от 1 до 5 фреймов
    t0 = tf.random.uniform([], 0, config.NUM_FRAMES - t, dtype=tf.int32)
    # Создаём маску: 1 везде, 0 в зоне маскирования
    mask_t = tf.concat([
        tf.ones([t0, config.NUM_MFCC]),
        tf.zeros([t, config.NUM_MFCC]),
        tf.ones([config.NUM_FRAMES - t0 - t, config.NUM_MFCC]),
    ], axis=0)
    mfcc = mfcc * mask_t

    # Frequency mask: обнулить f случайных коэффициентов
    f = tf.random.uniform([], 1, 3, dtype=tf.int32)         # от 1 до 2 коэффициентов
    f0 = tf.random.uniform([], 0, config.NUM_MFCC - f, dtype=tf.int32)
    mask_f = tf.concat([
        tf.ones([config.NUM_FRAMES, f0]),
        tf.zeros([config.NUM_FRAMES, f]),
        tf.ones([config.NUM_FRAMES, config.NUM_MFCC - f0 - f]),
    ], axis=1)
    mfcc = mfcc * mask_f

    return mfcc


# ============================================================================
# MFCC
# ============================================================================
_MEL_MATRIX: tf.Variable | None = None


def _get_mel_matrix() -> tf.Variable:
    """
    Mel-матрица тоже хранится как tf.Variable по той же причине —
    безопасный захват внутри ds.map().
    """
    global _MEL_MATRIX
    if _MEL_MATRIX is None:
        mat = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=config.NUM_MEL_BINS,
            num_spectrogram_bins=config.FFT_LENGTH // 2 + 1,
            sample_rate=config.SAMPLE_RATE,
            lower_edge_hertz=config.MEL_LOWER_HZ,
            upper_edge_hertz=config.MEL_UPPER_HZ,
        )
        _MEL_MATRIX = tf.Variable(mat, trainable=False)
    return _MEL_MATRIX


def compute_mfcc(audio: tf.Tensor) -> tf.Tensor:
    """audio: [CLIP_SAMPLES] float32  ->  mfcc: [NUM_FRAMES, NUM_MFCC, 1]"""
    stft = tf.signal.stft(
        audio,
        frame_length=config.WINDOW_SIZE_SAMPLES,
        frame_step=config.WINDOW_STRIDE_SAMPLES,
        fft_length=config.FFT_LENGTH,
        window_fn=tf.signal.hann_window,
        pad_end=False,
    )
    magnitude = tf.abs(stft)
    mel = tf.matmul(magnitude, _get_mel_matrix())
    log_mel = tf.math.log(mel + 1e-6)
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel)
    mfccs = mfccs[..., : config.NUM_MFCC]
    mfccs = mfccs[: config.NUM_FRAMES, :]
    mfccs = tf.ensure_shape(mfccs, [config.NUM_FRAMES, config.NUM_MFCC])
    return tf.expand_dims(mfccs, axis=-1)


# ============================================================================
# Pipeline
# ============================================================================
def _load_manifest(path: Path) -> tuple[list[str], list[int], list[int]]:
    df = pd.read_csv(path)
    paths = df["filepath"].tolist()
    labels = [config.LABEL_TO_INDEX[lbl] for lbl in df["label"].tolist()]
    is_silence = [
        1 if lbl == config.SILENCE_LABEL else 0 for lbl in df["label"].tolist()
    ]
    return paths, labels, is_silence


def build_dataset(
    manifest_path: Path,
    batch_size: int,
    training: bool,
    shuffle_buffer: int = 2000,
) -> tf.data.Dataset:
    paths, labels, is_silence = _load_manifest(manifest_path)

    # Загружаем фон и mel-матрицу один раз (tf.Variable — safe для ds.map)
    _ensure_bg_noise()
    _get_mel_matrix()

    ds = tf.data.Dataset.from_tensor_slices((paths, labels, is_silence))

    if training:
        ds = ds.shuffle(shuffle_buffer, seed=config.SEED, reshuffle_each_iteration=True)

    def _parse(path, label, is_sil):
        is_sil_b = tf.cast(is_sil, tf.bool)

        audio = tf.cond(
            is_sil_b,
            lambda: tf.zeros([config.CLIP_SAMPLES], dtype=tf.float32),
            lambda: _load_wav_file(path),
        )

        if training:
            audio = _augment(audio, is_sil_b)

        mfcc = compute_mfcc(audio)
        label_oh = tf.one_hot(label, depth=config.NUM_CLASSES)
        return mfcc, label_oh

    ds = ds.map(_parse, num_parallel_calls=AUTOTUNE)

    # Для val/test: кешируем в RAM после первого прохода.
    # MFCC без аугментаций одинаковый каждую эпоху — нет смысла считать заново.
    if not training:
        ds = ds.cache()

    ds = ds.batch(batch_size, drop_remainder=training)
    ds = ds.prefetch(AUTOTUNE)
    return ds


# быстрый пайплайн из предвычисленного кэша
CACHE_DIR = config.DATA_DIR / "cache"
def build_dataset_cached(
    split_name: str,
    batch_size: int,
    training: bool,
    shuffle_buffer: int = 2000,
) -> tf.data.Dataset:
    """
    Быстрый dataset из предвычисленного .npz кэша.

    Вместо: WAV с диска → аугментация → MFCC (6 мс/пример)
    Делает:  MFCC из RAM → SpecAugment маски (0.03 мс/пример)

    Ускорение: ~200× на data pipeline, общее обучение ~10× быстрее.
    """
    cache_path = CACHE_DIR / f"{split_name}_mfcc.npz"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Нет кэша {cache_path}. Запустите: python precompute_mfcc.py"
        )

    data = np.load(cache_path)
    mfccs = data["mfccs"]    # (N, 49, 10) float32
    labels = data["labels"]  # (N,) int32
    print(f"[dataset] loaded cache: {split_name} — {len(mfccs)} samples from {cache_path}")

    ds = tf.data.Dataset.from_tensor_slices((mfccs, labels))

    if training:
        ds = ds.shuffle(shuffle_buffer, seed=config.SEED, reshuffle_each_iteration=True)

    def _parse_cached(mfcc, label):
        # mfcc: (49, 10), label: int32
        if training:
            mfcc = _spec_augment(mfcc)

        mfcc = tf.expand_dims(mfcc, axis=-1)  # (49, 10, 1) — добавляем канал
        label_oh = tf.one_hot(label, depth=config.NUM_CLASSES)
        return mfcc, label_oh

    ds = ds.map(_parse_cached, num_parallel_calls=AUTOTUNE)

    if not training:
        ds = ds.cache()  # val/test — кэш после первого прохода

    ds = ds.batch(batch_size, drop_remainder=training)
    ds = ds.prefetch(AUTOTUNE)
    return ds

def count_examples(manifest_path: Path) -> int:
    return len(pd.read_csv(manifest_path))
