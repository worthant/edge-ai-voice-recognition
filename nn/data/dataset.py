"""
tf.data.Dataset pipeline для Speech Commands v2.

Загружает WAV, нормализует длину до 1 секунды, применяет аугментации
(time shift ± 100 мс, подмешивание фонового шума), считает MFCC (49×10)
и формирует батчи.

MFCC реализация — через tf.signal, параметры согласованы с config.py и
повторяются 1:1 в C++ реализации на ESP32.
"""

from __future__ import annotations

import glob
import random
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import tensorflow as tf

import config


AUTOTUNE = tf.data.AUTOTUNE


# ----------------------------------------------------------------------------
# Загрузка WAV и базовые преобразования
# ----------------------------------------------------------------------------
def _decode_wav(contents: tf.Tensor) -> tf.Tensor:
    """Декодирует wav-файл в tensor формата float32 [-1, 1] длиной CLIP_SAMPLES."""
    audio, sr = tf.audio.decode_wav(contents, desired_channels=1)
    audio = tf.squeeze(audio, axis=-1)
    # обрезка/padding до 1 секунды
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


# ----------------------------------------------------------------------------
# Аугментации
# ----------------------------------------------------------------------------
def _time_shift(audio: tf.Tensor) -> tf.Tensor:
    max_s = config.TIME_SHIFT_SAMPLES
    shift = tf.random.uniform([], -max_s, max_s + 1, dtype=tf.int32)
    pad_left = tf.maximum(shift, 0)
    pad_right = tf.maximum(-shift, 0)
    padded = tf.pad(audio, [[pad_left, pad_right]])
    start = tf.maximum(-shift, 0)
    return padded[start : start + config.CLIP_SAMPLES]


def _load_bg_noises() -> list[np.ndarray]:
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


_BG_CACHE: list[tf.Tensor] = []


def _get_bg_noise_tensors() -> list[tf.Tensor]:
    global _BG_CACHE
    if _BG_CACHE:
        return _BG_CACHE
    arrays = _load_bg_noises()
    if not arrays:
        print("[dataset] WARNING: _background_noise_ не найден, silence будет нулями")
        return []
    _BG_CACHE = [tf.constant(a, dtype=tf.float32) for a in arrays]
    return _BG_CACHE


def _random_bg_slice() -> tf.Tensor:
    """Случайный кусок из случайного фонового файла длиной CLIP_SAMPLES."""
    bgs = _get_bg_noise_tensors()
    if not bgs:
        return tf.zeros([config.CLIP_SAMPLES], dtype=tf.float32)
    idx = tf.random.uniform([], 0, len(bgs), dtype=tf.int32)

    # tf.switch_case для выбора тензора переменной длины
    def _pick(i):
        def _f():
            return bgs[i]

        return _f

    bg = tf.switch_case(idx, {i: _pick(i) for i in range(len(bgs))})
    bg_len = tf.shape(bg)[0]
    max_start = tf.maximum(bg_len - config.CLIP_SAMPLES, 1)
    start = tf.random.uniform([], 0, max_start, dtype=tf.int32)
    return bg[start : start + config.CLIP_SAMPLES]


def _mix_bg_noise(audio: tf.Tensor) -> tf.Tensor:
    vol = tf.random.uniform([], 0.0, config.BG_NOISE_VOLUME_MAX, dtype=tf.float32)
    bg = _random_bg_slice()
    return audio + vol * bg


def _maybe_augment(
    audio: tf.Tensor, label: tf.Tensor, is_silence: tf.Tensor
) -> tf.Tensor:
    """
    Для не-silence: time shift + (с вероятностью) фон.
    Для silence: чистый background slice с полной громкостью (1.0).
    """

    def silence_branch():
        return _random_bg_slice()

    def normal_branch():
        a = _time_shift(audio)

        def add_bg():
            return _mix_bg_noise(a)

        def no_bg():
            return a

        mix = tf.cond(
            tf.random.uniform([]) < config.BG_NOISE_PROB,
            add_bg,
            no_bg,
        )
        return mix

    return tf.cond(is_silence, silence_branch, normal_branch)


# ----------------------------------------------------------------------------
# MFCC
# ----------------------------------------------------------------------------
_MEL_MATRIX = None


def _get_mel_matrix() -> tf.Tensor:
    global _MEL_MATRIX
    if _MEL_MATRIX is None:
        num_spec_bins = config.FFT_LENGTH // 2 + 1
        _MEL_MATRIX = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=config.NUM_MEL_BINS,
            num_spectrogram_bins=num_spec_bins,
            sample_rate=config.SAMPLE_RATE,
            lower_edge_hertz=config.MEL_LOWER_HZ,
            upper_edge_hertz=config.MEL_UPPER_HZ,
        )
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
    magnitude = tf.abs(stft)  # [frames, bins]
    mel = tf.matmul(magnitude, _get_mel_matrix())  # [frames, NUM_MEL_BINS]
    log_mel = tf.math.log(mel + 1e-6)
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel)  # [frames, NUM_MEL_BINS]
    mfccs = mfccs[..., : config.NUM_MFCC]  # [frames, NUM_MFCC]
    # Жёстко режем до NUM_FRAMES (на случай pad_end)
    mfccs = mfccs[: config.NUM_FRAMES, :]
    mfccs = tf.ensure_shape(mfccs, [config.NUM_FRAMES, config.NUM_MFCC])
    return tf.expand_dims(mfccs, axis=-1)  # добавляем канал


# ----------------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------------
def _load_manifest(path: Path) -> tuple[list[str], list[int], list[int]]:
    df = pd.read_csv(path)
    paths = df["filepath"].tolist()
    labels = [config.LABEL_TO_INDEX[l] for l in df["label"].tolist()]
    is_silence = [1 if l == config.SILENCE_LABEL else 0 for l in df["label"].tolist()]
    return paths, labels, is_silence


def build_dataset(
    manifest_path: Path,
    batch_size: int,
    training: bool,
    shuffle_buffer: int = 2000,
) -> tf.data.Dataset:
    paths, labels, is_silence = _load_manifest(manifest_path)
    _get_bg_noise_tensors()  # прогреваем кэш в main-процессе

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
            audio = _maybe_augment(audio, label, is_sil_b)
        mfcc = compute_mfcc(audio)
        label_oh = tf.one_hot(label, depth=config.NUM_CLASSES)
        return mfcc, label_oh

    ds = ds.map(_parse, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=training)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def count_examples(manifest_path: Path) -> int:
    return len(pd.read_csv(manifest_path))
