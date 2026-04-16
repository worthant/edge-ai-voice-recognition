"""
Утилиты для работы с аудио и пересчёта MFCC (в т.ч. numpy-реализация
для отладки и экспорта эталонов под ESP32-side тесты).
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf

import config


def load_wav_to_float(path: str) -> np.ndarray:
    contents = tf.io.read_file(path)
    audio, _ = tf.audio.decode_wav(contents, desired_channels=1)
    a = tf.squeeze(audio, axis=-1).numpy().astype(np.float32)
    if a.shape[0] > config.CLIP_SAMPLES:
        a = a[: config.CLIP_SAMPLES]
    elif a.shape[0] < config.CLIP_SAMPLES:
        a = np.pad(a, (0, config.CLIP_SAMPLES - a.shape[0]))
    return a


def mfcc_reference_numpy(audio: np.ndarray) -> np.ndarray:
    """
    Эталонный MFCC через tf.signal (в numpy-обёртке).
    Используется для генерации золотых значений для unit-тестов
    на устройстве.
    """
    assert audio.shape == (config.CLIP_SAMPLES,), audio.shape
    a = tf.constant(audio, dtype=tf.float32)
    stft = tf.signal.stft(
        a,
        frame_length=config.WINDOW_SIZE_SAMPLES,
        frame_step=config.WINDOW_STRIDE_SAMPLES,
        fft_length=config.FFT_LENGTH,
        window_fn=tf.signal.hann_window,
        pad_end=False,
    )
    mag = tf.abs(stft)
    mel_w = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=config.NUM_MEL_BINS,
        num_spectrogram_bins=config.FFT_LENGTH // 2 + 1,
        sample_rate=config.SAMPLE_RATE,
        lower_edge_hertz=config.MEL_LOWER_HZ,
        upper_edge_hertz=config.MEL_UPPER_HZ,
    )
    mel = tf.matmul(mag, mel_w)
    log_mel = tf.math.log(mel + 1e-6)
    mfcc = tf.signal.mfccs_from_log_mel_spectrograms(log_mel)[..., : config.NUM_MFCC]
    mfcc = mfcc[: config.NUM_FRAMES, :]
    return mfcc.numpy()


def export_mel_matrix(path: str) -> None:
    """Сохраняет mel-матрицу в бинарный файл (float32, row-major)
    для проверки на стороне ESP32."""
    mel_w = (
        tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=config.NUM_MEL_BINS,
            num_spectrogram_bins=config.FFT_LENGTH // 2 + 1,
            sample_rate=config.SAMPLE_RATE,
            lower_edge_hertz=config.MEL_LOWER_HZ,
            upper_edge_hertz=config.MEL_UPPER_HZ,
        )
        .numpy()
        .astype(np.float32)
    )
    mel_w.tofile(path)
    print(f"[audio] mel matrix {mel_w.shape} -> {path}")
