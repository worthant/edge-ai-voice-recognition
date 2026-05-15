"""
Визуализация конвейера feature extraction для ВКР.

Показывает все промежуточные представления одного аудиосигнала:
  waveform → spectrogram (STFT) → log-mel spectrogram → MFCC

Параметры зашиты те же что в nn/config.py:
  16 kHz, окно 40 мс с шагом 20 мс, FFT 1024, 40 mel-фильтров,
  10 MFCC коэффициентов. Итог — 49 кадров × 10 коэффициентов,
  ровно то что подаётся на вход нейронки в kws.cpp.

Запуск:
    python make_feature_pipeline_figure.py /path/to/yes.wav
        --out figure_feature_pipeline.png
    # или с дефолтным синтетическим аудио для теста:
    python make_feature_pipeline_figure.py --synthetic
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.fft import dct

# Параметры из nn/config.py — должны совпадать со встроенными в прошивку.
SAMPLE_RATE = 16000
CLIP_SAMPLES = 16000
WINDOW_SAMPLES = 640  # 40 мс
STRIDE_SAMPLES = 320  # 20 мс
FFT_LENGTH = 1024
NUM_MEL_BINS = 40
MEL_LOWER_HZ = 20.0
MEL_UPPER_HZ = 4000.0
NUM_MFCC = 10
NUM_FRAMES = 49


def hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def build_mel_matrix(num_bins, fft_len, sample_rate, low_hz, high_hz):
    """Mel-filterbank матрица [num_spectrum_bins, num_mel_bins]."""
    num_spec_bins = fft_len // 2 + 1
    spec_hz = np.linspace(0, sample_rate / 2, num_spec_bins)

    mel_lo = hz_to_mel(low_hz)
    mel_hi = hz_to_mel(high_hz)
    mel_points = np.linspace(mel_lo, mel_hi, num_bins + 2)
    hz_points = mel_to_hz(mel_points)

    mat = np.zeros((num_spec_bins, num_bins), dtype=np.float32)
    for m in range(num_bins):
        left, center, right = hz_points[m], hz_points[m + 1], hz_points[m + 2]
        for k, hz in enumerate(spec_hz):
            if left <= hz <= center:
                mat[k, m] = (hz - left) / (center - left)
            elif center <= hz <= right:
                mat[k, m] = (right - hz) / (right - center)
    return mat


def compute_pipeline(audio):
    """Прогон audio через весь pipeline. Возвращает все промежуточные тензоры."""
    # 1. Waveform
    wav = audio[:CLIP_SAMPLES]
    if len(wav) < CLIP_SAMPLES:
        wav = np.pad(wav, (0, CLIP_SAMPLES - len(wav)))

    # 2. STFT → magnitude spectrogram
    # Hann window, FFT length 1024, frame 640, stride 320.
    f, t, Zxx = signal.stft(
        wav,
        fs=SAMPLE_RATE,
        window="hann",
        nperseg=WINDOW_SAMPLES,
        noverlap=WINDOW_SAMPLES - STRIDE_SAMPLES,
        nfft=FFT_LENGTH,
        return_onesided=True,
        boundary=None,
        padded=False,
    )
    spectrogram = np.abs(Zxx)  # [freq_bins=513, time_frames]
    # обрезаем по числу кадров до NUM_FRAMES для согласованности с моделью
    spectrogram = spectrogram[:, :NUM_FRAMES]

    # 3. Mel-spectrogram
    mel_matrix = build_mel_matrix(
        NUM_MEL_BINS, FFT_LENGTH, SAMPLE_RATE, MEL_LOWER_HZ, MEL_UPPER_HZ
    )
    mel_spec = mel_matrix.T @ spectrogram  # [40, 49]

    # 4. Log-mel
    log_mel = np.log(mel_spec + 1e-6)

    # 5. MFCC — DCT-II по mel-оси, берём первые 10
    mfcc = dct(log_mel, axis=0, type=2, norm="ortho")[:NUM_MFCC, :]

    return wav, spectrogram, log_mel, mfcc


def make_synthetic_audio():
    """Простое тестовое аудио: 0.4 с тишины, 0.4 с звука (шепот гласной), 0.2 с тишины."""
    t = np.arange(CLIP_SAMPLES) / SAMPLE_RATE
    audio = np.zeros(CLIP_SAMPLES)
    # формантоподобный сигнал в 0.4..0.8 с
    voice_mask = (t > 0.4) & (t < 0.8)
    base = np.sin(2 * np.pi * 130 * t)
    formant1 = 0.5 * np.sin(2 * np.pi * 700 * t)
    formant2 = 0.3 * np.sin(2 * np.pi * 1220 * t)
    envelope = np.exp(-((t - 0.6) ** 2) / 0.01)
    audio[voice_mask] = (base + formant1 + formant2)[voice_mask] * envelope[voice_mask]
    audio += 0.005 * np.random.randn(CLIP_SAMPLES)
    return audio.astype(np.float32)


def load_wav(path: Path):
    """Загрузить WAV без зависимости от librosa."""
    from scipy.io import wavfile

    sr, audio = wavfile.read(str(path))
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        print(
            f"WARNING: WAV sample rate {sr} ≠ expected {SAMPLE_RATE}; "
            f"resampling not implemented, results may look wrong"
        )
    return audio


def make_figure(wav, spectrogram, log_mel, mfcc, word_label, out_path):
    """Главная фигура: 4 представления в ряд."""
    fig = plt.figure(figsize=(16, 4.2), constrained_layout=True)
    gs = fig.add_gridspec(1, 4, width_ratios=[1.4, 1.0, 1.0, 1.0])

    # --- 1. Waveform ---
    ax1 = fig.add_subplot(gs[0])
    t_axis = np.arange(len(wav)) / SAMPLE_RATE * 1000  # мс
    ax1.plot(t_axis, wav, color="#222", linewidth=0.5)
    ax1.set_xlim(0, 1000)
    ax1.set_ylim(-1.05 * np.abs(wav).max() or 1, 1.05 * (np.abs(wav).max() or 1))
    ax1.set_xlabel("Время, мс")
    ax1.set_ylabel("Амплитуда")
    ax1.set_title(
        f"1. Сигнал (WAV)\nслово «{word_label}», 16 кГц × 1 с = 16000 отсчётов",
        fontsize=10,
        loc="left",
    )
    ax1.grid(alpha=0.3)

    # --- 2. Spectrogram (STFT magnitude) ---
    ax2 = fig.add_subplot(gs[1])
    # обрезаем до 4 кГц для наглядности (мел-диапазон у нас как раз до 4000)
    freq_cutoff = int(FFT_LENGTH * MEL_UPPER_HZ / SAMPLE_RATE) + 1
    im2 = ax2.imshow(
        20 * np.log10(spectrogram[:freq_cutoff] + 1e-6),
        aspect="auto",
        origin="lower",
        cmap="viridis",
        extent=[0, NUM_FRAMES, 0, MEL_UPPER_HZ / 1000],
    )
    ax2.set_xlabel("Кадр (всего 49)")
    ax2.set_ylabel("Частота, кГц")
    ax2.set_title(
        "2. Спектрограмма (STFT)\nразмер 49 × 513, шкала линейная",
        fontsize=10,
        loc="left",
    )

    # --- 3. Log-mel spectrogram ---
    ax3 = fig.add_subplot(gs[2])
    im3 = ax3.imshow(
        log_mel,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        extent=[0, NUM_FRAMES, 0, NUM_MEL_BINS],
    )
    ax3.set_xlabel("Кадр (всего 49)")
    ax3.set_ylabel("Mel-канал")
    ax3.set_title(
        "3. Log-mel спектрограмма\nразмер 49 × 40, шкала по слуху",
        fontsize=10,
        loc="left",
    )

    # --- 4. MFCC ---
    ax4 = fig.add_subplot(gs[3])
    im4 = ax4.imshow(
        mfcc,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        extent=[0, NUM_FRAMES, 0, NUM_MFCC],
    )
    ax4.set_xlabel("Кадр (всего 49)")
    ax4.set_ylabel("MFCC коэф.")
    ax4.set_title("4. MFCC (вход в нейронку)\nразмер 49 × 10", fontsize=10, loc="left")

    fig.suptitle(
        "Конвейер извлечения признаков: от сырого аудио до входа DS-CNN",
        fontsize=12,
        fontweight="bold",
        y=1.04,
    )

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    pdf_path = Path(out_path).with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"saved: {out_path}")
    print(f"saved: {pdf_path}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "wav",
        nargs="?",
        default=None,
        help="Путь к WAV (16 кГц mono); если не указан — синтетика",
    )
    ap.add_argument(
        "--synthetic",
        action="store_true",
        help="Принудительно использовать синтетическое аудио",
    )
    ap.add_argument("--word", default="yes", help="Подпись слова на графике")
    ap.add_argument("--out", default="figure_feature_pipeline.png")
    args = ap.parse_args()

    if args.wav and not args.synthetic:
        wav = load_wav(Path(args.wav))
        word = args.word or Path(args.wav).parent.name
    else:
        wav = make_synthetic_audio()
        word = "synthetic"

    waveform, spec, logmel, mfcc = compute_pipeline(wav)
    make_figure(waveform, spec, logmel, mfcc, word, args.out)


if __name__ == "__main__":
    main()
