"""
Единая точка истины для всех гиперпараметров и путей.
Нигде в других файлах значения не хардкодятся.
"""

from pathlib import Path

# ============================================================================
# Пути
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
DATASET_URL = "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
DATASET_ARCHIVE = DATA_DIR / "speech_commands_v0.02.tar.gz"
DATASET_ROOT = DATA_DIR / "speech_commands_v0.02"
MANIFEST_DIR = DATA_DIR / "manifests"
BG_NOISE_SUBDIR = "_background_noise_"

RESULTS_DIR = PROJECT_ROOT / "results"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
MODEL_DIR = RESULTS_DIR / "models"
LOG_DIR = RESULTS_DIR / "logs"
PLOT_DIR = RESULTS_DIR / "plots"
TENSORBOARD_DIR = LOG_DIR / "tensorboard"

for d in [CHECKPOINT_DIR, MODEL_DIR, LOG_DIR, PLOT_DIR, TENSORBOARD_DIR, MANIFEST_DIR]:
    d.mkdir(parents=True, exist_ok=True)

FP32_KERAS = MODEL_DIR / "ds_cnn_fp32.keras"
PTQ_TFLITE = MODEL_DIR / "ds_cnn_ptq_int8.tflite"
QAT_TFLITE = MODEL_DIR / "ds_cnn_qat_int8.tflite"

# ============================================================================
# Аудио и MFCC
# ============================================================================
SAMPLE_RATE = 16000  # Гц
CLIP_DURATION_MS = 1000  # все сэмплы приводятся к 1 секунде
CLIP_SAMPLES = SAMPLE_RATE * CLIP_DURATION_MS // 1000  # = 16000

WINDOW_SIZE_MS = 40
WINDOW_STRIDE_MS = 20
WINDOW_SIZE_SAMPLES = SAMPLE_RATE * WINDOW_SIZE_MS // 1000  # = 640
WINDOW_STRIDE_SAMPLES = SAMPLE_RATE * WINDOW_STRIDE_MS // 1000  # = 320
FFT_LENGTH = 1024  # ближайшая степень двойки >= WINDOW_SIZE_SAMPLES

NUM_MEL_BINS = 40
MEL_LOWER_HZ = 20.0
MEL_UPPER_HZ = 4000.0

NUM_MFCC = 10  # стандарт для KWS на MCU
# Количество фреймов: (CLIP_SAMPLES - WINDOW_SIZE_SAMPLES) / STRIDE + 1
NUM_FRAMES = (CLIP_SAMPLES - WINDOW_SIZE_SAMPLES) // WINDOW_STRIDE_SAMPLES + 1  # = 49

# Итоговый shape входа модели: (49, 10, 1)
INPUT_SHAPE = (NUM_FRAMES, NUM_MFCC, 1)

# ============================================================================
# Классы
# ============================================================================
TARGET_WORDS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
SILENCE_LABEL = "_silence_"
UNKNOWN_LABEL = "_unknown_"
ALL_LABELS = TARGET_WORDS + [SILENCE_LABEL, UNKNOWN_LABEL]  # 12 штук
NUM_CLASSES = len(ALL_LABELS)
LABEL_TO_INDEX = {lbl: i for i, lbl in enumerate(ALL_LABELS)}
INDEX_TO_LABEL = {i: lbl for lbl, i in LABEL_TO_INDEX.items()}

# Доля silence и unknown относительно среднего количества примеров target-класса
SILENCE_PERCENTAGE = 10.0  # %
UNKNOWN_PERCENTAGE = 10.0  # %

# ============================================================================
# Аугментации
# ============================================================================
TIME_SHIFT_MS = 100  # ± 100 мс
TIME_SHIFT_SAMPLES = SAMPLE_RATE * TIME_SHIFT_MS // 1000  # = 1600

BG_NOISE_PROB = 0.7  # вероятность подмешать фон в training пример
BG_NOISE_VOLUME_MAX = 0.1  # амплитуда фонового шума (доля от сигнала)

# ============================================================================
# Training
# ============================================================================
SEED = 42
BATCH_SIZE = 100
EPOCHS = 50
LEARNING_RATE_INIT = 1e-3
LEARNING_RATE_MIN = 1e-5  # для cosine decay
L2_REGULARIZATION = 1e-4
LABEL_SMOOTHING = 0.1

# ============================================================================
# DS-CNN (вариант S из Hello Edge)
# модификация: увеличил модель для увеличения точности, вариант M
# ============================================================================
DS_CNN_CONFIG = {
    "first_conv_filters": 172,
    "first_conv_kernel": (10, 4),
    "first_conv_stride": (2, 2),
    "num_ds_blocks": 6,
    "ds_filters": 172,
    "ds_kernel": (3, 3),
}

# ============================================================================
# Quantization
# ============================================================================
PTQ_REPRESENTATIVE_SAMPLES = 500
QAT_EPOCHS = 10
QAT_LEARNING_RATE = 1e-4
QAT_BATCH_SIZE = 100

# ============================================================================
# Evaluation
# ============================================================================
EVAL_LATENCY_WARMUP = 10
EVAL_LATENCY_RUNS = 200
