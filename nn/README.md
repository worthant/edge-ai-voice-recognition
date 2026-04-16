# KWS on ESP32-S3 — обучение и квантизация DS-CNN

Production-ready пайплайн для обучения Depthwise Separable CNN на Google Speech
Commands v2 с последующей INT8 квантизацией (PTQ и QAT) и экспортом под
TensorFlow Lite Micro на ESP32-S3.

Архитектура: DS-CNN вариант S из **"Hello Edge: Keyword Spotting on
Microcontrollers"** (Zhang et al., 2017), baseline из MLPerf Tiny.

## Целевые метрики

| Метрика                   | Цель                                |
| ------------------------- | ----------------------------------- |
| Accuracy на test set      | > 90%                               |
| Размер .tflite после INT8 | < 100 KB (желательно < 50 KB)       |
| Tensor arena на ESP32-S3  | ~60 KB (уточняется профилированием) |

## Требования

- **OS:** Ubuntu 22.04 / macOS 13+ / Windows 11 (через WSL2)
- **Python:** 3.10+
- **Диск:** ~5 GB (датасет ~2.3 GB + распакованный ~2.5 GB)
- **GPU:** опционально. На CPU обучение DS-CNN S занимает ~45 минут, на RTX 3060
  — ~10 минут
- **RAM:** минимум 8 GB (для `tf.data` pipeline)
