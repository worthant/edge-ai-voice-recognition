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

## Использование

### 1. сетапчик

```bash
cd nn
```

> if .venv not created

```bash
python3.10 -m venv .venv
```

```bash
source .venv/bin/activate.fish
```

### 2. датасетик

> [!IMPORTANT]  
> эта корова весит 2.3Gb.

```bash
python -m data.download
```

> на выходе получим `data/speech_commands_v0.02/` с поддиректориями по командам
> и `_background_noise_/`

### 3. манифестики

```bash
python -m data.preprocess
```

### 4. обучаем fp32 baseline

> [!IMPORTANT]  
> Для сохранения эмоционального спокойствия делайте это в гугл коллабе или на
> собственном кластере rtx5090

```bash
python train.py
```

**На выходе:**

- `results/models/ds_cnn_fp32.h5`
- `results/models/ds_cnn_fp32_saved_model/`
- `results/logs/train.csv`
- `results/logs/tensorboard/` (смотреть:
  `tensorboard --logdir results/logs/tensorboard`)
- В stdout: финальная test accuracy, confusion matrix, размер модели

### 5. ptq

```bash
python quantize_ptq.py
```

**На выходе:** `results/models/ds_cnn_ptq_int8.tflite` + в stdout: accuracy,
size, drop от FP32.

### 6. qat

```bash
python quantize_qat.py
```

**На выходе:** `results/models/ds_cnn_qat_int8.tflite` + сравнение с PTQ.

### 7. сравнительный анализ

```bash
python compare_models.py
```

**На выходе:** `results/comparison.md`, `results/plots/*.png`.

### 8. экспорт в C-массив весов для esp32

```bash
# По умолчанию берёт QAT модель как итоговую (она точнее PTQ)
python export_to_c.py --input results/models/ds_cnn_qat_int8.tflite \
                      --output ../esp32_firmware/components/nn_inference/src
```

**На выходе:** `model_data.cc` и `model_data.h` готовые к сборке в ESP-IDF.

## TODO: таблица результатов

| Модель        | Test Accuracy | Size (KB) | Inference (CPU, ms) | Acc drop vs FP32 |
| ------------- | ------------- | --------- | ------------------- | ---------------- |
| FP32 baseline | —             | —         | —                   | —                |
| PTQ INT8      | —             | —         | —                   | —                |
| QAT INT8      | —             | —         | —                   | —                |
