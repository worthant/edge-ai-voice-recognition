# ESP32-S3 Voice Recognition Firmware

Энергоэффективная прошивка для распознавания речи на ESP32-S3 с использованием
TensorFlow Lite Micro и ESP-DL.

## Описание

Эта прошивка реализует keyword spotting (распознавание ключевых слов) на
микроконтроллере ESP32-S3 с минимальным энергопотреблением. Использует MEMS
микрофон INMP441 для захвата аудио и квантизированные нейронные сети для
inference.

## Возможности

- Захват аудио через I2S (INMP441 микрофон)
- TensorFlow Lite Micro inference
- ESP-DL модели (keyword spotting, speech commands)
- Поддержка режимов энергосбережения (Light Sleep / Deep Sleep)
- Логирование энергопотребления (интеграция с внешним INA219)
- SIMD оптимизации (Xtensa LX7)

## Место в системе прототипа

Эта прошивка работает на **ESP32-S3 Zero** (выделена красным на схеме ниже) в
составе измерительной системы:

```mermaid
---
config:
  layout: dagre
---
flowchart LR
    subgraph system1["Power Domain №1 - ИЗМЕРЯЕМАЯ СИСТЕМА"]
        direction TB
        BAT1["18650~4V DC"]
        TP1["TP4056защита"]
        INA["INA219измерениеI2C addr 0x40"]
        S3["ESP32-S3 Zero===> ЭТА ПРОШИВКА <===TFLite Micro / ESP-DLsleep modes"]
        MIC["INMP441MEMS микрофонI2S slave"]
    end

    subgraph system2["Power Domain №2 - Логгер"]
        direction LR
        BAT2["18650~4V DC"]
        TP2["TP4056защита"]
        C3["ESP32-C3 Miniсбор данныхUART/WiFi/BT"]
    end

    BAT1 --> TP1
    TP1 --> INA
    INA --> S3
    MIC -->|I2S streamSD data| S3
    S3 -->|SCK/WS clocks| MIC
    S3 -.->|3.3V GND| MIC

    BAT2 --> TP2
    TP2 --> C3
    INA |I2Caddr 0x40| C3

    C3 -->|USB-UARTWiFiBLE| PC["PCанализmatplotlib"]

    BAT1:::powerDomain
    TP1:::powerDomain
    INA:::sensor
    S3:::target
    MIC:::audio
    BAT2:::logger
    TP2:::logger
    C3:::logger
    PC:::logger

    classDef powerDomain fill:#e1f5ff,stroke:#0066cc,stroke-width:2px
    classDef sensor fill:#fff4e1,stroke:#ff9800,stroke-width:2px
    classDef logger fill:#f0fff0,stroke:#4caf50,stroke-width:2px
    classDef audio fill:#ffe1f0,stroke:#e91e63,stroke-width:2px
    classDef target fill:#ffcccc,stroke:#cc0000,stroke-width:4px,color:#000
```

**Примечание:** Прошивка работает автономно на ESP32-S3 с микрофоном. INA219 и
ESP32-C3 используются только для измерения энергопотребления и не требуются для
базовой функциональности.

## Требования

### Железо

- ESP32-S3 (любая плата с USB, например Waveshare ESP32-S3-Zero)
- INMP441 MEMS микрофон
- (Опционально) INA219 для измерения энергопотребления

### Софт

- ESP-IDF v5.1 или новее
- Python 3.8+
- Git

## Быстрый старт

### 1. Установка ESP-IDF

```bash
# Клонируем ESP-IDF
git clone -b v5.1 --recursive https://github.com/espressif/esp-idf.git
cd esp-idf
./install.sh esp32s3

# Активируем окружение
. ./export.sh
```

### 2. Клонирование репозитория

```bash
git clone https://github.com/yourusername/edge-ai-voice-recognition.git
cd edge-ai-voice-recognition
```

### 3. Конфигурация

```bash
idf.py menuconfig
```

Настройки:

- **Serial flasher config → Flash size** → 4 MB (или больше)
- **Component config → ESP32S3-Specific → CPU frequency** → 240 MHz

### 4. Сборка и прошивка

```bash
# Сборка
idf.py build

# Прошивка (замени /dev/ttyUSB0 на свой порт)
idf.py -p /dev/ttyUSB0 flash

# Мониторинг логов
idf.py -p /dev/ttyUSB0 monitor
```

## Подключение оборудования

### INMP441 Микрофон

| INMP441 Pin | ESP32-S3 Pin | Описание           |
| ----------- | ------------ | ------------------ |
| VDD         | 3.3V         | Питание            |
| GND         | GND          | Земля              |
| WS          | GPIO4        | Word Select (I2S)  |
| SCK         | GPIO5        | Serial Clock (I2S) |
| SD          | GPIO6        | Serial Data (I2S)  |
| L/R         | GND          | Левый канал        |

### Схема подключения

```
ESP32-S3           INMP441
  3.3V  ────────── VDD
  GND   ────────── GND
  GPIO4 ────────── WS
  GPIO5 ────────── SCK
  GPIO6 ────────── SD
         ────────── L/R (→ GND)
```

## Структура проекта

```
edge-ai-voice-recognition/
├── main/
│   ├── main.c              # Точка входа
│   ├── audio_capture.c     # I2S захват аудио
│   ├── model_inference.c   # TFLite / ESP-DL inference
│   └── power_management.c  # Режимы сна
├── components/
│   ├── tflite-micro/       # TensorFlow Lite Micro
│   └── esp-dl/             # ESP-DL framework
├── models/
│   ├── keyword_model.tflite        # TFLite модель
│   └── speech_commands_espdl.bin   # ESP-DL модель
├── CMakeLists.txt
└── README.md
```

## Конфигурация I2S

Прошивка использует следующие настройки I2S:

```c
i2s_config_t i2s_config = {
    .mode = I2S_MODE_MASTER | I2S_MODE_RX,
    .sample_rate = 16000,           // 16 kHz для речи
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_I2S,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 4,
    .dma_buf_len = 1024,
};
```

## Режимы энергосбережения

Прошивка поддерживает три режима работы:

| Режим           | Потребление | Описание                 |
| --------------- | ----------- | ------------------------ |
| **Active**      | ~60-80 mA   | AI inference активен     |
| **Light Sleep** | ~0.8 mA     | CPU спит, wake on timer  |
| **Deep Sleep**  | ~10 µA      | Только RTC, wake on GPIO |

### Пример использования Light Sleep

```c
// Включить режим light sleep между inference
esp_sleep_enable_timer_wakeup(100000); // 100ms
esp_light_sleep_start();
```

## Модели

### TensorFlow Lite Micro

Прошивка поддерживает TFLite модели с квантизацией int8.

**Формат модели:**

- **Входной тензор:** `[1, 16000, 1]` (1 секунда аудио @ 16 kHz)
- **Выходной тензор:** `[1, NUM_CLASSES]` (вероятности классов)
- **Квантизация:** int8
- **Размер:** ~200-300 KB

### ESP-DL

Поддержка готовых моделей из ESP-DL:

- Keyword Spotting (10 команд)
- Speech Commands (35 слов)
- Wake Word Detection

**Преимущества ESP-DL:**

- Оптимизация под Xtensa LX7 (SIMD)
- Меньший размер моделей
- Готовые предобученные веса

### Замена модели

**TFLite:**

1. Положи свою модель в `models/custom_model.tflite`
2. Обнови `main/CMakeLists.txt`:

```cmake
   target_add_binary_data(${COMPONENT_TARGET}
       "../models/custom_model.tflite"
       BINARY)
```

**ESP-DL:**

```c
// В main.c
#include "esp_mn_speech_commands.h"
model_iface_data_t *model = esp_mn_speech_commands_create(...);
```

## Отладка

### Логи

```bash
# Включить debug логи для I2S
idf.py menuconfig
# → Component config → Log output → Default log verbosity → Debug
```

### Проверка I2S

```c
// В main.c добавь дамп первых 100 сэмплов
for (int i = 0; i < 100; i++) {
    ESP_LOGI(TAG, "Sample[%d]: %d", i, audio_buffer[i]);
}
```

## Производительность

На ESP32-S3 @ 240 MHz:

**TensorFlow Lite:**

- **Inference time:** ~50-80 ms (зависит от модели)
- **Audio capture latency:** ~64 ms (1024 samples @ 16 kHz)
- **Total loop time:** ~150 ms
- **Power consumption (active):** ~70 mA

**ESP-DL:**

- **Inference time:** ~30-40 ms (оптимизировано под SIMD)
- **Power consumption (active):** ~60 mA

## TODO

- [ ] Wake-on-voice через ULP
- [ ] Streaming inference (без буферизации)
- [ ] Поддержка микрофонов PDM
- [ ] ? OTA обновление моделей
- [ ] ?? Интеграция с Home Assistant

## Лицензия

MIT License

## Благодарности

- [ESP-IDF](https://github.com/espressif/esp-idf)
- [TensorFlow Lite Micro](https://github.com/tensorflow/tflite-micro)
- [ESP-DL](https://github.com/espressif/esp-dl)
