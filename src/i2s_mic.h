#pragma once

#include "esp_check.h"
#include "esp_err.h"
#include <stdint.h>

#define I2S_WS GPIO_NUM_4
#define I2S_SCK GPIO_NUM_5
#define I2S_SD GPIO_NUM_6
#define I2S_MIC_SAMPLE_RATE 16000 // 16 kHZ
#define I2S_MIC_CHUNK_SAMPLES 512

esp_err_t i2s_mic_init(void);
esp_err_t i2s_mic_read(int32_t *buf, size_t buf_sz, size_t *bytes_read);
void i2s_mic_demo(void);
esp_err_t i2s_mic_read_s16(int16_t *buf, size_t samples, size_t *samples_read);
