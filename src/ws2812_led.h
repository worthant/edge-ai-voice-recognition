#pragma once
#include "esp_err.h"
#include <stdint.h>

esp_err_t ws2812_init(void);
void ws2812_blink(uint8_t r, uint8_t g, uint8_t b);
void ws2812_blink_model(int model_idx);
void ws2812_clear(void);
