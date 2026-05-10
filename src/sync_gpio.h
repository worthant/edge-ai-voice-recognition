#pragma once
#include "esp_err.h"

esp_err_t sync_gpio_init(void);   // configure GPIO1 as output, LOW
void      sync_pulse(void);       // pulse HIGH ~1ms, then LOW
void      sync_reset(void);       // pull LOW (reset before new cycle)
