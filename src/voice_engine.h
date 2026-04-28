/*
 * voice_engine.h — unified interface for keyword detection.
 *
 * Two implementations:
 *   voice_engine_wakenet.c  — Espressif WakeNet9 (esp-sr)
 *   voice_engine_kws.c      — custom DS-CNN (TFLite Micro)
 *
 * Only one is compiled, selected by USE_CUSTOM_KWS define.
 */

#pragma once

#include "esp_err.h"
#include <stdint.h>

/* Called on each detection: engine-specific index + human-readable word */
typedef void (*voice_detect_cb_t)(int idx, const char *word);

esp_err_t voice_engine_init(void);
void voice_engine_run(uint32_t listen_ms, voice_detect_cb_t cb);
void voice_engine_deinit(void);
