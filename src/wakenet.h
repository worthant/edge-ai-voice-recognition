#pragma once

#include "esp_err.h"
#include "esp_wn_iface.h"
#include "esp_wn_models.h"
#include "model_path.h"
#include <stdint.h>

#define WN_MAX_MODELS 8

typedef struct {
    const esp_wn_iface_t *ops;
    model_iface_data_t *model;
    char *name;
} wn_model_t;

typedef struct {
    wn_model_t models[WN_MAX_MODELS];
    int count;
    int chunk_sz;
} wakenet_t;

// called on each detection: model index + word name
typedef void (*wakenet_detect_cb_t)(int model_idx, const char *word);

esp_err_t wakenet_init(wakenet_t *wn);
void wakenet_run_window(wakenet_t *wn, uint32_t ms, wakenet_detect_cb_t cb);
void wakenet_deinit(wakenet_t *wn);
