/*
 * KWS (Keyword Spotting) inference — C interface over TFLite Micro.
 *
 * Usage:
 *   kws_init()  → load model, allocate arena
 *   kws_classify(mfcc_buf, &result) → run inference
 *   kws_deinit() → free resources
 */

#pragma once

#include "esp_err.h"
#include "mfcc.h" /* MFCC_NUM_FRAMES, MFCC_NUM_COEFFS */

#ifdef __cplusplus
extern "C" {
#endif

#define KWS_NUM_CLASSES 12

typedef enum {
    KWS_YES = 0,
    KWS_NO,
    KWS_UP,
    KWS_DOWN,
    KWS_LEFT,
    KWS_RIGHT,
    KWS_ON,
    KWS_OFF,
    KWS_STOP,
    KWS_GO,
    KWS_SILENCE,
    KWS_UNKNOWN,
} kws_label_t;

typedef struct {
    kws_label_t label;
    float score; /* dequantized logit of winning class */
} kws_result_t;

esp_err_t kws_init(void);
esp_err_t kws_classify(const float mfcc[MFCC_NUM_FRAMES][MFCC_NUM_COEFFS],
                       kws_result_t *result);
const char *kws_label_name(kws_label_t label);
void kws_deinit(void);

#ifdef __cplusplus
}
#endif
