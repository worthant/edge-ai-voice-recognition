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

/*
 * Benchmark mode: run inference N times on the current input tensor contents
 * and stream a CSV profile over UART. Assumes kws_classify was called at least
 * once before, so the input tensor already contains quantized MFCC.
 *
 * Output format on UART:
 *   ===KWS_PROFILE_BEGIN===
 *   kws_profile,runs=100,warmup=5,arena_used=NNNN
 *   tag,total_ticks_us,event_count,avg_ticks_us,percent
 *   CONV_2D,12345,100,123.45,42.30
 *   ...
 *   ===KWS_PROFILE_END===
 */
esp_err_t kws_benchmark(int num_runs, int warmup_runs);

#ifdef __cplusplus
}
#endif
