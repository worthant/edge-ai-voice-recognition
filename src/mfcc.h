/*
 * MFCC feature extraction for ESP32-S3.
 *
 * Constants match nn/config.py exactly:
 *   sample_rate=16000, window=40ms, stride=20ms, fft=1024,
 *   40 mel bins (20-4000 Hz), 10 MFCC coefficients, 49 frames.
 *
 * Usage:
 *   mfcc_ctx_t *ctx;
 *   mfcc_init(&ctx);
 *   mfcc_compute(ctx, pcm_16k_1sec, out_49x10);
 *   mfcc_free(ctx);
 */

#pragma once

#include "esp_err.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Audio params — must match nn/config.py */
#define MFCC_SAMPLE_RATE      16000
#define MFCC_CLIP_SAMPLES     16000   /* 1 second */

/* STFT params */
#define MFCC_WINDOW_MS        40
#define MFCC_STRIDE_MS        20
#define MFCC_WINDOW_SAMPLES   640     /* SAMPLE_RATE * WINDOW_MS / 1000 */
#define MFCC_STRIDE_SAMPLES   320     /* SAMPLE_RATE * STRIDE_MS / 1000 */
#define MFCC_FFT_LENGTH       1024    /* next power of 2 >= 640 */

/* Mel / MFCC params */
#define MFCC_MEL_BINS         40
#define MFCC_MEL_LO_HZ       20.0f
#define MFCC_MEL_HI_HZ       4000.0f
#define MFCC_NUM_COEFFS       10

/* Derived */
#define MFCC_NUM_FRAMES       49      /* (16000 - 640) / 320 + 1 */
#define MFCC_SPEC_BINS        513     /* FFT_LENGTH / 2 + 1 */

/* Opaque context — holds precomputed tables + scratch buffers */
typedef struct mfcc_ctx mfcc_ctx_t;

esp_err_t mfcc_init(mfcc_ctx_t **out);
esp_err_t mfcc_compute(mfcc_ctx_t *ctx, const int16_t *pcm,
                       float out[MFCC_NUM_FRAMES][MFCC_NUM_COEFFS]);
void      mfcc_free(mfcc_ctx_t *ctx);

#ifdef __cplusplus
}
#endif
