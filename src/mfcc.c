/*
 * MFCC for ESP32-S3.  Uses esp-dsp FFT, plain C for mel+DCT.
 *
 * Pipeline matches TF training:
 *   stft(hann,640,320,1024) → abs → mel_weights → log → DCT-II[:10]
 */

#include "mfcc.h"
#include "dsps_fft2r.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "mel_matrix.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

static const char *TAG = "mfcc";

struct mfcc_ctx {
    float hann[MFCC_WINDOW_SAMPLES];
    float dct_matrix[MFCC_NUM_COEFFS][MFCC_MEL_BINS];
    float *fft_buf; /* complex interleaved, 2 * FFT_LENGTH */
    float *mag_buf; /* FFT_LENGTH / 2 + 1 */
};

static void build_hann(float *w, int n) {
    for (int i = 0; i < n; i++)
        w[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / n));
}

/* DCT-II ortho normalisation (matches TF mfccs_from_log_mel_spectrograms) */
static void build_dct(float d[MFCC_NUM_COEFFS][MFCC_MEL_BINS]) {
    float s = sqrtf(2.0f / MFCC_MEL_BINS);
    for (int c = 0; c < MFCC_NUM_COEFFS; c++)
        for (int m = 0; m < MFCC_MEL_BINS; m++)
            d[c][m] = s * cosf((float)M_PI * c * (2 * m + 1) /
                               (2.0f * MFCC_MEL_BINS));
}

esp_err_t mfcc_init(mfcc_ctx_t **out) {
    mfcc_ctx_t *c = heap_caps_calloc(1, sizeof(*c), MALLOC_CAP_SPIRAM);
    if (!c)
        return ESP_ERR_NO_MEM;

    c->fft_buf = heap_caps_malloc(2 * MFCC_FFT_LENGTH * sizeof(float),
                                  MALLOC_CAP_SPIRAM);
    c->mag_buf = heap_caps_malloc((MFCC_FFT_LENGTH / 2 + 1) * sizeof(float),
                                  MALLOC_CAP_SPIRAM);
    if (!c->fft_buf || !c->mag_buf) {
        free(c->fft_buf);
        free(c->mag_buf);
        free(c);
        return ESP_ERR_NO_MEM;
    }

    esp_err_t r = dsps_fft2r_init_fc32(NULL, MFCC_FFT_LENGTH);
    if (r != ESP_OK) {
        free(c->fft_buf);
        free(c->mag_buf);
        free(c);
        return r;
    }

    build_hann(c->hann, MFCC_WINDOW_SAMPLES);
    build_dct(c->dct_matrix);

    ESP_LOGI(TAG, "ok  win=%d stride=%d fft=%d mel=%d coeff=%d frames=%d",
             MFCC_WINDOW_SAMPLES, MFCC_STRIDE_SAMPLES, MFCC_FFT_LENGTH,
             MFCC_MEL_BINS, MFCC_NUM_COEFFS, MFCC_NUM_FRAMES);
    *out = c;
    return ESP_OK;
}

esp_err_t mfcc_compute(mfcc_ctx_t *c, const int16_t *pcm,
                       float out[MFCC_NUM_FRAMES][MFCC_NUM_COEFFS]) {
    const int K = MFCC_FFT_LENGTH / 2 + 1;

    for (int f = 0; f < MFCC_NUM_FRAMES; f++) {
        const int off = f * MFCC_STRIDE_SAMPLES;
        float *cb = c->fft_buf;

        /* 1. hann-window + zero-pad → complex */
        memset(cb, 0, 2 * MFCC_FFT_LENGTH * sizeof(float));
        for (int i = 0; i < MFCC_WINDOW_SAMPLES; i++)
            cb[2 * i] = ((float)pcm[off + i] / 32768.0f) * c->hann[i];

        /* 2. FFT */
        dsps_fft2r_fc32(cb, MFCC_FFT_LENGTH);
        dsps_bit_rev_fc32(cb, MFCC_FFT_LENGTH);

        /* 3. magnitude |X[k]| */
        float *mag = c->mag_buf;
        for (int k = 0; k < K; k++) {
            float re = cb[2 * k], im = cb[2 * k + 1];
            mag[k] = sqrtf(re * re + im * im);
        }

        /* 4. mel filterbank → log → DCT */
        float mel[MFCC_MEL_BINS];
        for (int m = 0; m < MFCC_MEL_BINS; m++) {
            float s = 0.0f;
            for (int k = 0; k < K; k++)
                s += g_mel_matrix[k * MFCC_MEL_BINS + m] *
                     mag[k]; // index into (513,40) row-major
            mel[m] = logf(s + 1e-6f);
        }
        for (int cc = 0; cc < MFCC_NUM_COEFFS; cc++) {
            float s = 0.0f;
            for (int m = 0; m < MFCC_MEL_BINS; m++)
                s += c->dct_matrix[cc][m] * mel[m];
            out[f][cc] = s;
        }
    }
    return ESP_OK;
}

void mfcc_free(mfcc_ctx_t *c) {
    if (!c)
        return;
    free(c->fft_buf);
    free(c->mag_buf);
    free(c);
    dsps_fft2r_deinit_fc32();
}
