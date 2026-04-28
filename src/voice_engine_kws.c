/*
 * voice_engine_kws.c — custom DS-CNN backend.
 * Compiled only when USE_CUSTOM_KWS is defined.
 */

#ifdef USE_CUSTOM_KWS

#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "i2s_mic.h"
#include "kws.h"
#include "mfcc.h"
#include "voice_engine.h"
#include "ws2812_led.h"

static const char *TAG = "ve_kws";

#define KWS_THRESHOLD 2.0f

static mfcc_ctx_t *mfcc_ctx;
static int16_t *audio_buf;
static float (*mfcc_buf)[MFCC_NUM_COEFFS];

static esp_err_t record_1s(int16_t *buf) {
    int filled = 0;
    while (filled < MFCC_CLIP_SAMPLES) {
        size_t got = 0;
        int want = MFCC_CLIP_SAMPLES - filled;
        if (want > I2S_MIC_CHUNK_SAMPLES)
            want = I2S_MIC_CHUNK_SAMPLES;
        esp_err_t r = i2s_mic_read_s16(buf + filled, want, &got);
        if (r != ESP_OK)
            return r;
        filled += got;
    }
    return ESP_OK;
}

esp_err_t voice_engine_init(void) {
    ESP_LOGI("kws_voice_engine_init", "KWS voice engine init");
    esp_err_t r = mfcc_init(&mfcc_ctx);
    if (r != ESP_OK)
        return r;

    r = kws_init();
    if (r != ESP_OK) {
        mfcc_free(mfcc_ctx);
        return r;
    }

    audio_buf = heap_caps_malloc(MFCC_CLIP_SAMPLES * sizeof(int16_t),
                                 MALLOC_CAP_SPIRAM);
    mfcc_buf = heap_caps_malloc(
        sizeof(float) * MFCC_NUM_FRAMES * MFCC_NUM_COEFFS, MALLOC_CAP_SPIRAM);
    if (!audio_buf || !mfcc_buf) {
        ESP_LOGE(TAG, "alloc failed");
        return ESP_ERR_NO_MEM;
    }
    return ESP_OK;
}

void voice_engine_run(uint32_t listen_ms, voice_detect_cb_t cb) {
    int64_t deadline = esp_timer_get_time() + (int64_t)listen_ms * 1000;
    int n = 0;

    ESP_LOGI(TAG, "listening for %lu ms...", (unsigned long)listen_ms);

    while (esp_timer_get_time() < deadline) {
        if (record_1s(audio_buf) != ESP_OK)
            continue;

        mfcc_compute(mfcc_ctx, audio_buf, mfcc_buf);

        kws_result_t res;
        kws_classify(mfcc_buf, &res);
        n++;

        ESP_LOGI(TAG, "[%d] %s (%.3f)", n, kws_label_name(res.label),
                 res.score);

        if (res.label != KWS_SILENCE && res.label != KWS_UNKNOWN &&
            res.score >= KWS_THRESHOLD) {
            if (cb)
                cb((int)res.label, kws_label_name(res.label));
        }
    }

    ESP_LOGI(TAG, "done, %d inferences", n);
}

void voice_engine_deinit(void) {
    free(mfcc_buf);
    mfcc_buf = NULL;
    free(audio_buf);
    audio_buf = NULL;
    kws_deinit();
    mfcc_free(mfcc_ctx);
    mfcc_ctx = NULL;
}

#endif /* USE_CUSTOM_KWS */
