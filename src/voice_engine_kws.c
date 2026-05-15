/*
 * voice_engine_kws.c — custom DS-CNN backend.
 *
 * Single-shot mode:
 *   1. Record listen_ms of audio after wake-up
 *   2. Find the 1-second window with highest energy
 *   3. Run MFCC + inference on that window
 *   4. Report result via callback
 */

#include "sync_gpio.h"
#ifdef USE_CUSTOM_KWS

#include "display.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "i2s_mic.h"
#include "kws.h"
#include "mfcc.h"
#include "voice_engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *TAG = "ve_kws";

#define KWS_THRESHOLD 2.0f

static mfcc_ctx_t *mfcc_ctx;

static esp_err_t record_samples_with_progress(int16_t *buf, int total) {
    int filled = 0;
    int last_pct = -1;
    while (filled < total) {
        size_t got = 0;
        int want = total - filled;
        if (want > I2S_MIC_CHUNK_SAMPLES)
            want = I2S_MIC_CHUNK_SAMPLES;
        esp_err_t r = i2s_mic_read_s16(buf + filled, want, &got);
        if (r != ESP_OK)
            return r;
        filled += got;

        int pct = (filled * 100) / total;
        if (pct != last_pct) {
            display_progress(pct, DISP_BLACK);
            last_pct = pct;
        }
    }
    return ESP_OK;
}

static int find_best_window(const int16_t *buf, int total_samples) {
    int window = MFCC_CLIP_SAMPLES;
    int stride = MFCC_STRIDE_SAMPLES;
    int best_offset = 0;
    float best_energy = 0.0f;

    for (int off = 0; off + window <= total_samples; off += stride) {
        float energy = 0.0f;
        for (int i = off; i < off + window; i++) {
            float s = (float)buf[i];
            energy += s * s;
        }
        if (energy > best_energy) {
            best_energy = energy;
            best_offset = off;
        }
    }
    return best_offset;
}

esp_err_t voice_engine_init(void) {
    esp_err_t r = mfcc_init(&mfcc_ctx);
    if (r != ESP_OK)
        return r;

    r = kws_init();
    if (r != ESP_OK) {
        mfcc_free(mfcc_ctx);
        return r;
    }
    return ESP_OK;
}

void voice_engine_run(uint32_t listen_ms, voice_detect_cb_t cb) {
    int record_count = (MFCC_SAMPLE_RATE * listen_ms) / 1000;
    if (record_count < MFCC_CLIP_SAMPLES)
        record_count = MFCC_CLIP_SAMPLES;

    int16_t *audio =
        heap_caps_malloc(record_count * sizeof(int16_t), MALLOC_CAP_SPIRAM);
    float (*mfcc)[MFCC_NUM_COEFFS] = heap_caps_malloc(
        sizeof(float) * MFCC_NUM_FRAMES * MFCC_NUM_COEFFS, MALLOC_CAP_SPIRAM);

    if (!audio || !mfcc) {
        ESP_LOGE(TAG, "alloc failed");
        free(audio);
        free(mfcc);
        return;
    }

    /* record */
    sync_pulse(); // signal record state to logger
    char buf[32];
    snprintf(buf, sizeof(buf), "%lums", (unsigned long)listen_ms);
    display_fsm("RECORD", buf, DISP_YELLOW, DISP_BLACK);

    int64_t t0 = esp_timer_get_time();
    esp_err_t r = record_samples_with_progress(audio, record_count);
    int64_t t_rec = esp_timer_get_time() - t0;

    if (r != ESP_OK) {
        ESP_LOGE(TAG, "recording failed");
        free(audio);
        free(mfcc);
        return;
    }
    ESP_LOGI(TAG, "recorded %lldms", (long long)(t_rec / 1000));

    /* mfcc calc */
    sync_pulse(); // signal mfcc state to logger
    int offset = find_best_window(audio, record_count);
    display_fsm("MFCC", "49x10", DISP_ORANGE, DISP_BLACK);

    t0 = esp_timer_get_time();
    mfcc_compute(mfcc_ctx, audio + offset, mfcc);
    int64_t t_mfcc = esp_timer_get_time() - t0;
    ESP_LOGI(TAG, "MFCC %lldms", (long long)(t_mfcc / 1000));

    /* inference */
    sync_pulse(); // signal inference state to logger
    display_fsm("INFERENCE", "DS-CNN-M", DISP_PURPLE, DISP_BLACK);

    kws_result_t res;
    t0 = esp_timer_get_time();
    kws_classify(mfcc, &res);
    int64_t t_inf = esp_timer_get_time() - t0;
    int inf_ms = (int)(t_inf / 1000);

    ESP_LOGI(TAG, "inf=%dms -> %s (%.3f)", inf_ms, kws_label_name(res.label),
             res.score);

    /* Benchmark trigger: if user said "stop" with high confidence, run
     * 100 inferences on this MFCC and dump per-layer profile to SPIFFS. */
    extern esp_err_t kws_benchmark(int num_runs, int warmup_runs);
    if (res.label == KWS_STOP && res.score >= KWS_THRESHOLD) {
        display_fsm("BENCH", "100 runs", DISP_PURPLE, DISP_BLACK);
        ESP_LOGW(TAG, ">>> BENCH MODE (said 'stop') <<<");
        esp_err_t br = kws_benchmark(100, 5);
        if (br == ESP_OK) {
            display_fsm("BENCH OK", "/spiffs", DISP_GREEN, DISP_BLACK);
            ESP_LOGW(TAG, ">>> BENCH DONE — dumping CSV <<<");

            /* Dump CSV to UART one time, framed with markers. */
            FILE *f = fopen("/spiffs/profile.csv", "r");
            if (f) {
                printf("\n===PROFILE_CSV_BEGIN===\n");
                char line[256];
                while (fgets(line, sizeof(line), f)) {
                    fputs(line, stdout);
                }
                fclose(f);
                printf("===PROFILE_CSV_END===\n");
                fflush(stdout);
                ESP_LOGW(TAG, ">>> CSV dumped — copy between markers <<<");
            } else {
                ESP_LOGE(TAG, "fopen /spiffs/profile.csv for read failed");
            }

            display_fsm("DONE", "halted", DISP_GREEN, DISP_BLACK);
            while (1) {
                vTaskDelay(pdMS_TO_TICKS(1000));
            }
        }
    }

    /* result */
    sync_pulse(); // signal result state to logger
    if (res.label != KWS_SILENCE && res.label != KWS_UNKNOWN &&
        res.score >= KWS_THRESHOLD) {
        ESP_LOGW(TAG, ">>> DETECTED: %s (%.3f) <<<", kws_label_name(res.label),
                 res.score);
        display_detection(kws_label_name(res.label), res.score, inf_ms);
        if (cb)
            cb((int)res.label, kws_label_name(res.label));
    } else {
        snprintf(buf, sizeof(buf), "%.2f", res.score);
        display_fsm("SILENCE", buf, DISP_BLACK, DISP_GRAY);
        ESP_LOGI(TAG, "no keyword (%s %.3f)", kws_label_name(res.label),
                 res.score);
    }

    free(mfcc);
    free(audio);
}

void voice_engine_deinit(void) {
    kws_deinit();
    mfcc_free(mfcc_ctx);
    mfcc_ctx = NULL;
}

#endif
