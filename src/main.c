/*
 * main.c — esp32-s3 voice recognition
 *
 * Two modes (compile-time):
 *   USE_CUSTOM_KWS=1 : DS-CNN with TFLite Micro (my model)
 *   USE_CUSTOM_KWS=0 : WakeNet9 via esp-sr (Espressif proprietary)
 */
#include "display.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "i2s_mic.h"
#include "vad_sleep.h"
#include "voice_engine.h"
#include "ws2812_led.h"
#include <stdio.h>

static const char *TAG = "main";

static void on_detect(int idx, const char *word) {
    ESP_LOGW(TAG, ">>> DETECTED: %s <<<", word);
    ws2812_blink_model(idx);
}

void app_main(void) {
    int64_t t_boot = esp_timer_get_time();

    ESP_ERROR_CHECK(ws2812_init());
    ESP_ERROR_CHECK(display_init());

    if (!vad_sleep_wakeup_by_sound()) {
        ESP_LOGI(TAG, "cold boot -> sleep");
        display_fsm("SLEEP", "zzz...", DISP_DKGREEN, DISP_BLACK);
        vTaskDelay(pdMS_TO_TICKS(500));
        vad_sleep_enter(true);
        return;
    }

    /* wake tf up! */
    char buf[32];
    snprintf(buf, sizeof(buf), "%lldms", (long long)(t_boot / 1000));
    display_fsm("WAKE UP", buf, DISP_RED, DISP_BLACK);
    ESP_LOGI(TAG, "sound wakeup -> inference (boot %lldms)",
             (long long)(t_boot / 1000));

    ESP_ERROR_CHECK(i2s_mic_init());
    ESP_ERROR_CHECK(voice_engine_init());

    /* record + mfcc calc + inference */
    int64_t t0 = esp_timer_get_time();
    voice_engine_run(2500, on_detect);
    int total_ms = (int)((esp_timer_get_time() - t0) / 1000);
    ESP_LOGI(TAG, "total pipeline: %dms", total_ms);

    /* hold result on screen */
    vTaskDelay(pdMS_TO_TICKS(3000));

    /* go sleep */
    ws2812_clear();
    display_fsm("SLEEP", "zzz...", DISP_DKGREEN, DISP_BLACK);
    vTaskDelay(pdMS_TO_TICKS(500));

    voice_engine_deinit();
    vad_sleep_enter(true);
}
