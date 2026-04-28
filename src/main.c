/*
 * main.c — ESP32-S3 voice recognition
 *
 * Two modes (compile-time):
 *   USE_CUSTOM_KWS=1  → DS-CNN with TFLite Micro (our model)
 *   USE_CUSTOM_KWS=0  → WakeNet9 via esp-sr (Espressif proprietary)
 *
 * Both use the same wake-from-sleep + I2S mic infrastructure.
 */
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "i2s_mic.h"
#include "vad_sleep.h"
#include "voice_engine.h"
#include "ws2812_led.h"

static const char *TAG = "main";

static void on_detect(int idx, const char *word) {
    ESP_LOGW(TAG, ">>> DETECTED: %s <<<", word);
    ws2812_blink_model(idx);
}

void app_main(void) {
    ESP_ERROR_CHECK(ws2812_init());

    if (!vad_sleep_wakeup_by_sound()) {
        ESP_LOGI(TAG, "cold boot → sleep");
        vad_sleep_enter(true);
        return;
    }

    ESP_LOGI(TAG, "sound wakeup → inference");
    ESP_ERROR_CHECK(i2s_mic_init());
    ESP_ERROR_CHECK(voice_engine_init());

    /* Single-shot: record, detect, done */
    voice_engine_run(2500, on_detect);

    vTaskDelay(pdMS_TO_TICKS(2000));
    ws2812_clear();
    voice_engine_deinit();
    vad_sleep_enter(true);
}
