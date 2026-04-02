#include "esp_log.h"
#include "i2s_mic.h"
#include "vad_sleep.h"
#include "wakenet.h"
#include "ws2812_led.h"

static const char *TAG = "main";

static void on_detect(int model_idx, const char *word) {
    ws2812_blink_model(model_idx);
}

void app_main(void) {
    ESP_ERROR_CHECK(ws2812_init());

    if (vad_sleep_wakeup_by_sound()) {
        ESP_LOGI(TAG, "sound wakeup → inference");
        ESP_ERROR_CHECK(i2s_mic_init());

        static wakenet_t wn;
        ESP_ERROR_CHECK(wakenet_init(&wn));
        wakenet_run_window(&wn, 3000, on_detect);
        wakenet_deinit(&wn);
    } else {
        ESP_LOGI(TAG, "cold boot → sleep");
    }

    vad_sleep_enter();
}
