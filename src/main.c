#include "driver/i2s_common.h"
#include "driver/i2s_std.h"
#include "driver/i2s_types.h"
#include "esp_err.h"
#include "esp_log.h"
#include "esp_system.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "hal/i2s_types.h"
#include "i2s_mic.h"
#include "portmacro.h"
#include <math.h>
#include <stdio.h>

static const char *TAG = "main";

void app_main() {
    ESP_LOGI(TAG, "--- entering app main ---");
    ESP_ERROR_CHECK(i2s_mic_init());
    i2s_mic_demo();
}
