#include "driver/i2s_common.h"
#include "driver/i2s_std.h"
#include "driver/i2s_types.h"
#include "esp_err.h"
#include "esp_log.h"
#include "esp_system.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "hal/i2s_types.h"
#include "portmacro.h"
#include <math.h>
#include <stdio.h>

static const char *TAG = "main";

#define I2S_WS GPIO_NUM_4
#define I2S_SCK GPIO_NUM_5
#define I2S_SD GPIO_NUM_6

i2s_chan_handle_t rx_hdl;

// inmp441 mems i2s mic
void init_mic() {
    ESP_LOGI(TAG, "--- i2s init START ---");
    i2s_chan_config_t chn_cfg =
        I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
    ESP_ERROR_CHECK(i2s_new_channel(&chn_cfg, NULL, &rx_hdl));

    i2s_std_config_t std_cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(16000),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(
            I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_MONO),
        .gpio_cfg =
            {
                .mclk = I2S_GPIO_UNUSED,
                .bclk = I2S_SCK,
                .ws = I2S_WS,
                .din = I2S_SD,
                .dout = I2S_GPIO_UNUSED,
                .invert_flags =
                    {
                        .mclk_inv = false,
                        .bclk_inv = false,
                        .ws_inv = false,
                    },
            },
    };

    ESP_ERROR_CHECK(i2s_channel_init_std_mode(rx_hdl, &std_cfg));
    ESP_ERROR_CHECK(i2s_channel_enable(rx_hdl));
    ESP_LOGI(TAG, "--- i2s init DONE ---");
}

void app_main() {
    ESP_LOGI(TAG, "--- entering app main ---");

    init_mic();

    const size_t chunk = 512 * sizeof(int32_t);
    int32_t *buf = malloc(chunk);
    size_t bytes = 0;

    while (1) {
        esp_err_t res =
            i2s_channel_read(rx_hdl, buf, chunk, &bytes, portMAX_DELAY);

        if (res == ESP_OK && bytes > 0) {
            int16_t max = 0;
            // samples are int16_t each
            int sample_cnt = bytes / sizeof(int32_t);

            // amplitude
            for (int i = 0; i < sample_cnt; i++) {
                int16_t sample = (int16_t)(buf[i] >> 14);
                int16_t absv = abs(sample);
                if (absv > max)
                    max = absv;
            }

            int len = max / 500;
            char bar[65];
            int j;
            for (j = 0; j < len && j < 64; j++)
                bar[j] = '#';
            bar[j] = '\0';

            printf("Lvl: [%-64s] | MAX: %d\n", bar, max);
        }

        vTaskDelay(pdMS_TO_TICKS(10));
    }
}
