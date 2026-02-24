#include "i2s_mic.h"
#include "driver/i2s_std.h"
#include "esp_check.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "portmacro.h"

static const char *TAG = "i2s_mic";
static i2s_chan_handle_t rx_hdl;

// inmp441 mems i2s mic
esp_err_t i2s_mic_init(void) {
    ESP_LOGI(TAG, "--- i2s init START ---");

    i2s_chan_config_t chn_cfg =
        I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
    ESP_RETURN_ON_ERROR(i2s_new_channel(&chn_cfg, NULL, &rx_hdl), TAG,
                        "new channel failed");

    i2s_std_config_t std_cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(I2S_MIC_SAMPLE_RATE),
        // inmp441 gives 24-bit samples
        // but i2s only has 16 or 32
        // so we receive 32bit
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

    ESP_RETURN_ON_ERROR(i2s_channel_init_std_mode(rx_hdl, &std_cfg), TAG,
                        "init std mode failed");
    ESP_RETURN_ON_ERROR(i2s_channel_enable(rx_hdl), TAG, "enable failed");

    ESP_LOGI(TAG, "--- i2s init DONE ---");
    return ESP_OK;
}

esp_err_t i2s_mic_read(int32_t *buf, size_t buf_size, size_t *bytes_read) {
    return i2s_channel_read(rx_hdl, buf, buf_size, bytes_read, portMAX_DELAY);
}

void i2s_mic_demo(void) {
    const size_t chunk = I2S_MIC_CHUNK_SAMPLES * sizeof(int32_t);
    int32_t *buf = malloc(chunk);
    size_t bytes = 0;

    while (1) {
        if (i2s_mic_read(buf, chunk, &bytes) == ESP_OK && bytes > 0) {
            int16_t max = 0;
            int n = bytes / sizeof(int32_t);
            for (int i = 0; i < n; i++) {
                // 32 - 24 = 8, so the shift should be >>8
                // but philips adds 1 tic delay and to detect
                // level properly it was experimentally deduced by me
                // to make >>14 shift
                int16_t s = (int16_t)(buf[i] >> 14);
                int16_t a = s < 0 ? -s : s;
                if (a > max)
                    max = a;
            }
            int len = max / 500;
            char bar[65];
            int j;
            for (j = 0; j < len && j < 64; j++)
                bar[j] = '#';
            bar[j] = '\0';
            printf("Lvl: [%-64s] | MAX: %d\n", bar, max);
        }
        vTaskDelay(pdMS_TO_TICKS(100));
    }
}
