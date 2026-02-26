#include "esp_err.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_wn_iface.h"
#include "esp_wn_models.h"
#include "i2s_mic.h"
#include "led_strip.h"
#include "model_path.h"
#include <stdlib.h>
#include <string.h>

static const char *TAG = "main";

#define WN_MAX_MODELS 8
#define LED_GPIO 21

// rgb led hdl
static led_strip_handle_t led_hdl;

const uint8_t model_colors[WN_MAX_MODELS][3] = {
    {255, 0, 0},   // 0: red
    {0, 255, 0},   // 1: green
    {0, 0, 255},   // 2: blue
    {128, 0, 255}, // 4: violet
    {0, 255, 255}, // 5: cyan
    {255, 128, 0}, // 6: orange
    {255, 255, 0}, // 7: yellow
    {255, 0, 255}, // 8: purple
};

static void init_led(void) {
    led_strip_config_t strip_config = {
        .strip_gpio_num =
            LED_GPIO,  // The GPIO that connected to the LED strip's data line
        .max_leds = 1, // The number of LEDs in the strip,
        .led_model =
            LED_MODEL_WS2812, // LED strip model, it determines the bit timing
        .color_component_format =
            LED_STRIP_COLOR_COMPONENT_FMT_GRB, // The color component format is
                                               // G-R-B
        .flags = {
            .invert_out = false, // don't invert the output signal
        }};

    /// RMT backend specific configuration
    led_strip_rmt_config_t rmt_config = {
        .clk_src = RMT_CLK_SRC_DEFAULT,    // different clock source can lead to
                                           // different power consumption
        .resolution_hz = 10 * 1000 * 1000, // RMT counter clock frequency: 10MHz
        .mem_block_symbols =
            64, // the memory size of each RMT channel, in words (4 bytes)
        .flags = {
            .with_dma =
                false, // DMA feature is available on chips like ESP32-S3/P4
        }};
    ESP_ERROR_CHECK(
        led_strip_new_rmt_device(&strip_config, &rmt_config, &led_hdl));
    led_strip_clear(led_hdl); // Turn off on start
}

static void blink_led(int model_idx) {
    int idx = model_idx % WN_MAX_MODELS;
    led_strip_set_pixel(led_hdl, 0, model_colors[idx][0], model_colors[idx][1],
                        model_colors[idx][2]);
    led_strip_refresh(led_hdl);

    vTaskDelay(pdMS_TO_TICKS(150));

    led_strip_clear(led_hdl);
}

typedef struct {
    const esp_wn_iface_t *ops;
    model_iface_data_t *model;
    char *name;
} wn_model_t;

typedef struct {
    wn_model_t models[WN_MAX_MODELS];
    int count;
    int chunk_sz;
} wakenet;

static esp_err_t wn_model_init(wn_model_t *m, const char *name) {
    m->ops = esp_wn_handle_from_name(name);
    if (!m->ops) {
        ESP_LOGE(TAG, "no handle for '%s'", name);
        return ESP_FAIL;
    }

    m->model = m->ops->create(name, DET_MODE_90);
    if (!m->model) {
        ESP_LOGE(TAG, "create failed for '%s'", name);
        return ESP_ERR_NO_MEM;
    }

    m->name = strdup(name);
    return ESP_OK;
}

static void wn_model_destroy(wn_model_t *m) {
    if (m->model)
        m->ops->destroy(m->model);
    free(m->name);
    m->model = NULL;
    m->name = NULL;
}

static void wn_model_log(wn_model_t *m) {
    int word_num = m->ops->get_word_num(m->model);
    int sr = m->ops->get_samp_rate(m->model);
    float gain = m->ops->get_vol_gain(m->model, -26.0f);
    ESP_LOGI(TAG, "model '%s': %d word(s), %dHz, gain=%.1f", m->name, word_num,
             sr, gain);
    for (int i = 1; i <= word_num; i++) {
        ESP_LOGI(TAG, "  [%d] '%s' thr=%.4f", i,
                 m->ops->get_word_name(m->model, i),
                 m->ops->get_det_threshold(m->model, i));
    }
}

esp_err_t setup_wakenet(wakenet *wn) {
    memset(wn, 0, sizeof(wakenet));

    srmodel_list_t *models = esp_srmodel_init("model");
    if (!models) {
        ESP_LOGE(TAG, "srmodel_init failed");
        return ESP_FAIL;
    }

    for (int i = 0; i < models->num; i++) {
        const char *name = models->model_name[i];
        if (!strstr(name, ESP_WN_PREFIX))
            continue;

        if (wn->count >= WN_MAX_MODELS) {
            ESP_LOGW(TAG, "max models reached, skipping '%s'", name);
            continue;
        }

        esp_err_t ret = wn_model_init(&wn->models[wn->count], name);
        if (ret != ESP_OK)
            continue;

        wn_model_log(&wn->models[wn->count]);
        wn->count++;
    }

    if (wn->count == 0) {
        ESP_LOGE(TAG, "no models loaded");
        return ESP_FAIL;
    }

    // chunk_sz одинаковый для всех wn9 — берём у первой
    wn->chunk_sz = wn->models[0].ops->get_samp_chunksize(wn->models[0].model);
    ESP_LOGI(TAG, "%d model(s) loaded, chunk=%d", wn->count, wn->chunk_sz);
    return ESP_OK;
}

static void wakenet_loop(wakenet *wn) {
    int16_t *buf =
        heap_caps_malloc(wn->chunk_sz * sizeof(int16_t), MALLOC_CAP_INTERNAL);
    if (!buf) {
        ESP_LOGE(TAG, "buf alloc failed");
        return;
    }

    size_t got = 0;

    while (1) {
        if (i2s_mic_read_s16(buf, wn->chunk_sz, &got) != ESP_OK)
            continue;
        if ((int)got < wn->chunk_sz)
            continue;

        /* log audio level for debugging
        int frame = 0;
        if (++frame % 32 == 0) {
            int16_t max = 0;
            for (int i = 0; i < wn->chunk_sz; i++) {
                int16_t a = buf[i] < 0 ? -buf[i] : buf[i];
                if (a > max)
                    max = a;
            }
            ESP_LOGI(TAG, "audio level: %d", max);
        }
        */

        for (int m = 0; m < wn->count; m++) {
            wakenet_state_t state =
                wn->models[m].ops->detect(wn->models[m].model, buf);
            if (state != WAKENET_DETECTED)
                continue;

            int ch =
                wn->models[m].ops->get_triggered_channel(wn->models[m].model);
            char *word =
                wn->models[m].ops->get_word_name(wn->models[m].model, ch + 1);
            ESP_LOGW(TAG, ">>> DETECTED: '%s' <<<", word ? word : "unknown");
            blink_led(m);
        }
    }
}

void app_main(void) {
    ESP_LOGI(TAG, "start");

    init_led();

    static wakenet wn;
    ESP_ERROR_CHECK(setup_wakenet(&wn));
    ESP_ERROR_CHECK(i2s_mic_init());
    wakenet_loop(&wn);
}
