#include "ws2812_led.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "led_strip.h"

#define LED_GPIO    21
#define WN_COLORS   8

static const char *TAG = "ws2812";
static led_strip_handle_t led_hdl;
static esp_timer_handle_t blink_timer;

static const uint8_t model_colors[WN_COLORS][3] = {
    {255, 0,   0},    // 0: red
    {0,   255, 0},    // 1: green
    {0,   0,   255},  // 2: blue
    {128, 0,   255},  // 3: violet
    {0,   255, 255},  // 4: cyan
    {255, 128, 0},    // 5: orange
    {255, 255, 0},    // 6: yellow
    {255, 0,   255},  // 7: purple
};

static void blink_timeout_cb(void *arg) {
    ws2812_clear();
}

esp_err_t ws2812_init(void) {
    led_strip_config_t strip_cfg = {
        .strip_gpio_num         = LED_GPIO,
        .max_leds               = 1,
        .led_model              = LED_MODEL_WS2812,
        .color_component_format = LED_STRIP_COLOR_COMPONENT_FMT_GRB,
        .flags.invert_out       = false,
    };
    led_strip_rmt_config_t rmt_cfg = {
        .clk_src         = RMT_CLK_SRC_DEFAULT,
        .resolution_hz   = 10 * 1000 * 1000,
        .mem_block_symbols = 64,
        .flags.with_dma  = false,
    };
    ESP_ERROR_CHECK(led_strip_new_rmt_device(&strip_cfg, &rmt_cfg, &led_hdl));
    led_strip_clear(led_hdl);

    const esp_timer_create_args_t timer_args = {
        .callback = blink_timeout_cb,
        .name     = "led_off",
    };
    ESP_ERROR_CHECK(esp_timer_create(&timer_args, &blink_timer));

    ESP_LOGI(TAG, "init ok, GPIO%d", LED_GPIO);
    return ESP_OK;
}

void ws2812_blink(uint8_t r, uint8_t g, uint8_t b) {
    led_strip_set_pixel(led_hdl, 0, r, g, b);
    led_strip_refresh(led_hdl);
    esp_timer_stop(blink_timer);
    esp_timer_start_once(blink_timer, 150 * 1000);
}

void ws2812_blink_model(int model_idx) {
    int i = model_idx % WN_COLORS;
    ws2812_blink(model_colors[i][0], model_colors[i][1], model_colors[i][2]);
}

void ws2812_clear(void) {
    led_strip_clear(led_hdl);
}
