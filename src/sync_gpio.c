#include "sync_gpio.h"
#include "driver/gpio.h"
#include "rom/ets_sys.h"

#define SYNC_PIN GPIO_NUM_1
#define PULSE_US 1000

esp_err_t sync_gpio_init(void) {
    gpio_config_t cfg = {
        .pin_bit_mask = 1ULL << SYNC_PIN,
        .mode = GPIO_MODE_OUTPUT,
        .pull_up_en = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_ENABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };
    esp_err_t r = gpio_config(&cfg);
    if (r == ESP_OK)
        gpio_set_level(SYNC_PIN, 0);
    return r;
}

/* my FSM is completely sequential,
 * so i can send 1ms pulse to signal
 * esp32-c3 logger a new state
 */
void sync_pulse(void) {
    gpio_set_level(SYNC_PIN, 1);
    ets_delay_us(PULSE_US);
    gpio_set_level(SYNC_PIN, 0);
}

void sync_reset(void) { gpio_set_level(SYNC_PIN, 0); }
