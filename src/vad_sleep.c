#include "vad_sleep.h"
#include "driver/rtc_io.h"
#include "esp_log.h"
#include "esp_sleep.h"

static const char *TAG = "vad_sleep";

bool vad_sleep_wakeup_by_sound(void) {
    rtc_gpio_hold_dis(SLEEP_LED_GPIO);
    rtc_gpio_set_level(SLEEP_LED_GPIO, 0);
    rtc_gpio_deinit(SLEEP_LED_GPIO);

    esp_sleep_wakeup_cause_t cause = esp_sleep_get_wakeup_cause();
    ESP_LOGI(TAG, "wakeup cause: %d", cause);
    return cause == ESP_SLEEP_WAKEUP_EXT1;
}

void vad_sleep_enter(void) {
    // led: light, hold
    rtc_gpio_init(SLEEP_LED_GPIO);
    rtc_gpio_set_direction(SLEEP_LED_GPIO, RTC_GPIO_MODE_OUTPUT_ONLY);
    rtc_gpio_set_level(SLEEP_LED_GPIO, 1);
    rtc_gpio_hold_en(SLEEP_LED_GPIO); // фиксируем HIGH в deep sleep

    // sound sensor:
    //   ext1, active low logic, 10 kOhm ext pullup on sensor
    // RTC_PERIPH off,
    //   using hold for pin state
    rtc_gpio_init(SOUND_GPIO);
    rtc_gpio_set_direction(SOUND_GPIO, RTC_GPIO_MODE_INPUT_ONLY);
    rtc_gpio_pullup_dis(SOUND_GPIO);
    rtc_gpio_pulldown_dis(SOUND_GPIO);

    ESP_ERROR_CHECK(esp_sleep_enable_ext1_wakeup(1ULL << SOUND_GPIO,
                                                 ESP_EXT1_WAKEUP_ANY_LOW));

    ESP_LOGI(TAG, "entering deep sleep, LED on GPIO%d is ON", SLEEP_LED_GPIO);
    esp_deep_sleep_start();
}
