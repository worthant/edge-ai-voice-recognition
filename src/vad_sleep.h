#pragma once
#include "esp_sleep.h"
#include <stdbool.h>

#define SOUND_GPIO GPIO_NUM_11
#define SLEEP_LED_GPIO GPIO_NUM_12

// true if woke up from sound sensor
bool vad_sleep_wakeup_by_sound(void);

// setup ext1, led; go to deep sleep
void vad_sleep_enter(void);
