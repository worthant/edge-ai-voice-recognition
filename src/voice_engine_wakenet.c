/*
 * voice_engine_wakenet.c — WakeNet9 backend.
 * Compiled only when USE_CUSTOM_KWS is NOT defined.
 */

#ifndef USE_CUSTOM_KWS

#include "voice_engine.h"
#include "wakenet.h"
#include "esp_log.h"

static wakenet_t wn;

esp_err_t voice_engine_init(void) { 
    ESP_LOGI("wn_voice_engine_init", "wakenet voice engine init");
    return wakenet_init(&wn); }

void voice_engine_run(uint32_t listen_ms, voice_detect_cb_t cb) {
    wakenet_run_window(&wn, listen_ms, cb);
}

void voice_engine_deinit(void) { wakenet_deinit(&wn); }

#endif /* !USE_CUSTOM_KWS */
