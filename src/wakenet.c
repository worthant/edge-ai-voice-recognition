#include "wakenet.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "i2s_mic.h"
#include <stdlib.h>
#include <string.h>

static const char *TAG = "wakenet";

/* private helpers */

static esp_err_t model_init(wn_model_t *m, const char *name) {
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

static void model_log(wn_model_t *m) {
    int word_num = m->ops->get_word_num(m->model);
    int sr = m->ops->get_samp_rate(m->model);
    ESP_LOGI(TAG, "model '%s': %d word(s), %dHz", m->name, word_num, sr);
    for (int i = 1; i <= word_num; i++) {
        ESP_LOGI(TAG, "  [%d] '%s' thr=%.4f", i,
                 m->ops->get_word_name(m->model, i),
                 m->ops->get_det_threshold(m->model, i));
    }
}

static void model_destroy(wn_model_t *m) {
    if (m->model)
        m->ops->destroy(m->model);
    free(m->name);
    m->model = NULL;
    m->name = NULL;
}

/* public api */

esp_err_t wakenet_init(wakenet_t *wn) {
    memset(wn, 0, sizeof(wakenet_t));

    srmodel_list_t *models = esp_srmodel_init("model");
    if (!models) {
        ESP_LOGE(TAG, "srmodel_init failed");
        return ESP_FAIL;
    }

    for (int i = 0; i < models->num && wn->count < WN_MAX_MODELS; i++) {
        const char *name = models->model_name[i];
        if (!strstr(name, ESP_WN_PREFIX))
            continue;
        if (model_init(&wn->models[wn->count], name) == ESP_OK) {
            model_log(&wn->models[wn->count]);
            wn->count++;
        }
    }

    if (wn->count == 0) {
        ESP_LOGE(TAG, "no models loaded");
        return ESP_FAIL;
    }

    wn->chunk_sz = wn->models[0].ops->get_samp_chunksize(wn->models[0].model);
    ESP_LOGI(TAG, "%d model(s) loaded, chunk=%d samples", wn->count,
             wn->chunk_sz);
    return ESP_OK;
}

void wakenet_run_window(wakenet_t *wn, uint32_t ms, wakenet_detect_cb_t cb) {
    int16_t *buf =
        heap_caps_malloc(wn->chunk_sz * sizeof(int16_t), MALLOC_CAP_INTERNAL);
    if (!buf) {
        ESP_LOGE(TAG, "buf alloc failed");
        return;
    }

    int64_t deadline = esp_timer_get_time() + (int64_t)ms * 1000;
    size_t got = 0;

    ESP_LOGI(TAG, "listening for %lu ms", (unsigned long)ms);

    while (esp_timer_get_time() < deadline) {
        if (i2s_mic_read_s16(buf, wn->chunk_sz, &got) != ESP_OK)
            continue;
        if ((int)got < wn->chunk_sz)
            continue;

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

            if (cb)
                cb(m, word ? word : "unknown");
        }
    }

    free(buf);
    ESP_LOGI(TAG, "window closed");
}

void wakenet_deinit(wakenet_t *wn) {
    for (int i = 0; i < wn->count; i++)
        model_destroy(&wn->models[i]);
    wn->count = 0;
}
