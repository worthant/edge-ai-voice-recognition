#include "esp_log.h"
#include "esp_wn_iface.h"
#include "esp_wn_models.h"
#include "i2s_mic.h"
#include "model_path.h"
#include <stdio.h>
#include <stdlib.h>

static const char *TAG = "main";

void app_main() {
    ESP_LOGI(TAG, "--- entering app main ---");

    srmodel_list_t *models = esp_srmodel_init("model");
    if (models == NULL) {
        ESP_LOGE(TAG,
                 "Failed to initialize model list. Check partition table!");
        return;
    }

    char *wn_name = esp_srmodel_filter(models, ESP_WN_PREFIX, "hiesp");
    if (wn_name == NULL) {
        ESP_LOGE(TAG, "Model 'hiesp' not found in Flash. Check menuconfig!");
        return;
    }
    ESP_LOGI(TAG, "--- found model: %s ---", wn_name);

    esp_wn_iface_t *wakenet = esp_wn_handle_from_name(wn_name);

    model_iface_data_t *model_data = wakenet->create(wn_name, DET_MODE_90);
    if (model_data == NULL) {
        ESP_LOGE(TAG, "Failed to create model instance. Not enough PSRAM?");
        return;
    }

    ESP_LOGI(TAG, "--- WakeNet init DONE ---");

    int chunk_size = wakenet->get_samp_chunksize(model_data);
    ESP_LOGI(TAG, "Model expects chunks of %d samples (int16_t)", chunk_size);

    ESP_ERROR_CHECK(i2s_mic_init());
    //i2s_mic_demo();
}
