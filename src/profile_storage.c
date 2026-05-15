/*
 * SPIFFS init for storing benchmark profile CSV.
 * Mounted at /spiffs, 1.5 MB partition, label "storage".
 */

#include "profile_storage.h"
#include "esp_log.h"
#include "esp_spiffs.h"

static const char *TAG = "prof_storage";

esp_err_t profile_storage_init(void) {
    esp_vfs_spiffs_conf_t conf = {
        .base_path = "/spiffs",
        .partition_label = "storage",
        .max_files = 4,
        .format_if_mount_failed = true,
    };

    esp_err_t r = esp_vfs_spiffs_register(&conf);
    if (r != ESP_OK) {
        ESP_LOGE(TAG, "spiffs mount failed: %s", esp_err_to_name(r));
        return r;
    }

    size_t total = 0, used = 0;
    if (esp_spiffs_info("storage", &total, &used) == ESP_OK) {
        ESP_LOGI(TAG, "spiffs ok  total=%u used=%u", (unsigned)total,
                 (unsigned)used);
    }
    return ESP_OK;
}
