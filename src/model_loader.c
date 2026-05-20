#include "model_loader.h"
#include "esp_log.h"
#include "esp_partition.h"

static const char *TAG = "model_loader";

/* Custom subtype 0x40 — must match partitions.csv */
#define MODEL_PARTITION_SUBTYPE 0x40

esp_err_t model_loader_mmap(const void **out_ptr, size_t *out_size) {
    const esp_partition_t *part = esp_partition_find_first(
        ESP_PARTITION_TYPE_DATA, MODEL_PARTITION_SUBTYPE, "model");
    if (!part) {
        ESP_LOGE(TAG, "'model' partition not found");
        return ESP_ERR_NOT_FOUND;
    }

    const void *mapped_ptr = NULL;
    esp_partition_mmap_handle_t handle;
    esp_err_t err = esp_partition_mmap(
        part, 0, part->size, ESP_PARTITION_MMAP_DATA, &mapped_ptr, &handle);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "mmap failed: %s", esp_err_to_name(err));
        return err;
    }

    ESP_LOGI(TAG, "model partition mapped: addr=%p offset=0x%x size=%u KB",
             mapped_ptr, (unsigned)part->address, (unsigned)part->size / 1024);

    *out_ptr = mapped_ptr;
    *out_size = part->size;
    return ESP_OK;
}
