/*
 * KWS inference using TFLite Micro on ESP32-S3.
 *
 * Loads the INT8 DS-CNN model exported by export_to_c.py.
 * Input:  float[49][10] MFCC → quantized to int8 using model's scale/zp
 * Output: int8[12] logits    → dequantized, argmax
 */

#include "kws.h"
#include "model_data.h" /* g_model_data, g_model_data_size from export_to_c.py */

#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include <cmath>
#include <cstring>

static const char *TAG = "kws";

/* --- Tensor arena ---
 * DS-CNN-M with 172 filters × 6 blocks ≈ 300K params → ~300 KB tflite.
 * Tensor arena needs ~60-120 KB depending on intermediate activations.
 * Start with 100 KB; tune down using interpreter->arena_used_bytes(). */
static constexpr int kArenaSize = 140 * 1024;
static uint8_t *tensor_arena = nullptr;

static const tflite::Model *model = nullptr;
static tflite::MicroInterpreter *interpreter = nullptr;
static TfLiteTensor *input_tensor = nullptr;
static TfLiteTensor *output_tensor = nullptr;

static const char *label_names[KWS_NUM_CLASSES] = {
    "yes", "no",  "up",   "down", "left",      "right",
    "on",  "off", "stop", "go",   "_silence_", "_unknown_"};

extern "C" {

esp_err_t kws_init(void) {
    /* 1. Load model flatbuffer */
    model = tflite::GetModel(g_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "model schema %lu != expected %d",
                 (unsigned long)model->version(), TFLITE_SCHEMA_VERSION);
        return ESP_FAIL;
    }

    /* 2. Op resolver — register only what DS-CNN needs */
    static tflite::MicroMutableOpResolver<8> resolver;
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddFullyConnected();
    resolver.AddMean(); /* GlobalAveragePooling2D */
    resolver.AddReshape();
    resolver.AddQuantize();
    resolver.AddDequantize();
    resolver.AddSoftmax(); /* in case converter added it */

    /* 3. Tensor arena in internal RAM (esp-nn simd cores don't work with psram directly) */
    tensor_arena = (uint8_t *)heap_caps_malloc(kArenaSize, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (!tensor_arena) {
        ESP_LOGE(TAG, "arena alloc %d failed", kArenaSize);
        return ESP_ERR_NO_MEM;
    }

    /* 4. Build interpreter */
    static tflite::MicroInterpreter static_interp(model, resolver, tensor_arena,
                                                  kArenaSize);
    interpreter = &static_interp;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors failed");
        return ESP_FAIL;
    }

    ESP_LOGI(TAG, "free internal: %zu KB  free PSRAM: %zu KB",
         heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024,
         heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024);

    input_tensor = interpreter->input(0);
    output_tensor = interpreter->output(0);

    ESP_LOGI(TAG, "model loaded  arena_used=%zu/%d",
             interpreter->arena_used_bytes(), kArenaSize);
    ESP_LOGI(TAG, "input:  type=%d dims=[%d,%d,%d,%d] scale=%.6f zp=%d",
             (int)input_tensor->type, (int)input_tensor->dims->data[0],
             (int)input_tensor->dims->data[1], (int)input_tensor->dims->data[2],
             (int)input_tensor->dims->data[3], input_tensor->params.scale,
             (int)input_tensor->params.zero_point);
    ESP_LOGI(TAG, "output: type=%d dims=[%d,%d] scale=%.6f zp=%d",
             (int)output_tensor->type, (int)output_tensor->dims->data[0],
             (int)output_tensor->dims->data[1], output_tensor->params.scale,
             (int)output_tensor->params.zero_point);

    return ESP_OK;
}

esp_err_t kws_classify(const float mfcc[MFCC_NUM_FRAMES][MFCC_NUM_COEFFS],
                       kws_result_t *result) {
    if (!interpreter || !input_tensor || !output_tensor)
        return ESP_ERR_INVALID_STATE;

    /* Quantize float MFCC → int8 using input tensor's scale & zero_point */
    float in_scale = input_tensor->params.scale;
    int in_zp = input_tensor->params.zero_point;
    int8_t *in_data = input_tensor->data.int8;

    for (int f = 0; f < MFCC_NUM_FRAMES; f++) {
        for (int c = 0; c < MFCC_NUM_COEFFS; c++) {
            float val = mfcc[f][c];
            int q = (int)roundf(val / in_scale) + in_zp;
            if (q < -128)
                q = -128;
            if (q > 127)
                q = 127;
            in_data[f * MFCC_NUM_COEFFS + c] = (int8_t)q;
        }
    }

    /* Invoke */
    int64_t t0 = esp_timer_get_time();
    TfLiteStatus status = interpreter->Invoke();
    int64_t dt = esp_timer_get_time() - t0;

    if (status != kTfLiteOk) {
        ESP_LOGE(TAG, "Invoke failed");
        return ESP_FAIL;
    }
    ESP_LOGI(TAG, "invoke: %lld us", (long long)dt);

    /* Dequantize output → find argmax */
    float out_scale = output_tensor->params.scale;
    int out_zp = output_tensor->params.zero_point;
    int8_t *out_data = output_tensor->data.int8;

    int best_idx = 0;
    float best_val = -1e9f;
    for (int i = 0; i < KWS_NUM_CLASSES; i++) {
        float v = ((float)out_data[i] - out_zp) * out_scale;
        if (v > best_val) {
            best_val = v;
            best_idx = i;
        }
    }

    result->label = (kws_label_t)best_idx;
    result->score = best_val;

    ESP_LOGI(TAG, "result: %s (%.4f)", label_names[best_idx], best_val);
    return ESP_OK;
}

const char *kws_label_name(kws_label_t label) {
    if (label >= 0 && label < KWS_NUM_CLASSES)
        return label_names[label];
    return "???";
}

void kws_deinit(void) {
    /* interpreter is static, nothing to free there */
    if (tensor_arena) {
        heap_caps_free(tensor_arena);
        tensor_arena = nullptr;
    }
    interpreter = nullptr;
}

} /* extern "C" */
