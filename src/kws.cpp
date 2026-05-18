/*
 * KWS inference using TFLite Micro on ESP32-S3.
 *
 * Loads the INT8 DS-CNN model exported by export_to_c.py.
 * Input:  float[49][10] MFCC → quantized to int8 using model's scale/zp
 * Output: int8[12] logits    → dequantized, argmax
 */

#include "kws.h"
#include "model_data.h" /* g_model_data, g_model_data_size from export_to_c.py */

#include "esp_cpu.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_memory_utils.h"
#include "esp_spiffs.h"
#include "esp_timer.h"
#include "kws_profiler.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include <cmath>
#include <cstring>
#include <new>

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

/* Profiler for benchmark mode. Lives in .bss, allocated once. */
static kws::LayerProfiler g_profiler;
static bool g_profiler_attached = false;

/* Buffers for placement-new of interpreter (avoids heap fragmentation
 * when we rebuild the interpreter to attach the profiler). */
alignas(tflite::MicroInterpreter) static uint8_t
    g_interp_buf[sizeof(tflite::MicroInterpreter)];

static const char *label_names[KWS_NUM_CLASSES] = {
    "yes", "no",  "up",   "down", "left",      "right",
    "on",  "off", "stop", "go",   "_silence_", "_unknown_"};

extern "C" {

esp_err_t kws_init(void) {
    ESP_LOGI(TAG, "build sanity: icache=%d kB dcache=%d kB",
             CONFIG_ESP32S3_INSTRUCTION_CACHE_SIZE / 1024,
             CONFIG_ESP32S3_DATA_CACHE_SIZE / 1024);

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

    size_t largest_internal =
        heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL);
    size_t free_internal = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
    ESP_LOGI(TAG,
             "before arena: free_internal=%uKB largest_block=%uKB needed=%uKB",
             (unsigned)free_internal / 1024, (unsigned)largest_internal / 1024,
             (unsigned)kArenaSize / 1024);

    /* 3. Tensor arena in internal RAM (esp-nn simd cores don't work with psram
     * directly) */
    tensor_arena = (uint8_t *)heap_caps_malloc(kArenaSize, MALLOC_CAP_INTERNAL |
                                                               MALLOC_CAP_8BIT);
    if (!tensor_arena) {
        ESP_LOGE(TAG, "arena alloc %d failed", kArenaSize);
        return ESP_ERR_NO_MEM;
    }

    /* 4. Build interpreter — placement-new into static buffer so we can
     * rebuild later (with profiler attached) without heap churn. */
    interpreter = new (g_interp_buf)
        tflite::MicroInterpreter(model, resolver, tensor_arena, kArenaSize);

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
    ESP_LOGI(TAG, "model size: %u bytes", g_model_data_size);

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

    ESP_LOGI(TAG, "tensor diagnostics:");
    for (size_t i = 0; i < interpreter->inputs_size(); i++) {
        TfLiteTensor *t = interpreter->input(i);
        ESP_LOGI(TAG, "  input[%zu]: data=%p bytes=%zu (internal=%d psram=%d)",
                 i, t->data.raw, t->bytes, esp_ptr_internal(t->data.raw),
                 esp_ptr_external_ram(t->data.raw));
    }
    ESP_LOGI(TAG, "  arena: %p (internal=%d psram=%d)", tensor_arena,
             esp_ptr_internal(tensor_arena),
             esp_ptr_external_ram(tensor_arena));
    ESP_LOGI(TAG, "  model_data: %p (internal=%d psram=%d flash=%d)",
             g_model_data, esp_ptr_internal(g_model_data),
             esp_ptr_external_ram(g_model_data), esp_ptr_in_drom(g_model_data));

    /* Invoke */
    uint32_t start = esp_cpu_get_cycle_count();
    interpreter->Invoke();
    uint32_t end = esp_cpu_get_cycle_count();
    ESP_LOGI(TAG, "invoke cycles: %u (%.1f ms @ 240 MHz)",
             (unsigned)(end - start), (end - start) / 240000.0f);

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

esp_err_t kws_benchmark(int num_runs, int warmup_runs) {
    if (!interpreter || !input_tensor) {
        ESP_LOGE(TAG, "benchmark: not initialized");
        return ESP_ERR_INVALID_STATE;
    }

    /* Lazy-init profiler event buffer in PSRAM. */
    if (!g_profiler.init(3200)) {
        return ESP_ERR_NO_MEM;
    }

    /* Save current input tensor contents — we'll restore them after rebuild. */
    size_t input_bytes = input_tensor->bytes;
    int8_t *saved_input =
        (int8_t *)heap_caps_malloc(input_bytes, MALLOC_CAP_INTERNAL);
    if (!saved_input) {
        ESP_LOGE(TAG, "save input alloc failed");
        return ESP_ERR_NO_MEM;
    }
    memcpy(saved_input, input_tensor->data.int8, input_bytes);

    /* Rebuild interpreter with profiler attached. We reuse the same model,
     * resolver, and arena — only the interpreter object is recreated. */
    if (!g_profiler_attached) {
        ESP_LOGI(TAG, "attaching profiler, rebuilding interpreter");

        /* Op resolver must outlive the interpreter. Same as in kws_init. */
        static tflite::MicroMutableOpResolver<8> bench_resolver;
        static bool bench_resolver_inited = false;
        if (!bench_resolver_inited) {
            bench_resolver.AddConv2D();
            bench_resolver.AddDepthwiseConv2D();
            bench_resolver.AddFullyConnected();
            bench_resolver.AddMean();
            bench_resolver.AddReshape();
            bench_resolver.AddQuantize();
            bench_resolver.AddDequantize();
            bench_resolver.AddSoftmax();
            bench_resolver_inited = true;
        }

        /* Destroy old interpreter, build new one in same buffer with profiler.
         */
        interpreter->~MicroInterpreter();
        interpreter = new (g_interp_buf)
            tflite::MicroInterpreter(model, bench_resolver, tensor_arena,
                                     kArenaSize, nullptr, &g_profiler);

        if (interpreter->AllocateTensors() != kTfLiteOk) {
            ESP_LOGE(TAG, "rebuild AllocateTensors failed");
            free(saved_input);
            return ESP_FAIL;
        }
        input_tensor = interpreter->input(0);
        output_tensor = interpreter->output(0);
        g_profiler_attached = true;
    }

    /* Restore input (rebuild may have zeroed the arena). */
    memcpy(input_tensor->data.int8, saved_input, input_bytes);
    free(saved_input);

    ESP_LOGI(TAG, "benchmark: warmup=%d runs=%d", warmup_runs, num_runs);

    /* Warmup. Profiler records but we don't care — we'll see events anyway
     * but begin_run() bookkeeping isn't called, so they get run_id=-1. */
    for (int i = 0; i < warmup_runs; i++) {
        if (interpreter->Invoke() != kTfLiteOk) {
            ESP_LOGE(TAG, "warmup invoke %d failed", i);
            return ESP_FAIL;
        }
    }

    /* Actual measurement. begin_run() bumps run counter & resets op index. */
    int64_t t_total_start = esp_timer_get_time();
    for (int i = 0; i < num_runs; i++) {
        g_profiler.begin_run();
        if (interpreter->Invoke() != kTfLiteOk) {
            ESP_LOGE(TAG, "bench invoke %d failed", i);
            return ESP_FAIL;
        }
    }
    int64_t t_total = esp_timer_get_time() - t_total_start;

    ESP_LOGI(TAG, "benchmark done: total=%lldms avg_invoke=%lldus events=%d",
             (long long)(t_total / 1000), (long long)(t_total / num_runs),
             g_profiler.num_events());

    /* Dump CSV to SPIFFS. */
    FILE *f = fopen("/spiffs/profile.csv", "w");
    if (!f) {
        ESP_LOGE(TAG, "fopen /spiffs/profile.csv failed");
        return ESP_FAIL;
    }
    /* Header line with metadata as CSV comment. */
    fprintf(f,
            "# kws_profile runs=%d warmup=%d arena_used=%u "
            "total_us=%lld avg_invoke_us=%lld\n",
            num_runs, warmup_runs, (unsigned)interpreter->arena_used_bytes(),
            (long long)t_total, (long long)(t_total / num_runs));
    g_profiler.dump_csv(f);
    fclose(f);

    ESP_LOGI(TAG, "profile written: /spiffs/profile.csv");
    return ESP_OK;
}

} /* extern "C" */
