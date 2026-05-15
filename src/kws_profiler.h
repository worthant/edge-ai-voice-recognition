/*
 * Per-layer profiler for TFLite Micro inference.
 *
 * Events stored dynamically in PSRAM (allocated once via init()), so the
 * .bss footprint is just a few pointers. PSRAM access latency (~few hundred
 * ns) is negligible compared to ms-scale per-op times.
 *
 * Layer position in graph is preserved (op_index increments with each
 * BeginEvent call within a run), so all 6 PW-conv layers stay distinguishable.
 */

#pragma once

#ifdef __cplusplus

#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "tensorflow/lite/micro/micro_profiler_interface.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace kws {

class LayerProfiler : public tflite::MicroProfilerInterface {
  public:
    struct Event {
        int run_id;
        int op_index;
        const char *tag;
        int64_t start_us;
        int64_t end_us;
    };

    LayerProfiler() = default;
    ~LayerProfiler() override {
        if (events_)
            heap_caps_free(events_);
    }

    /* Allocate event buffer in PSRAM. Returns false on OOM. */
    bool init(int max_events) {
        if (events_)
            return true; /* already inited */
        max_events_ = max_events;
        events_ = (Event *)heap_caps_malloc(sizeof(Event) * max_events_,
                                            MALLOC_CAP_SPIRAM);
        if (!events_) {
            ESP_LOGE("layer_prof", "PSRAM alloc %d events failed", max_events_);
            return false;
        }
        memset(events_, 0, sizeof(Event) * max_events_);
        ESP_LOGI("layer_prof", "ok  events_max=%d size=%uKB", max_events_,
                 (unsigned)(sizeof(Event) * max_events_) / 1024);
        return true;
    }

    /* Call just before interpreter->Invoke(): bumps run, resets op index. */
    void begin_run() {
        current_run_++;
        op_index_in_run_ = 0;
    }

    uint32_t BeginEvent(const char *tag) override {
        if (!events_ || num_events_ >= max_events_)
            return 0;
        int idx = num_events_++;
        events_[idx].run_id = current_run_;
        events_[idx].op_index = op_index_in_run_++;
        events_[idx].tag = tag;
        events_[idx].start_us = esp_timer_get_time();
        events_[idx].end_us = 0;
        return (uint32_t)idx;
    }

    void EndEvent(uint32_t handle) override {
        if (!events_ || handle >= (uint32_t)num_events_)
            return;
        events_[handle].end_us = esp_timer_get_time();
    }

    void dump_csv(FILE *f) const {
        fprintf(f, "run_id,op_index,op_tag,ticks_us\n");
        for (int i = 0; i < num_events_; i++) {
            const auto &e = events_[i];
            int64_t dt = e.end_us - e.start_us;
            fprintf(f, "%d,%d,%s,%lld\n", e.run_id, e.op_index, e.tag,
                    (long long)dt);
        }
    }

    int num_events() const { return num_events_; }

  private:
    Event *events_ = nullptr;
    int max_events_ = 0;
    int num_events_ = 0;
    int current_run_ = -1;
    int op_index_in_run_ = 0;
};

} // namespace kws

#endif /* __cplusplus */
