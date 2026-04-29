/*
 * display.h — high-level display for KWS pipeline visualization.
 *
 * Uses st7789 driver underneath. Provides:
 *   - FSM state display with colored backgrounds
 *   - Detection result with score/latency
 *   - Recording progress bar
 */

#pragma once

#include "esp_err.h"
#include <stdint.h>

/* Colors (RGB565) */
#define DISP_BLACK 0x0000
#define DISP_WHITE 0xFFFF
#define DISP_RED 0xF800
#define DISP_GREEN 0x07E0
#define DISP_BLUE 0x001F
#define DISP_CYAN 0x07FF
#define DISP_YELLOW 0xFFE0
#define DISP_ORANGE 0xFD20
#define DISP_GRAY 0x7BEF
#define DISP_PURPLE 0x780F
#define DISP_DKGREEN 0x0400

esp_err_t display_init(void);

/* Full-screen FSM state: big title + smaller detail line */
void display_fsm(const char *state, const char *detail, uint16_t bg,
                 uint16_t fg);

/* Detection result: huge word + score + latency */
void display_detection(const char *word, float score, int latency_ms);

/* Progress bar at bottom of screen (0-100%) */
void display_progress(int percent, uint16_t color);
