/*
 * display.c — high-level display for KWS pipeline.
 * Calls st7789 for all rendering.
 */

#include "display.h"
#include "st7789.h"
#include <stdio.h>
#include <string.h>

/* Draw a string at pixel position with given scale */
static void draw_text(int x, int y, const char *str,
                      uint16_t fg, uint16_t bg, int scale) {
    int char_w = 8 * scale;
    while (*str) {
        st7789_draw_char(x, y, *str, fg, bg, scale);
        x += char_w;
        str++;
    }
}

/* Center text horizontally */
static int center_x(const char *str, int scale) {
    int w = strlen(str) * 8 * scale;
    int x = (ST7789_WIDTH - w) / 2;
    return x < 0 ? 0 : x;
}

esp_err_t display_init(void) {
    esp_err_t r = st7789_init();
    if (r != ESP_OK) return r;
    st7789_fill_screen(DISP_BLACK);
    return ESP_OK;
}

void display_fsm(const char *state, const char *detail,
                 uint16_t bg, uint16_t fg) {
    st7789_fill_screen(bg);

    /* State name — scale 3 (24x48), centered */
    draw_text(center_x(state, 3), 80, state, fg, bg, 3);

    /* Detail — scale 2 (16x32), below */
    if (detail && detail[0]) {
        draw_text(center_x(detail, 2), 150, detail, fg, bg, 2);
    }
}

void display_detection(const char *word, float score, int latency_ms) {
    st7789_fill_screen(DISP_DKGREEN);

    /* Word — scale 4 (32x64) */
    draw_text(center_x(word, 4), 30, word, DISP_WHITE, DISP_DKGREEN, 4);

    /* Score — scale 2 */
    char buf[32];
    snprintf(buf, sizeof(buf), "score: %.2f", score);
    draw_text(center_x(buf, 2), 120, buf, DISP_WHITE, DISP_DKGREEN, 2);

    /* Latency — scale 2 */
    snprintf(buf, sizeof(buf), "%dms", latency_ms);
    draw_text(center_x(buf, 2), 170, buf, DISP_GREEN, DISP_DKGREEN, 2);
}

void display_progress(int percent, uint16_t color) {
    if (percent < 0) percent = 0;
    if (percent > 100) percent = 100;

    int bar_x = 20, bar_y = 200, bar_w = 200, bar_h = 20;

    st7789_fill_rect(bar_x, bar_y, bar_w, bar_h, DISP_GRAY);

    int filled = bar_w * percent / 100;
    if (filled > 0) {
        st7789_fill_rect(bar_x, bar_y, filled, bar_h, color);
    }
}
