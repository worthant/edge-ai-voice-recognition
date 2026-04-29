/*
 * st7789.h — low-level ST7789 240x240 SPI LCD driver.
 *
 * Handles: SPI init, hardware reset, controller init sequence,
 * pixel-level operations (fill rect, draw char).
 *
 * Pins (ESP32-S3-Zero):
 *   MOSI=GPIO8, SCLK=GPIO7, DC=GPIO10, RST=GPIO9, CS=tied to GND
 */

#pragma once

#include "esp_err.h"
#include <stdint.h>

#define ST7789_WIDTH 240
#define ST7789_HEIGHT 240

esp_err_t st7789_init(void);

void st7789_fill_rect(int x, int y, int w, int h, uint16_t color);
void st7789_fill_screen(uint16_t color);

/* Draw one character from built-in 8x16 font.
 * scale=1: 8x16, scale=2: 16x32, scale=3: 24x48, scale=4: 32x64 */
void st7789_draw_char(int x, int y, char c, uint16_t fg, uint16_t bg,
                      int scale);
