/*
 * st7789.c — low-level ST7789 240x240 SPI LCD driver.
 */

#include "st7789.h"
#include "driver/gpio.h"
#include "driver/spi_master.h"
#include "esp_check.h"
#include "esp_log.h"
#include "font8x16.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include <string.h>

static const char *TAG = "st7789";

#define PIN_SCLK GPIO_NUM_7
#define PIN_MOSI GPIO_NUM_8
#define PIN_RST GPIO_NUM_9
#define PIN_DC GPIO_NUM_10
#define PIN_CS -1

static spi_device_handle_t spi;

/* --- SPI helpers --- */

static void spi_write(const uint8_t *data, int len) {
    if (len == 0)
        return;
    spi_transaction_t t = {.length = len * 8, .tx_buffer = data};
    spi_device_transmit(spi, &t);
}

static void cmd(uint8_t c) {
    gpio_set_level(PIN_DC, 0);
    spi_write(&c, 1);
}

static void data8(uint8_t d) {
    gpio_set_level(PIN_DC, 1);
    spi_write(&d, 1);
}

static void data16(uint16_t d) {
    uint8_t buf[2] = {d >> 8, d & 0xFF};
    gpio_set_level(PIN_DC, 1);
    spi_write(buf, 2);
}

static void set_window(int x0, int y0, int x1, int y1) {
    cmd(0x2A);
    data16(x0);
    data16(x1);
    cmd(0x2B);
    data16(y0);
    data16(y1);
    cmd(0x2C);
}

/* --- Init --- */

static void hw_reset(void) {
    gpio_set_level(PIN_RST, 0);
    vTaskDelay(pdMS_TO_TICKS(20));
    gpio_set_level(PIN_RST, 1);
    vTaskDelay(pdMS_TO_TICKS(120));
}

static void init_registers(void) {
    cmd(0x01);
    vTaskDelay(pdMS_TO_TICKS(150));
    cmd(0x11);
    vTaskDelay(pdMS_TO_TICKS(120));

    cmd(0xB2);
    data8(0x0C);
    data8(0x0C);
    data8(0x00);
    data8(0x33);
    data8(0x33);
    cmd(0xB7);
    data8(0x35);
    cmd(0xBB);
    data8(0x19);
    cmd(0xC0);
    data8(0x2C);
    cmd(0xC2);
    data8(0x01);
    cmd(0xC3);
    data8(0x12);
    cmd(0xC4);
    data8(0x20);
    cmd(0xC6);
    data8(0x0F);
    cmd(0xD0);
    data8(0xA4);
    data8(0xA1);

    cmd(0x3A);
    data8(0x55);
    cmd(0x36);
    data8(0x00);
    cmd(0x21);
    cmd(0x29);
    vTaskDelay(pdMS_TO_TICKS(20));
}

esp_err_t st7789_init(void) {
    gpio_config_t io = {
        .pin_bit_mask = (1ULL << PIN_DC) | (1ULL << PIN_RST),
        .mode = GPIO_MODE_OUTPUT,
    };
    gpio_config(&io);

    spi_bus_config_t bus = {
        .mosi_io_num = PIN_MOSI,
        .miso_io_num = -1,
        .sclk_io_num = PIN_SCLK,
        .quadwp_io_num = -1,
        .quadhd_io_num = -1,
        .max_transfer_sz = ST7789_WIDTH * 16 * 2,
    };
    ESP_RETURN_ON_ERROR(spi_bus_initialize(SPI2_HOST, &bus, SPI_DMA_CH_AUTO),
                        TAG, "SPI bus init");

    spi_device_interface_config_t dev = {
        .clock_speed_hz = 40 * 1000 * 1000,
        .mode = 3,
        .spics_io_num = PIN_CS,
        .queue_size = 7,
    };
    ESP_RETURN_ON_ERROR(spi_bus_add_device(SPI2_HOST, &dev, &spi), TAG,
                        "SPI add device");

    hw_reset();
    init_registers();

    ESP_LOGI(TAG, "init ok (MOSI=%d SCLK=%d DC=%d RST=%d)", PIN_MOSI, PIN_SCLK,
             PIN_DC, PIN_RST);
    return ESP_OK;
}

/* --- Drawing primitives --- */

void st7789_fill_rect(int x, int y, int w, int h, uint16_t color) {
    if (x + w > ST7789_WIDTH)
        w = ST7789_WIDTH - x;
    if (y + h > ST7789_HEIGHT)
        h = ST7789_HEIGHT - y;
    if (w <= 0 || h <= 0)
        return;

    set_window(x, y, x + w - 1, y + h - 1);
    gpio_set_level(PIN_DC, 1);

    uint8_t rowbuf[ST7789_WIDTH * 2];
    for (int i = 0; i < w; i++) {
        rowbuf[i * 2] = color >> 8;
        rowbuf[i * 2 + 1] = color & 0xFF;
    }
    for (int row = 0; row < h; row++) {
        spi_write(rowbuf, w * 2);
    }
}

void st7789_fill_screen(uint16_t color) {
    st7789_fill_rect(0, 0, ST7789_WIDTH, ST7789_HEIGHT, color);
}

void st7789_draw_char(int x, int y, char c, uint16_t fg, uint16_t bg,
                      int scale) {
    if (c < 0x20 || c > 0x7E)
        c = '?';
    int idx = (c - 0x20) * 16;
    int w = 8 * scale, h = 16 * scale;
    if (x + w > ST7789_WIDTH || y + h > ST7789_HEIGHT)
        return;

    set_window(x, y, x + w - 1, y + h - 1);
    gpio_set_level(PIN_DC, 1);

    uint8_t rowbuf[8 * 4 * 2]; /* max scale=4 */
    for (int row = 0; row < 16; row++) {
        uint8_t bits = font8x16_data[idx + row];
        int pos = 0;
        for (int col = 0; col < 8; col++) {
            uint16_t color = (bits & (0x80 >> col)) ? fg : bg;
            for (int sx = 0; sx < scale; sx++) {
                rowbuf[pos++] = color >> 8;
                rowbuf[pos++] = color & 0xFF;
            }
        }
        for (int sy = 0; sy < scale; sy++) {
            spi_write(rowbuf, pos);
        }
    }
}
