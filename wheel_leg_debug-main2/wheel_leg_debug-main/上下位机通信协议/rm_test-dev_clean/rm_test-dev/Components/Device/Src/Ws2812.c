#include "Ws2812.h"

#define WS2812_LowLevel 0xC0  // 0码 (即逻辑0)
#define WS2812_HighLevel 0xF0 // 1码 (即逻辑1)

void WS2812_Ctrl(uint8_t r, uint8_t g, uint8_t b)
{
    uint8_t txbuf[24];
    uint8_t res = 0;
    for (int i = 0; i < 8; i++)
    {
        txbuf[7 - i] = (((g >> i) & 0x01) ? WS2812_HighLevel : WS2812_LowLevel) >> 1;
        txbuf[15 - i] = (((r >> i) & 0x01) ? WS2812_HighLevel : WS2812_LowLevel) >> 1;
        txbuf[23 - i] = (((b >> i) & 0x01) ? WS2812_HighLevel : WS2812_LowLevel) >> 1;
    }
    HAL_SPI_Transmit(&WS2812_SPI_UNIT, &res, 0, 0xFFFF);
    while (WS2812_SPI_UNIT.State != HAL_SPI_STATE_READY)
        ;
    HAL_SPI_Transmit(&WS2812_SPI_UNIT, txbuf, 24, 0xFFFF);
    for (int i = 0; i < 100; i++)
    {
        HAL_SPI_Transmit(&WS2812_SPI_UNIT, &res, 1, 0xFFFF);
    }
}

void WS2812_Show_ARGB(uint32_t aRGB)
{
    uint8_t alpha;
    uint8_t r, g, b;

    // 1. 提取 Alpha 通道
    alpha = (aRGB & 0xFF000000) >> 24;

    // 2. 提取 RGB 并应用 Alpha 混合
    // 逻辑：(颜色 * Alpha) / 255
    // 这里的 >> 8 相当于除以 256 (快速近似除以255)，将结果从 16位 缩放回 8位

    r = (uint8_t)((((aRGB & 0x00FF0000) >> 16) * alpha) >> 8);
    g = (uint8_t)((((aRGB & 0x0000FF00) >> 8) * alpha) >> 8);
    b = (uint8_t)((((aRGB & 0x000000FF) >> 0) * alpha) >> 8);

    // 3. 调用底层驱动
    WS2812_Ctrl(r, g, b);
}