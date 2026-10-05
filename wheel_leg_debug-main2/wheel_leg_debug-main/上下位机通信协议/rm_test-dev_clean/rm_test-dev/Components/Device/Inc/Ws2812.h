#ifndef WS2812_H
#define WS2812_H

#include "main.h"

#define WS2812_SPI_UNIT hspi6
extern SPI_HandleTypeDef WS2812_SPI_UNIT;

void WS2812_Ctrl(uint8_t r, uint8_t g, uint8_t b);
void WS2812_Show_ARGB(uint32_t aRGB);
#endif
