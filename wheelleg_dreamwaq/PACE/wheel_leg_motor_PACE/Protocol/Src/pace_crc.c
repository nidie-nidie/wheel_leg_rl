#include "pace_crc.h"

uint32_t pace_crc32_iso_hdlc(const uint8_t *data, size_t length)
{
    uint32_t crc = 0xFFFFFFFFUL;
    size_t index;

    if (data == 0)
    {
        return 0U;
    }
    for (index = 0U; index < length; ++index)
    {
        uint8_t bit;
        crc ^= data[index];
        for (bit = 0U; bit < 8U; ++bit)
        {
            crc = (crc >> 1) ^ ((crc & 1U) ? 0xEDB88320UL : 0U);
        }
    }
    return crc ^ 0xFFFFFFFFUL;
}
