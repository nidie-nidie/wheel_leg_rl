#ifndef PACE_CRC_H
#define PACE_CRC_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

uint32_t pace_crc32_iso_hdlc(const uint8_t *data, size_t length);

#ifdef __cplusplus
}
#endif

#endif
